"""
model_v4/training_loop.py — v4 training loop for CUT + Transformer.

Adds a Transformer encoder and PatchNCE contrastive loss on top of the
baseline GAN. PatchNCE is computed on encoder features sampled at shared
spatial locations between real and generated images.

Data flow:
    real_A → G_AB → fake_B   (unstained → stained)
    real_B → G_BA → fake_A   (stained   → unstained)

Losses:
    Generator:     λ_gan * LSGAN + λ_nce * PatchNCE
    Discriminator: LSGAN — (D(real) − 1)² + D(fake)²
"""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from config import V4Config
from shared.data_loader import getDataLoader
from shared.metrics import MetricsCalculator
from shared.replay_buffer import ReplayBuffer
from shared.validation import save_images_with_title

from model_v4.generator import getGeneratorV4
from model_v4.discriminator import getDiscriminatorV4
from model_v4.patch_sampler import PatchSampler
from model_v4.nce_loss import PatchNCELoss


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    """Enable or disable gradient computation for all parameters in *module*."""
    for p in module.parameters():
        p.requires_grad = flag


def _global_grad_norm(parameters) -> float:
    """
    Compute the global L2 gradient norm across all parameters that have a grad.

    Equivalent to ``torch.nn.utils.clip_grad_norm_`` with no clipping, but
    returns a plain Python float for logging.
    """
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


def _make_lr_lambda(warmup: int, decay_start: int, total: int):
    """
    Return a LambdaLR multiplier function with linear warmup then linear decay.

    Schedule:
        [0, warmup)          — ramp from ~0 to 1
        [warmup, decay_start) — constant 1
        [decay_start, total)  — linear decay from 1 to 0

    Args:
        warmup:      Number of warmup epochs.
        decay_start: Epoch at which linear decay begins.
        total:       Total number of training epochs.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-8, (epoch + 1) / max(1, warmup))
        if epoch < decay_start:
            return 1.0
        remaining = total - decay_start
        if remaining <= 0:
            return 0.0
        return max(0.0, 1.0 - (epoch - decay_start) / remaining)

    return lr_lambda


def _lsgan_gen_loss(pred_fake: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: E[(D(fake) − 1)²].  Target is 1 (real label)."""
    return torch.mean((pred_fake - 1.0) ** 2)


def _lsgan_disc_loss(pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
    """
    LSGAN discriminator loss: E[(D(real) − 1)²] + E[D(fake)²].

    Real target is 1, fake target is 0.
    """
    loss_real = torch.mean((pred_real - 1.0) ** 2)
    loss_fake = torch.mean(pred_fake**2)
    return loss_real + loss_fake


def _run_validation_v4(
    epoch: int,
    G_AB: torch.nn.Module,
    G_BA: torch.nn.Module,
    test_loader,
    device: torch.device,
    save_dir: str,
    num_samples: int,
    calculator: MetricsCalculator,
    max_batches: int = 50,
    fid_max_samples: int = 200,
    fid_min_samples: int = 50,
    writer: SummaryWriter | None = None,
    is_test: bool = False,
) -> dict[str, float]:
    """
    Run one validation pass and optionally save comparison images.

    Iterates over up to *max_batches* batches from *test_loader*, computes
    per-batch SSIM and PSNR for both translation directions, then averages
    the results.  The first *num_samples* batches are also saved as side-by-
    side PNG grids under *save_dir*.

    Args:
        epoch:       Current epoch number (used for TensorBoard step and
                     image filenames).
        G_AB:        Generator mapping domain A → B.
        G_BA:        Generator mapping domain B → A.
        test_loader: DataLoader yielding ``{"A": tensor, "B": tensor}`` dicts.
        device:      Device on which to run inference.
        save_dir:    Directory where comparison images are written.
        num_samples: Number of image grids to save.
        calculator:  MetricsCalculator instance for SSIM/PSNR.
        max_batches: Maximum number of batches to evaluate.
        fid_max_samples: Maximum samples used for FID computation.
        fid_min_samples: Minimum samples required to compute FID; skipped
                     if fewer are available (unless ``is_test`` is True).
        writer:      Optional TensorBoard SummaryWriter.
        is_test:     If True, logs under the ``Testing`` prefix and always
                     attempts FID regardless of sample count.

    Returns:
        Dict with keys ``ssim_A``, ``psnr_A``, ``ssim_B``, ``psnr_B``
        (and optionally ``fid_A``, ``fid_B``) containing epoch-averaged
        metric values.
    """
    G_AB.eval()
    G_BA.eval()
    os.makedirs(save_dir, exist_ok=True)
    print(
        f"{'Testing' if is_test else 'Validation'}: Running on {min(len(test_loader), max_batches)} batches, saving {num_samples} sample grids to {save_dir}"
    )  # noqa: E501

    metrics = {"ssim_A": [], "psnr_A": [], "ssim_B": [], "psnr_B": []}
    real_A_list = []
    real_B_list = []
    fake_A_list = []
    fake_B_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)

            batch_metrics = calculator.evaluate_batch(real_A, real_B, fake_A, fake_B)
            for key, value in batch_metrics.items():
                metrics[key].append(float(value))

            real_A_list.append(real_A)
            real_B_list.append(real_B)
            fake_A_list.append(fake_A)
            fake_B_list.append(fake_B)

            if i < num_samples:
                row_A = torch.cat(
                    [real_A[:1], fake_B[:1], rec_A[:1], real_B[:1]], dim=0
                ).cpu()
                out_path_A = os.path.join(save_dir, f"image_{i + 1}_A.png")
                save_images_with_title(
                    row_A,
                    labels=["Real A", "Fake B", "Reconstructed A", "Real B"],
                    out_path=out_path_A,
                    value_range=(-1, 1),
                )

                row_B = torch.cat(
                    [real_B[:1], fake_A[:1], rec_B[:1], real_A[:1]], dim=0
                ).cpu()
                out_path_B = os.path.join(save_dir, f"image_{i + 1}_B.png")
                save_images_with_title(
                    row_B,
                    labels=["Real B", "Fake A", "Reconstructed B", "Real A"],
                    out_path=out_path_B,
                    value_range=(-1, 1),
                )
                if (i + 1) % 20 == 0:
                    print(f"Saved sample grid {i + 1}.")

    avg_metrics = {
        "ssim_A": float(sum(metrics["ssim_A"]) / max(1, len(metrics["ssim_A"]))),
        "psnr_A": float(sum(metrics["psnr_A"]) / max(1, len(metrics["psnr_A"]))),
        "ssim_B": float(sum(metrics["ssim_B"]) / max(1, len(metrics["ssim_B"]))),
        "psnr_B": float(sum(metrics["psnr_B"]) / max(1, len(metrics["psnr_B"]))),
    }

    # Optional FID (domain A and B) if enough samples are collected.
    fid_count = min(fid_max_samples, len(real_B_list))
    if (
        fid_count >= fid_min_samples
        or is_test
        or (fid_count < fid_min_samples and epoch % 10 == 0)
    ) and fid_count > 0:
        real_B_tensor = torch.cat(real_B_list[:fid_count])
        fake_B_tensor = torch.cat(fake_B_list[:fid_count])
        avg_metrics["fid_B"] = float(
            calculator.evaluate_fid(real_B_tensor, fake_B_tensor)
        )

    fid_count_A = min(fid_max_samples, len(real_A_list))
    if (
        fid_count_A >= fid_min_samples
        or is_test
        or (fid_count_A < fid_min_samples and epoch % 10 == 0)
    ) and fid_count_A > 0:
        real_A_tensor = torch.cat(real_A_list[:fid_count_A])
        fake_A_tensor = torch.cat(fake_A_list[:fid_count_A])
        avg_metrics["fid_A"] = float(
            calculator.evaluate_fid(real_A_tensor, fake_A_tensor)
        )

    prefix = "Testing" if is_test else "Validation"
    if writer is not None:
        writer.add_scalar(f"{prefix}/ssim_A", avg_metrics["ssim_A"], epoch)
        writer.add_scalar(f"{prefix}/psnr_A", avg_metrics["psnr_A"], epoch)
        writer.add_scalar(f"{prefix}/ssim_B", avg_metrics["ssim_B"], epoch)
        writer.add_scalar(f"{prefix}/psnr_B", avg_metrics["psnr_B"], epoch)
        if "fid_A" in avg_metrics:
            writer.add_scalar(f"{prefix}/fid_A", avg_metrics["fid_A"], epoch)
        if "fid_B" in avg_metrics:
            writer.add_scalar(f"{prefix}/fid_B", avg_metrics["fid_B"], epoch)

    print(
        f"{prefix} Metrics - SSIM_A: {avg_metrics['ssim_A']:.4f}, "
        f"SSIM_B: {avg_metrics['ssim_B']:.4f}, "
        f"PSNR_A: {avg_metrics['psnr_A']:.2f}, "
        f"PSNR_B: {avg_metrics['psnr_B']:.2f}"
    )
    if "fid_A" in avg_metrics:
        print(f"{prefix} FID_A: {avg_metrics['fid_A']:.2f}")
    if "fid_B" in avg_metrics:
        print(f"{prefix} FID_B: {avg_metrics['fid_B']:.2f}")

    G_AB.train()
    G_BA.train()
    return avg_metrics


def train_v4(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    cfg: Optional[V4Config] = None,
):
    """
    Train the v4 GAN baseline (A ↔ B, Transformer encoder + PatchNCE).

    Any argument that is not None overrides the corresponding field in *cfg*
    before training begins, allowing the caller to pass a pre-built config and
    still tweak individual values at call time.

    Args:
        epoch_size: Samples drawn per epoch (overrides cfg.training.epoch_size).
        num_epochs: Total training epochs (overrides cfg.training.num_epochs).
        model_dir:  Root output directory for checkpoints and logs
                    (overrides cfg.model_dir).
        val_dir:    Directory for validation image grids
                    (overrides cfg.val_dir).
        cfg:        V4Config instance.  Defaults to V4Config() if None.

    Returns:
        tuple: (history, G_AB, G_BA, D_A, D_B)
            - history: dict keyed by epoch number, each value is a dict
              keyed by batch index containing per-batch loss scalars.
            - G_AB, G_BA: trained generator modules (raw weights).
            - D_A, D_B:   trained discriminator modules.

        EMA weights (if ``cfg.training.use_ema`` is True) are saved in the
        checkpoint under ``ema_G_AB_state_dict`` / ``ema_G_BA_state_dict``
        but are not returned directly; use the checkpoint for inference.
    """
    if cfg is None:
        cfg = V4Config()

    if epoch_size is not None:
        cfg.training.epoch_size = epoch_size
    if num_epochs is not None:
        cfg.training.num_epochs = num_epochs
    if model_dir is not None:
        cfg.model_dir = model_dir
    if val_dir is not None:
        cfg.val_dir = val_dir
    tcfg = cfg.training
    dcfg = cfg.data
    mcfg = cfg.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    # Unpaired loader returns {"A": tensor, "B": tensor} dicts each iteration.
    train_loader, test_loader = getDataLoader(
        epoch_size=tcfg.epoch_size,
        image_size=dcfg.image_size,
        batch_size=dcfg.batch_size,
        num_workers=dcfg.num_workers,
        prefetch_factor=dcfg.prefetch_factor,
    )

    # ---- Models ----
    # G_AB: unstained → stained;  G_BA: stained → unstained.
    # D_A discriminates real/fake unstained; D_B discriminates real/fake stained.
    if mcfg.image_size != dcfg.image_size:
        print(
            f"[warn] V4 model image_size ({mcfg.image_size}) does not match "
            f"data image_size ({dcfg.image_size}); using data image_size."
        )
    G_AB = getGeneratorV4(
        input_nc=mcfg.input_nc,
        output_nc=mcfg.output_nc,
        base_channels=mcfg.base_channels,
        num_res_blocks=mcfg.num_res_blocks,
        use_transformer_encoder=mcfg.use_transformer_encoder,
        image_size=dcfg.image_size,
        patch_size=mcfg.patch_size,
        encoder_dim=mcfg.encoder_dim,
        encoder_depth=mcfg.encoder_depth,
        encoder_heads=mcfg.encoder_heads,
        encoder_mlp_ratio=mcfg.encoder_mlp_ratio,
        encoder_dropout=mcfg.encoder_dropout,
        use_gradient_checkpointing=mcfg.use_gradient_checkpointing,
        device=device,
        run_smoke_test=False,
    )
    G_BA = getGeneratorV4(
        input_nc=mcfg.output_nc,
        output_nc=mcfg.input_nc,
        base_channels=mcfg.base_channels,
        num_res_blocks=mcfg.num_res_blocks,
        use_transformer_encoder=mcfg.use_transformer_encoder,
        image_size=dcfg.image_size,
        patch_size=mcfg.patch_size,
        encoder_dim=mcfg.encoder_dim,
        encoder_depth=mcfg.encoder_depth,
        encoder_heads=mcfg.encoder_heads,
        encoder_mlp_ratio=mcfg.encoder_mlp_ratio,
        encoder_dropout=mcfg.encoder_dropout,
        use_gradient_checkpointing=mcfg.use_gradient_checkpointing,
        device=device,
        run_smoke_test=False,
    )
    D_A = getDiscriminatorV4(
        input_nc=mcfg.input_nc,
        base_channels=mcfg.disc_base_channels,
        n_layers=mcfg.disc_n_layers,
        device=device,
        run_smoke_test=False,
    )
    D_B = getDiscriminatorV4(
        input_nc=mcfg.output_nc,
        base_channels=mcfg.disc_base_channels,
        n_layers=mcfg.disc_n_layers,
        device=device,
        run_smoke_test=False,
    )

    # ---- Optimizers ----
    # Single Adam for both generators so their gradients accumulate together.
    optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=tcfg.lr,
        betas=(tcfg.beta1, tcfg.beta2),
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )

    # ---- LR schedulers ----
    lr_scheduler_G = None
    lr_scheduler_D_A = None
    lr_scheduler_D_B = None
    if tcfg.use_lr_schedule:
        lr_lambda = _make_lr_lambda(
            warmup=tcfg.lr_warmup_epochs,
            decay_start=tcfg.lr_decay_start_epoch,
            total=tcfg.num_epochs,
        )
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=lr_lambda
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=lr_lambda
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=lr_lambda
        )

    use_amp = tcfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- EMA ----
    ema_G_AB = None
    ema_G_BA = None
    if tcfg.use_ema:
        ema_G_AB = copy.deepcopy(G_AB).to(device)
        ema_G_BA = copy.deepcopy(G_BA).to(device)
        ema_G_AB.requires_grad_(False)
        ema_G_BA.requires_grad_(False)

    # ---- Metrics / PatchNCE ----
    # Shared SSIM/PSNR calculator reused across validation calls.
    metrics_calculator = MetricsCalculator(device=device)
    idt_loss = torch.nn.L1Loss()
    nce_layers = list(tcfg.nce_layers)
    max_layers = mcfg.encoder_depth if mcfg.use_transformer_encoder else 4
    nce_layers = [idx for idx in nce_layers if idx < max_layers]
    if not nce_layers:
        nce_layers = list(range(max_layers))
        print(
            f"[warn] nce_layers was empty after filtering; using all layers 0..{max_layers-1}"
        )
    patch_sampler = PatchSampler(num_patches=tcfg.nce_num_patches)
    nce_criterion = PatchNCELoss(
        temperature=tcfg.nce_temperature, proj_dim=tcfg.nce_proj_dim
    ).to(device)
    use_nce = tcfg.lambda_nce > 0.0

    # ---- Replay buffers ----
    replay_A = ReplayBuffer(tcfg.replay_buffer_size) if tcfg.use_replay_buffer else None
    replay_B = ReplayBuffer(tcfg.replay_buffer_size) if tcfg.use_replay_buffer else None

    # ---- Output dirs / TensorBoard ----
    # Fall back to a default path when model_dir was not supplied via cfg.
    model_dir = cfg.model_dir or os.path.join(
        "data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v4"
    )
    val_dir = cfg.val_dir or os.path.join(model_dir, "validation_images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    tb_dir = os.path.join(model_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    history = {}
    accumulate = max(1, tcfg.accumulate_grads)
    accum_count = 0

    if accumulate > 1:
        print(
            f"[train_v4] Gradient accumulation: {accumulate} "
            f"(effective batch {accumulate * dcfg.batch_size})"
        )

    for epoch in range(tcfg.num_epochs):
        print()
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_step = {}
        epoch_loss_G = 0.0
        epoch_loss_G_gan = 0.0
        epoch_loss_NCE = 0.0
        epoch_loss_Id = 0.0
        epoch_loss_G_AB = 0.0
        epoch_loss_G_BA = 0.0
        epoch_loss_D_A = 0.0
        epoch_loss_D_B = 0.0
        epoch_grad_norm = 0.0
        accum_count = 0

        writer.add_scalar("Epoch", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            if not (real_A.isfinite().all() and real_B.isfinite().all()):
                print(f"[warn] non-finite input at epoch {epoch+1} batch {i}, skipping")
                continue

            # -------------------------
            # Discriminator steps
            # -------------------------
            # Freeze generators so their weights are not updated here;
            # generate fakes under no_grad to avoid storing the graph.
            _set_requires_grad(D_A, True)
            _set_requires_grad(D_B, True)
            _set_requires_grad(G_AB, False)
            _set_requires_grad(G_BA, False)

            with torch.no_grad():
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)

            optimizer_D_A.zero_grad(set_to_none=True)
            fake_A_detached = (
                replay_A.push_and_pop(fake_A.detach())
                if replay_A is not None
                else fake_A.detach()
            )
            with autocast("cuda", enabled=False):
                pred_real_A = D_A(real_A.float())
                pred_fake_A = D_A(fake_A_detached.float())
                loss_D_A = _lsgan_disc_loss(pred_real_A, pred_fake_A)
            if not torch.isfinite(loss_D_A):
                print(
                    f"[warn] non-finite D_A loss at epoch {epoch+1} batch {i}, skipping"
                )
                optimizer_D_A.zero_grad(set_to_none=True)
                continue
            loss_D_A.backward()
            if tcfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(D_A.parameters(), tcfg.grad_clip_norm)
            optimizer_D_A.step()

            optimizer_D_B.zero_grad(set_to_none=True)
            fake_B_detached = (
                replay_B.push_and_pop(fake_B.detach())
                if replay_B is not None
                else fake_B.detach()
            )
            with autocast("cuda", enabled=False):
                pred_real_B = D_B(real_B.float())
                pred_fake_B = D_B(fake_B_detached.float())
                loss_D_B = _lsgan_disc_loss(pred_real_B, pred_fake_B)
            if not torch.isfinite(loss_D_B):
                print(
                    f"[warn] non-finite D_B loss at epoch {epoch+1} batch {i}, skipping"
                )
                optimizer_D_B.zero_grad(set_to_none=True)
                continue
            loss_D_B.backward()
            if tcfg.grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(D_B.parameters(), tcfg.grad_clip_norm)
            optimizer_D_B.step()

            # -------------------------
            # Generator step
            # -------------------------
            # Unfreeze generators, freeze discriminators so D gradients
            # are not computed during the generator backward pass.
            _set_requires_grad(D_A, False)
            _set_requires_grad(D_B, False)
            _set_requires_grad(G_AB, True)
            _set_requires_grad(G_BA, True)

            if accum_count == 0:
                optimizer_G.zero_grad(
                    set_to_none=True
                )  # reset at start of accumulation window

            with autocast("cuda", enabled=use_amp):
                fake_B, feats_real_A = G_AB(
                    real_A, return_features=True, nce_layers=nce_layers
                )
                fake_A, feats_real_B = G_BA(
                    real_B, return_features=True, nce_layers=nce_layers
                )
                pred_fake_B = D_B(fake_B)
                pred_fake_A = D_A(fake_A)
                loss_G_AB = _lsgan_gen_loss(pred_fake_B)
                loss_G_BA = _lsgan_gen_loss(pred_fake_A)
                loss_G_gan = loss_G_AB + loss_G_BA

                loss_nce = torch.tensor(0.0, device=device)
                loss_nce_AB = torch.tensor(0.0, device=device)
                loss_nce_BA = torch.tensor(0.0, device=device)
                if use_nce:
                    feats_fake_B = G_AB.encode_features(fake_B, nce_layers=nce_layers)
                    feats_fake_A = G_BA.encode_features(fake_A, nce_layers=nce_layers)

                    patches_real_A, patch_ids_A = patch_sampler.sample(
                        feats_real_A, num_patches=tcfg.nce_num_patches
                    )
                    patches_fake_B, _ = patch_sampler.sample(
                        feats_fake_B,
                        num_patches=tcfg.nce_num_patches,
                        patch_ids=patch_ids_A,
                    )
                    patches_real_B, patch_ids_B = patch_sampler.sample(
                        feats_real_B, num_patches=tcfg.nce_num_patches
                    )
                    patches_fake_A, _ = patch_sampler.sample(
                        feats_fake_A,
                        num_patches=tcfg.nce_num_patches,
                        patch_ids=patch_ids_B,
                    )
                    loss_nce_AB = nce_criterion(patches_fake_B, patches_real_A)
                    loss_nce_BA = nce_criterion(patches_fake_A, patches_real_B)
                    loss_nce = 0.5 * (loss_nce_AB + loss_nce_BA)

                loss_id = torch.tensor(0.0, device=device)
                if tcfg.lambda_identity > 0.0:
                    idt_B = G_AB(real_B)
                    idt_A = G_BA(real_A)
                    loss_id = idt_loss(idt_B, real_B) + idt_loss(idt_A, real_A)

                loss_G = (
                    tcfg.lambda_gan * loss_G_gan
                    + tcfg.lambda_nce * loss_nce
                    + tcfg.lambda_identity * loss_id
                )
                loss_G_scaled = (
                    loss_G / accumulate
                )  # scale before backward for correct gradient magnitude

            if not torch.isfinite(loss_G):
                print(
                    f"[warn] non-finite G loss at epoch {epoch+1} batch {i}, skipping"
                )
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            if use_amp:
                scaler.scale(loss_G_scaled).backward()
            else:
                loss_G_scaled.backward()
            accum_count += 1

            # Step optimizer once the accumulation window is full or the
            # loader is exhausted (handles non-divisible dataset sizes).
            if accum_count == accumulate or (
                i == len(train_loader) and accum_count > 0
            ):
                if tcfg.grad_clip_norm > 0.0:
                    if use_amp:
                        scaler.unscale_(optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(G_AB.parameters()) + list(G_BA.parameters()),
                        tcfg.grad_clip_norm,
                    )
                grad_norm = _global_grad_norm(
                    list(G_AB.parameters()) + list(G_BA.parameters())
                )
                if use_amp:
                    scaler.step(optimizer_G)
                    scaler.update()
                else:
                    optimizer_G.step()

                if tcfg.use_ema and ema_G_AB is not None and ema_G_BA is not None:
                    with torch.no_grad():
                        for ema_p, p in zip(ema_G_AB.parameters(), G_AB.parameters()):
                            ema_p.data.mul_(tcfg.ema_decay).add_(
                                p.data, alpha=1 - tcfg.ema_decay
                            )
                        for ema_p, p in zip(ema_G_BA.parameters(), G_BA.parameters()):
                            ema_p.data.mul_(tcfg.ema_decay).add_(
                                p.data, alpha=1 - tcfg.ema_decay
                            )
                accum_count = 0
            else:
                grad_norm = 0.0

            # ---- Logging ----
            # Accumulate per-batch scalars for epoch-level averaging.
            epoch_step[i] = {
                "Batch": i,
                "Loss_G": float(loss_G.item()),
                "Loss_G_GAN": float(loss_G_gan.item()),
                "Loss_NCE": float(loss_nce.item()),
                "Loss_NCE_AB": float(loss_nce_AB.item()),
                "Loss_NCE_BA": float(loss_nce_BA.item()),
                "Loss_Id": float(loss_id.item()),
                "Loss_G_AB": float(loss_G_AB.item()),
                "Loss_G_BA": float(loss_G_BA.item()),
                "Loss_D_A": float(loss_D_A.item()),
                "Loss_D_B": float(loss_D_B.item()),
                "GradNorm_G": float(grad_norm),
            }
            epoch_loss_G += float(loss_G.item())
            epoch_loss_G_gan += float(loss_G_gan.item())
            epoch_loss_NCE += float(loss_nce.item())
            epoch_loss_Id += float(loss_id.item())
            epoch_loss_G_AB += float(loss_G_AB.item())
            epoch_loss_G_BA += float(loss_G_BA.item())
            epoch_loss_D_A += float(loss_D_A.item())
            epoch_loss_D_B += float(loss_D_B.item())
            epoch_grad_norm += float(grad_norm)

            if i == 1 or i == len(train_loader) or i % tcfg.log_every == 0:
                print(
                    f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_G_GAN: {loss_G_gan.item():.4f} "
                    f"Loss_NCE: {loss_nce.item():.4f} "
                    f"Loss_Id: {loss_id.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f} "
                    f"GradNorm_G: {grad_norm:.4f}"
                )

        # ---- Epoch-level summary ----
        n_batches = max(1, len(train_loader))
        avg_loss_G = epoch_loss_G / n_batches
        avg_loss_G_gan = epoch_loss_G_gan / n_batches
        avg_loss_NCE = epoch_loss_NCE / n_batches
        avg_loss_Id = epoch_loss_Id / n_batches
        avg_loss_G_AB = epoch_loss_G_AB / n_batches
        avg_loss_G_BA = epoch_loss_G_BA / n_batches
        avg_loss_D_A = epoch_loss_D_A / n_batches
        avg_loss_D_B = epoch_loss_D_B / n_batches
        avg_grad_norm = epoch_grad_norm / n_batches

        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/Generator", avg_loss_G, epoch + 1)
        writer.add_scalar("Loss/GAN", avg_loss_G_gan, epoch + 1)
        writer.add_scalar("Loss/NCE", avg_loss_NCE, epoch + 1)
        writer.add_scalar("Loss/Identity", avg_loss_Id, epoch + 1)
        writer.add_scalar("Loss/Generator_AB", avg_loss_G_AB, epoch + 1)
        writer.add_scalar("Loss/Generator_BA", avg_loss_G_BA, epoch + 1)
        writer.add_scalar("Loss/Discriminator_A", avg_loss_D_A, epoch + 1)
        writer.add_scalar("Loss/Discriminator_B", avg_loss_D_B, epoch + 1)
        writer.add_scalar("Diagnostics/GradNorm_G", avg_grad_norm, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
            f"Avg Loss_G: {avg_loss_G:.4f} "
            f"Avg Loss_G_GAN: {avg_loss_G_gan:.4f} "
            f"Avg Loss_NCE: {avg_loss_NCE:.4f} "
            f"Avg Loss_D_A: {avg_loss_D_A:.4f} "
            f"Avg Loss_D_B: {avg_loss_D_B:.4f}"
        )

        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
            writer.add_scalar(
                "LR/Generator", lr_scheduler_G.get_last_lr()[0], epoch + 1
            )

        if lr_scheduler_D_A is not None:
            lr_scheduler_D_A.step()
            writer.add_scalar(
                "LR/Discriminator_A", lr_scheduler_D_A.get_last_lr()[0], epoch + 1
            )

        if lr_scheduler_D_B is not None:
            lr_scheduler_D_B.step()
            writer.add_scalar(
                "LR/Discriminator_B", lr_scheduler_D_B.get_last_lr()[0], epoch + 1
            )

        if (epoch + 1) % tcfg.save_every == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_AB_state_dict": G_AB.state_dict(),
                    "G_BA_state_dict": G_BA.state_dict(),
                    "D_A_state_dict": D_A.state_dict(),
                    "D_B_state_dict": D_B.state_dict(),
                    "ema_G_AB_state_dict": (
                        ema_G_AB.state_dict() if ema_G_AB is not None else None
                    ),
                    "ema_G_BA_state_dict": (
                        ema_G_BA.state_dict() if ema_G_BA is not None else None
                    ),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_A_state_dict": optimizer_D_A.state_dict(),
                    "optimizer_D_B_state_dict": optimizer_D_B.state_dict(),
                    "lr_scheduler_G_state_dict": (
                        lr_scheduler_G.state_dict()
                        if lr_scheduler_G is not None
                        else None
                    ),
                    "lr_scheduler_D_A_state_dict": (
                        lr_scheduler_D_A.state_dict()
                        if lr_scheduler_D_A is not None
                        else None
                    ),
                    "lr_scheduler_D_B_state_dict": (
                        lr_scheduler_D_B.state_dict()
                        if lr_scheduler_D_B is not None
                        else None
                    ),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

        if (epoch + 1) >= tcfg.validation_every:
            save_dir = os.path.join(val_dir, f"epoch_{epoch + 1}")
            val_G_AB = ema_G_AB if tcfg.use_ema and ema_G_AB is not None else G_AB
            val_G_BA = ema_G_BA if tcfg.use_ema and ema_G_BA is not None else G_BA
            _run_validation_v4(
                epoch=epoch + 1,
                G_AB=val_G_AB,
                G_BA=val_G_BA,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir,
                num_samples=tcfg.validation_samples,
                calculator=metrics_calculator,
                max_batches=tcfg.validation_max_batches,
                fid_max_samples=tcfg.validation_fid_samples,
                fid_min_samples=tcfg.validation_fid_min_samples,
                writer=writer,
                is_test=False,
            )

    final_ckpt = os.path.join(model_dir, "final_checkpoint.pth")
    torch.save(
        {
            "epoch": tcfg.num_epochs,
            "G_AB_state_dict": G_AB.state_dict(),
            "G_BA_state_dict": G_BA.state_dict(),
            "D_A_state_dict": D_A.state_dict(),
            "D_B_state_dict": D_B.state_dict(),
            "ema_G_AB_state_dict": (
                ema_G_AB.state_dict() if ema_G_AB is not None else None
            ),
            "ema_G_BA_state_dict": (
                ema_G_BA.state_dict() if ema_G_BA is not None else None
            ),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": optimizer_D_B.state_dict(),
            "lr_scheduler_G_state_dict": (
                lr_scheduler_G.state_dict() if lr_scheduler_G is not None else None
            ),
            "lr_scheduler_D_A_state_dict": (
                lr_scheduler_D_A.state_dict() if lr_scheduler_D_A is not None else None
            ),
            "lr_scheduler_D_B_state_dict": (
                lr_scheduler_D_B.state_dict() if lr_scheduler_D_B is not None else None
            ),
        },
        final_ckpt,
    )
    print(f"Final checkpoint saved: {final_ckpt}")

    # ---- Final test-set export ----
    test_dir = os.path.join(model_dir, "test_images")
    writer.add_scalar("Testing Started", tcfg.num_epochs, tcfg.num_epochs)
    test_G_AB = ema_G_AB if tcfg.use_ema and ema_G_AB is not None else G_AB
    test_G_BA = ema_G_BA if tcfg.use_ema and ema_G_BA is not None else G_BA
    _run_validation_v4(
        epoch=tcfg.num_epochs,
        G_AB=test_G_AB,
        G_BA=test_G_BA,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        num_samples=tcfg.test_size,
        calculator=metrics_calculator,
        max_batches=tcfg.test_size,
        fid_max_samples=tcfg.validation_fid_samples,
        fid_min_samples=tcfg.validation_fid_min_samples,
        writer=writer,
        is_test=True,
    )
    writer.add_scalar("Training Completed", tcfg.num_epochs, tcfg.num_epochs)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B


if __name__ == "__main__":
    train_v4()
