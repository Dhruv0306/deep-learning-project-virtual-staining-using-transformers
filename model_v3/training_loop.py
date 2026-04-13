"""
model_v3/training_loop.py — v3 CycleDiT latent diffusion training loop.

Component structure:
    1) LR schedule helper
    2) Gradient-norm helper
    3) Validation runner
    4) train_v3 main loop

Per-batch shape flow:
    real_A:(N,3,256,256), real_B:(N,3,256,256)
      → z0 via VAE encode: (N,4,32,32)
      → z_t via add_noise: (N,4,32,32)
      → model_pred from DiT: (N,4,32,32)

Two-stage generator update per batch:
    Stage 1 — diffusion denoising loss (MSE ± Min-SNR ± perceptual).
    Stage 2 — adversarial + cycle-consistency + identity losses on a
               fresh forward pass so Stage 1 activations can be freed
               before the larger Stage 2 graph is built.
"""

from __future__ import annotations

import copy
import math
import os
import pickle
from typing import Callable, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from model_v2.losses import VGGPerceptualLossV2
from config import UVCGANConfig, get_dit_config, get_dit_8gb_config
from shared.data_loader import getDataLoader
from shared.EarlyStopping import EarlyStopping
from shared.replay_buffer import ReplayBuffer
from model_v3.history_utils import append_history_to_csv_v3, load_history_from_csv_v3
from shared.metrics import MetricsCalculator
from shared.validation import save_images_with_title

from model_v3.generator import getGeneratorV3
from model_v3.discriminator import getDiscriminatorsV3
from model_v3.noise_scheduler import DDPMScheduler, DDIMSampler
from model_v3.vae_wrapper import VAEWrapper
from model_v3.losses import (
    compute_diffusion_loss,
    _lsgan_gen_loss,
    _lsgan_disc_loss,
    _r1_penalty_loss,
    _ddim_shortcut_from_xt,
    _compute_cycle_loss,
    _compute_identity_loss,
    _compute_identity_weight,
)


def _load_checkpoint_compat(checkpoint_path: str, map_location):
    """
    Load a local checkpoint safely across PyTorch versions.

    PyTorch 2.6+ defaults weights_only=True, which breaks checkpoints that
    contain config dataclasses.  Falls back gracefully for older versions.
    """
    try:
        return torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )


def _make_cosine_warmup_lambda(
    warmup: int, total: int, lr_min_ratio: float
) -> Callable[[int], float]:
    """
    Return a LambdaLR multiplier with linear warmup then cosine decay.

    Schedule:
        [0, warmup)     — linear ramp from ~0 to 1.
        [warmup, total) — cosine decay from 1 down to lr_min_ratio.

    Args:
        warmup:       Number of warmup epochs.
        total:        Total training epochs.
        lr_min_ratio: Minimum LR as a fraction of the base LR.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-8, (epoch + 1) / max(1, warmup))
        if total <= warmup:
            return lr_min_ratio
        progress = (epoch - warmup) / max(1, total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return lr_lambda


def _global_grad_norm(parameters) -> float:
    """
    Return the global L2 gradient norm across all parameters with a grad.

    Returns a plain Python float for logging; no clipping is applied.
    """
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    """Enable or disable gradient computation for all parameters in module."""
    for p in module.parameters():
        p.requires_grad = flag


def _run_validation_v3(
    epoch: int,
    ema_model: torch.nn.Module,
    vae: VAEWrapper,
    sampler: DDIMSampler,
    test_loader,
    device: torch.device,
    save_dir: str,
    calculator: MetricsCalculator,
    num_steps: int,
    writer: SummaryWriter,
    max_batches: int = 50,
    num_samples: int = 6,
    fid_max_samples: int = 200,
    fid_min_samples: int = 50,
    prediction_type: str = "v",
    cfg_scale: float = 1.0,
    is_test: bool = False,
):
    """
    Run one validation (or test) pass and save comparison image grids.

    Generates fake_B (A→B) for every batch up to max_batches, computes
    SSIM/PSNR for both domains, and optionally computes FID when enough
    samples are available.  For the first num_samples batches, also
    generates fake_A (B→A) and cycle reconstructions and saves 4-panel
    PNG grids to save_dir.

    Args:
        epoch:          Current epoch number (used for TensorBoard and filenames).
        ema_model:      EMA copy of CycleDiTGenerator in eval mode.
        vae:            Frozen VAEWrapper for decoding latents.
        sampler:        DDIMSampler instance.
        test_loader:    DataLoader yielding {"A": tensor, "B": tensor} dicts.
        device:         Inference device.
        save_dir:       Directory for comparison PNG grids.
        calculator:     MetricsCalculator for SSIM/PSNR/FID.
        num_steps:      DDIM inference steps.
        writer:         TensorBoard SummaryWriter.
        max_batches:    Maximum batches to evaluate.
        num_samples:    Number of image grids to save.
        fid_max_samples: Maximum samples used for FID.
        fid_min_samples: Minimum samples required to compute FID.
        prediction_type: ``"v"`` or ``"eps"``.
        cfg_scale:      Classifier-free guidance scale.
        is_test:        If True, logs under ``Testing`` prefix and always
                        attempts FID regardless of sample count.

    Returns:
        Dict with keys ``ssim_A``, ``psnr_A``, ``ssim_B``, ``psnr_B``
        (and optionally ``fid_A``, ``fid_B``).
    """
    ema_model.eval()
    vae.eval()
    print(f"[{'Testing' if is_test else 'Validation'}] Starting run at epoch {epoch}")
    # Track metrics for both domains; domain A metrics are computed only for
    # the first num_samples batches where the reverse pass is also run.
    metrics = {"ssim_A": [], "psnr_A": [], "ssim_B": [], "psnr_B": []}
    real_A_list = []
    fake_A_list = []
    real_B_list = []
    fake_B_list = []

    os.makedirs(save_dir, exist_ok=True)
    prefix = "Testing" if is_test else "Validation"

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            if i == 0:
                print(f"[{prefix}] Running first batch...")
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            uncond = torch.zeros_like(real_A)
            z0 = sampler.sample(
                ema_model,
                real_A,
                shape=(real_A.size(0), 4, 32, 32),
                device=device,
                num_steps=num_steps,
                eta=0.0,
                prediction_type=prediction_type,
                cfg_scale=cfg_scale,
                uncond_condition=uncond,
                target_domain=1,
            )
            fake_B = vae.decode(z0).clamp(-1.0, 1.0)

            metrics["ssim_B"].append(calculator.calculate_ssim(real_B, fake_B))
            metrics["psnr_B"].append(calculator.calculate_psnr(real_B, fake_B))

            real_B_list.append(real_B)
            fake_B_list.append(fake_B)

            if i < num_samples:
                # For qualitative grids, also run the reverse direction and cycle reconstructions.
                uncond_B = torch.zeros_like(real_B)
                z0_A = sampler.sample(
                    ema_model,
                    real_B,
                    shape=(real_B.size(0), 4, 32, 32),
                    device=device,
                    num_steps=num_steps,
                    eta=0.0,
                    prediction_type=prediction_type,
                    cfg_scale=cfg_scale,
                    uncond_condition=uncond_B,
                    target_domain=0,
                )
                fake_A = vae.decode(z0_A).clamp(-1.0, 1.0)

                metrics["ssim_A"].append(calculator.calculate_ssim(real_A, fake_A))
                metrics["psnr_A"].append(calculator.calculate_psnr(real_A, fake_A))

                real_A_list.append(real_A)
                fake_A_list.append(fake_A)

                uncond_fake_B = torch.zeros_like(fake_B)
                z0_rec_A = sampler.sample(
                    ema_model,
                    fake_B,
                    shape=(fake_B.size(0), 4, 32, 32),
                    device=device,
                    num_steps=num_steps,
                    eta=0.0,
                    prediction_type=prediction_type,
                    cfg_scale=cfg_scale,
                    uncond_condition=uncond_fake_B,
                    target_domain=0,
                )
                rec_A = vae.decode(z0_rec_A).clamp(-1.0, 1.0)

                uncond_fake_A = torch.zeros_like(fake_A)
                z0_rec_B = sampler.sample(
                    ema_model,
                    fake_A,
                    shape=(fake_A.size(0), 4, 32, 32),
                    device=device,
                    num_steps=num_steps,
                    eta=0.0,
                    prediction_type=prediction_type,
                    cfg_scale=cfg_scale,
                    uncond_condition=uncond_fake_A,
                    target_domain=1,
                )
                rec_B = vae.decode(z0_rec_B).clamp(-1.0, 1.0)

                row_A = torch.cat(
                    [real_A[:1], fake_B[:1], rec_A[:1], real_B[:1]], dim=0
                ).cpu()
                out_path_A = os.path.join(save_dir, f"image_{i + 1}_A.png")
                save_images_with_title(
                    row_A,
                    labels=["Real A", "Fake B", "Rec A", "Real B"],
                    out_path=out_path_A,
                    value_range=(-1, 1),
                )

                row_B = torch.cat(
                    [real_B[:1], fake_A[:1], rec_B[:1], real_A[:1]], dim=0
                ).cpu()
                out_path_B = os.path.join(save_dir, f"image_{i + 1}_B.png")
                save_images_with_title(
                    row_B,
                    labels=["Real B", "Fake A", "Rec B", "Real A"],
                    out_path=out_path_B,
                    value_range=(-1, 1),
                )
            if (i + 1) % 10 == 0:
                print(f"[{prefix}] Processed {i + 1} batches")

    avg_metrics = {
        "ssim_A": float(sum(metrics["ssim_A"]) / max(1, len(metrics["ssim_A"]))),
        "psnr_A": float(sum(metrics["psnr_A"]) / max(1, len(metrics["psnr_A"]))),
        "ssim_B": float(sum(metrics["ssim_B"]) / max(1, len(metrics["ssim_B"]))),
        "psnr_B": float(sum(metrics["psnr_B"]) / max(1, len(metrics["psnr_B"]))),
    }

    # Early stopping score is the mean of both domain SSIMs.
    early_stopping_score = 0.5 * (avg_metrics["ssim_A"] + avg_metrics["ssim_B"])

    fid_count = min(fid_max_samples, len(real_B_list))
    if (
        fid_count >= fid_min_samples
        or is_test
        or (fid_count < fid_min_samples and len(real_B_list) > 0 and epoch % 5 == 0)
    ):
        real_B_tensor = torch.cat(real_B_list[:fid_count])
        fake_B_tensor = torch.cat(fake_B_list[:fid_count])
        avg_metrics["fid_B"] = calculator.evaluate_fid(real_B_tensor, fake_B_tensor)

    fid_count_A = min(fid_max_samples, len(real_A_list))  # FID for domain A
    if (
        fid_count_A >= fid_min_samples
        or is_test
        or (fid_count_A < fid_min_samples and len(real_A_list) > 0 and epoch % 5 == 0)
    ):
        real_A_tensor = torch.cat(real_A_list[:fid_count_A])
        fake_A_tensor = torch.cat(fake_A_list[:fid_count_A])
        avg_metrics["fid_A"] = calculator.evaluate_fid(real_A_tensor, fake_A_tensor)

    for metric_name, value in avg_metrics.items():
        writer.add_scalar(f"{prefix}/{metric_name}", value, epoch)

    print(
        f"{prefix} Metrics - SSIM_A: {avg_metrics['ssim_A']:.4f}, "
        f"SSIM_B: {avg_metrics['ssim_B']:.4f}, "
        f"PSNR_A: {avg_metrics['psnr_A']:.2f}, "
        f"PSNR_B: {avg_metrics['psnr_B']:.2f}"
    )
    if "fid_A" in avg_metrics:
        print(f"{prefix} FID_A Score: {avg_metrics['fid_A']:.2f}")
    if "fid_B" in avg_metrics:
        print(f"{prefix} FID_B Score: {avg_metrics['fid_B']:.2f}")
    print(f"[{prefix}] Completed run at epoch {epoch}")

    # ema_model.train()
    return avg_metrics


def train_v3(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint: Optional[str] = None,
    cfg: Optional[UVCGANConfig] = None,
):
    """
    Train the v3 CycleDiT latent diffusion model.

    Builds the DiT backbone, frozen VAE, dual ProjectionDiscriminators,
    DDPM scheduler, and EMA model, then runs the two-stage per-batch loop:

        Stage 1 — diffusion denoising loss (MSE ± Min-SNR ± perceptual).
        Stage 2 — adversarial + cycle-consistency + identity losses on a
                   fresh forward pass to allow Stage 1 activations to be
                   freed before the larger Stage 2 graph is built.

    Discriminators are updated after the generator step using replay
    buffers and optional R1 gradient penalty.

    Any argument that is not None overrides the corresponding field in cfg
    before training begins.

    Args:
        epoch_size:         Samples per epoch (overrides cfg.training.epoch_size).
        num_epochs:         Total epochs (overrides cfg.training.num_epochs).
        model_dir:          Root output directory for checkpoints and logs.
        val_dir:            Directory for per-epoch validation image grids.
        test_size:          Test samples exported at end of training.
        resume_checkpoint:  Path to a v3 ``.pth`` checkpoint to resume from.
                            Must contain ``dit_state_dict``.
        cfg:                UVCGANConfig with model_version=3.  Defaults to
                            ``get_dit_8gb_config()`` when None.

    Returns:
        tuple: ``(history, dit_model, ema_model, None)``
            - history:   Full training history reloaded from CSV.
            - dit_model: Trained CycleDiTGenerator (raw weights).
            - ema_model: EMA copy of the generator.
            - None:      Placeholder for API compatibility with v1/v2/v4.
    """
    if cfg is None:
        cfg = get_dit_8gb_config()

    if epoch_size is not None:
        cfg.training.epoch_size = epoch_size
    if num_epochs is not None:
        cfg.training.num_epochs = num_epochs
    if model_dir is not None:
        cfg.model_dir = model_dir
    if val_dir is not None:
        cfg.val_dir = val_dir
    if test_size is not None:
        cfg.training.test_size = test_size

    tcfg = cfg.training
    dtcfg = cfg.data
    dcfg = cfg.diffusion
    lcfg = cfg.loss

    assert tcfg.num_epochs is not None, "num_epochs must be specified"
    assert tcfg.epoch_size is not None, "epoch_size must be specified"
    assert tcfg.validation_size is not None, "validation_size must be specified"
    assert (
        tcfg.validation_warmup_epochs < tcfg.early_stopping_warmup
    ), "validation_warmup_epochs must be less than early_stopping_warmup"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    # Unpaired loader yields {"A": tensor, "B": tensor} dicts each iteration.
    train_loader, test_loader = getDataLoader(
        epoch_size=tcfg.epoch_size,
        image_size=dtcfg.image_size,
        batch_size=dtcfg.batch_size,
        num_workers=dtcfg.num_workers,
        prefetch_factor=dtcfg.prefetch_factor,
    )

    # ---- Models ----
    # VAE is frozen throughout; only the DiT and discriminators are updated.
    vae = VAEWrapper(dcfg.vae_model_id).to(device)
    vae.eval()

    dit_model = getGeneratorV3(dcfg, device=device)
    ema_model = copy.deepcopy(dit_model).to(device)
    ema_model.requires_grad_(False)

    D_A, D_B = getDiscriminatorsV3(
        input_nc=3,
        base_channels=dcfg.disc_base_channels,
        n_layers=dcfg.disc_n_layers,
        global_base_channels=dcfg.disc_global_base_channels,
        fft_base_channels=dcfg.disc_fft_base_channels,
        use_local=dcfg.disc_use_local,
        use_global=dcfg.disc_use_global,
        use_fft=dcfg.disc_use_fft,
        device=device,
    )

    replay_A = ReplayBuffer(tcfg.replay_buffer_size)
    replay_B = ReplayBuffer(tcfg.replay_buffer_size)

    scheduler = DDPMScheduler(dcfg.num_timesteps, dcfg.beta_schedule).to(device)
    sampler = DDIMSampler(scheduler)

    # ---- Optimizers ----
    # AdamW for the DiT generator (weight decay stabilises large Transformers);
    # standard Adam for discriminators (matching v2 convention).
    optimizer_G = torch.optim.AdamW(
        list(dit_model.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )

    lr_min_ratio = 1e-6 / 1e-4
    lr_lambda = _make_cosine_warmup_lambda(
        warmup=tcfg.warmup_epochs,
        total=tcfg.num_epochs,
        lr_min_ratio=lr_min_ratio,
    )
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lr_lambda
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lr_lambda
    )

    use_amp = tcfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Metrics / early stopping ----
    metrics_calculator = MetricsCalculator(device=device)
    early_stopping = EarlyStopping(
        patience=max(
            1,
            math.ceil(tcfg.early_stopping_patience / tcfg.early_stopping_interval),
        ),
        min_delta=1e-5,
        divergence_threshold=tcfg.divergence_threshold,
        divergence_patience=tcfg.divergence_patience,
    )

    # ---- Perceptual loss ----
    # Only instantiated when lambda_perceptual_v3 > 0 to avoid loading VGG19
    # when the perceptual term is disabled.
    perceptual_loss = None
    if dcfg.lambda_perceptual_v3 > 0.0:
        perceptual_resize = (
            lcfg.perceptual_resize if lcfg.perceptual_resize is not None else 128
        )
        perceptual_loss = VGGPerceptualLossV2(resize_to=perceptual_resize).to(device)

    # ---- Output dirs / TensorBoard ----
    model_dir = cfg.model_dir or os.path.join(
        "data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v3"
    )
    val_dir = cfg.val_dir or os.path.join(model_dir, "validation_images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    tb_dir = os.path.join(model_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path) and not resume_checkpoint:
        os.remove(history_csv_path)

    history = {}
    stopped_epoch = tcfg.num_epochs
    start_epoch = 0
    accumulate = max(1, tcfg.accumulate_grads)
    accum_count = 0

    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"resume_checkpoint does not exist: {resume_checkpoint}"
            )
        print(f"[train_v3] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = _load_checkpoint_compat(resume_checkpoint, map_location=device)

        if "dit_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing 'dit_state_dict' for v3 resume.")
        dit_model.load_state_dict(checkpoint["dit_state_dict"])

        if "ema_state_dict" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        else:
            ema_model.load_state_dict(dit_model.state_dict())

        if "D_A_state_dict" in checkpoint:
            D_A.load_state_dict(checkpoint["D_A_state_dict"])
        if "D_B_state_dict" in checkpoint:
            D_B.load_state_dict(checkpoint["D_B_state_dict"])

        if "optimizer_G_state_dict" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        elif "optimizer_state_dict" in checkpoint:
            # Backward compatibility with older save schema.
            optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"])

        if "optimizer_D_A_state_dict" in checkpoint:
            optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A_state_dict"])
        if "optimizer_D_B_state_dict" in checkpoint:
            optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B_state_dict"])

        if "lr_scheduler_G_state_dict" in checkpoint:
            lr_scheduler_G.load_state_dict(checkpoint["lr_scheduler_G_state_dict"])
        if "lr_scheduler_D_A_state_dict" in checkpoint:
            lr_scheduler_D_A.load_state_dict(checkpoint["lr_scheduler_D_A_state_dict"])
        if "lr_scheduler_D_B_state_dict" in checkpoint:
            lr_scheduler_D_B.load_state_dict(checkpoint["lr_scheduler_D_B_state_dict"])

        if use_amp and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])

        start_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = min(max(0, start_epoch), tcfg.num_epochs)
        print(f"[train_v3] Resume start epoch: {start_epoch + 1}")

    if accumulate > 1:
        print(
            f"[train_v3] Gradient accumulation enabled: accumulate_grads={accumulate}, "
            f"batch_size={dtcfg.batch_size} -> effective batch = {accumulate * dtcfg.batch_size}"
        )
    if dcfg.use_gradient_checkpointing:
        print(
            "[train_v3] Gradient checkpointing enabled: ~30-40% less activation VRAM."
        )

    for epoch in range(start_epoch, tcfg.num_epochs):
        print()
        dit_model.train()

        epoch_step = {}
        epoch_loss = 0.0
        epoch_loss_perc = 0.0
        epoch_grad_norm = 0.0
        t_epoch_sum = 0.0
        t_epoch_sq_sum = 0.0
        t_epoch_count = 0
        accum_count = 0

        writer.add_scalar("Epoch", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)
            stepped_any = False

            if not (real_A.isfinite().all() and real_B.isfinite().all()):
                print(f"[warn] non-finite input at epoch {epoch+1} batch {i}, skipping")
                continue

            with torch.no_grad():
                z0_A = vae.encode(real_A)
                z0_B = vae.encode(real_B)

            t_A = torch.randint(0, dcfg.num_timesteps, (real_A.size(0),), device=device)
            t_B = torch.randint(0, dcfg.num_timesteps, (real_B.size(0),), device=device)
            noise_A = torch.randn_like(z0_A)
            noise_B = torch.randn_like(z0_B)
            z_t_A = scheduler.add_noise(z0_A, noise_A, t_A)
            z_t_B = scheduler.add_noise(z0_B, noise_B, t_B)

            t_epoch_sum += float(t_A.float().sum().item() + t_B.float().sum().item())
            t_epoch_sq_sum += float(
                (t_A.float().pow(2).sum().item() + t_B.float().pow(2).sum().item())
            )
            t_epoch_count += int(t_A.numel() + t_B.numel())

            if accum_count == 0:
                optimizer_G.zero_grad(set_to_none=True)

            # Generator step precedes discriminator steps.
            # Stage 1 (diffusion) activations are freed before Stage 2
            # (adversarial/cycle/identity) graph is built to reduce peak VRAM.
            _set_requires_grad(D_A, False)
            _set_requires_grad(D_B, False)

            global_step = epoch * len(train_loader) + i
            if dcfg.lambda_adv_warmup_steps > 0:
                warm = min(1.0, global_step / float(dcfg.lambda_adv_warmup_steps))
            else:
                warm = 1.0
            lambda_adv_curr = dcfg.lambda_adv_v3 * warm

            lambda_identity_curr = _compute_identity_weight(
                epoch=epoch,
                num_epochs=tcfg.num_epochs,
                l_start=dcfg.lambda_identity_v3_start,
                l_end=dcfg.lambda_identity_v3_end,
                decay_ratio=dcfg.identity_decay_end_ratio,
            )

            # Stage 1: diffusion-only objective (MSE noise prediction).
            with autocast("cuda", enabled=use_amp):
                out_A2B = dit_model(
                    z_t_A,
                    t_A,
                    real_A,
                    target_domain=1,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )
                out_B2A = dit_model(
                    z_t_B,
                    t_B,
                    real_B,
                    target_domain=0,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )

            loss_A2B, loss_simple_A2B, loss_perc_A2B = compute_diffusion_loss(
                z0=z0_A,
                z_t=z_t_A,
                t=t_A,
                noise=noise_A,
                model_pred=out_A2B["v_pred"],
                real_B=real_B,
                scheduler=scheduler,
                vae=vae,
                perceptual_loss=perceptual_loss,
                lambda_perc=dcfg.lambda_perceptual_v3,
                prediction_type=dcfg.prediction_type,
                min_snr_gamma=dcfg.min_snr_gamma,
                global_step=global_step,
                perceptual_every_n_steps=dcfg.perceptual_every_n_steps,
                perceptual_batch_fraction=dcfg.perceptual_batch_fraction,
            )
            loss_B2A, loss_simple_B2A, loss_perc_B2A = compute_diffusion_loss(
                z0=z0_B,
                z_t=z_t_B,
                t=t_B,
                noise=noise_B,
                model_pred=out_B2A["v_pred"],
                real_B=real_A,
                scheduler=scheduler,
                vae=vae,
                perceptual_loss=perceptual_loss,
                lambda_perc=dcfg.lambda_perceptual_v3,
                prediction_type=dcfg.prediction_type,
                min_snr_gamma=dcfg.min_snr_gamma,
                global_step=global_step,
                perceptual_every_n_steps=dcfg.perceptual_every_n_steps,
                perceptual_batch_fraction=dcfg.perceptual_batch_fraction,
            )

            denoise_loss = 0.5 * (loss_A2B + loss_B2A)
            loss_simple = 0.5 * (loss_simple_A2B + loss_simple_B2A)
            loss_perc_val = 0.5 * (loss_perc_A2B + loss_perc_B2A)
            loss_denoise = dcfg.lambda_denoising * denoise_loss

            if not torch.isfinite(loss_denoise):
                print(
                    f"[warn] non-finite diffusion loss at epoch {epoch+1} batch {i}, skipping"
                )
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                _set_requires_grad(D_A, True)
                _set_requires_grad(D_B, True)
                continue

            scaler.scale(loss_denoise / accumulate).backward()

            del out_A2B, out_B2A, loss_A2B, loss_B2A

            # Stage 2: adversarial, cycle, and identity losses on a fresh forward pass.
            # Recomputing here allows Stage 1 activations to be freed first.
            with autocast("cuda", enabled=use_amp):
                out_A2B = dit_model(
                    z_t_A,
                    t_A,
                    real_A,
                    target_domain=1,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )
                out_B2A = dit_model(
                    z_t_B,
                    t_B,
                    real_B,
                    target_domain=0,
                    scheduler=scheduler,
                    prediction_type=dcfg.prediction_type,
                )

                z0_fake_B = out_A2B["x0_pred"]
                z0_fake_A = out_B2A["x0_pred"]
                # Decode x0 predictions to pixel space; fp32 avoids dtype mismatches.
                fake_B_img = vae.decode(z0_fake_B).clamp(-1.0, 1.0).float()
                fake_A_img = vae.decode(z0_fake_A).clamp(-1.0, 1.0).float()
                loss_adv_G_B = _lsgan_gen_loss(D_B(fake_B_img))
                loss_adv_G_A = _lsgan_gen_loss(D_A(fake_A_img))
                loss_adv_G = 0.5 * (loss_adv_G_A + loss_adv_G_B)

            # Cycle consistency loss
            with autocast("cuda", enabled=use_amp):
                loss_cyc = _compute_cycle_loss(
                    model=dit_model,
                    scheduler=scheduler,
                    z0_A=z0_A,
                    z0_B=z0_B,
                    z0_fake_B=z0_fake_B,
                    z0_fake_A=z0_fake_A,
                    noise_A=noise_A,
                    noise_B=noise_B,
                    t_A=t_A,
                    t_B=t_B,
                    fake_B_img=fake_B_img,
                    fake_A_img=fake_A_img,
                    real_A=real_A,
                    real_B=real_B,
                    prediction_type=dcfg.prediction_type,
                    cycle_ddim_steps=dcfg.cycle_ddim_steps,
                    cycle_ddim_eta=dcfg.cycle_ddim_eta,
                )

            # Identity loss (only if lambda > 0)
            loss_id = torch.tensor(0.0, device=device, dtype=torch.float32)
            if lambda_identity_curr > 0.0:
                with autocast("cuda", enabled=use_amp):
                    loss_id = _compute_identity_loss(
                        model=dit_model,
                        scheduler=scheduler,
                        z0_A=z0_A,
                        z0_B=z0_B,
                        real_A=real_A,
                        real_B=real_B,
                        device=device,
                        prediction_type=dcfg.prediction_type,
                    )

            aux_loss = (
                lambda_adv_curr * loss_adv_G
                + dcfg.lambda_cycle_v3 * loss_cyc
                + lambda_identity_curr * loss_id
            )

            if not torch.isfinite(aux_loss):
                print(
                    f"[warn] non-finite auxiliary loss at epoch {epoch+1} batch {i}, skipping"
                )
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                _set_requires_grad(D_A, True)
                _set_requires_grad(D_B, True)
                continue

            scaler.scale(aux_loss / accumulate).backward()

            loss = loss_denoise + aux_loss
            accum_count += 1

            if accum_count == accumulate or (
                i == len(train_loader) and accum_count > 0
            ):
                grad_clip = (
                    dcfg.grad_clip_norm_g
                    if getattr(dcfg, "grad_clip_norm_g", None) is not None
                    else tcfg.grad_clip_norm
                )
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(dit_model.parameters()),
                        grad_clip,
                    )
                grad_norm = _global_grad_norm(list(dit_model.parameters()))
                scaler.step(optimizer_G)
                stepped_any = True

                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), dit_model.parameters()):
                        ema_p.data.mul_(0.9999).add_(p.data, alpha=1 - 0.9999)

                accum_count = 0
            else:
                grad_norm = 0.0

            _set_requires_grad(D_A, True)
            _set_requires_grad(D_B, True)

            # Replay buffers mix old and new fakes to stabilise discriminator training.
            fake_A_detached = replay_A.push_and_pop(fake_A_img.detach())
            fake_B_detached = replay_B.push_and_pop(fake_B_img.detach())

            optimizer_D_A.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_A = _lsgan_disc_loss(D_A(real_A), D_A(fake_A_detached))
            if (
                dcfg.use_r1_penalty
                and dcfg.r1_interval > 0
                and global_step % dcfg.r1_interval == 0
            ):
                with autocast("cuda", enabled=False):
                    real_A_r1 = real_A.detach().float().requires_grad_(True)
                    r1_A = _r1_penalty_loss(D_A, real_A_r1, dcfg.r1_gamma)
                loss_D_A = loss_D_A + r1_A.to(loss_D_A.dtype)
            if not (
                dcfg.adaptive_d_update
                and float(loss_D_A.item()) < dcfg.adaptive_d_loss_threshold
            ):
                scaler.scale(loss_D_A).backward()
                scaler.step(optimizer_D_A)
                stepped_any = True

            optimizer_D_B.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_B = _lsgan_disc_loss(D_B(real_B), D_B(fake_B_detached))
            if (
                dcfg.use_r1_penalty
                and dcfg.r1_interval > 0
                and global_step % dcfg.r1_interval == 0
            ):
                with autocast("cuda", enabled=False):
                    real_B_r1 = real_B.detach().float().requires_grad_(True)
                    r1_B = _r1_penalty_loss(D_B, real_B_r1, dcfg.r1_gamma)
                loss_D_B = loss_D_B + r1_B.to(loss_D_B.dtype)
            if not (
                dcfg.adaptive_d_update
                and float(loss_D_B.item()) < dcfg.adaptive_d_loss_threshold
            ):
                scaler.scale(loss_D_B).backward()
                scaler.step(optimizer_D_B)
                stepped_any = True

            if stepped_any:
                scaler.update()

            epoch_step[i] = {
                "Batch": i,
                "Loss_DiT_A2B": float(loss_simple_A2B.item()),
                "Loss_DiT_B2A": float(loss_simple_B2A.item()),
                "Loss_DiT": float(loss_simple.item()),
                "Loss_G_Adv": float(loss_adv_G.item()),
                "Loss_Cyc": float(loss_cyc.item()),
                "Loss_Id": float(loss_id.item()),
                "Loss_D_A": float(loss_D_A.item()),
                "Loss_D_B": float(loss_D_B.item()),
                "Lambda_Adv": float(lambda_adv_curr),
                "Lambda_Id": float(lambda_identity_curr),
                "Loss_Perceptual": float(loss_perc_val),
                "Loss Total": float(loss.item()),
                "GradNorm": float(grad_norm),
            }
            epoch_loss += float(loss_simple.item())
            epoch_loss_perc += float(loss_perc_val)
            epoch_grad_norm += float(grad_norm)

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                if dcfg.lambda_perceptual_v3 > 0.0:
                    print(
                        f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss A2B: {loss_simple_A2B.item():.4f} "
                        f"Loss B2A: {loss_simple_B2A.item():.4f} "
                        f"Loss DiT: {loss_simple.item():.4f} "
                        f"Loss G_adv: {loss_adv_G.item():.4f} "
                        f"Loss Cyc: {loss_cyc.item():.4f} "
                        f"Loss Id: {loss_id.item():.4f} "
                        f"Loss D_A: {loss_D_A.item():.4f} "
                        f"Loss D_B: {loss_D_B.item():.4f} "
                        f"Loss Perceptual: {loss_perc_val:.4f} "
                        f"Loss Total: {loss.item():.4f} "
                        f"GradNorm: {grad_norm:.4f}"
                    )
                else:
                    print(
                        f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
                        f"Batch [{i}/{len(train_loader)}] "
                        f"Loss A2B: {loss_simple_A2B.item():.4f} "
                        f"Loss B2A: {loss_simple_B2A.item():.4f} "
                        f"Loss DiT: {loss_simple.item():.4f} "
                        f"Loss G_adv: {loss_adv_G.item():.4f} "
                        f"Loss Cyc: {loss_cyc.item():.4f} "
                        f"Loss Id: {loss_id.item():.4f} "
                        f"Loss D_A: {loss_D_A.item():.4f} "
                        f"Loss D_B: {loss_D_B.item():.4f} "
                        f"Loss Total: {loss.item():.4f} "
                        f"GradNorm: {grad_norm:.4f}"
                    )

        # ---- Epoch-level aggregation ----
        n_batches = max(1, len(train_loader))
        avg_loss = epoch_loss / n_batches
        avg_loss_perc = epoch_loss_perc / n_batches
        avg_grad_norm = epoch_grad_norm / n_batches

        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/DiT", avg_loss, epoch + 1)
        writer.add_scalar("Loss/Perceptual", avg_loss_perc, epoch + 1)
        writer.add_scalar("Diagnostics/GradNorm", avg_grad_norm, epoch + 1)
        # Log last-batch GAN scalars and epoch-average lambda values.
        if epoch_step:
            last_key = max(epoch_step.keys())
            lambda_adv_epoch_avg = float(
                sum(v["Lambda_Adv"] for v in epoch_step.values()) / len(epoch_step)
            )
            lambda_id_epoch_avg = float(
                sum(v["Lambda_Id"] for v in epoch_step.values()) / len(epoch_step)
            )
            writer.add_scalar(
                "Loss/G_adv", epoch_step[last_key]["Loss_G_Adv"], epoch + 1
            )
            writer.add_scalar("Loss/D_A", epoch_step[last_key]["Loss_D_A"], epoch + 1)
            writer.add_scalar("Loss/D_B", epoch_step[last_key]["Loss_D_B"], epoch + 1)
            writer.add_scalar(
                "Weights/lambda_adv_current",
                epoch_step[last_key]["Lambda_Adv"],
                epoch + 1,
            )
            writer.add_scalar(
                "Weights/lambda_adv_epoch_avg",
                lambda_adv_epoch_avg,
                epoch + 1,
            )
            writer.add_scalar("Loss/Cycle", epoch_step[last_key]["Loss_Cyc"], epoch + 1)
            writer.add_scalar(
                "Loss/Identity", epoch_step[last_key]["Loss_Id"], epoch + 1
            )
            writer.add_scalar(
                "Weights/lambda_identity_current",
                epoch_step[last_key]["Lambda_Id"],
                epoch + 1,
            )
            writer.add_scalar(
                "Weights/lambda_identity_epoch_avg",
                lambda_id_epoch_avg,
                epoch + 1,
            )
        if t_epoch_count > 0:
            t_mean = t_epoch_sum / t_epoch_count
            t_var = max(0.0, (t_epoch_sq_sum / t_epoch_count) - (t_mean * t_mean))
            t_std = math.sqrt(t_var)
            writer.add_scalar("Diagnostics/TimestepMean", t_mean, epoch + 1)
            writer.add_scalar("Diagnostics/TimestepStd", t_std, epoch + 1)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        current_lr = lr_scheduler_G.get_last_lr()[0]
        writer.add_scalar("LR/DiT", current_lr, epoch + 1)
        writer.add_scalar("LR/D_A", lr_scheduler_D_A.get_last_lr()[0], epoch + 1)
        writer.add_scalar("LR/D_B", lr_scheduler_D_B.get_last_lr()[0], epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{tcfg.num_epochs}] "
            f"Avg Loss: {avg_loss:.4f}  "
            f"LR: {current_lr:.6f}"
        )

        if (epoch + 1) % 5 == 0:
            append_history_to_csv_v3(history, history_csv_path)
            history.clear()

        if (epoch + 1) % tcfg.save_checkpoint_every == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "checkpoint_format_version": 2,
                    "epoch": epoch + 1,
                    "dit_state_dict": dit_model.state_dict(),
                    "D_A_state_dict": D_A.state_dict(),
                    "D_B_state_dict": D_B.state_dict(),
                    "ema_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer_G.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_A_state_dict": optimizer_D_A.state_dict(),
                    "optimizer_D_B_state_dict": optimizer_D_B.state_dict(),
                    "lr_scheduler_G_state_dict": lr_scheduler_G.state_dict(),
                    "lr_scheduler_D_A_state_dict": lr_scheduler_D_A.state_dict(),
                    "lr_scheduler_D_B_state_dict": lr_scheduler_D_B.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "early_stopping_state": early_stopping.state_dict(),
                    "config": dcfg,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

        val_metrics = None
        if (epoch + 1) > tcfg.validation_warmup_epochs:
            save_dir = os.path.join(val_dir, f"epoch_{epoch + 1}")
            val_metrics = _run_validation_v3(
                epoch=epoch + 1,
                ema_model=ema_model,
                vae=vae,
                sampler=sampler,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir,
                calculator=metrics_calculator,
                num_steps=dcfg.num_inference_steps,
                writer=writer,
                is_test=False,
                max_batches=tcfg.validation_size,
                num_samples=max(6, tcfg.validation_size),
                fid_max_samples=tcfg.validation_fid_samples,
                fid_min_samples=tcfg.validation_fid_min_samples,
                prediction_type=dcfg.prediction_type,
                cfg_scale=dcfg.cfg_scale,
            )

        if (
            (epoch + 1) % tcfg.early_stopping_interval == 0
            and epoch + 1 >= tcfg.early_stopping_warmup
            and val_metrics is not None
        ):
            # Phase 2: early stopping score uses mean(SSIM_A, SSIM_B).
            avg_ssim = 0.5 * (
                val_metrics.get("ssim_A", 0.0) + val_metrics.get("ssim_B", 0.0)
            )
            should_stop = early_stopping(ssim=avg_ssim, losses={"DiT": avg_loss})
            writer.add_scalar("EarlyStopping/ssim", avg_ssim, epoch + 1)
            writer.add_scalar(
                "EarlyStopping/counter", early_stopping.counter, epoch + 1
            )
            writer.add_scalar(
                "EarlyStopping/divergence_counter",
                early_stopping.divergence_counter,
                epoch + 1,
            )

            if should_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                stopped_epoch = epoch + 1
                break

    final_ckpt = os.path.join(model_dir, f"final_checkpoint_epoch_{stopped_epoch}.pth")
    torch.save(
        {
            "checkpoint_format_version": 2,
            "epoch": stopped_epoch,
            "dit_state_dict": dit_model.state_dict(),
            "D_A_state_dict": D_A.state_dict(),
            "D_B_state_dict": D_B.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer_G.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": optimizer_D_B.state_dict(),
            "lr_scheduler_G_state_dict": lr_scheduler_G.state_dict(),
            "lr_scheduler_D_A_state_dict": lr_scheduler_D_A.state_dict(),
            "lr_scheduler_D_B_state_dict": lr_scheduler_D_B.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "early_stopping_state": early_stopping.state_dict(),
            "config": dcfg,
        },
        final_ckpt,
    )
    print(f"Final checkpoint saved: {final_ckpt}")

    # ---- Final test-set inference ----
    test_dir = os.path.join(model_dir, "test_images")
    writer.add_scalar("Testing Started", stopped_epoch, stopped_epoch)
    _run_validation_v3(
        epoch=stopped_epoch,
        ema_model=ema_model,
        vae=vae,
        sampler=sampler,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        calculator=metrics_calculator,
        num_steps=dcfg.num_inference_steps,
        writer=writer,
        max_batches=tcfg.test_size,
        num_samples=max(6, tcfg.test_size),
        fid_max_samples=tcfg.validation_fid_samples,
        fid_min_samples=tcfg.validation_fid_min_samples,
        prediction_type=dcfg.prediction_type,
        cfg_scale=dcfg.cfg_scale,
        is_test=True,
    )
    writer.add_scalar("Training Completed", stopped_epoch, stopped_epoch)

    append_history_to_csv_v3(history, history_csv_path)
    history = load_history_from_csv_v3(history_csv_path)

    writer.close()
    return history, dit_model, ema_model, None
