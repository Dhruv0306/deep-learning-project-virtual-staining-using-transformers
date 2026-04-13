"""
model_v2/training_loop.py — UVCGAN v2 training loop (Prokopenko et al., 2023).

GAN objective   : LSGAN — losses are always ≥ 0, numerically stable.
Gradient penalty: one-sided GP, gamma=100, lambda=0.1 (inside UVCGANLoss).
n_critic        : 1 — standard for LSGAN.
Adam betas      : (0.5, 0.999), lr=2e-4.
AMP safety      : GP always runs in float32; autocast is disabled inside
                  UVCGANLoss.discriminator_loss regardless of use_amp.
Cross-domain    : generator_loss() calls forward_with_cross_domain()
                  automatically when both generators support it.

Engineering additions:
    - Three-phase LR schedule: linear warm-up → constant → linear decay.
    - Gradient clipping on G and D.
    - EarlyStopping on SSIM + loss-divergence detection.
    - Replay buffer for discriminator stabilisation.
    - Last-known-good G snapshot for non-finite loss rollback.
    - TensorBoard logging: losses, LR, grad norms, early-stopping counters.
    - Periodic CSV history flush and epoch checkpoints.

Entry point: train_v2().
"""

import math
import os
import copy
import pickle

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from model_v2.losses import UVCGANLoss
from config import UVCGANConfig, get_default_config
from shared.data_loader import getDataLoader
from shared.EarlyStopping import EarlyStopping
from shared.history_utils import append_history_to_csv, load_history_from_csv
from shared.metrics import MetricsCalculator
from model_v2.discriminator import getDiscriminatorsV2
from shared.testing import run_testing
from model_v2.generator import getGeneratorsV2
from shared.validation import calculate_metrics, run_validation


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


def _load_state_dict_with_compat(module: torch.nn.Module, state_dict: dict, tag: str):
    """
    Load a state dict strictly when possible, else fall back to shape-compatible keys.

    Helps resume from older checkpoints when the model schema has changed.
    """
    try:
        module.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        model_state = module.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        module.load_state_dict(compatible, strict=False)
        print(
            f"[train_v2][warn] {tag}: strict load failed; loaded "
            f"{len(compatible)}/{len(model_state)} shape-compatible tensors."
        )
        print(f"[train_v2][warn] {tag}: strict-load error: {exc}")


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def _make_lr_lambda(warmup: int, decay_start: int, total: int):
    """
    Return a LambdaLR callable implementing a three-phase LR schedule.

    Phase 1 — warm-up  [0, warmup):          linear ramp from ~0 to 1.
    Phase 2 — plateau  [warmup, decay_start): constant at 1.
    Phase 3 — decay    [decay_start, total):  linear decay from 1 to 0.

    Args:
        warmup:      Warm-up duration in epochs.
        decay_start: First epoch of the linear decay phase.
        total:       Total training epochs.

    Returns:
        Callable[[int], float]: Multiplicative factor for LambdaLR.
    """

    def lr_lambda(epoch: int) -> float:
        """Return the LR multiplier for epoch (0-based)."""
        if epoch < warmup:
            return max(1e-8, epoch / max(1, warmup))
        if epoch < decay_start:
            return 1.0
        remaining = total - decay_start
        if remaining <= 0:
            return 0.0
        return max(0.0, 1.0 - (epoch - decay_start) / remaining)

    return lr_lambda


# ---------------------------------------------------------------------------
# Gradient-norm helper
# ---------------------------------------------------------------------------


def _global_grad_norm(parameters) -> float:
    """
    Return the global L2 gradient norm across parameters as a float.

    Parameters with None gradients are skipped.  Returns 0.0 when no
    parameter has a gradient.
    """
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


def _snapshot_module_to_cpu(module: torch.nn.Module) -> dict:
    """
    Return a detached CPU copy of module's state_dict for rollback.

    CPU placement avoids holding extra GPU memory between steps.
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    """Deep-copy optimizer state for rollback independent of live tensors."""
    return copy.deepcopy(optimizer.state_dict())


def _snapshot_scaler_state(scaler: GradScaler) -> dict:
    """Deep-copy AMP GradScaler state for rollback alongside the G snapshot."""
    return copy.deepcopy(scaler.state_dict())


def _module_parameters_are_finite(module: torch.nn.Module) -> bool:
    """Return True if every parameter in module is finite."""
    return all(torch.isfinite(p).all() for p in module.parameters())


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_v2(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint=None,
    cfg: Optional[UVCGANConfig] = None,
):
    """
    Train UVCGAN v2 generators and multi-scale discriminators.

    Args:
        epoch_size: Max samples drawn per epoch (overrides cfg.training.epoch_size).
        num_epochs: Total training epochs (overrides cfg.training.num_epochs).
        model_dir:  Output directory for checkpoints and logs (overrides cfg.model_dir).
        val_dir:    Directory for validation images (overrides cfg.val_dir).
        test_size:  Number of test samples to export (overrides cfg.training.test_size).
        resume_checkpoint: Path to a checkpoint to resume from.
        cfg:        UVCGANConfig instance. Defaults to get_default_config(model_version=2).

    Returns:
        tuple[dict, nn.Module, nn.Module, nn.Module, nn.Module]:
            (history, G_AB, G_BA, D_A, D_B)
    """
    if cfg is None:
        cfg = get_default_config(model_version=2)

    # Apply argument overrides.
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

    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"resume_checkpoint does not exist: {resume_checkpoint}"
            )
        print(f"[train_v2] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = _load_checkpoint_compat(resume_checkpoint, map_location="cpu")
        ckpt_cfg = checkpoint.get("config")
        if isinstance(ckpt_cfg, UVCGANConfig):
            cfg = copy.deepcopy(ckpt_cfg)
            if epoch_size is not None:
                cfg.training.epoch_size = epoch_size
            if num_epochs is not None:
                cfg.training.num_epochs = num_epochs
            if test_size is not None:
                cfg.training.test_size = test_size
            if model_dir is not None:
                cfg.model_dir = model_dir
            if val_dir is not None:
                cfg.val_dir = val_dir
        print("[train_v2] Using training config from checkpoint.")
    else:
        checkpoint = None

    tcfg = cfg.training
    lcfg = cfg.loss
    gcfg = cfg.generator
    dcfg = cfg.discriminator
    dtcfg = cfg.data

    n_critic = tcfg.n_critic  # 1 for LSGAN (paper default)

    # ---- Backend tuning ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    train_loader, test_loader = getDataLoader(
        epoch_size=tcfg.epoch_size,
        image_size=dtcfg.image_size,
        batch_size=dtcfg.batch_size,
        num_workers=dtcfg.num_workers,
    )

    # ---- Models ----
    G_AB, G_BA = getGeneratorsV2(
        base_channels=gcfg.base_channels,
        vit_depth=gcfg.vit_depth,
        vit_heads=gcfg.vit_heads,
        vit_mlp_ratio=gcfg.vit_mlp_ratio,
        vit_dropout=gcfg.vit_dropout,
        layerscale_init=gcfg.layerscale_init,
        use_cross_domain=gcfg.use_cross_domain,
        use_gradient_checkpointing=gcfg.use_gradient_checkpointing,
    )
    D_A, D_B = getDiscriminatorsV2(
        input_nc=dcfg.input_nc,
        base_channels=dcfg.base_channels,
        n_layers=dcfg.n_layers,
        num_scales=dcfg.num_scales,
        use_spectral_norm=dcfg.use_spectral_norm,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    # ---- Loss ----
    loss_fn = UVCGANLoss(
        lambda_cycle=lcfg.lambda_cycle,
        lambda_identity=lcfg.lambda_identity,
        lambda_cycle_perceptual=lcfg.lambda_cycle_perceptual,
        lambda_identity_perceptual=lcfg.lambda_identity_perceptual,
        lambda_gp=lcfg.lambda_gp,
        lambda_contrastive=lcfg.lambda_contrastive,
        lambda_spectral=lcfg.lambda_spectral,
        perceptual_resize=lcfg.perceptual_resize,
        contrastive_temperature=lcfg.contrastive_temperature,
        device=device,
    )

    # ---- AMP ----
    # G step uses AMP; D step (including GP) always runs in float32 via
    # autocast(enabled=False) inside discriminator_loss.
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

    # ---- Optimisers ----
    # AdamW for G (Transformer bottleneck); Adam for discriminators.
    optimizer_G = optim.AdamW(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=tcfg.lr,
        betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=0.01,
    )
    optimizer_D_A = optim.Adam(
        D_A.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )
    optimizer_D_B = optim.Adam(
        D_B.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2)
    )

    # ---- LR schedulers ----
    lr_lambda = _make_lr_lambda(
        warmup=tcfg.warmup_epochs,
        decay_start=tcfg.decay_start_epoch,
        total=tcfg.num_epochs,
    )
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

    # ---- Output dirs / TensorBoard ----
    model_dir = cfg.model_dir or os.path.join(
        "data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models_v2"
    )
    val_dir = cfg.val_dir or os.path.join(model_dir, "validation_images")
    os.makedirs(model_dir, exist_ok=True)
    tb_dir = os.path.join(model_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path) and not resume_checkpoint:
        os.remove(history_csv_path)

    start_epoch = 0
    if resume_checkpoint:
        if checkpoint is None:
            raise ValueError(f"Failed to load checkpoint: {resume_checkpoint}")
        # Support both old key names ("G_AB") and new ones ("G_AB_state_dict").
        G_AB_state = checkpoint.get("G_AB") or checkpoint.get("G_AB_state_dict")
        G_BA_state = checkpoint.get("G_BA") or checkpoint.get("G_BA_state_dict")
        D_A_state = checkpoint.get("D_A") or checkpoint.get("D_A_state_dict")
        D_B_state = checkpoint.get("D_B") or checkpoint.get("D_B_state_dict")
        if G_AB_state is None or G_BA_state is None:
            raise KeyError(
                "Checkpoint missing generator weights required for v2 resume."
            )

        _load_state_dict_with_compat(G_AB, G_AB_state, "G_AB")
        _load_state_dict_with_compat(G_BA, G_BA_state, "G_BA")
        if D_A_state is not None:
            _load_state_dict_with_compat(D_A, D_A_state, "D_A")
        if D_B_state is not None:
            _load_state_dict_with_compat(D_B, D_B_state, "D_B")

        if "optimizer_G" in checkpoint:
            try:
                optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            except (ValueError, RuntimeError) as exc:
                print(
                    "[train_v2][warn] optimizer_G state is incompatible; starting fresh. "
                    f"Details: {exc}"
                )
        if "optimizer_D_A" in checkpoint:
            try:
                optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A"])
            except (ValueError, RuntimeError) as exc:
                print(
                    "[train_v2][warn] optimizer_D_A state is incompatible; starting fresh. "
                    f"Details: {exc}"
                )
        if "optimizer_D_B" in checkpoint:
            try:
                optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B"])
            except (ValueError, RuntimeError) as exc:
                print(
                    "[train_v2][warn] optimizer_D_B state is incompatible; starting fresh. "
                    f"Details: {exc}"
                )

        resume_epoch = int(checkpoint.get("epoch", 0))
        if "lr_scheduler_G_state_dict" in checkpoint:
            lr_scheduler_G.load_state_dict(checkpoint["lr_scheduler_G_state_dict"])
        else:
            lr_scheduler_G.last_epoch = max(-1, resume_epoch - 1)
        if "lr_scheduler_D_A_state_dict" in checkpoint:
            lr_scheduler_D_A.load_state_dict(checkpoint["lr_scheduler_D_A_state_dict"])
        else:
            lr_scheduler_D_A.last_epoch = max(-1, resume_epoch - 1)
        if "lr_scheduler_D_B_state_dict" in checkpoint:
            lr_scheduler_D_B.load_state_dict(checkpoint["lr_scheduler_D_B_state_dict"])
        else:
            lr_scheduler_D_B.last_epoch = max(-1, resume_epoch - 1)

        if use_amp and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])

        start_epoch = resume_epoch
        if start_epoch >= tcfg.num_epochs:
            raise ValueError(
                f"num_epochs ({tcfg.num_epochs}) must be greater than checkpoint epoch ({start_epoch})."
            )
        print(f"[train_v2] Resume start epoch: {start_epoch + 1}")

    num_epochs = tcfg.num_epochs
    history = {}
    stopped_epoch = num_epochs
    accumulate = max(1, tcfg.accumulate_grads)
    accum_count = 0
    nonfinite_g_streak = 0
    nonfinite_d_streak = 0

    # Last-known-good G snapshot: restored when Loss_G becomes non-finite.
    last_good_G_AB_state = _snapshot_module_to_cpu(G_AB)
    last_good_G_BA_state = _snapshot_module_to_cpu(G_BA)
    last_good_optimizer_G_state = _snapshot_optimizer_state(optimizer_G)
    last_good_scaler_state = _snapshot_scaler_state(scaler)
    last_good_D_A_state = _snapshot_module_to_cpu(D_A)
    last_good_D_B_state = _snapshot_module_to_cpu(D_B)
    last_good_optimizer_D_A_state = _snapshot_optimizer_state(optimizer_D_A)
    last_good_optimizer_D_B_state = _snapshot_optimizer_state(optimizer_D_B)

    if accumulate > 1:
        print(
            f"[train_v2] Gradient accumulation enabled: accumulate_grads={accumulate}, "
            f"batch_size={dtcfg.batch_size} → effective batch = {accumulate * dtcfg.batch_size}"
        )
    if gcfg.use_gradient_checkpointing:
        print(
            "[train_v2] Gradient checkpointing enabled: ~30-40% less activation VRAM, ~20% slower backward."
        )

    for epoch in range(start_epoch, num_epochs):
        print()
        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        epoch_step = {}
        epoch_loss_G = 0.0
        epoch_loss_D_A = 0.0
        epoch_loss_D_B = 0.0
        epoch_grad_norm_G = 0.0
        accum_count = 0
        warn_near_uniform = 0

        writer.add_scalar("Epoch", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

        # Skip batches with non-finite values; warn on near-uniform patches.
        if not (real_A.isfinite().all() and real_B.isfinite().all()):
            print(f"[warn] non-finite input at epoch {epoch+1} batch {i}, skipping")
            continue

            real_A_std = real_A.std(dim=[1, 2, 3])
            real_B_std = real_B.std(dim=[1, 2, 3])
            if (real_A_std < 1e-4).any() or (real_B_std < 1e-4).any():
                warn_near_uniform += 1
                if warn_near_uniform % 100 == 0:
                    print(
                        f"[warn] near-uniform input detected at epoch {epoch+1} batch {i} "
                        f"(A std: {real_A_std}, B std: {real_B_std})"
                    )

            # --- Discriminator step (n_critic=1 for LSGAN) ---
            for p in G_AB.parameters():
                p.requires_grad_(False)
            for p in G_BA.parameters():
                p.requires_grad_(False)
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            skip_batch = False
            loss_D_A_accum = 0.0
            loss_D_B_accum = 0.0

            for _ in range(n_critic):
                with torch.no_grad():
                    fake_B_d = G_AB(real_A)
                    fake_A_d = G_BA(real_B)

                # GP is computed in float32 inside discriminator_loss.
                optimizer_D_A.zero_grad(set_to_none=True)
                optimizer_D_B.zero_grad(set_to_none=True)
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A_d, loss_fn.fake_A_buffer
                )
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B_d, loss_fn.fake_B_buffer
                )
                if not (torch.isfinite(loss_D_A) and torch.isfinite(loss_D_B)):
                    nonfinite_d_streak += 1
                    print(
                        f"[warn] non-finite discriminator loss at epoch {epoch+1} batch {i}, rolling back D to last good state (streak={nonfinite_d_streak})"
                    )
                    D_A.load_state_dict(last_good_D_A_state, strict=True)
                    D_B.load_state_dict(last_good_D_B_state, strict=True)
                    optimizer_D_A.load_state_dict(last_good_optimizer_D_A_state)
                    optimizer_D_B.load_state_dict(last_good_optimizer_D_B_state)
                    optimizer_D_A.zero_grad(set_to_none=True)
                    optimizer_D_B.zero_grad(set_to_none=True)
                    skip_batch = True
                    break

                loss_D_A.backward()
                if tcfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        D_A.parameters(), tcfg.grad_clip_norm
                    )
                optimizer_D_A.step()

                loss_D_B.backward()
                if tcfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        D_B.parameters(), tcfg.grad_clip_norm
                    )
                optimizer_D_B.step()

                if not (
                    _module_parameters_are_finite(D_A)
                    and _module_parameters_are_finite(D_B)
                ):
                    nonfinite_d_streak += 1
                    print(
                        f"[warn] non-finite discriminator params after step at epoch {epoch+1} batch {i}, rolling back D to last good state (streak={nonfinite_d_streak})"
                    )
                    D_A.load_state_dict(last_good_D_A_state, strict=True)
                    D_B.load_state_dict(last_good_D_B_state, strict=True)
                    optimizer_D_A.load_state_dict(last_good_optimizer_D_A_state)
                    optimizer_D_B.load_state_dict(last_good_optimizer_D_B_state)
                    optimizer_D_A.zero_grad(set_to_none=True)
                    optimizer_D_B.zero_grad(set_to_none=True)
                    skip_batch = True
                    break

                # Refresh D rollback snapshots only after a validated finite step.
                last_good_D_A_state = _snapshot_module_to_cpu(D_A)
                last_good_D_B_state = _snapshot_module_to_cpu(D_B)
                last_good_optimizer_D_A_state = _snapshot_optimizer_state(optimizer_D_A)
                last_good_optimizer_D_B_state = _snapshot_optimizer_state(optimizer_D_B)
                nonfinite_d_streak = 0

                loss_D_A_item = loss_D_A.item()
                loss_D_B_item = loss_D_B.item()
                loss_D_A_accum += loss_D_A_item
                loss_D_B_accum += loss_D_B_item

            if skip_batch:
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                continue

            loss_D_A_val = loss_D_A_accum / n_critic
            loss_D_B_val = loss_D_B_accum / n_critic

            # --- Generator step (gradient accumulation over `accumulate` batches) ---
            for p in G_AB.parameters():
                p.requires_grad_(True)
            for p in G_BA.parameters():
                p.requires_grad_(True)
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)

            # Zero grads only at the start of each accumulation window.
            if accum_count == 0:
                optimizer_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
                )
                # Divide by accumulate so gradients match a single large batch.
                loss_G_scaled = loss_G / accumulate

            loss_G_val = loss_G.item()
            if not torch.isfinite(loss_G):
                nonfinite_g_streak += 1
                print(
                    f"[warn] non-finite Loss_G at epoch {epoch+1} batch {i}, rolling back G to last good state (streak={nonfinite_g_streak})"
                )
                G_AB.load_state_dict(last_good_G_AB_state, strict=True)
                G_BA.load_state_dict(last_good_G_BA_state, strict=True)
                optimizer_G.load_state_dict(last_good_optimizer_G_state)
                scaler.load_state_dict(last_good_scaler_state)
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                if use_amp:
                    scaler.update(new_scale=max(1.0, scaler.get_scale() / 2.0))
                continue
            if not (torch.isfinite(fake_A).all() and torch.isfinite(fake_B).all()):
                nonfinite_g_streak += 1
                print(
                    f"[warn] non-finite generator output at epoch {epoch+1} batch {i}, rolling back G to last good state (streak={nonfinite_g_streak})"
                )
                G_AB.load_state_dict(last_good_G_AB_state, strict=True)
                G_BA.load_state_dict(last_good_G_BA_state, strict=True)
                optimizer_G.load_state_dict(last_good_optimizer_G_state)
                scaler.load_state_dict(last_good_scaler_state)
                optimizer_G.zero_grad(set_to_none=True)
                accum_count = 0
                if use_amp:
                    scaler.update(new_scale=max(1.0, scaler.get_scale() / 2.0))
                continue
            scaler.scale(loss_G_scaled).backward()
            accum_count += 1

            # Step only when the accumulation window is complete.
            if accum_count == accumulate or (
                i == len(train_loader) and accum_count > 0
            ):
                if tcfg.grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        list(G_AB.parameters()) + list(G_BA.parameters()),
                        tcfg.grad_clip_norm,
                    )
                grad_norm_G = _global_grad_norm(
                    list(G_AB.parameters()) + list(G_BA.parameters())
                )
                scaler.step(optimizer_G)
                scaler.update()
                current_scale = scaler.get_scale()
                if current_scale < 1.0:
                    print(
                        f"[warn] AMP scaler scale dropped below 1.0 ({current_scale}) at epoch {epoch+1} batch {i} — persistent overflow, check VGG/OOM"
                    )

                if not (
                    _module_parameters_are_finite(G_AB)
                    and _module_parameters_are_finite(G_BA)
                ):
                    nonfinite_g_streak += 1
                    print(
                        f"[warn] non-finite generator params after step at epoch {epoch+1} batch {i}, rolling back G to last good state (streak={nonfinite_g_streak})"
                    )
                    G_AB.load_state_dict(last_good_G_AB_state, strict=True)
                    G_BA.load_state_dict(last_good_G_BA_state, strict=True)
                    optimizer_G.load_state_dict(last_good_optimizer_G_state)
                    scaler.load_state_dict(last_good_scaler_state)
                    optimizer_G.zero_grad(set_to_none=True)
                    accum_count = 0
                    if use_amp:
                        scaler.update(new_scale=max(1.0, scaler.get_scale() / 2.0))
                    continue

                # Refresh G rollback snapshots only after finite loss/output and finite params.
                last_good_G_AB_state = _snapshot_module_to_cpu(G_AB)
                last_good_G_BA_state = _snapshot_module_to_cpu(G_BA)
                last_good_optimizer_G_state = _snapshot_optimizer_state(optimizer_G)
                last_good_scaler_state = _snapshot_scaler_state(scaler)
                nonfinite_g_streak = 0
                accum_count = 0
            else:
                grad_norm_G = 0.0  # accumulating; no step this batch

            # ---- Per-batch book-keeping ----
            loss_D_A_val = loss_D_A_val if math.isfinite(loss_D_A_val) else 0.0
            loss_D_B_val = loss_D_B_val if math.isfinite(loss_D_B_val) else 0.0

            epoch_step[i] = {
                "Batch": i,
                "Loss_G": loss_G_val,
                "Loss_D_A": loss_D_A_val,
                "Loss_D_B": loss_D_B_val,
            }
            epoch_loss_G += loss_G_val
            epoch_loss_D_A += loss_D_A_val
            epoch_loss_D_B += loss_D_B_val
            epoch_grad_norm_G += grad_norm_G

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A_val:.4f} "
                    f"Loss_D_B: {loss_D_B_val:.4f} "
                    f"GradNorm_G: {grad_norm_G:.4f}"
                )

        # ---- Epoch-level aggregation ----
        n_batches = max(1, len(train_loader))
        avg_loss_G = epoch_loss_G / n_batches
        avg_loss_D_A = epoch_loss_D_A / n_batches
        avg_loss_D_B = epoch_loss_D_B / n_batches
        avg_grad_norm_G = epoch_grad_norm_G / n_batches

        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/Generator", avg_loss_G, epoch + 1)
        writer.add_scalar("Loss/Discriminator_A", avg_loss_D_A, epoch + 1)
        writer.add_scalar("Loss/Discriminator_B", avg_loss_D_B, epoch + 1)
        writer.add_scalar("Diagnostics/GradNorm_G", avg_grad_norm_G, epoch + 1)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        current_lr_G = lr_scheduler_G.get_last_lr()[0]
        current_lr_DA = lr_scheduler_D_A.get_last_lr()[0]
        current_lr_DB = lr_scheduler_D_B.get_last_lr()[0]
        writer.add_scalar("LR/Generator", current_lr_G, epoch + 1)
        writer.add_scalar("LR/Discriminator_A", current_lr_DA, epoch + 1)
        writer.add_scalar("LR/Discriminator_B", current_lr_DB, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Avg Loss_G: {avg_loss_G:.4f}  "
            f"Avg Loss_D_A: {avg_loss_D_A:.4f}  "
            f"Avg Loss_D_B: {avg_loss_D_B:.4f}  "
            f"LR_G: {current_lr_G:.6f}"
        )

        # ---- Flush history to CSV every 5 epochs ----
        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

        # ---- Periodic checkpoint ----
        if (epoch + 1) % tcfg.save_checkpoint_every == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "config": cfg,
                    "G_AB": G_AB.state_dict(),
                    "G_BA": G_BA.state_dict(),
                    "D_A": D_A.state_dict(),
                    "D_B": D_B.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_A": optimizer_D_A.state_dict(),
                    "optimizer_D_B": optimizer_D_B.state_dict(),
                    "lr_scheduler_G_state_dict": lr_scheduler_G.state_dict(),
                    "lr_scheduler_D_A_state_dict": lr_scheduler_D_A.state_dict(),
                    "lr_scheduler_D_B_state_dict": lr_scheduler_D_B.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if use_amp else None,
                    "early_stopping_state": early_stopping.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

        # ---- Validation + early stopping ----
        if (epoch + 1) > tcfg.validation_warmup_epochs:
            save_dir = os.path.join(val_dir, f"epoch_{epoch + 1}")
            run_validation(
                epoch=epoch + 1,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                save_dir=save_dir,
                num_samples=10,
                writer=writer,
            )

        if (
            (epoch + 1) % tcfg.early_stopping_interval == 0
            and epoch + 1 >= tcfg.early_stopping_warmup
        ):
            val_metrics = calculate_metrics(
                calculator=metrics_calculator,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                writer=writer,
                epoch=epoch + 1,
            )

            avg_ssim = (
                val_metrics.get("ssim_A", 0.0) + val_metrics.get("ssim_B", 0.0)
            ) / 2.0

            # LSGAN losses are always >= 0; divergence check needs no sign fix.
            should_stop = early_stopping(
                ssim=avg_ssim,
                losses={
                    "G": avg_loss_G,
                    "D_A": avg_loss_D_A,
                    "D_B": avg_loss_D_B,
                },
            )

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

    print("\n")

    # ---- Finalization ----
    calculate_metrics(
        calculator=metrics_calculator,
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        writer=writer,
        epoch=stopped_epoch,
    )

    # Test-set inference.
    test_dir = os.path.join(model_dir, "test_images")
    writer.add_scalar("Testing Started", stopped_epoch, stopped_epoch)
    run_testing(
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        save_dir=test_dir,
        writer=writer,
        epoch=stopped_epoch,
        num_samples=tcfg.test_size,
    )

    # Final checkpoint.
    writer.add_scalar("Training Completed", stopped_epoch, stopped_epoch)
    final_ckpt = os.path.join(model_dir, f"final_checkpoint_epoch_{stopped_epoch}.pth")
    torch.save(
        {
            "epoch": stopped_epoch,
            "config": cfg,
            "G_AB": G_AB.state_dict(),
            "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D_A": optimizer_D_A.state_dict(),
            "optimizer_D_B": optimizer_D_B.state_dict(),
            "lr_scheduler_G_state_dict": lr_scheduler_G.state_dict(),
            "lr_scheduler_D_A_state_dict": lr_scheduler_D_A.state_dict(),
            "lr_scheduler_D_B_state_dict": lr_scheduler_D_B.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "early_stopping_state": early_stopping.state_dict(),
        },
        final_ckpt,
    )
    print(f"Final checkpoint saved: {final_ckpt}")

    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B
