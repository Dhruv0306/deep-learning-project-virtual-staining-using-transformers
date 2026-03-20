"""
Improved training loop for the UVCGAN v2 pipeline.

Paper-aligned implementation (Prokopenko et al., UVCGAN v2, 2023):
  - GAN objective  : LSGAN -- all losses are >= 0, always stable.
  - Gradient penalty: one-sided, gamma=100, lambda=0.1 (see LSGANGradientPenalty).
  - n_critic        : 1 -- standard for LSGAN, no multi-step D updates needed.
  - Adam betas      : (0.5, 0.999), lr=2e-4.
  - AMP safety      : GP is always computed in float32 outside autocast.
  - Cross-domain    : generator_loss() activates forward_with_cross_domain()
                      automatically when both generators support it.

Additional engineering improvements (kept from previous version):
  - Warm-up + linear decay LR schedule.
  - Gradient clipping.
  - EarlyStopping based on SSIM + loss divergence.
    NOTE: With LSGAN all losses are >= 0, so the divergence check works
    correctly without any sign correction (no _abs_losses wrapper needed).
  - Replay buffer for discriminator stabilisation.
  - TensorBoard logging of losses, LR, and grad norms.
  - Periodic CSV history flush and epoch checkpoints.

Entry point: train_v2().
"""

import math
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from advanced_losses import UVCGANLoss
from config import UVCGANConfig, get_default_config
from data_loader import getDataLoader
from EarlyStopping import EarlyStopping
from history_utils import append_history_to_csv, load_history_from_csv
from metrics import MetricsCalculator
from spectral_norm_discriminator import getDiscriminatorsV2
from testing import run_testing
from uvcgan_v2_generator import getGeneratorsV2
from validation import calculate_metrics, run_validation


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def _make_lr_lambda(warmup: int, decay_start: int, total: int):
    """
    LambdaLR schedule:
      [0,        warmup)      : linear ramp  0 -> 1
      [warmup,   decay_start) : constant 1
      [decay_start, total)    : linear decay 1 -> 0
    """

    def lr_lambda(epoch: int) -> float:
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
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_v2(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    cfg: Optional[UVCGANConfig] = None,
):
    """
    Train the UVCGAN v2 generators and multi-scale discriminators.

    Args:
        epoch_size  : Max samples per epoch (overrides cfg).
        num_epochs  : Number of epochs to train (overrides cfg).
        model_dir   : Directory for checkpoints and logs (overrides cfg).
        val_dir     : Directory for validation images (overrides cfg).
        test_size   : Number of test samples to export (overrides cfg).
        cfg         : UVCGANConfig with all hyperparameters.
                      A default v2 config is created when None.

    Returns:
        tuple: (history, G_AB, G_BA, D_A, D_B)
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
    # Generator step uses AMP when available.
    # Discriminator step (including GP) always runs in float32 -- the
    # autocast(enabled=False) in UVCGANLoss.discriminator_loss handles this.
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
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=tcfg.lr,
        betas=(tcfg.beta1, tcfg.beta2),
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
    if os.path.exists(history_csv_path):
        os.remove(history_csv_path)

    num_epochs = tcfg.num_epochs
    history = {}
    stopped_epoch = num_epochs
    accumulate = max(1, tcfg.accumulate_grads)

    if accumulate > 1:
        print(
            f"[train_v2] Gradient accumulation enabled: accumulate_grads={accumulate}, "
            f"batch_size={dtcfg.batch_size} → effective batch = {accumulate * dtcfg.batch_size}"
        )
    if gcfg.use_gradient_checkpointing:
        print(
            "[train_v2] Gradient checkpointing enabled: ~30-40% less activation VRAM, ~20% slower backward."
        )

    for epoch in range(num_epochs):
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

        writer.add_scalar("Epoch", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ==================================================
            # Discriminator step(s)  (n_critic times per G step)
            # For LSGAN n_critic=1, matching the standard CycleGAN protocol.
            # ==================================================
            for p in G_AB.parameters():
                p.requires_grad_(False)
            for p in G_BA.parameters():
                p.requires_grad_(False)
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            loss_D_A_accum = 0.0
            loss_D_B_accum = 0.0

            for _ in range(n_critic):
                with torch.no_grad():
                    fake_B_d = G_AB(real_A)
                    fake_A_d = G_BA(real_B)

                # Discriminator loss call includes the one-sided GP.
                # UVCGANLoss.discriminator_loss internally disables autocast
                # for the GP computation -- safe regardless of use_amp.
                optimizer_D_A.zero_grad(set_to_none=True)
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A_d, loss_fn.fake_A_buffer
                )
                loss_D_A.backward()
                if tcfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        D_A.parameters(), tcfg.grad_clip_norm
                    )
                optimizer_D_A.step()

                optimizer_D_B.zero_grad(set_to_none=True)
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B_d, loss_fn.fake_B_buffer
                )
                loss_D_B.backward()
                if tcfg.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        D_B.parameters(), tcfg.grad_clip_norm
                    )
                optimizer_D_B.step()

                loss_D_A_accum += loss_D_A.item()
                loss_D_B_accum += loss_D_B.item()

            loss_D_A_val = loss_D_A_accum / n_critic
            loss_D_B_val = loss_D_B_accum / n_critic

            # ==================================================
            # Generator step  (with gradient accumulation)
            # Gradients are accumulated over `accumulate` batches before
            # the optimiser steps, so the effective batch size is
            # batch_size * accumulate without increasing peak VRAM.
            # ==================================================
            for p in G_AB.parameters():
                p.requires_grad_(True)
            for p in G_BA.parameters():
                p.requires_grad_(True)
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)

            # Zero gradients only at the start of an accumulation window.
            if (i - 1) % accumulate == 0:
                optimizer_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
                )
                # Scale loss so gradients are equivalent to a single large batch.
                loss_G_scaled = loss_G / accumulate

            scaler.scale(loss_G_scaled).backward()

            # Step optimiser only when the accumulation window is complete.
            if i % accumulate == 0 or i == len(train_loader):
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
            else:
                grad_norm_G = 0.0  # not stepping this batch

            # ---- Book-keeping ----
            epoch_step[i] = {
                "Batch": i,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A_val,
                "Loss_D_B": loss_D_B_val,
            }
            epoch_loss_G += loss_G.item()
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

        # ---- Periodic CSV flush ----
        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

        # ---- Checkpoint every 20 epochs ----
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_AB": G_AB.state_dict(),
                    "G_BA": G_BA.state_dict(),
                    "D_A": D_A.state_dict(),
                    "D_B": D_B.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_A": optimizer_D_A.state_dict(),
                    "optimizer_D_B": optimizer_D_B.state_dict(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

        # ---- Periodic validation + early stopping ----
        if (epoch + 1) > tcfg.validation_warmup_epochs:
            run_validation(
                epoch=epoch + 1,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                save_dir=val_dir,
                num_samples=3,
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

            # With LSGAN all losses are >= 0, so the divergence check works
            # correctly without any sign correction.
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

    # ---- Final metrics on test set ----
    calculate_metrics(
        calculator=metrics_calculator,
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        writer=writer,
        epoch=stopped_epoch,
    )

    # ---- Final test-set inference ----
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

    # ---- Final checkpoint ----
    writer.add_scalar("Training Completed", stopped_epoch, stopped_epoch)
    final_ckpt = os.path.join(model_dir, f"final_checkpoint_epoch_{stopped_epoch}.pth")
    torch.save(
        {
            "epoch": stopped_epoch,
            "G_AB": G_AB.state_dict(),
            "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D_A": optimizer_D_A.state_dict(),
            "optimizer_D_B": optimizer_D_B.state_dict(),
        },
        final_ckpt,
    )
    print(f"Final checkpoint saved: {final_ckpt}")

    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B
