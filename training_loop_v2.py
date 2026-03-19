"""
Improved training loop for the UVCGAN v2 pipeline.

Key improvements over ``training_loop.py`` (v1):

* **Gradient clipping** – clips the global gradient norm before every
  optimizer step to prevent explosive gradients.
* **Warm-up + linear decay learning rate schedule** – a short warm-up phase
  ramps the LR from zero to the base value before the standard linear decay
  schedule takes over.
* **Multi-scale discriminator support** – handles the list of logit maps
  returned by :class:`~spectral_norm_discriminator.MultiScaleDiscriminator`.
* **Better logging and diagnostics** – logs additional scalars (LR, gradient
  norms, contrastive/spectral loss components) to TensorBoard.
* **Configurable via** :class:`~config.UVCGANConfig` – all hyperparameters
  are read from the config object, making experiments reproducible.

Entry point: :func:`train_v2`.
"""

import math
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from advanced_losses import AdvancedCycleGANLoss
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
# Learning rate schedule helpers
# ---------------------------------------------------------------------------


def _make_lr_lambda(warmup: int, decay_start: int, total: int):
    """
    Return a ``lr_lambda`` function for :class:`~torch.optim.lr_scheduler.LambdaLR`.

    Schedule:
    * Epochs [0, warmup): linear ramp from 0 → 1.
    * Epochs [warmup, decay_start): constant at 1.
    * Epochs [decay_start, total): linear decay from 1 → 0.

    Args:
        warmup: Number of warm-up epochs.
        decay_start: Epoch at which linear decay begins.
        total: Total number of training epochs.

    Returns:
        Callable ``(epoch) -> float``.
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
    """
    Compute the global gradient L2 norm across an iterable of parameters.

    Args:
        parameters: Iterable of ``nn.Parameter`` objects.

    Returns:
        Gradient norm as a Python float (0.0 when no gradients exist).
    """
    grads = [
        p.grad.detach().float()
        for p in parameters
        if p.grad is not None
    ]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack([g.norm() for g in grads])))


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_v2(cfg: UVCGANConfig = None):
    """
    Train the UVCGAN v2 generators and multi-scale discriminators.

    Args:
        cfg: :class:`~config.UVCGANConfig` with all hyperparameters.  A
            default v2 config is created when ``None`` is passed.

    Returns:
        tuple: ``(history, G_AB, G_BA, D_A, D_B)``
    """
    if cfg is None:
        cfg = get_default_config(model_version=2)

    tcfg = cfg.training
    lcfg = cfg.loss
    gcfg = cfg.generator
    dcfg = cfg.discriminator

    # ---- Backend tuning ----
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---- Data ----
    train_loader, test_loader = getDataLoader(epoch_size=tcfg.epoch_size)

    # ---- Models ----
    G_AB, G_BA = getGeneratorsV2(
        base_channels=gcfg.base_channels,
        vit_depth=gcfg.vit_depth,
        vit_heads=gcfg.vit_heads,
        vit_mlp_ratio=gcfg.vit_mlp_ratio,
        vit_dropout=gcfg.vit_dropout,
        layerscale_init=gcfg.layerscale_init,
        use_cross_domain=gcfg.use_cross_domain,
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
    loss_fn = AdvancedCycleGANLoss(
        lambda_cycle=lcfg.lambda_cycle,
        lambda_identity=lcfg.lambda_identity,
        lambda_cycle_perceptual=lcfg.lambda_cycle_perceptual,
        lambda_identity_perceptual=lcfg.lambda_identity_perceptual,
        lambda_gp=lcfg.lambda_gp,
        lambda_contrastive=lcfg.lambda_contrastive,
        lambda_spectral=lcfg.lambda_spectral,
        perceptual_resize=lcfg.perceptual_resize,
        use_wgan_gp=lcfg.use_wgan_gp,
        contrastive_temperature=lcfg.contrastive_temperature,
        device=device,
    )

    # ---- AMP ----
    use_amp = tcfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Metrics / early stopping ----
    metrics_calculator = MetricsCalculator(device=device)
    early_stopping = EarlyStopping(
        patience=max(
            1,
            math.ceil(
                tcfg.early_stopping_patience / tcfg.early_stopping_interval
            ),
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

            # --------------------------------------------------
            # Generator step
            # --------------------------------------------------
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            optimizer_G.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                loss_G, fake_A, fake_B = loss_fn.generator_loss(
                    real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
                )

            scaler.scale(loss_G).backward()
            # Unscale before clipping so the threshold is in the same units.
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

            # --------------------------------------------------
            # Discriminator steps
            # --------------------------------------------------
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)

            optimizer_D_A.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_A = loss_fn.discriminator_loss(
                    D_A, real_A, fake_A, loss_fn.fake_A_buffer
                )
            scaler.scale(loss_D_A).backward()
            if tcfg.grad_clip_norm > 0.0:
                scaler.unscale_(optimizer_D_A)
                torch.nn.utils.clip_grad_norm_(
                    D_A.parameters(), tcfg.grad_clip_norm
                )
            scaler.step(optimizer_D_A)
            scaler.update()

            optimizer_D_B.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B, loss_fn.fake_B_buffer
                )
            scaler.scale(loss_D_B).backward()
            if tcfg.grad_clip_norm > 0.0:
                scaler.unscale_(optimizer_D_B)
                torch.nn.utils.clip_grad_norm_(
                    D_B.parameters(), tcfg.grad_clip_norm
                )
            scaler.step(optimizer_D_B)
            scaler.update()

            # --------------------------------------------------
            # Book-keeping
            # --------------------------------------------------
            epoch_step[i] = {
                "Batch": i,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A.item(),
                "Loss_D_B": loss_D_B.item(),
            }
            epoch_loss_G += loss_G.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()
            epoch_grad_norm_G += grad_norm_G

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f} "
                    f"GradNorm_G: {grad_norm_G:.4f}"
                )

        # ---- Epoch-level logging ----
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
            f"LR_G: {current_lr_G:.6f} "
            f"LR_D_A: {current_lr_DA:.6f} "
            f"LR_D_B: {current_lr_DB:.6f}"
        )

        # ---- Periodic CSV flush ----
        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

        # ---- Checkpoint ----
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(
                model_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
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
            writer.add_scalar("Checkpoint saved", epoch + 1, epoch + 1)

        # ---- Validation images ----
        save_dir = os.path.join(val_dir, f"epoch_{epoch + 1}")
        writer.add_scalar("Validation Started", epoch + 1, epoch + 1)
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

        # ---- Early stopping ----
        if (epoch + 1) % tcfg.early_stopping_interval == 0:
            avg_metrics = calculate_metrics(
                calculator=metrics_calculator,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                writer=writer,
                epoch=epoch + 1,
            )
            avg_ssim = (
                avg_metrics.get("ssim_A", 0) + avg_metrics.get("ssim_B", 0)
            ) / 2
            tracked_losses = {
                "G": avg_loss_G,
                "D_A": avg_loss_D_A,
                "D_B": avg_loss_D_B,
            }

            should_stop = False
            if (epoch + 1) >= tcfg.early_stopping_warmup:
                should_stop = early_stopping(avg_ssim, tracked_losses)

            print(
                f"EarlyStopping | epoch={epoch + 1} "
                f"avg_ssim={avg_ssim:.6f} "
                f"loss_G={avg_loss_G:.6f} "
                f"counter={early_stopping.counter}/{early_stopping.patience} "
                f"stop={should_stop}"
            )
            writer.add_scalar("EarlyStopping/avg_ssim", avg_ssim, epoch + 1)
            writer.add_scalar("EarlyStopping/loss_G", avg_loss_G, epoch + 1)
            writer.add_scalar("EarlyStopping/counter", early_stopping.counter, epoch + 1)

            if should_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                stopped_epoch = epoch + 1
                break

    print()

    # ---- Final evaluation ----
    calculate_metrics(
        calculator=metrics_calculator,
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        writer=writer,
        epoch=stopped_epoch,
    )

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

    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B
