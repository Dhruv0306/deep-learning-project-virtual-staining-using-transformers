"""
model_v1/training_loop.py — v1 hybrid CycleGAN/UVCGAN training loop.

Entry points:
    train(...)    — primary implementation.
    train_v1(...) — thin alias for import consistency with v2/v3/v4.

Per-batch update order:
    1. Generator step   — freeze D, compute loss_G via CycleGANLoss, AMP backward.
    2. Discriminator A  — unfreeze D, compute loss_D_A, AMP backward.
    3. Discriminator B  — compute loss_D_B, AMP backward.

Loss terms (CycleGANLoss):
    LSGAN (1.0) + cycle (λ=10) + identity (λ=5, decays after 50%) +
    perceptual cycle (λ=0.2) + perceptual identity (λ=0.1) + two-sided GP (λ=10).

Scheduler: LambdaLR — constant until epoch 100, then linear decay to 0.
"""

import os
import math
import pickle

# Disable OneDNN optimizations to avoid numerical differences across systems.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from shared.data_loader import getDataLoader
from model_v1.discriminator import getDiscriminators
from shared.EarlyStopping import EarlyStopping
from model_v1.generator import getGenerators
from shared.history_utils import append_history_to_csv, load_history_from_csv
from model_v1.losses import CycleGANLoss
from shared.metrics import MetricsCalculator
from shared.testing import run_testing
from shared.validation import calculate_metrics, run_validation
from config import get_default_config


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


def train(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint=None,
    cfg=None,
):
    """
    Train v1 generators and discriminators.

    Args:
        epoch_size: Max samples per epoch (defaults to 3000).
        num_epochs:  Total training epochs (defaults to 200).
        model_dir:   Output directory for checkpoints and logs.
        val_dir:     Directory for per-epoch validation image grids.
        test_size:   Images exported during final test-set inference.
        resume_checkpoint: Path to a v1 .pth checkpoint to resume from.
        cfg:         UVCGANConfig instance (defaults to get_default_config(1)).

    Returns:
        tuple: (history, G_AB, G_BA, D_A, D_B)
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = cfg if cfg is not None else get_default_config(model_version=1)
    tcfg = cfg.training

    train_loader, test_loader = getDataLoader(
        epoch_size=3000 if epoch_size is None else epoch_size
    )
    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CycleGANLoss bundles LSGAN + cycle + identity + perceptual + two-sided GP.
    loss_fn = CycleGANLoss(
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_cycle_perceptual=0.2,
        lambda_identity_perceptual=0.1,
        lambda_gp=10.0,
        perceptual_resize=160,
        device=device,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    metrics_calculator = MetricsCalculator(device=device)
    # Patience is in epochs; convert to check-count for EarlyStopping.
    early_stopping_check_interval = 10
    early_stopping_patience_epochs = 40
    early_stopping_warmup_epochs = 80
    early_stopping = EarlyStopping(
        patience=max(
            1, math.ceil(early_stopping_patience_epochs / early_stopping_check_interval)
        ),
        min_delta=0.00001,
        divergence_threshold=5.0,
        divergence_patience=2,
    )

    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    # AdamW for the Transformer-based generator; Adam for discriminators.
    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.AdamW(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr,
        betas=(beta1, 0.999),
        weight_decay=0.01,
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    # Constant LR until epoch 100, then linear decay to 0.
    lr_scheduler_G = optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100
    )

    num_epochs = 200 if num_epochs is None else num_epochs
    history = {}

    model_dir = (
        os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset", "models")
        if model_dir is None
        else model_dir
    )
    os.makedirs(model_dir, exist_ok=True)
    tb_dir = os.path.join(model_dir, "tensorboard_logs")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path) and not resume_checkpoint:
        os.remove(history_csv_path)

    start_epoch = 0
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(
                f"resume_checkpoint does not exist: {resume_checkpoint}"
            )
        print(f"[train_v1] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = _load_checkpoint_compat(resume_checkpoint, map_location=device)

        # Support both old key names ("G_AB") and new ones ("G_AB_state_dict").
        G_AB_state = checkpoint.get("G_AB") or checkpoint.get("G_AB_state_dict")
        G_BA_state = checkpoint.get("G_BA") or checkpoint.get("G_BA_state_dict")
        D_A_state = checkpoint.get("D_A") or checkpoint.get("D_A_state_dict")
        D_B_state = checkpoint.get("D_B") or checkpoint.get("D_B_state_dict")
        if G_AB_state is None or G_BA_state is None:
            raise KeyError(
                "Checkpoint missing generator weights required for v1 resume."
            )

        G_AB.load_state_dict(G_AB_state)
        G_BA.load_state_dict(G_BA_state)
        if D_A_state is not None:
            D_A.load_state_dict(D_A_state)
        if D_B_state is not None:
            D_B.load_state_dict(D_B_state)

        if "optimizer_G" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        if "optimizer_D_A" in checkpoint:
            optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A"])
        if "optimizer_D_B" in checkpoint:
            optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B"])

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
        if start_epoch >= num_epochs:
            raise ValueError(
                f"num_epochs ({num_epochs}) must be greater than checkpoint epoch ({start_epoch})."
            )
        print(f"[train_v1] Resume start epoch: {start_epoch + 1}")

    stopped_epoch = num_epochs
    for epoch in range(start_epoch, num_epochs):
        print("\n")

        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()
        epoch_step = {}
        epoch_loss_G = 0.0
        epoch_loss_D_A = 0.0
        epoch_loss_D_B = 0.0

        writer.add_scalar("Epoch: ", epoch + 1, epoch + 1)

        for i, batch in enumerate(train_loader):
            i += 1

            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ---- Generator step ----
            # Freeze D so its gradients are not computed during G backward.
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
            scaler.step(optimizer=optimizer_G)
            scaler.update()

            # ---- Discriminator steps ----
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
            scaler.step(optimizer_D_A)
            scaler.update()

            optimizer_D_B.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss_D_B = loss_fn.discriminator_loss(
                    D_B, real_B, fake_B, loss_fn.fake_B_buffer
                )
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
            scaler.update()

            epoch_step[i] = {
                "Batch": i,
                "Loss_G": loss_G.item(),
                "Loss_D_A": loss_D_A.item(),
                "Loss_D_B": loss_D_B.item(),
            }
            epoch_loss_G += loss_G.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()

            if i == 1 or i == len(train_loader) or i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/Generator", epoch_loss_G / len(train_loader), epoch + 1)
        writer.add_scalar(
            "Loss/Discriminator_A", epoch_loss_D_A / len(train_loader), epoch + 1
        )
        writer.add_scalar(
            "Loss/Discriminator_B", epoch_loss_D_B / len(train_loader), epoch + 1
        )

        # Flush history to CSV every 5 epochs to bound in-memory growth.
        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

        # Periodic checkpoint (every save_checkpoint_every epochs).
        if (epoch + 1) % tcfg.save_checkpoint_every == 0:
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
                os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )
            writer.add_scalar("Checkpoint saved", epoch + 1, epoch + 1)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Learning Rate G: {lr_scheduler_G.get_last_lr()[0]:.6f} "
            f"Learning Rate D_A: {lr_scheduler_D_A.get_last_lr()[0]:.6f} "
            f"Learning Rate D_B: {lr_scheduler_D_B.get_last_lr()[0]:.6f}"
        )

        writer.add_scalar(
            "Learning Rate/Generator", lr_scheduler_G.get_last_lr()[0], epoch + 1
        )
        writer.add_scalar(
            "Learning Rate/Discriminator_A",
            lr_scheduler_D_A.get_last_lr()[0],
            epoch + 1,
        )
        writer.add_scalar(
            "Learning Rate/Discriminator_B",
            lr_scheduler_D_B.get_last_lr()[0],
            epoch + 1,
        )

        # Full checkpoint with optimizer/scheduler/scaler states for resume.
        if (epoch + 1) % tcfg.save_checkpoint_every == 0:
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
                os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )
            writer.add_scalar("Checkpoint saved", epoch + 1, epoch + 1)

        if val_dir is None:
            val_dir = os.path.join(model_dir, "validation_images")
        save_dir = os.path.join(val_dir, f"epoch_{epoch+1}")
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

        # Compute metrics and check early stopping every check_interval epochs.
        if (epoch + 1) % early_stopping_check_interval == 0:
            avg_metrics = calculate_metrics(
                calculator=metrics_calculator,
                G_AB=G_AB,
                G_BA=G_BA,
                test_loader=test_loader,
                device=device,
                writer=writer,
                epoch=epoch + 1,
            )
            print(f"Epoch {epoch + 1} Validation Metrics: {avg_metrics}")
            avg_ssim = (avg_metrics.get("ssim_A", 0) + avg_metrics.get("ssim_B", 0)) / 2
            num_train_batches = max(1, len(train_loader))
            avg_loss_G = epoch_loss_G / num_train_batches
            avg_loss_D_A = epoch_loss_D_A / num_train_batches
            avg_loss_D_B = epoch_loss_D_B / num_train_batches
            tracked_losses = {
                "G": avg_loss_G,
                "D_A": avg_loss_D_A,
                "D_B": avg_loss_D_B,
            }

            should_stop = False
            if (epoch + 1) >= early_stopping_warmup_epochs:
                should_stop = early_stopping(avg_ssim, tracked_losses)
            print(
                f"EarlyStopping status | "
                f"epoch={epoch + 1} "
                f"avg_ssim={avg_ssim:.6f} "
                f"loss_G={avg_loss_G:.6f} "
                f"loss_D_A={avg_loss_D_A:.6f} "
                f"loss_D_B={avg_loss_D_B:.6f} "
                f"best_ssim={early_stopping.best_ssim:.6f} "
                f"best_loss_G={early_stopping.best_losses.get('G', float('nan')):.6f} "
                f"best_loss_D_A={early_stopping.best_losses.get('D_A', float('nan')):.6f} "
                f"best_loss_D_B={early_stopping.best_losses.get('D_B', float('nan')):.6f} "
                f"counter={early_stopping.counter}/{early_stopping.patience} "
                f"div_counter={early_stopping.divergence_counter}/{early_stopping.divergence_patience} "
                f"warmup_until={early_stopping_warmup_epochs} "
                f"stop={should_stop}"
            )
            writer.add_scalar("EarlyStopping/avg_ssim", avg_ssim, epoch + 1)
            writer.add_scalar("EarlyStopping/loss_G", avg_loss_G, epoch + 1)
            writer.add_scalar("EarlyStopping/loss_D_A", avg_loss_D_A, epoch + 1)
            writer.add_scalar("EarlyStopping/loss_D_B", avg_loss_D_B, epoch + 1)
            writer.add_scalar(
                "EarlyStopping/best_ssim", early_stopping.best_ssim, epoch + 1
            )
            writer.add_scalar(
                "EarlyStopping/best_loss_G",
                early_stopping.best_losses.get("G", float("nan")),
                epoch + 1,
            )
            writer.add_scalar(
                "EarlyStopping/best_loss_D_A",
                early_stopping.best_losses.get("D_A", float("nan")),
                epoch + 1,
            )
            writer.add_scalar(
                "EarlyStopping/best_loss_D_B",
                early_stopping.best_losses.get("D_B", float("nan")),
                epoch + 1,
            )
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
        num_samples=200 if test_size is None else int(test_size),
    )

    writer.add_scalar("Training Completed", stopped_epoch, stopped_epoch)
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
            "lr_scheduler_G_state_dict": lr_scheduler_G.state_dict(),
            "lr_scheduler_D_A_state_dict": lr_scheduler_D_A.state_dict(),
            "lr_scheduler_D_B_state_dict": lr_scheduler_D_B.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "early_stopping_state": early_stopping.state_dict(),
        },
        os.path.join(model_dir, f"final_checkpoint_epoch_{stopped_epoch}.pth"),
    )

    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B


def train_v1(*args, **kwargs):
    """Alias for train() — keeps import paths consistent with v2/v3/v4."""
    return train(*args, **kwargs)
