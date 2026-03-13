"""
Training loop for the CycleGAN model.

Handles data loading, model setup, loss computation, optimization, logging,
validation, early stopping, and final evaluation.
"""

import os
import math

# Disable OneDNN optimizations to avoid numerical differences across systems.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from data_loader import getDataLoader
from discriminator import getDiscriminators
from EarlyStopping import EarlyStopping
from generator import getGenerators
from history_utils import append_history_to_csv, load_history_from_csv
from losses import CycleGANLoss
from metrics import MetricsCalculator
from testing import run_testing
from validation import calculate_metrics, run_validation


def train(epoch_size=None, num_epochs=None, model_dir=None, val_dir=None, test_size=None):
    """
    Train CycleGAN generators and discriminators.

    Args:
        epoch_size (int | None): Max samples per epoch (defaults to loader default).
        num_epochs (int | None): Number of epochs to train.
        model_dir (str | None): Directory for checkpoints and logs.
        val_dir (str | None): Directory for validation image outputs.
        test_size (float | None): Number of test samples to export in testing.

    Returns:
        tuple: (history, G_AB, G_BA, D_A, D_B)
    """
    # Backend tuning for faster convolutions on GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load training and test data.
    train_loader, test_loader = getDataLoader(
        epoch_size=3000 if epoch_size is None else epoch_size
    )
    # Initialize models.
    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Composite CycleGAN loss with perceptual components.
    loss_fn = CycleGANLoss(
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_cycle_perceptual=0.1,
        lambda_identity_perceptual=0.05,
        device=device,
    )
    # Mixed precision is used only when CUDA is available.
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    metrics_calculator = MetricsCalculator(device=device)
    early_stopping_check_interval = 10
    early_stopping_patience_epochs = 40
    early_stopping_warmup_epochs = 80
    # Early stopping is triggered by validation SSIM and loss trends.
    early_stopping = EarlyStopping(
        patience=max(
            1, math.ceil(early_stopping_patience_epochs / early_stopping_check_interval)
        ),
        min_delta=0.00001,
        divergence_threshold=5.0,
        divergence_patience=2,
    )

    # Move models to the selected device.
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)

    # Optimizers share the standard CycleGAN hyperparameters.
    lr = 0.0002
    beta1 = 0.5
    optimizer_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(beta1, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    # Linear learning rate decay after epoch 100.
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

    # Set up output directories and TensorBoard logging.
    model_dir = (
        "data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models"
        if model_dir is None
        else model_dir
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{model_dir}\\tensorboard_logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"{model_dir}\\tensorboard_logs")
    history_csv_path = os.path.join(model_dir, "training_history.csv")
    if os.path.exists(history_csv_path):
        os.remove(history_csv_path)

    stopped_epoch = num_epochs
    for epoch in range(num_epochs):
        print("\n")

        # Switch all networks to train mode each epoch.
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

            # -----------------------------
            # Generator step
            # -----------------------------
            # Freeze discriminators so they do not update during generator loss.
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

            # -----------------------------
            # Discriminator steps
            # -----------------------------
            # Re-enable discriminator gradients.
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

            # Progress log every 50 batches (and at ends).
            if i == 1 or i == len(train_loader) or i % 50 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{i}/{len(train_loader)}] "
                    f"Loss_G: {loss_G.item():.4f} "
                    f"Loss_D_A: {loss_D_A.item():.4f} "
                    f"Loss_D_B: {loss_D_B.item():.4f}"
                )

        # Store batch-level history for this epoch.
        history[epoch + 1] = epoch_step
        writer.add_scalar("Loss/Generator", epoch_loss_G / len(train_loader), epoch + 1)
        writer.add_scalar("Loss/Discriminator_A", epoch_loss_D_A / len(train_loader), epoch + 1)
        writer.add_scalar("Loss/Discriminator_B", epoch_loss_D_B / len(train_loader), epoch + 1)

        # Periodically flush training history to CSV to avoid large memory usage.
        if (epoch + 1) % 5 == 0:
            append_history_to_csv(history, history_csv_path)
            history.clear()

        # Save checkpoints every 20 epochs.
        if (epoch + 1) % 20 == 0:
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
                f"{model_dir}\\checkpoint_epoch_{epoch+1}.pth",
            )
            writer.add_scalar("Checkpoint saved", epoch + 1, epoch + 1)

        # Step learning rate schedulers.
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

        # Run qualitative validation image generation each epoch.
        if val_dir is None:
            val_dir = f"{model_dir}\\validation_images"
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

        # Compute validation metrics and check early stopping at intervals.
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
    # Final metrics on the best/last checkpoint.
    calculate_metrics(
        calculator=metrics_calculator,
        G_AB=G_AB,
        G_BA=G_BA,
        test_loader=test_loader,
        device=device,
        writer=writer,
        epoch=stopped_epoch,
    )

    # Run test set inference and save example outputs.
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

    # Save final checkpoint.
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
        },
        f"{model_dir}\\final_checkpoint_epoch_{stopped_epoch}.pth",
    )

    # Persist any remaining history and reload for a consistent return value.
    append_history_to_csv(history, history_csv_path)
    history = load_history_from_csv(history_csv_path)

    writer.close()
    return history, G_AB, G_BA, D_A, D_B
