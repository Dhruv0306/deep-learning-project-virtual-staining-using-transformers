"""
Testing helper for CycleGAN.

Runs inference on the test loader, logs summary losses, and writes
comparison images to disk.
"""

import os

import torch
import torch.nn as nn

from validation import save_images


def run_testing(
    G_AB, G_BA, test_loader, device, save_dir, writer=None, epoch=None, num_samples=None
):
    """
    Run testing/inference on the test set.

    Args:
        G_AB, G_BA (nn.Module): Generators.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Target device.
        save_dir (str): Output directory for images.
        writer (SummaryWriter | None): TensorBoard writer.
        epoch (int | None): Epoch index for logging/filenames.
        num_samples (int | None): Max number of test samples to process.
    """
    G_AB.eval()
    G_BA.eval()

    total_cycle_loss = 0
    total_identity_loss = 0
    num_samples = (
        len(test_loader) if num_samples is None else min(num_samples, len(test_loader))
    )
    num_samples = max(1, num_samples)

    os.makedirs(save_dir, exist_ok=True)

    idt_A_loss = nn.L1Loss()
    idt_B_loss = nn.L1Loss()
    cycle_A_loss = nn.L1Loss()
    cycle_B_loss = nn.L1Loss()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            print(f"Testing Image {i}.")

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)
            idt_A = G_BA(real_A)
            idt_B = G_AB(real_B)

            # Compute identity and cycle losses for logging only.
            loss_idt_A = idt_A_loss(idt_A, real_A)
            loss_idt_B = idt_B_loss(idt_B, real_B)
            loss_cycle_A = cycle_A_loss(rec_A, real_A)
            loss_cycle_B = cycle_B_loss(rec_B, real_B)
            total_cycle_loss += loss_cycle_A.item() + loss_cycle_B.item()
            total_identity_loss += loss_idt_A.item() + loss_idt_B.item()

            save_images(
                img_id=i + 1,
                real_A=real_A,
                fake_B=fake_B,
                rec_A=rec_A,
                real_B=real_B,
                fake_A=fake_A,
                rec_B=rec_B,
                epoch=epoch if epoch is not None else "test",
                save_dir=save_dir,
                is_test=True,
            )

    total_cycle_loss /= num_samples
    total_identity_loss /= num_samples
    print(
        f"Testing: Average Cycle Loss: {total_cycle_loss:.4f}, "
        f"Average Identity Loss: {total_identity_loss:.4f}"
    )

    if writer is not None:
        log_step = num_samples if epoch is None else epoch
        writer.add_scalar("Testing/Average Cycle Loss", total_cycle_loss, log_step)
        writer.add_scalar(
            "Testing/Average Identity Loss", total_identity_loss, log_step
        )

    G_AB.train()
    G_BA.train()
