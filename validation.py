"""
Validation helpers for CycleGAN.

Provides metric computation and image saving utilities used during
training and testing.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont


def calculate_metrics(calculator, G_AB, G_BA, test_loader, device, writer, epoch):
    """
    Compute validation metrics on a subset of the test loader.

    Args:
        calculator (MetricsCalculator): Metric computation helper.
        G_AB, G_BA (nn.Module): Generators.
        test_loader (DataLoader): Test/validation data loader.
        device (torch.device): Target device.
        writer (SummaryWriter): TensorBoard writer.
        epoch (int): Current epoch number.

    Returns:
        dict: Average metrics (ssim, psnr, optional fid).
    """
    G_AB.eval()
    G_BA.eval()

    val_metrics = {"ssim_A": [], "ssim_B": [], "psnr_A": [], "psnr_B": []}
    real_B_list, fake_B_list = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 50:
                break

            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            batch_metrics = calculator.evaluate_batch(real_A, real_B, fake_A, fake_B)
            for key, value in batch_metrics.items():
                val_metrics[key].append(value)

            real_B_list.append(real_B)
            fake_B_list.append(fake_B)

    avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}

    # Compute FID on a small subset to reduce overhead.
    if len(real_B_list) > 10:
        real_B_tensor = torch.cat(real_B_list[:10])
        fake_B_tensor = torch.cat(fake_B_list[:10])
        fid_score = calculator.evaluate_fid(real_B_tensor, fake_B_tensor)
        avg_metrics["fid"] = fid_score

    for metric_name, value in avg_metrics.items():
        writer.add_scalar(f"Validation/{metric_name}", value, epoch)

    print(
        f"Validation Metrics - SSIM_A: {avg_metrics['ssim_A']:.4f}, "
        f"SSIM_B: {avg_metrics['ssim_B']:.4f}, "
        f"PSNR_A: {avg_metrics['psnr_A']:.2f}, "
        f"PSNR_B: {avg_metrics['psnr_B']:.2f}"
    )

    if "fid" in avg_metrics:
        print(f"FID Score: {avg_metrics['fid']:.2f}")

    G_AB.train()
    G_BA.train()
    return avg_metrics


def run_validation(
    epoch, G_AB, G_BA, test_loader, device, save_dir, num_samples=3, writer=None
):
    """
    Run qualitative validation and save comparison images.

    Args:
        epoch (int): Current epoch.
        G_AB, G_BA (nn.Module): Generators.
        test_loader (DataLoader): Test/validation loader.
        device (torch.device): Target device.
        save_dir (str): Output directory for images.
        num_samples (int): Number of samples to visualize.
        writer (SummaryWriter | None): TensorBoard writer.
    """
    G_AB.eval()
    G_BA.eval()

    total_cycle_loss = 0
    total_identity_loss = 0
    num_samples = min(num_samples, len(test_loader))
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
            print(f"Validating Image {i}.")

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
                epoch=epoch,
                save_dir=save_dir,
            )

    total_cycle_loss /= num_samples
    total_identity_loss /= num_samples
    print(
        f"Validation Epoch {epoch}: Average Cycle Loss: {total_cycle_loss:.4f}, "
        f"Average Identity Loss: {total_identity_loss:.4f}"
    )
    if writer is not None:
        writer.add_scalar("Validation/Average Cycle Loss", total_cycle_loss, epoch)
        writer.add_scalar(
            "Validation/Average Identity Loss", total_identity_loss, epoch
        )
    G_AB.train()
    G_BA.train()


def save_images_with_title(
    row_tensor, labels, out_path, value_range=(-1, 1), header_h=36
):
    """
    Save a 1x4 grid of images with a text header row.

    Args:
        row_tensor (torch.Tensor): Tensor with shape [4, C, H, W].
        labels (list[str]): Text labels for each column.
        out_path (str): Output image path.
        value_range (tuple): Min/max for normalization.
        header_h (int): Header height in pixels.
    """
    grid = make_grid(row_tensor, nrow=4, normalize=True, value_range=value_range)
    grid_img = to_pil_image(grid)  # PIL RGB

    w, h = grid_img.size
    cell_w = w // 4

    canvas = Image.new("RGB", (w, h + header_h), color=(255, 255, 255))
    canvas.paste(grid_img, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for i, text in enumerate(labels):
        x_center = i * cell_w + cell_w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x_center - tw // 2, (header_h - th) // 2), text, fill=(0, 0, 0), font=font
        )

    canvas.save(out_path)


def save_images(
    img_id,
    real_A,
    fake_B,
    rec_A,
    real_B,
    fake_A,
    rec_B,
    epoch,
    save_dir=None,
    is_test=False,
):
    """
    Save validation or test image rows for both domains.

    Args:
        img_id (int): Image index for naming.
        real_A, fake_B, rec_A, real_B, fake_A, rec_B (torch.Tensor): Image tensors.
        epoch (int | str): Epoch identifier for filenames.
        save_dir (str | None): Output directory.
        is_test (bool): Whether this is a test run.
    """
    filename_A = (
        (
            f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\validation_images\\epoch_{epoch}_A.png"
            if not is_test
            else f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\test_images\\epoch_{epoch}_A.png"
        )
        if save_dir is None
        else f"{save_dir}\\image_{img_id}_A.png"
    )

    filename_B = (
        (
            f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\validation_images\\epoch_{epoch}_B.png"
            if not is_test
            else f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models\\test_images\\epoch_{epoch}_B.png"
        )
        if save_dir is None
        else f"{save_dir}\\image_{img_id}_B.png"
    )

    row_A = (
        torch.cat([real_A[:1], fake_B[:1], rec_A[:1], real_B[:1]], dim=0).detach().cpu()
    )
    row_B = (
        torch.cat([real_B[:1], fake_A[:1], rec_B[:1], real_A[:1]], dim=0).detach().cpu()
    )

    save_images_with_title(
        row_A,
        labels=["Real A", "Fake B", "Rec A", "Real B"],
        out_path=filename_A,
        value_range=(-1, 1),
    )
    save_images_with_title(
        row_B,
        labels=["Real B", "Fake A", "Rec B", "Real A"],
        out_path=filename_B,
        value_range=(-1, 1),
    )
    print("Validation images saved." if not is_test else "Testing images saved.")
