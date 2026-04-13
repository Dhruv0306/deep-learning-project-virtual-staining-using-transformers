"""
Compare image quality metrics across three folders with matched filenames.

Usage example:
    python compare_three_folder_metrics.py \
        --source_dir data/E_Staining_DermaRepo/H_E-Staining_dataset/testA \
        --generated1_dir data/E_Staining_DermaRepo/H_E-Staining_dataset/models_v2_xxx/validation_images/genA \
        --generated2_dir data/E_Staining_DermaRepo/H_E-Staining_dataset/models_v4_xxx/validation_images/genA

Outputs are written to the parent directory of generated2_dir:
    - <model_name>_three_folder_metrics_<timestamp>.csv
    - <model_name>_three_folder_metrics_<timestamp>.md
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy import linalg

from shared.metrics import MetricsCalculator


# Allow very large pathology slides and silence PIL decompression-bomb warnings.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PairMetrics:
    image_name: str
    ssim_src_gen1: float
    psnr_src_gen1: float
    mae_src_gen1: float
    ssim_src_gen2: float
    psnr_src_gen2: float
    mae_src_gen2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare source folder with two generated folders using matched filenames. "
            "Computes per-image SSIM/PSNR/MAE and dataset-level FID."
        )
    )
    parser.add_argument("--source_dir", required=True, help="Path to source/original images")
    parser.add_argument("--generated1_dir", required=True, help="Path to generated image folder 1")
    parser.add_argument("--generated2_dir", required=True, help="Path to generated image folder 2")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for Inception feature extraction when computing FID",
    )
    parser.add_argument(
        "--output_prefix",
        default="three_folder_metrics",
        help="Prefix for output CSV/Markdown filenames",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device for Inception feature extraction and metric compute. "
            "auto uses CUDA when available."
        ),
    )
    parser.add_argument(
        "--match_mode",
        choices=["exact", "prefix"],
        default="exact",
        help=(
            "Filename matching mode: exact uses full filename (case-insensitive, with extension), "
            "prefix uses text before first dot in filename (recommended when generated files append suffixes)."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images_by_name(folder: str) -> Dict[str, str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")

    images: Dict[str, str] = {}
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            images[entry] = path
    return images


def get_match_key(filename: str, match_mode: str) -> str:
    lower_name = filename.strip().lower()
    if match_mode == "exact":
        return lower_name

    stem = os.path.splitext(lower_name)[0]
    prefix = stem.split(".")[0].strip()
    return re.sub(r"\s+", " ", prefix)


def build_keyed_map(images: Dict[str, str], match_mode: str, folder_label: str) -> Dict[str, str]:
    keyed: Dict[str, str] = {}
    duplicates = 0
    for name, path in sorted(images.items()):
        key = get_match_key(name, match_mode)
        if key in keyed:
            duplicates += 1
            continue
        keyed[key] = path

    if duplicates > 0:
        print(
            f"[WARN] {folder_label}: ignored {duplicates} duplicate keys under match mode '{match_mode}'."
        )
    return keyed


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def load_triplet_tensors(
    src_path: str,
    gen1_path: str,
    gen2_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src_img = Image.open(src_path).convert("RGB")
    gen1_img = Image.open(gen1_path).convert("RGB")
    gen2_img = Image.open(gen2_path).convert("RGB")

    if gen1_img.size != src_img.size:
        gen1_img = gen1_img.resize(src_img.size, Image.Resampling.BICUBIC)
    if gen2_img.size != src_img.size:
        gen2_img = gen2_img.resize(src_img.size, Image.Resampling.BICUBIC)

    src_t = to_normalized_tensor(src_img)
    gen1_t = to_normalized_tensor(gen1_img)
    gen2_t = to_normalized_tensor(gen2_img)

    return src_t, gen1_t, gen2_t


def compute_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    return torch.mean(torch.abs(img1 - img2)).item()


def compute_psnr_torch(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0) -> float:
    mse = torch.mean((img1 - img2) ** 2)
    if torch.isclose(mse, torch.tensor(0.0, device=mse.device, dtype=mse.dtype)):
        return float("inf")
    psnr = 20.0 * torch.log10(torch.tensor(data_range, device=mse.device, dtype=mse.dtype))
    psnr -= 10.0 * torch.log10(mse)
    return float(psnr.item())


def compute_ssim_torch(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0,
    kernel_size: int = 11,
) -> float:
    """Compute SSIM on GPU using channel-wise pooling to avoid huge CPU allocations."""
    if img1.shape != img2.shape:
        raise ValueError("SSIM input tensors must have the same shape.")
    if img1.ndim != 4:
        raise ValueError("SSIM expects tensors of shape [N, C, H, W].")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = kernel_size // 2
    channels = img1.shape[1]

    channel_scores: List[torch.Tensor] = []
    for ch in range(channels):
        x = img1[:, ch : ch + 1]
        y = img2[:, ch : ch + 1]

        mu_x = torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mu_y = torch.nn.functional.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            torch.nn.functional.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
            - mu_x_sq
        )
        sigma_y_sq = (
            torch.nn.functional.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=pad)
            - mu_y_sq
        )
        sigma_xy = (
            torch.nn.functional.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=pad)
            - mu_xy
        )

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / (denominator + 1e-12)
        channel_scores.append(ssim_map.mean())

    return float(torch.stack(channel_scores).mean().item())


def compute_fid_from_features(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("Features must be 2D arrays of shape [N, D].")

    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("Feature dimensions do not match for FID computation.")

    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("FID requires at least 2 images in each set.")

    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)

    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    eps = 1e-4
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0], dtype=sigma1.dtype)
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0], dtype=sigma2.dtype)

    diff = mu1 - mu2

    covmean_result = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    covmean = covmean_result[0] if isinstance(covmean_result, tuple) else covmean_result
    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def batched_inception_features(
    calculator: MetricsCalculator,
    tensors: Sequence[torch.Tensor],
    batch_size: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    for i in range(0, len(tensors), batch_size):
        batch_slice: List[torch.Tensor] = list(tensors[i : i + batch_size])
        batch = torch.stack(batch_slice, dim=0)
        features.append(calculator.get_inception_features(batch))
    return np.concatenate(features, axis=0)


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value)
    return float(value)


def write_csv(csv_path: str, rows: Sequence[PairMetrics]) -> None:
    fieldnames = [
        "image_name",
        "ssim_src_gen1",
        "psnr_src_gen1",
        "mae_src_gen1",
        "ssim_src_gen2",
        "psnr_src_gen2",
        "mae_src_gen2",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "ssim_src_gen1": f"{row.ssim_src_gen1:.6f}",
                    "psnr_src_gen1": f"{row.psnr_src_gen1:.6f}",
                    "mae_src_gen1": f"{row.mae_src_gen1:.6f}",
                    "ssim_src_gen2": f"{row.ssim_src_gen2:.6f}",
                    "psnr_src_gen2": f"{row.psnr_src_gen2:.6f}",
                    "mae_src_gen2": f"{row.mae_src_gen2:.6f}",
                }
            )


def write_markdown(
    md_path: str,
    model_name: str,
    source_dir: str,
    generated1_dir: str,
    generated2_dir: str,
    match_mode: str,
    total_src: int,
    total_gen1: int,
    total_gen2: int,
    matched_count: int,
    fid_src_gen1: float,
    fid_src_gen2: float,
    rows: Sequence[PairMetrics],
) -> None:
    lines: List[str] = []
    lines.append(f"# Three-Folder Metrics Report ({model_name})")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Source folder: `{source_dir}`")
    lines.append(f"- Generated1 folder: `{generated1_dir}`")
    lines.append(f"- Generated2 folder: `{generated2_dir}`")
    lines.append(f"- Match mode: `{match_mode}`")
    lines.append(f"- Source images: {total_src}")
    lines.append(f"- Generated1 images: {total_gen1}")
    lines.append(f"- Generated2 images: {total_gen2}")
    lines.append(f"- Matched filenames in all 3 folders: {matched_count}")
    lines.append("")
    lines.append("## Dataset-Level FID")
    lines.append("")
    lines.append("| Pair | FID |")
    lines.append("|---|---:|")
    lines.append(f"| Source vs Generated1 | {fid_src_gen1:.6f} |")
    lines.append(f"| Source vs Generated2 | {fid_src_gen2:.6f} |")
    lines.append("")
    lines.append("## Per-Image Metrics")
    lines.append("")
    lines.append(
        "| image_name | ssim_src_gen1 | psnr_src_gen1 | mae_src_gen1 | ssim_src_gen2 | psnr_src_gen2 | mae_src_gen2 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| "
            f"{row.image_name} | {row.ssim_src_gen1:.6f} | {row.psnr_src_gen1:.6f} | {row.mae_src_gen1:.6f} | "
            f"{row.ssim_src_gen2:.6f} | {row.psnr_src_gen2:.6f} | {row.mae_src_gen2:.6f} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    print("[DEBUG] Starting three-folder comparison...")
    print(f"[DEBUG] source_dir: {args.source_dir}")
    print(f"[DEBUG] generated1_dir: {args.generated1_dir}")
    print(f"[DEBUG] generated2_dir: {args.generated2_dir}")
    print(f"[DEBUG] match_mode: {args.match_mode}")
    print(f"[DEBUG] batch_size: {args.batch_size}")
    print(f"[DEBUG] device arg: {args.device}")

    source_map = list_images_by_name(args.source_dir)
    gen1_map = list_images_by_name(args.generated1_dir)
    gen2_map = list_images_by_name(args.generated2_dir)

    print(f"[DEBUG] source image count: {len(source_map)}")
    print(f"[DEBUG] generated1 image count: {len(gen1_map)}")
    print(f"[DEBUG] generated2 image count: {len(gen2_map)}")

    source_keyed = build_keyed_map(source_map, args.match_mode, "source")
    gen1_keyed = build_keyed_map(gen1_map, args.match_mode, "generated1")
    gen2_keyed = build_keyed_map(gen2_map, args.match_mode, "generated2")

    print(f"[DEBUG] source keyed count: {len(source_keyed)}")
    print(f"[DEBUG] generated1 keyed count: {len(gen1_keyed)}")
    print(f"[DEBUG] generated2 keyed count: {len(gen2_keyed)}")

    matched_keys = sorted(set(source_keyed) & set(gen1_keyed) & set(gen2_keyed))
    print(f"[DEBUG] matched_keys count: {len(matched_keys)}")
    if len(matched_keys) < 2:
        raise ValueError(
            "Need at least 2 matched filenames across all folders to compute reliable dataset FID. "
            "Try --match_mode prefix if generated filenames add suffixes."
        )

    output_dir = os.path.dirname(os.path.abspath(args.generated2_dir))
    model_name = os.path.basename(output_dir) or "model"
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    csv_path = os.path.join(
        output_dir,
        f"{model_name}_{args.output_prefix}_{timestamp}.csv",
    )
    md_path = os.path.join(
        output_dir,
        f"{model_name}_{args.output_prefix}_{timestamp}.md",
    )

    device = resolve_device(args.device)
    calculator = MetricsCalculator(device=device)
    print(f"Using device: {device}")
    print("[DEBUG] MetricsCalculator initialized for FID feature extraction.")

    per_image_rows: List[PairMetrics] = []
    src_tensors: List[torch.Tensor] = []
    gen1_tensors: List[torch.Tensor] = []
    gen2_tensors: List[torch.Tensor] = []

    skipped = 0
    for idx, match_key in enumerate(matched_keys, start=1):
        src_path = source_keyed[match_key]
        gen1_path = gen1_keyed[match_key]
        gen2_path = gen2_keyed[match_key]
        image_name = os.path.basename(src_path)

        print(f"[DEBUG] [{idx}/{len(matched_keys)}] key='{match_key}' image='{image_name}'")

        try:
            src_t, gen1_t, gen2_t = load_triplet_tensors(src_path, gen1_path, gen2_path)
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping unreadable image '{image_name}' (key='{match_key}'): {exc}")
            continue

        src_b = src_t.unsqueeze(0).to(device, non_blocking=True)
        gen1_b = gen1_t.unsqueeze(0).to(device, non_blocking=True)
        gen2_b = gen2_t.unsqueeze(0).to(device, non_blocking=True)

        ssim_1 = to_float(compute_ssim_torch(src_b, gen1_b))
        psnr_1 = to_float(compute_psnr_torch(src_b, gen1_b))
        mae_1 = to_float(compute_mae(src_b, gen1_b))

        ssim_2 = to_float(compute_ssim_torch(src_b, gen2_b))
        psnr_2 = to_float(compute_psnr_torch(src_b, gen2_b))
        mae_2 = to_float(compute_mae(src_b, gen2_b))

        print(
            "[DEBUG] "
            f"key='{match_key}' | src-vs-gen1: SSIM={ssim_1:.6f}, PSNR={psnr_1:.6f}, MAE={mae_1:.6f} | "
            f"src-vs-gen2: SSIM={ssim_2:.6f}, PSNR={psnr_2:.6f}, MAE={mae_2:.6f}"
        )

        per_image_rows.append(
            PairMetrics(
                image_name=image_name,
                ssim_src_gen1=ssim_1,
                psnr_src_gen1=psnr_1,
                mae_src_gen1=mae_1,
                ssim_src_gen2=ssim_2,
                psnr_src_gen2=psnr_2,
                mae_src_gen2=mae_2,
            )
        )

        src_tensors.append(src_t)
        gen1_tensors.append(gen1_t)
        gen2_tensors.append(gen2_t)

    if len(src_tensors) < 2:
        raise ValueError("Less than 2 valid matched images after filtering unreadable files.")

    src_features = batched_inception_features(calculator, src_tensors, args.batch_size)
    gen1_features = batched_inception_features(calculator, gen1_tensors, args.batch_size)
    gen2_features = batched_inception_features(calculator, gen2_tensors, args.batch_size)

    print("[DEBUG] Finished Inception feature extraction for dataset-level FID.")

    fid_src_gen1 = compute_fid_from_features(src_features, gen1_features)
    fid_src_gen2 = compute_fid_from_features(src_features, gen2_features)

    write_csv(csv_path, per_image_rows)
    write_markdown(
        md_path=md_path,
        model_name=model_name,
        source_dir=args.source_dir,
        generated1_dir=args.generated1_dir,
        generated2_dir=args.generated2_dir,
        match_mode=args.match_mode,
        total_src=len(source_map),
        total_gen1=len(gen1_map),
        total_gen2=len(gen2_map),
        matched_count=len(matched_keys),
        fid_src_gen1=fid_src_gen1,
        fid_src_gen2=fid_src_gen2,
        rows=per_image_rows,
    )

    print("Comparison complete.")
    print(f"Model label: {model_name}")
    print(f"Match mode: {args.match_mode}")
    print(f"Matched images: {len(matched_keys)}")
    if skipped > 0:
        print(f"Skipped unreadable images: {skipped}")
    print(f"Dataset FID (source vs generated1): {fid_src_gen1:.6f}")
    print(f"Dataset FID (source vs generated2): {fid_src_gen2:.6f}")
    print(f"CSV report: {csv_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
