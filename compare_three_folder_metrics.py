"""
Compare image quality metrics across three folders with matched filenames.
FULLY GPU-OPTIMIZED VERSION - All operations run on GPU.

Usage example:
    python compare_three_folder_metrics_gpu_optimized.py \
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

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import io as tv_io
from torchvision.transforms import functional as TF

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
            "Computes per-image SSIM/PSNR/MAE and dataset-level FID. FULLY GPU-OPTIMIZED."
        )
    )
    parser.add_argument(
        "--source_dir", required=True, help="Path to source/original images"
    )
    parser.add_argument(
        "--generated1_dir", required=True, help="Path to generated image folder 1"
    )
    parser.add_argument(
        "--generated2_dir", required=True, help="Path to generated image folder 2"
    )
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
    parser.add_argument(
        "--use_torchvision_io",
        action="store_true",
        help="Use torchvision.io for direct GPU image loading (faster, but may have compatibility issues with some formats)",
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


def build_keyed_map(
    images: Dict[str, str], match_mode: str, folder_label: str
) -> Dict[str, str]:
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


def load_image_gpu_torchvision(image_path: str, device: torch.device) -> torch.Tensor:
    """Load image directly to GPU using torchvision.io (faster but less compatible)."""
    # Read image as uint8 tensor [C, H, W] on CPU
    img = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.RGB)
    # Convert to float32 and normalize to [-1, 1]
    img = img.to(device=device, dtype=torch.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img


def load_image_gpu_pil(image_path: str, device: torch.device) -> torch.Tensor:
    """Load image using PIL then transfer to GPU (more compatible, slightly slower)."""
    img = Image.open(image_path).convert("RGB")
    # Convert to tensor on CPU, then move to GPU
    tensor = TF.to_tensor(img).to(device=device, non_blocking=True)
    # Normalize to [-1, 1]
    tensor = (tensor - 0.5) / 0.5
    return tensor


def load_triplet_tensors(
    src_path: str,
    gen1_path: str,
    gen2_path: str,
    device: torch.device,
    use_torchvision_io: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load three images and return normalized tensors on GPU."""

    load_fn = load_image_gpu_torchvision if use_torchvision_io else load_image_gpu_pil

    src_t = load_fn(src_path, device)
    gen1_t = load_fn(gen1_path, device)
    gen2_t = load_fn(gen2_path, device)

    # Resize if needed (on GPU)
    if gen1_t.shape[1:] != src_t.shape[1:]:
        gen1_t = F.interpolate(
            gen1_t.unsqueeze(0),
            size=src_t.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    if gen2_t.shape[1:] != src_t.shape[1:]:
        gen2_t = F.interpolate(
            gen2_t.unsqueeze(0),
            size=src_t.shape[1:],
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    return src_t, gen1_t, gen2_t


def compute_mae_gpu(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute MAE on GPU, return GPU tensor."""
    return torch.mean(torch.abs(img1 - img2))


def compute_psnr_gpu(
    img1: torch.Tensor, img2: torch.Tensor, data_range: float = 2.0
) -> torch.Tensor:
    """Compute PSNR on GPU, return GPU tensor."""
    mse = torch.mean((img1 - img2) ** 2)
    if torch.isclose(mse, torch.tensor(0.0, device=mse.device, dtype=mse.dtype)):
        return torch.tensor(float("inf"), device=mse.device, dtype=mse.dtype)

    psnr = 20.0 * torch.log10(
        torch.tensor(data_range, device=mse.device, dtype=mse.dtype)
    )
    psnr -= 10.0 * torch.log10(mse)
    return psnr


def compute_ssim_gpu(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Compute SSIM on GPU using channel-wise pooling, return GPU tensor."""
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

        mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=pad)

        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x_sq = (
            F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=pad)
            - mu_x_sq
        )
        sigma_y_sq = (
            F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=pad)
            - mu_y_sq
        )
        sigma_xy = (
            F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=pad) - mu_xy
        )

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / (denominator + 1e-12)
        channel_scores.append(ssim_map.mean())

    return torch.stack(channel_scores).mean()


def compute_fid_from_features_gpu(
    real_features: torch.Tensor, fake_features: torch.Tensor
) -> torch.Tensor:
    """
    Compute FID entirely on GPU using PyTorch operations.
    Returns GPU tensor.
    """
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("Features must be 2D tensors of shape [N, D].")

    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("Feature dimensions do not match for FID computation.")

    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        raise ValueError("FID requires at least 2 images in each set.")

    # Compute means
    mu1 = real_features.mean(dim=0)
    mu2 = fake_features.mean(dim=0)

    # Compute covariance matrices (on GPU)
    real_centered = real_features - mu1
    fake_centered = fake_features - mu2

    sigma1 = (real_centered.T @ real_centered) / (real_features.shape[0] - 1)
    sigma2 = (fake_centered.T @ fake_centered) / (fake_features.shape[0] - 1)

    # Add small epsilon for numerical stability
    eps = 1e-4
    sigma1 = sigma1 + eps * torch.eye(
        sigma1.shape[0], device=sigma1.device, dtype=sigma1.dtype
    )
    sigma2 = sigma2 + eps * torch.eye(
        sigma2.shape[0], device=sigma2.device, dtype=sigma2.dtype
    )

    # Compute difference of means
    diff = mu1 - mu2

    # Compute sqrt of product of covariance matrices using eigendecomposition
    # sqrtm(A @ B) using eigendecomposition for symmetric positive definite matrices
    product = sigma1 @ sigma2

    # For symmetric matrices, we can use torch.linalg.eigh (faster and more stable)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(product)
        # Ensure eigenvalues are positive (numerical stability)
        eigenvalues = torch.clamp(eigenvalues, min=0)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        covmean = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    except:
        # Fallback: use general square root (slower but more robust)
        covmean = torch.linalg.matrix_power(product, 0.5)

    # Handle potential complex values by taking real part
    if torch.is_complex(covmean):
        covmean = covmean.real

    # Compute FID
    fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


def batched_inception_features_gpu(
    calculator: MetricsCalculator,
    tensors: List[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract inception features in batches, keeping everything on GPU.
    Returns GPU tensor instead of NumPy array.
    """
    features_list: List[torch.Tensor] = []

    for i in range(0, len(tensors), batch_size):
        batch_tensors = tensors[i : i + batch_size]
        # Stack tensors (all already on GPU)
        batch = torch.stack(batch_tensors, dim=0)

        # Get features - modify calculator to return torch tensor
        batch_features = calculator.get_inception_features(batch)

        # If calculator returns numpy, convert to torch and move to GPU
        if isinstance(batch_features, torch.Tensor):
            features_list.append(batch_features)
        else:
            # Convert numpy to torch tensor on GPU
            features_list.append(torch.from_numpy(batch_features).to(device))

    # Concatenate all features on GPU
    return torch.cat(features_list, dim=0)


def to_float(value: Any) -> float:
    """Convert tensor/array to Python float."""
    if isinstance(value, torch.Tensor):
        return float(value.cpu().item())
    elif hasattr(value, "item"):
        return float(value.item())
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

    print("[DEBUG] Starting three-folder comparison (GPU-OPTIMIZED)...")
    print(f"[DEBUG] source_dir: {args.source_dir}")
    print(f"[DEBUG] generated1_dir: {args.generated1_dir}")
    print(f"[DEBUG] generated2_dir: {args.generated2_dir}")
    print(f"[DEBUG] match_mode: {args.match_mode}")
    print(f"[DEBUG] batch_size: {args.batch_size}")
    print(f"[DEBUG] device arg: {args.device}")
    print(f"[DEBUG] use_torchvision_io: {args.use_torchvision_io}")

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

    # Process images with GPU operations
    with torch.no_grad():  # Disable gradient computation for inference
        for idx, match_key in enumerate(matched_keys, start=1):
            src_path = source_keyed[match_key]
            gen1_path = gen1_keyed[match_key]
            gen2_path = gen2_keyed[match_key]
            image_name = os.path.basename(src_path)

            print(
                f"[DEBUG] [{idx}/{len(matched_keys)}] key='{match_key}' image='{image_name}'"
            )

            try:
                src_t, gen1_t, gen2_t = load_triplet_tensors(
                    src_path, gen1_path, gen2_path, device, args.use_torchvision_io
                )
            except Exception as exc:
                skipped += 1
                print(
                    f"[WARN] Skipping unreadable image '{image_name}' (key='{match_key}'): {exc}"
                )
                continue

            # Add batch dimension for metrics computation
            src_b = src_t.unsqueeze(0)
            gen1_b = gen1_t.unsqueeze(0)
            gen2_b = gen2_t.unsqueeze(0)

            # Compute metrics (all on GPU)
            ssim_1_tensor = compute_ssim_gpu(src_b, gen1_b)
            psnr_1_tensor = compute_psnr_gpu(src_b, gen1_b)
            mae_1_tensor = compute_mae_gpu(src_b, gen1_b)

            ssim_2_tensor = compute_ssim_gpu(src_b, gen2_b)
            psnr_2_tensor = compute_psnr_gpu(src_b, gen2_b)
            mae_2_tensor = compute_mae_gpu(src_b, gen2_b)

            # Convert to float only when needed for display
            ssim_1 = to_float(ssim_1_tensor)
            psnr_1 = to_float(psnr_1_tensor)
            mae_1 = to_float(mae_1_tensor)
            ssim_2 = to_float(ssim_2_tensor)
            psnr_2 = to_float(psnr_2_tensor)
            mae_2 = to_float(mae_2_tensor)

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

            # Store tensors on GPU for FID computation
            src_tensors.append(src_t)
            gen1_tensors.append(gen1_t)
            gen2_tensors.append(gen2_t)

    if len(src_tensors) < 2:
        raise ValueError(
            "Less than 2 valid matched images after filtering unreadable files."
        )

    print("[DEBUG] Computing Inception features on GPU...")
    src_features = batched_inception_features_gpu(
        calculator, src_tensors, args.batch_size, device
    )
    gen1_features = batched_inception_features_gpu(
        calculator, gen1_tensors, args.batch_size, device
    )
    gen2_features = batched_inception_features_gpu(
        calculator, gen2_tensors, args.batch_size, device
    )

    print("[DEBUG] Finished Inception feature extraction for dataset-level FID.")
    print("[DEBUG] Computing FID on GPU...")

    # Compute FID entirely on GPU
    with torch.no_grad():
        fid_src_gen1_tensor = compute_fid_from_features_gpu(src_features, gen1_features)
        fid_src_gen2_tensor = compute_fid_from_features_gpu(src_features, gen2_features)

    # Convert to float only for output
    fid_src_gen1 = to_float(fid_src_gen1_tensor)
    fid_src_gen2 = to_float(fid_src_gen2_tensor)

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
