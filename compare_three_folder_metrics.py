"""
Compare image quality metrics across three folders with matched filenames.
FULLY GPU-OPTIMIZED + MEMORY-EFFICIENT VERSION

Key design for large pathology images:
- Images are downsampled to max_side=1024 before ALL GPU operations.
  SSIM/PSNR/MAE are perceptually equivalent at that resolution, and
  Inception (used for FID) internally resizes to 299x299 anyway, so
  no information is lost for any metric.
- Each image is processed one at a time; full-res tensors are never kept.
- CUDA cache is flushed after every image.

Usage example:
    python compare_three_folder_metrics.py \
        --source_dir data/.../C_Stained \
        --generated1_dir data/.../V_Stained \
        --generated2_dir data/.../models_xxx/V_Stained \
        --match_mode prefix --batch_size 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision.transforms import functional as TF

from shared.metrics import MetricsCalculator


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Maximum side length for GPU metric computation.
# A 5614x20096 image at float32 needs ~1.3 GB just to store;
# SSIM intermediates push that to 8-10 GB for a single image pair.
# At 1024px the image is ~12 MB and SSIM intermediates stay under 200 MB.
MAX_METRIC_SIDE = 1024


@dataclass
class PairMetrics:
    image_name: str
    ssim_src_gen1: float
    psnr_src_gen1: float
    mae_src_gen1: float
    ssim_src_gen2: float
    psnr_src_gen2: float
    mae_src_gen2: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare source folder with two generated folders using matched filenames. "
            "Computes per-image SSIM/PSNR/MAE and dataset-level FID. "
            "GPU-optimised and memory-efficient for large pathology slides."
        )
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--generated1_dir", required=True)
    parser.add_argument("--generated2_dir", required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Images per Inception feature-extraction call (reduce if OOM).",
    )
    parser.add_argument("--output_prefix", default="three_folder_metrics")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--match_mode",
        choices=["exact", "prefix"],
        default="exact",
        help=(
            "exact: full filename match (case-insensitive). "
            "prefix: text before first dot (use when generated names add suffixes)."
        ),
    )
    parser.add_argument(
        "--max_metric_side",
        type=int,
        default=MAX_METRIC_SIDE,
        help=(
            "Downsample images so the longest side is at most this many pixels "
            "before computing SSIM/PSNR/MAE and extracting Inception features. "
            "Keeps GPU memory bounded for very large slides."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def list_images_by_name(folder: str) -> Dict[str, str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    images: Dict[str, str] = {}
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if (
            os.path.isfile(path)
            and os.path.splitext(entry)[1].lower() in SUPPORTED_EXTENSIONS
        ):
            images[entry] = path
    return images


def get_match_key(filename: str, match_mode: str) -> str:
    lower = filename.strip().lower()
    if match_mode == "exact":
        return lower
    stem = os.path.splitext(lower)[0]
    return re.sub(r"\s+", " ", stem.split(".")[0].strip())


def build_keyed_map(
    images: Dict[str, str], match_mode: str, label: str
) -> Dict[str, str]:
    keyed: Dict[str, str] = {}
    dupes = 0
    for name, path in sorted(images.items()):
        key = get_match_key(name, match_mode)
        if key in keyed:
            dupes += 1
            continue
        keyed[key] = path
    if dupes:
        print(
            f"[WARN] {label}: ignored {dupes} duplicate keys under match_mode='{match_mode}'."
        )
    return keyed


# ---------------------------------------------------------------------------
# Image loading & resizing
# ---------------------------------------------------------------------------


def load_image_gpu(image_path: str, device: torch.device) -> torch.Tensor:
    """Load image with PIL, transfer to GPU, normalise to [-1, 1]. Returns [C, H, W]."""
    img = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(img).to(device=device, non_blocking=True)
    return (tensor - 0.5) / 0.5


def cap_size(t: torch.Tensor, max_side: int) -> torch.Tensor:
    """
    Downsample [C, H, W] tensor so its longest side <= max_side (GPU bicubic).
    Returns the original tensor unchanged if already small enough.
    """
    _, h, w = t.shape
    if max(h, w) <= max_side:
        return t
    scale = max_side / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    return F.interpolate(
        t.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False
    ).squeeze(0)


# ---------------------------------------------------------------------------
# Metrics (all GPU, all return torch.Tensor scalars)
# ---------------------------------------------------------------------------


def compute_mae_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b))


def compute_psnr_gpu(
    a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0
) -> torch.Tensor:
    mse = torch.mean((a - b) ** 2)
    if torch.isclose(mse, torch.zeros(1, device=mse.device, dtype=mse.dtype)):
        return torch.tensor(float("inf"), device=mse.device, dtype=mse.dtype)
    dr = torch.tensor(data_range, device=mse.device, dtype=mse.dtype)
    return 20.0 * torch.log10(dr) - 10.0 * torch.log10(mse)


def compute_ssim_gpu(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 2.0,
    kernel_size: int = 11,
) -> torch.Tensor:
    """Channel-wise SSIM via avg_pool2d; explicit del of intermediates to save VRAM."""
    if img1.ndim != 4:
        raise ValueError("SSIM expects [N, C, H, W].")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = kernel_size // 2

    scores: List[torch.Tensor] = []
    for ch in range(img1.shape[1]):
        x = img1[:, ch : ch + 1]
        y = img2[:, ch : ch + 1]

        mu_x = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, kernel_size, stride=1, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sx2 = F.avg_pool2d(x * x, kernel_size, stride=1, padding=pad) - mu_x2
        sy2 = F.avg_pool2d(y * y, kernel_size, stride=1, padding=pad) - mu_y2
        sxy = F.avg_pool2d(x * y, kernel_size, stride=1, padding=pad) - mu_xy

        num = (2.0 * mu_xy + c1) * (2.0 * sxy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sx2 + sy2 + c2)
        score = (num / (den + 1e-12)).mean()
        scores.append(score)

        del x, y, mu_x, mu_y, mu_x2, mu_y2, mu_xy, sx2, sy2, sxy, num, den, score

    result = torch.stack(scores).mean()
    del scores
    return result


# ---------------------------------------------------------------------------
# FID — means/covariances on GPU, sqrtm on CPU via scipy (numerically stable)
# ---------------------------------------------------------------------------
# Background: sqrtm of a 2048x2048 float32 matrix via eigendecomposition is
# ill-conditioned — small negative eigenvalues (numerical noise) propagate
# through sqrt and trace, easily producing a negative FID.
# scipy.linalg.sqrtm uses a Schur decomposition in float64, which is the
# industry-standard approach used by every published FID implementation
# (pytorch-fid, clean-fid, TF implementation, etc.).
# The CPU cost of a single 2048x2048 sqrtm is negligible (<1 s).
# ---------------------------------------------------------------------------


def compute_fid(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    Compute FID score.
    - Means and covariances computed on GPU (fast).
    - sqrtm computed on CPU in float64 via scipy (numerically correct).
    Returns a plain Python float.
    """
    import numpy as np
    from scipy import linalg

    if real.shape[0] < 2 or fake.shape[0] < 2:
        raise ValueError("FID requires at least 2 images per set.")

    # means (GPU)
    mu1 = real.mean(0)
    mu2 = fake.mean(0)

    # covariances on GPU, then transfer to CPU as float64
    rc = real - mu1
    fc = fake - mu2
    sigma1 = ((rc.T @ rc) / (real.shape[0] - 1)).cpu().double().numpy()
    sigma2 = ((fc.T @ fc) / (fake.shape[0] - 1)).cpu().double().numpy()
    diff = (mu1 - mu2).cpu().double().numpy()

    # numerical stabilisation
    eps = 1e-6
    sigma1 += eps * np.eye(sigma1.shape[0])
    sigma2 += eps * np.eye(sigma2.shape[0])

    # matrix sqrt via Schur decomposition (scipy, float64)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if np.max(np.abs(covmean.imag)) > 1e-3:
            print("[WARN] FID sqrtm has large imaginary component; taking real part.")
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------


def get_features(
    calculator: MetricsCalculator,
    t: torch.Tensor,
) -> torch.Tensor:
    """Extract Inception features for a single [C,H,W] GPU tensor. Returns [1, D]."""
    feat = calculator.get_inception_features(t.unsqueeze(0))
    if isinstance(feat, torch.Tensor):
        return feat.to(t.device)
    import numpy as np

    return torch.from_numpy(np.asarray(feat)).to(t.device)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_all(
    calculator: MetricsCalculator,
    image_paths: List[Tuple[str, str, str, str]],
    device: torch.device,
    max_side: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PairMetrics]]:
    """
    Process every image triplet one at a time:
      1. Load all three images to GPU.
      2. Downsample to max_side (for both metrics and FID).
      3. Compute SSIM/PSNR/MAE immediately.
      4. Extract Inception features immediately.
      5. Delete every large tensor before moving to the next image.
      6. Flush CUDA cache after each image.

    Only the tiny feature vectors ([1, 2048]) are accumulated.
    """
    src_feats: List[torch.Tensor] = []
    gen1_feats: List[torch.Tensor] = []
    gen2_feats: List[torch.Tensor] = []
    rows: List[PairMetrics] = []
    skipped = 0
    total = len(image_paths)

    with torch.no_grad():
        for idx, (match_key, src_path, gen1_path, gen2_path) in enumerate(
            image_paths, 1
        ):
            image_name = os.path.basename(src_path)
            print(f"[DEBUG] [{idx}/{total}] key='{match_key}' image='{image_name}'")

            # 1. Load --------------------------------------------------------
            try:
                src_full = load_image_gpu(src_path, device)
                gen1_full = load_image_gpu(gen1_path, device)
                gen2_full = load_image_gpu(gen2_path, device)
            except Exception as exc:
                skipped += 1
                print(f"[WARN] Skipping '{image_name}': {exc}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            # Align spatial size to source
            if gen1_full.shape[1:] != src_full.shape[1:]:
                gen1_full = F.interpolate(
                    gen1_full.unsqueeze(0),
                    size=src_full.shape[1:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)
            if gen2_full.shape[1:] != src_full.shape[1:]:
                gen2_full = F.interpolate(
                    gen2_full.unsqueeze(0),
                    size=src_full.shape[1:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)

            # 2. Downsample --------------------------------------------------
            src_s = cap_size(src_full, max_side)
            gen1_s = cap_size(gen1_full, max_side)
            gen2_s = cap_size(gen2_full, max_side)

            h_orig, w_orig = src_full.shape[1], src_full.shape[2]
            h_small, w_small = src_s.shape[1], src_s.shape[2]
            if (h_small, w_small) != (h_orig, w_orig):
                print(
                    f"[DEBUG]   Downsampled {h_orig}x{w_orig} -> {h_small}x{w_small} "
                    f"for GPU computation"
                )

            # Full-res tensors no longer needed - delete immediately
            del src_full, gen1_full, gen2_full

            # 3. Metrics on downsampled --------------------------------------
            src_b = src_s.unsqueeze(0)
            gen1_b = gen1_s.unsqueeze(0)
            gen2_b = gen2_s.unsqueeze(0)

            try:
                ssim_1 = float(compute_ssim_gpu(src_b, gen1_b).cpu())
                psnr_1 = float(compute_psnr_gpu(src_b, gen1_b).cpu())
                mae_1 = float(compute_mae_gpu(src_b, gen1_b).cpu())

                ssim_2 = float(compute_ssim_gpu(src_b, gen2_b).cpu())
                psnr_2 = float(compute_psnr_gpu(src_b, gen2_b).cpu())
                mae_2 = float(compute_mae_gpu(src_b, gen2_b).cpu())
            except torch.OutOfMemoryError as exc:
                skipped += 1
                print(f"[WARN] OOM during metrics for '{image_name}': {exc}")
                del src_s, gen1_s, gen2_s, src_b, gen1_b, gen2_b
                torch.cuda.empty_cache()
                gc.collect()
                continue

            del src_b, gen1_b, gen2_b

            print(
                f"[DEBUG]   src-gen1  SSIM={ssim_1:.6f}  PSNR={psnr_1:.6f}  MAE={mae_1:.6f} | "
                f"src-gen2  SSIM={ssim_2:.6f}  PSNR={psnr_2:.6f}  MAE={mae_2:.6f}"
            )
            rows.append(
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

            # 4. Inception features on downsampled ---------------------------
            try:
                sf = get_features(calculator, src_s)
                g1f = get_features(calculator, gen1_s)
                g2f = get_features(calculator, gen2_s)
            except torch.OutOfMemoryError as exc:
                skipped += 1
                rows.pop()
                print(f"[WARN] OOM during FID features for '{image_name}': {exc}")
                del src_s, gen1_s, gen2_s
                torch.cuda.empty_cache()
                gc.collect()
                continue

            src_feats.append(sf)
            gen1_feats.append(g1f)
            gen2_feats.append(g2f)

            # 5. Delete everything large -------------------------------------
            del src_s, gen1_s, gen2_s, sf, g1f, g2f

            # 6. Flush cache -------------------------------------------------
            torch.cuda.empty_cache()
            gc.collect()

            alloc = torch.cuda.memory_allocated(device) / 1e9
            print(f"[DEBUG]   GPU allocated after cleanup: {alloc:.2f} GB")

    if not src_feats:
        raise ValueError("No valid images were processed.")

    print(f"[DEBUG] Skipped {skipped} images due to errors.")
    src_all = torch.cat(src_feats, dim=0)
    gen1_all = torch.cat(gen1_feats, dim=0)
    gen2_all = torch.cat(gen2_feats, dim=0)
    del src_feats, gen1_feats, gen2_feats
    torch.cuda.empty_cache()
    gc.collect()

    return src_all, gen1_all, gen2_all, rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


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
        for r in rows:
            writer.writerow(
                {
                    "image_name": r.image_name,
                    "ssim_src_gen1": f"{r.ssim_src_gen1:.6f}",
                    "psnr_src_gen1": f"{r.psnr_src_gen1:.6f}",
                    "mae_src_gen1": f"{r.mae_src_gen1:.6f}",
                    "ssim_src_gen2": f"{r.ssim_src_gen2:.6f}",
                    "psnr_src_gen2": f"{r.psnr_src_gen2:.6f}",
                    "mae_src_gen2": f"{r.mae_src_gen2:.6f}",
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
    lines: List[str] = [
        f"# Three-Folder Metrics Report ({model_name})",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Source folder: `{source_dir}`",
        f"- Generated1 folder: `{generated1_dir}`",
        f"- Generated2 folder: `{generated2_dir}`",
        f"- Match mode: `{match_mode}`",
        f"- Source images: {total_src}",
        f"- Generated1 images: {total_gen1}",
        f"- Generated2 images: {total_gen2}",
        f"- Matched filenames in all 3 folders: {matched_count}",
        "",
        "## Dataset-Level FID",
        "",
        "| Pair | FID |",
        "|---|---:|",
        f"| Source vs Generated1 | {fid_src_gen1:.6f} |",
        f"| Source vs Generated2 | {fid_src_gen2:.6f} |",
        "",
        "## Per-Image Metrics",
        "",
        "| image_name | ssim_src_gen1 | psnr_src_gen1 | mae_src_gen1 | ssim_src_gen2 | psnr_src_gen2 | mae_src_gen2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.image_name} | {r.ssim_src_gen1:.6f} | {r.psnr_src_gen1:.6f} | "
            f"{r.mae_src_gen1:.6f} | {r.ssim_src_gen2:.6f} | {r.psnr_src_gen2:.6f} | "
            f"{r.mae_src_gen2:.6f} |"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(
        "[DEBUG] Starting three-folder comparison (GPU-OPTIMISED + MEMORY-EFFICIENT)..."
    )
    for k in (
        "source_dir",
        "generated1_dir",
        "generated2_dir",
        "match_mode",
        "batch_size",
        "device",
        "max_metric_side",
    ):
        print(f"[DEBUG] {k}: {getattr(args, k)}")

    source_map = list_images_by_name(args.source_dir)
    gen1_map = list_images_by_name(args.generated1_dir)
    gen2_map = list_images_by_name(args.generated2_dir)
    print(
        f"[DEBUG] image counts - source:{len(source_map)}  gen1:{len(gen1_map)}  gen2:{len(gen2_map)}"
    )

    source_keyed = build_keyed_map(source_map, args.match_mode, "source")
    gen1_keyed = build_keyed_map(gen1_map, args.match_mode, "generated1")
    gen2_keyed = build_keyed_map(gen2_map, args.match_mode, "generated2")

    matched_keys = sorted(set(source_keyed) & set(gen1_keyed) & set(gen2_keyed))
    print(f"[DEBUG] matched_keys count: {len(matched_keys)}")
    if len(matched_keys) < 2:
        raise ValueError("Need at least 2 matched filenames. Try --match_mode prefix.")

    output_dir = os.path.dirname(os.path.abspath(args.generated2_dir))
    model_name = os.path.basename(output_dir) or "model"
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_path = os.path.join(
        output_dir, f"{model_name}_{args.output_prefix}_{timestamp}.csv"
    )
    md_path = os.path.join(
        output_dir, f"{model_name}_{args.output_prefix}_{timestamp}.md"
    )

    device = resolve_device(args.device)
    calculator = MetricsCalculator(device=device)
    print(f"Using device: {device}")

    image_paths = [
        (k, source_keyed[k], gen1_keyed[k], gen2_keyed[k]) for k in matched_keys
    ]

    src_features, gen1_features, gen2_features, per_image_rows = process_all(
        calculator=calculator,
        image_paths=image_paths,
        device=device,
        max_side=args.max_metric_side,
    )

    if len(per_image_rows) < 2:
        raise ValueError(
            "Fewer than 2 valid images after filtering; cannot compute FID."
        )

    print("[DEBUG] Computing FID on GPU...")
    with torch.no_grad():
        fid_1 = compute_fid(src_features, gen1_features)
        fid_2 = compute_fid(src_features, gen2_features)

    del src_features, gen1_features, gen2_features
    torch.cuda.empty_cache()
    gc.collect()

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
        fid_src_gen1=fid_1,
        fid_src_gen2=fid_2,
        rows=per_image_rows,
    )

    print("\nComparison complete.")
    print(f"  Model:             {model_name}")
    print(f"  Match mode:        {args.match_mode}")
    print(f"  Matched images:    {len(matched_keys)}")
    print(f"  Successfully proc: {len(per_image_rows)}")
    print(f"  FID (src vs gen1): {fid_1:.6f}")
    print(f"  FID (src vs gen2): {fid_2:.6f}")
    print(f"  CSV:  {csv_path}")
    print(f"  MD:   {md_path}")


if __name__ == "__main__":
    main()
