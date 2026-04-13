# `compare_three_folder_metrics.py` — Three-Folder Evaluation Report

Source of truth: `../compare_three_folder_metrics.py`

**Role:** Compares images with the same filename across three folders:
- source (original)
- generated set 1
- generated set 2

The script computes:
- **Per-image metrics** for both pairs (`source vs generated1`, `source vs generated2`): SSIM, PSNR, MAE
- **Dataset-level metrics** for both pairs: FID

It writes both a CSV and a Markdown report in the **parent directory of `generated2_dir`**, labeled with that parent folder name as model name.

---

## Why This Script Exists

Training/validation logs often show aggregate quality, but model comparison is easier when you can:
- enforce filename-aligned comparison across different output folders
- keep one report containing both generated variants
- inspect both per-image behavior and full-distribution quality

This script standardizes that comparison flow in one command.

---

## Command-Line Interface

```bash
python compare_three_folder_metrics.py \
  --source_dir <PATH_TO_SOURCE> \
  --generated1_dir <PATH_TO_GENERATED1> \
  --generated2_dir <PATH_TO_GENERATED2> \
  [--batch_size 16] \
  [--device auto|cpu|cuda] \
  [--output_prefix three_folder_metrics] \
  [--match_mode exact|prefix]
```

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--source_dir` | Yes | - | Directory with original/reference images |
| `--generated1_dir` | Yes | - | Directory with generated images for model/set 1 |
| `--generated2_dir` | Yes | - | Directory with generated images for model/set 2 |
| `--batch_size` | No | `16` | Batch size for Inception feature extraction used in FID |
| `--device` | No | `auto` | Compute device for metric inference. `auto` picks CUDA when available |
| `--output_prefix` | No | `three_folder_metrics` | Filename prefix for output reports |
| `--match_mode` | No | `exact` | Matching strategy: `exact` uses full filename; `prefix` uses text before first dot in filename |

### Match mode guidance

- Use `exact` when filenames are identical across all three folders.
- Use `prefix` when generated outputs add suffixes, extra descriptors, or case variations.

Example where `prefix` helps:
- source: `HC22-01151(C1-1).20X.jpg`
- generated1: `HC22-01151(C1-1).blended_test.jpg.jpg`
- generated2: `HC22-01151(C1-1).20X UNSTAINED.jpg`

---

## Folder Matching Rules

The script:
1. Reads image files from all three folders (top-level only; no recursion).
2. Builds comparison keys per `--match_mode`.
3. Keeps only keys that exist in all three folders.
3. Sorts matched filenames for deterministic output.

Supported image extensions:
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`

Matching key behavior:
- `exact`: lowercase full filename (including extension)
- `prefix`: lowercase text before first dot in filename, with extra spaces normalized

If duplicate keys appear within the same folder for a selected match mode, the script keeps the first key occurrence (sorted by filename) and prints a warning.

### Minimum data requirement

At least **2 valid matched images** are required after filtering/skipping; otherwise dataset-level FID is not computed and the script raises an error.

---

## Preprocessing and Alignment

For every matched filename:
1. Load all three images as RGB.
2. If generated image size differs from source, resize generated image to source size using bicubic interpolation.
3. Convert to tensor and normalize from `[0, 1]` to `[-1, 1]` with mean/std `(0.5, 0.5, 0.5)`.

Large-image handling:
- PIL pixel limit is disabled (`Image.MAX_IMAGE_PIXELS = None`) for very large whole-slide style images.
- Decompression bomb warnings are suppressed for this script.
- Truncated image loading is enabled (`ImageFile.LOAD_TRUNCATED_IMAGES = True`).

This aligns all pairwise computations with the shared project metric conventions.

Device behavior:
- Per-image SSIM/PSNR/MAE are computed using torch tensors on the selected device.
- Dataset FID Inception feature extraction also runs on the selected device.
- `--device auto` uses CUDA when available.

---

## Metrics Computed

## Per-Image (row-wise)

For each filename:
- `ssim_src_gen1`
- `psnr_src_gen1`
- `mae_src_gen1`
- `ssim_src_gen2`
- `psnr_src_gen2`
- `mae_src_gen2`

Implementation detail:
- SSIM and PSNR are computed via torch-based implementations (GPU-capable).
- MAE is computed with torch L1 mean.

### Interpretation

| Metric | Better direction | Note |
|---|---|---|
| SSIM | Higher | Structural similarity |
| PSNR | Higher | Pixel-level fidelity in dB |
| MAE | Lower | Average absolute pixel error |

## Dataset-Level (global)

Computed once over all matched images:
- `FID(source, generated1)`
- `FID(source, generated2)`

FID is computed in Inception feature space using Fréchet distance between feature distributions.

---

## Output Location and Naming

Let:
- `generated2_dir = <...>/<parent>/<generated2_subfolder>`

The script writes both files into `<...>/<parent>`.

Model label is inferred as:
- `model_name = basename(parent)`

Output files:
- `<model_name>_<output_prefix>_<YYYY_MM_DD_HH_MM_SS>.csv`
- `<model_name>_<output_prefix>_<YYYY_MM_DD_HH_MM_SS>.md`

---

## CSV Schema

Columns written in this exact order:

| Column | Description |
|---|---|
| `image_name` | Matched filename |
| `ssim_src_gen1` | SSIM source vs generated1 |
| `psnr_src_gen1` | PSNR source vs generated1 |
| `mae_src_gen1` | MAE source vs generated1 |
| `ssim_src_gen2` | SSIM source vs generated2 |
| `psnr_src_gen2` | PSNR source vs generated2 |
| `mae_src_gen2` | MAE source vs generated2 |

Values are formatted to 6 decimal places.

---

## Markdown Report Structure

The Markdown report includes:
1. Title with model label
2. Summary metadata:
   - timestamp
   - all input folder paths
   - image counts per folder
   - matched count
3. Dataset-level FID table (two rows)
4. Per-image metrics table (same fields as CSV)

---

## Runtime Messages

During execution, the script prints debug information from `main`:
- input paths, match mode, batch size, device argument
- source/generated folder counts
- keyed counts and matched key count
- per-key progress (`[i/N]`) and per-key metrics for both pairs
- warning lines for skipped unreadable images

On completion, it also prints:
- selected runtime device
- match mode
- model label
- matched image count
- skipped unreadable image count (if any)
- both dataset-level FID values
- output CSV path
- output Markdown path

---

## Error Handling and Edge Cases

- Missing folder path -> `FileNotFoundError`
- Less than 2 matched filenames -> `ValueError`
- Corrupt/unreadable image -> skipped with warning, processing continues
- Less than 2 valid images after skipping -> `ValueError`

---

## Example Commands

### Windows (project venv)

```powershell
.\.venv\Scripts\python .\compare_three_folder_metrics.py `
  --source_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\testA" `
  --generated1_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\models_v2_2026_03_23_16_27_18\validation_images\genA" `
  --generated2_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\models_v4_2026_04_08_14_37_03\validation_images\genA" `
  --match_mode exact
```

### Custom output prefix and larger FID batch size

```powershell
.\.venv\Scripts\python .\compare_three_folder_metrics.py `
  --source_dir "<source>" `
  --generated1_dir "<gen1>" `
  --generated2_dir "<gen2>" `
  --batch_size 32 `
  --device cuda `
  --output_prefix "ablation_setA"
```

### Your current folder style (recommended)

```powershell
.\.venv\Scripts\python .\compare_three_folder_metrics.py `
  --source_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\C_Stained" `
  --generated1_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\V_Stained" `
  --generated2_dir "data\E_Staining_DermaRepo\H_E-Staining_dataset\models_2026_03_16_18_48_27\V_Stained" `
  --match_mode prefix
```

---

## Practical Notes

- FID is a distribution metric and should be interpreted at dataset level.
- Per-image SSIM/PSNR/MAE are useful for locating outliers and failure cases.
- If folders are not true one-to-one exports, matched count may be much smaller than folder counts.
- For stable FID, use as many matched images as possible.
