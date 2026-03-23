# `preprocess_data.py` — Data Preprocessing

**Role:** Converts whole-slide histology images into 256×256 patches, filters out uninformative background tiles, and splits them into CycleGAN-style `trainA / trainB / testA / testB` directories.

Run this script **once** before training to prepare the dataset.

---

## Overview

Whole-slide pathology images can be gigapixels in size. The preprocessing pipeline:

1. Slides a 256×256 window across each source image (no overlap by default).
2. Estimates what fraction of each patch is tissue vs. background using colour heuristics.
3. Keeps patches that contain enough tissue, and randomly keeps a fraction of background patches to avoid a purely tissue-only bias.
4. Writes accepted patches as PNG files into the four CycleGAN dataset folders.

```
Un_Stained/  ──┐
               ├── split 80/20 ──► trainA/ , testA/
               │
C_Stained/   ──┘
               └── split 80/20 ──► trainB/ , testB/
```

---

## Functions

### `extract_patches_pil(img, patch_size=256, stride=256)`

Tiles a PIL image with a sliding window.

| Parameter | Default | Description |
|---|---|---|
| `img` | — | Source `PIL.Image.Image` |
| `patch_size` | 256 | Square tile side length in pixels |
| `stride` | 256 | Step between tiles. `stride == patch_size` produces non-overlapping tiles |

Returns a `list[PIL.Image.Image]` in row-major (top-to-bottom, left-to-right) order. Tiles that would extend beyond the image boundary are not extracted (only full-sized patches are kept).

---

### `estimate_tissue_fraction(patch, white_thresh=220, sat_thresh=0.05)`

Estimates the proportion of pixels in a patch that are likely tissue rather than slide background.

| Parameter | Default | Description |
|---|---|---|
| `patch` | — | RGB PIL patch |
| `white_thresh` | 220 | Pixels with all channels above this value are classified as near-white background |
| `sat_thresh` | 0.05 | Pixels with colour saturation below this value are classified as low-colour background |

**Algorithm:**
```
arr = float32 RGB array of patch
maxc = per-pixel max channel
minc = per-pixel min channel
sat  = (maxc - minc) / (maxc + ε)      ← saturation approximation

is_white   = all channels > white_thresh
is_low_sat = sat < sat_thresh
background = is_white | is_low_sat

tissue_fraction = 1 - mean(background)
```

Returns a `float` in `[0, 1]`. A value of `0.0` means the entire patch is background; `1.0` means all pixels are classified as tissue.

---

### `split_filenames(file_list, train_ratio=0.8, seed=42)`

Randomly splits a list of filenames into train and test subsets.

| Parameter | Default | Description |
|---|---|---|
| `file_list` | — | List of image filenames (not full paths) |
| `train_ratio` | 0.8 | Fraction of files assigned to training |
| `seed` | 42 | NumPy RNG seed for reproducibility |

The list is sorted before shuffling so the split is deterministic given the same `seed`.

Returns `(train_files, test_files)`.

---

### `save_patches(image_path, save_dir, patch_size, tissue_threshold, background_keep_ratio, white_thresh, sat_thresh)`

Loads one whole-slide image, extracts all patches, filters by tissue content, and writes accepted patches as PNG files.

| Parameter | Default | Description |
|---|---|---|
| `image_path` | — | Full path to source image |
| `save_dir` | — | Output directory (`trainA`, `testB`, etc.) |
| `patch_size` | 256 | Patch side length |
| `tissue_threshold` | 0.1 | Minimum tissue fraction to classify a patch as tissue and keep it unconditionally |
| `background_keep_ratio` | 0.1 | Probability of keeping a background patch (random sub-sampling) |
| `white_thresh` | 220 | Near-white pixel threshold passed to `estimate_tissue_fraction` |
| `sat_thresh` | 0.05 | Low-saturation threshold passed to `estimate_tissue_fraction` |

**Filtering logic:**
```
for each patch:
    tissue_fraction = estimate_tissue_fraction(patch)
    is_tissue       = tissue_fraction >= tissue_threshold
    keep_background = random() < background_keep_ratio

    if is_tissue or keep_background:
        save patch as {base}_{i}.png
```

The background sub-sampling (`background_keep_ratio`) prevents the dataset from being entirely dominated by blank slide regions while still including some context.

Output filenames follow the pattern `{source_image_base}_{patch_index}.png`.

---

### `main()`

Orchestrates the full preprocessing pipeline.

**Default settings used by `main()`:**

| Setting | Value |
|---|---|
| `tissue_threshold` | 0.1 |
| `background_keep_ratio` | 0.2 |
| `white_thresh` | 220 |
| `sat_thresh` | 0.05 |
| Train/test split | 80% / 20% |
| RNG seed | 42 |

**Steps:**
1. Lists all `.png`/`.jpg`/`.jpeg` files in `Un_Stained/` and `C_Stained/`.
2. Splits each list into train/test using `split_filenames`.
3. Calls `save_patches` for each image into the appropriate output folder.
4. Prints patch counts at each step.

The script sets `Image.MAX_IMAGE_PIXELS = None` before running to suppress PIL's `DecompressionBombWarning` for large whole-slide images.

---

## Output Structure

```
data/E_Staining_DermaRepo/H_E-Staining_dataset/
    trainA/   ← unstained training patches  (PNG, 256×256, RGB)
    trainB/   ← stained   training patches  (PNG, 256×256, RGB)
    testA/    ← unstained test patches      (PNG, 256×256, RGB)
    testB/    ← stained   test patches      (PNG, 256×256, RGB)
```

These folders are consumed directly by `data_loader.py`.
