# `app.py` — Inference / Whole-Slide Translation

**Role:** Loads a trained checkpoint and performs patch-based stain/unstain translation on whole-slide images. Uses overlapping patches with cosine blending to produce seamless full-image outputs.

---

## Overview

Whole-slide images are too large to process in one GPU forward pass. `app.py` divides the input image into 256×256 patches (with a configurable stride), translates each patch independently, and reconstructs the full image by averaging overlapping patch contributions through a smooth blending window.

```
Input image (arbitrary size)
     │
     ▼ pad_to_patch_multiple (white padding to exact multiple of patch_size)
     │
     ▼ extract_patches_with_coords (stride = patch_size / 2 by default)
     │
     ▼ [G_AB or G_BA forward pass on each 256×256 patch]
     │
     ▼ reconstruct_tensor_from_patches (weighted average with blend window)
     │
     ▼ crop back to original_size
     │
     ▼ save_image (normalize=True, value_range=(-1, 1))
     │
Output image (same size as input)
```

---

## Functions

### `load_model(checkpoint_path, device, model_version)`

Loads a saved checkpoint and returns two generators.

| Parameter | Default | Description |
|---|---|---|
| `checkpoint_path` | required | Path to a `.pth` file saved by the training loop |
| `device` | `"cpu"` | Device to load the model onto (`"cuda"` or `"cpu"`) |
| `model_version` | 2 | `1` loads `ViTUNetGenerator` (v1); `2` loads `ViTUNetGeneratorV2` (v2) |

For `model_version=2`, the architecture hyperparameters (`vit_depth`, `use_cross_domain`) are auto-detected from the checkpoint state dict via `_infer_v2_kwargs`. This ensures the loaded model always matches the saved architecture regardless of which config was active during training.

**Returns:** `(G_AB, G_BA)` — both in `eval()` mode on the specified device.

---

### `_infer_v2_kwargs(state_dict)`

Inspects a v2 checkpoint's state dict keys to recover the architecture parameters used during training.

| Detected parameter | Method |
|---|---|
| `vit_depth` | Count distinct block indices in keys like `vit.blocks.{i}.*` |
| `use_cross_domain` | Check if any key starts with `fuse` |

---

### `translate_image_from_patches(input_image_path, model, transform, output_path, patch_size, stride, device)`

Full pipeline for translating one whole-slide image.

| Parameter | Default | Description |
|---|---|---|
| `input_image_path` | — | Path to the input image (any PIL-supported format) |
| `model` | — | Generator model (`G_AB` for staining, `G_BA` for unstaining) |
| `transform` | — | Preprocessing transform (resize + normalize) |
| `output_path` | — | Path to save the translated output image |
| `patch_size` | 256 | Patch side length in pixels |
| `stride` | 256 | Stride between patch centres. `stride < patch_size` creates overlap |
| `device` | `"cpu"` | Computation device |

**Returns:** `(original_size, padded_size, num_patches, output_path)` — useful for logging and validation.

---

### `pad_to_patch_multiple(image, patch_size=256)`

Pads a PIL image with white pixels on the right and bottom to make its dimensions exact multiples of `patch_size`. Returns `(padded_image, original_size)`.

---

### `extract_patches_with_coords(pil_image, patch_size=256, stride=256)`

Extracts all `patch_size × patch_size` patches and their `(top, left)` coordinates from a PIL image. Ensures the final row/column of patches is always included (handles non-divisible sizes).

Returns `(patches, positions)` where `positions[i] = (top, left)` is the pixel offset of patch `i`.

---

### `reconstruct_tensor_from_patches(patches, positions, image_size, patch_size, stride)`

Reassembles translated patches into a full-resolution image tensor.

When `stride < patch_size` (overlapping patches), each patch is weighted by a **2D Hann window** (`sin²(π × position / patch_size)`) before accumulation. The weight map is used to normalise the final output, producing smooth blending across patch boundaries.

| Parameter | Description |
|---|---|
| `patches` | List of translated patch tensors, each `(3, patch_size, patch_size)` in `[-1, 1]` |
| `positions` | List of `(top, left)` coordinates matching `patches` |
| `image_size` | `(width, height)` of the padded image |

Returns a `(3, height, width)` float tensor in `[-1, 1]`.

---

### `_blend_window(patch_size, device, dtype, eps=0.05)`

Generates a 2D sinusoidal blending window for seamless patch reconstruction.

```
window_1d = sin(linspace(0, π, patch_size))²       ← peak at centre, 0 at edges
window_1d = window_1d × (1 - eps) + eps             ← avoid exact zeros at edges
window_2d = window_1d[:, None] × window_1d[None, :] ← outer product
```

---

### `stain_image(image, model, device)` / `unstain_image(image, model, device)`

Simple single-patch helpers for applying a generator to a pre-processed tensor.

---

## CLI Usage

```
$ python app.py
Using device: cuda
Enter model path: data/.../final_model.pth
Enter 1 for Hybrid UVCGAN based or 2 for True-UVCGAN based generator model: 2
Provide Path to Unstained Image: data/.../HC21-01338(A3-1).jpg
Provide Path to Stained Image:   data/.../HC21-01338(A3-2).jpg
```

Outputs:
- `data/reconstructed_stained_output.png` — unstained→stained translation
- `data/reconstructed_unstained_output.png` — stained→unstained translation

The default stride is `patch_size // 2 = 128`, so adjacent patches overlap by 50% and are blended together for a seamless result.
