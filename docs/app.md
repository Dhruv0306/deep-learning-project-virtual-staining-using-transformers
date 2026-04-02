# app.py Inference and Whole-Slide Translation

Source of truth: ../app.py

This script performs patch-based whole-slide translation for all model
versions in the project. It supports both non-overlapping and overlapping
tile inference, then reconstructs the full image with seam-aware blending.

## Supported Model Versions

- version 1: ViTUNetGenerator (hybrid baseline)
- version 2: ViTUNetGeneratorV2 (true UVCGAN v2)
- version 3: DiT diffusion pipeline
- version 4: CUT + Transformer generator pipeline

## Runtime Defaults

- `patch_size = 256`
- `stride = patch_size // 2` (50% overlap)
- normalized tensor range: `[-1, 1]`

Overlapping tiles are blended with a 2-D Hann-style window to reduce visible
stitching artifacts.

## Checkpoint Loading Paths

- `load_model(checkpoint_path, device, model_version)`
  - supports v1/v2
  - v2 architecture (`vit_depth`, `use_cross_domain`) is auto-inferred from
    checkpoint keys by `_infer_v2_kwargs`

- `load_v3_components(checkpoint_path, device)`
  - loads DiT, VAE wrapper, DDPM scheduler, DDIM sampler
  - uses checkpoint `config` if available, else `get_dit_config().diffusion`

- `load_v4_model(checkpoint_path, device, image_size)`
  - prefers EMA weights when present
  - infers generator architecture from saved weights via `_infer_v4_kwargs`
  - supports both Transformer and ResNet v4 checkpoints

## Patch Pipeline

- `pad_to_patch_multiple(image, patch_size)`
  - pads right and bottom with white pixels so dimensions are divisible

- `extract_patches_with_coords(image, patch_size, stride)`
  - extracts row-major patches and records `(top, left)` positions

- `_blend_window(patch_size, device, dtype, eps)`
  - creates smooth per-pixel blending weights for overlaps

- `reconstruct_tensor_from_patches(...)`
  - accumulates weighted patch predictions and normalizes by weight map

- `translate_image_from_patches(...)`
  - generic generator-based translation (v1/v2/v4)

- `translate_image_from_patches_v3(...)`
  - batched diffusion sampling path for v3

## Direction Behavior by Version

- v1/v2: bidirectional (`A -> B` and `B -> A`)
- v3: currently unstained-to-stained path in patch diffusion function
- v4: bidirectional (`A -> B` and `B -> A`)

## Outputs

- `data/reconstructed_stained_output.png`
- `data/reconstructed_unstained_output.png` (v1/v2/v4)

## CLI Usage

```bash
python app.py
```

Prompts:

- model checkpoint path
- model version (1, 2, 3, or 4)
- unstained image path
- stained image path (for v1/v2/v4)
