# app.py Inference and Whole-Slide Translation

Source of truth: ../app.py

This script runs patch-based translation for large histology images.

## Supported Model Versions

- version 1: ViTUNetGenerator (CycleGAN-style)
- version 2: ViTUNetGeneratorV2 (true UVCGAN)
- version 3: DiT diffusion pipeline

## Runtime Behavior

- All paths use patch_size = 256 by default in main().
- main() sets stride = patch_size // 2 for overlapping patches and smoother blending.

Version-specific behavior:

- v1 and v2:
  - loads G_AB and G_BA from checkpoint
  - performs both directions:
    - unstained -> stained
    - stained -> unstained

- v3:
  - loads DiT model, condition encoder, VAE wrapper, and DDIM sampler
  - performs unstained -> stained only

## Core Functions

- load_model(checkpoint_path, device, model_version)
  - supports model_version 1 and 2
  - v2 architecture args are inferred from checkpoint keys via _infer_v2_kwargs

- load_v3_components(checkpoint_path, device)
  - loads diffusion components and config used for inference step count

- pad_to_patch_multiple(image, patch_size)
  - right/bottom white padding so dimensions are multiples of patch size

- extract_patches_with_coords(image, patch_size, stride)
  - extracts patches and stores (top, left) positions

- reconstruct_tensor_from_patches(...)
  - blends overlapping outputs using a 2D Hann-style window

- translate_image_from_patches(...)
  - v1/v2 patch inference path

- translate_image_from_patches_v3(...)
  - v3 batched diffusion inference path

## Outputs

- data/reconstructed_stained_output.png
- data/reconstructed_unstained_output.png (v1/v2 only)

## CLI Prompts

python app.py

Prompts:

- Enter model path
- Enter model version (1, 2, or 3)
- Path to unstained input
- Path to stained input (v1/v2 path)
