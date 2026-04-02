# config.py Configuration Reference

Source of truth: ../config.py

This module centralizes runtime configuration for all four model families.

## Configuration Containers

### UVCGANConfig (v1/v2/v3)

Fields:

- `model_version`: 1, 2, or 3
- `generator`: `GeneratorConfig`
- `discriminator`: `DiscriminatorConfig`
- `loss`: `LossConfig`
- `training`: `TrainingConfig`
- `data`: `DataConfig`
- `diffusion`: `DiffusionConfig` (used by v3)
- `model_dir`, `val_dir`

Validation in `__post_init__`:

- `model_version` must be one of 1, 2, 3
- `training.decay_start_epoch` must be at least 2 epochs before
  `training.num_epochs`

### V4Config (v4 only)

Fields:

- `model_version`: must be 4
- `model`: `V4ModelConfig`
- `training`: `V4TrainingConfig`
- `data`: `V4DataConfig`
- `model_dir`, `val_dir`

Validation in `V4Config.__post_init__` enforces `model_version == 4`.

## Sub-Config Groups

### v1/v2/v3 groups

- `GeneratorConfig`: v2 generator architecture controls (`base_channels`, ViT
  depth/heads, LayerScale, cross-domain fusion, gradient checkpointing)
- `DiscriminatorConfig`: multi-scale PatchGAN settings (spectral norm, scales)
- `LossConfig`: cycle, identity, perceptual, GP, contrastive, spectral
- `TrainingConfig`: optimizer/scheduler, AMP, accumulation, validation,
  early-stop/divergence controls
- `DataConfig`: dataset root, image size, batch size, workers, prefetch
- `DiffusionConfig`: DiT architecture, DDPM/DDIM settings, guidance,
  perceptual cadence, adversarial/cycle/identity ramps, discriminator branch
  toggles for v3

### v4 groups

- `V4ModelConfig`: generator/discriminator architecture (Transformer encoder
  toggles, patch size, encoder depth/width/heads, discriminator width/layers)
- `V4TrainingConfig`: GAN + PatchNCE + identity weights, NCE layers/patches,
  replay buffer, EMA, LR schedule, accumulation, validation/save cadence
- `V4DataConfig`: image size, batch size, worker count, prefetch factor

## Factory Functions

- `get_default_config(model_version=2)`
  - default profile for v1/v2/v3
  - applies legacy v1 overrides when `model_version == 1`

- `get_8gb_config()`
  - tuned v2 profile for 8 GB GPUs
  - currently uses:
    - `data.batch_size = 2`
    - `training.accumulate_grads = 2`
    - `generator.use_gradient_checkpointing = True`

- `get_dit_config()`
  - baseline v3 profile

- `get_dit_8gb_config()`
  - v3 profile tuned for 8 GB VRAM
  - includes diffusion checkpointing and lighter validation settings

- `get_v4_config()`
  - default v4 profile

- `get_v4_8gb_config()`
  - v4 profile with Transformer gradient checkpointing enabled

## Practical Guidance

- Use `get_8gb_config()` for v2 on constrained VRAM.
- Use `get_dit_8gb_config()` for v3 on constrained VRAM.
- Use `get_v4_8gb_config()` for v4 on constrained VRAM.
