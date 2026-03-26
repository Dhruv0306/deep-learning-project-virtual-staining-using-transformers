# config.py Configuration Reference

Source of truth: ../config.py

This module centralizes hyperparameters for all model versions.

## Top-Level Container

UVCGANConfig groups all sub-configs:

- model_version (1, 2, or 3)
- generator
- discriminator
- loss
- training
- data
- diffusion
- model_dir
- val_dir

Validation in __post_init__ enforces:

- model_version must be one of 1, 2, 3
- decay_start_epoch must be at least 2 epochs before num_epochs

## Sub-Configs

- GeneratorConfig
  - v2 generator architecture settings (ViT UNet based)
  - includes use_cross_domain and use_gradient_checkpointing

- DiscriminatorConfig
  - multi-scale PatchGAN settings
  - includes spectral normalization toggle and number of scales

- LossConfig
  - cycle, identity, perceptual, and regularization weights
  - includes lambda_gp, lambda_contrastive, lambda_spectral

- TrainingConfig
  - optimizer and schedule controls
  - includes warmup, decay, AMP, gradient accumulation, and stopping controls

- DataConfig
  - dataset root, image size, batch size, and dataloader workers

- DiffusionConfig
  - v3 DiT/DDPM settings
  - includes timesteps, beta schedule, architecture dimensions, and VAE id

## Factory Functions

- get_default_config(model_version=2)
  - baseline config for selected version
  - applies legacy overrides when model_version == 1

- get_8gb_config()
  - v2 profile for 8 GB GPUs
  - keeps full v2 architecture defaults and reduces memory mainly through:
    - batch_size = 2
    - accumulate_grads = 2

- get_dit_config()
  - default v3 diffusion config

- get_dit_8gb_config()
  - v3 profile for 8 GB GPUs
  - enables diffusion gradient checkpointing and lighter validation setup

## Practical Note

For v2 training, start with get_8gb_config() on limited VRAM hardware.
For v3 training, start with get_dit_8gb_config() on limited VRAM hardware.
