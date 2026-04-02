# model_v4/discriminator.py

Source of truth: ../../model_v4/discriminator.py

Defines the v4 discriminator as a standard N-layer PatchGAN with optional
depth/width control.

## Building Block

## _conv_block(in_channels, out_channels, stride=2, use_norm=True)

Each block is:

- `Conv2d(kernel=4, padding=1, stride=stride)`
- optional `InstanceNorm2d`
- `LeakyReLU(0.2)`

Per PatchGAN convention, the first block disables normalization.

## PatchGANDiscriminator

Architecture:

1. first downsampling block (no norm)
2. `n_layers - 1` additional stride-2 downsampling blocks
3. one stride-1 refinement block
4. final `Conv2d(..., out_channels=1)` score-map head

Key points:

- outputs a spatial realism map (not a single scalar)
- default settings (`base_channels=64`, `n_layers=3`) match typical
  CycleGAN/CUT-style 70x70 PatchGAN behavior

## Initialization and Factory

## init_weights_v4(net)

- Conv / ConvTranspose: Normal(0, 0.02)
- InstanceNorm scale: Normal(1, 0.02), bias 0

## getDiscriminatorV4(...)

- builds `PatchGANDiscriminator`
- applies `init_weights_v4`
- auto-selects CUDA/CPU if `device` is omitted
- optional smoke test prints output shape for random 256x256 input

Return value:

- initialized discriminator on target device
