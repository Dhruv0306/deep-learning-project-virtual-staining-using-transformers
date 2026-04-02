# model_v3/discriminator.py - v3 ProjectionDiscriminator

Source of truth: ../../model_v3/discriminator.py

This module defines the dual v3 discriminators used for adversarial training.
Each discriminator combines local, global, and frequency-domain signals so the
model can judge both fine texture and whole-image structure.

## Public Components

1. `GlobalDiscriminatorBranch`
2. `FFTDiscriminatorBranch`
3. `ProjectionDiscriminator`
4. `getDiscriminatorsV3`

## Local Branch

The local branch is imported from v2 as `SpectralNormDiscriminator`.

Purpose:

- capture local stain granularity and texture

Behavior:

- standard PatchGAN-style spatial logits
- compatible with the list-aware LSGAN losses in `model_v3/losses.py`

## GlobalDiscriminatorBranch

Purpose:

- judge overall image structure, stain distribution, and tissue layout

Shape flow:

1. aggressive stride-4 downsampling
2. collapse to a single scalar logit per image

Output:

- `(N, 1)`

## FFTDiscriminatorBranch

Purpose:

- detect periodic VAE decode artifacts in the frequency domain

Shape flow:

1. convert RGB image to grayscale
2. compute real FFT magnitude
3. apply `log1p`
4. normalize per sample for brightness invariance
5. run a small spectral CNN and linear head

Output:

- `(N, 1)`

## ProjectionDiscriminator

This is the top-level discriminator used in training.

Enabled branches:

- local PatchGAN branch
- global image branch
- FFT branch

Forward output:

- list of logit tensors, one per enabled branch

The list is consumed directly by the list-aware LSGAN helper functions.

## getDiscriminatorsV3

Builds two independent discriminators, `D_A` and `D_B`.

Important config values from `DiffusionConfig`:

- `disc_base_channels`
- `disc_n_layers`
- `disc_global_base_channels`
- `disc_fft_base_channels`
- `disc_use_local`
- `disc_use_global`
- `disc_use_fft`

The factory also runs a smoke test on a random `256x256` input and prints the
output shapes for each active branch.
