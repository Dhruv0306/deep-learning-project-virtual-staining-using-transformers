# model_v4/patch_sampler.py

Source of truth: ../../model_v4/patch_sampler.py

Implements CUT-style patch sampling for PatchNCE loss.

## PatchSampler

## Purpose

Given one or more feature maps `(B, C, H, W)`, randomly sample spatial
positions and return patches in token format `(B, N, C)`.

This enables positive-pair alignment between real and generated features by
reusing sampled patch indices.

## Constructor

- `PatchSampler(num_patches=128)`

## Methods

## _sample_ids(b, hw, num_patches, device)

- samples random patch indices per batch item
- if `num_patches <= 0` or `num_patches >= hw`, returns all indices

## sample(features, num_patches=None, patch_ids=None)

Inputs:

- `features`: iterable of `(B, C, H, W)` tensors
- `num_patches`: optional override for sample count
- `patch_ids`: optional precomputed indices to reuse

Behavior:

- flattens each feature map to `(B, H*W, C)`
- chooses indices (random or provided)
- gathers selected tokens to `(B, N, C)`

Returns:

- sampled patch tensors list
- effective patch-index list

Typical use in training:

1. sample ids on real feature maps
2. reuse the same ids for corresponding fake feature maps
3. compute PatchNCE with matched spatial locations
