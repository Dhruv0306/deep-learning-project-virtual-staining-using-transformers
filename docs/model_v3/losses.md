# model_v3/losses.py - v3 Loss Helpers

Source of truth: ../../model_v3/losses.py

This module contains the loss functions used by the v3 diffusion training
loop: LSGAN objectives, R1 regularization, cycle and identity losses, and
the main diffusion denoising loss.

## Public Functions

1. `_lsgan_gen_loss`
2. `_lsgan_disc_loss`
3. `_r1_penalty_loss`
4. `_ddim_shortcut_from_xt`
5. `_compute_cycle_loss`
6. `_compute_identity_loss`
7. `_compute_identity_weight`
8. `compute_diffusion_loss`

## LSGAN Helpers

Both LSGAN helpers accept either a single tensor or a list of tensors.
That keeps them compatible with the v3 multi-branch discriminator.

- `_lsgan_gen_loss(fake_outputs)` minimizes `MSE(D(fake), 1)`
- `_lsgan_disc_loss(real_outputs, fake_outputs)` minimizes
  `0.5 * (MSE(D(real), 1) + MSE(D(fake), 0))`

## R1 Penalty

`_r1_penalty_loss(discriminator, real_images, gamma)` computes
`0.5 * gamma * ||∇_x D(x_real)||^2` on real images.

It is always evaluated in float32 outside autocast for stability.

## DDIM Cycle Shortcut

`_ddim_shortcut_from_xt(...)` starts from an existing noisy latent `z_t` and
performs a short reverse DDIM trajectory back to `z0`.

This is used by cycle consistency so the model reconstructs from the same
noise and timestep that were used in the primary forward pass.

## Cycle Loss

`_compute_cycle_loss(...)`:

1. rebuilds noisy latents for the fake outputs using the shared noise/timestep
2. runs short DDIM reconstruction in the opposite domain
3. decodes the reconstructed latent with the VAE
4. applies L1 reconstruction against the original clean latents

## Identity Loss

`_compute_identity_loss(...)` runs the model at `t=0` on same-domain inputs and
encourages the output latent to stay close to the original latent.

## Identity Weight Schedule

`_compute_identity_weight(epoch, num_epochs, l_start, l_end, decay_ratio)`
linearly decays the identity weight over the first portion of training and then
holds it constant.

## Diffusion Loss

`compute_diffusion_loss(...)` returns three values:

- total diffusion loss
- MSE-only diffusion loss for logging
- perceptual loss scalar for logging

Behavior:

1. compute the primary v- or epsilon-prediction loss
2. optionally apply Min-SNR weighting
3. optionally compute a VGG perceptual term on decoded predictions

The perceptual branch is scheduled by `perceptual_every_n_steps` and may use
only a fraction of the batch via `perceptual_batch_fraction`.

## Shapes

- latents: `(N, 4, 32, 32)`
- images: `(N, 3, 256, 256)`
- returned losses: scalar tensors plus a float perceptual mirror for logging
