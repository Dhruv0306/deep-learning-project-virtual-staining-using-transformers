# model_v1/losses.py - v1 Losses

Source of truth: ../../model_v1/losses.py

Model: Hybrid CycleGAN/UVCGAN v1
Role: Implements generator/discriminator objectives, perceptual terms, replay buffering, and optional gradient penalty.

---

## Composite Objective

CycleGANLoss combines multiple loss components for bidirectional translation.

Generator total:

L_G = L_GAN_AB + L_GAN_BA
    + L_cycle_A + L_cycle_B
    + L_identity_A + L_identity_B
    + L_cycle_perceptual_A + L_cycle_perceptual_B
    + L_identity_perceptual_A + L_identity_perceptual_B

where each family is scaled by configurable lambda values.

Discriminator total (per domain):

L_D = 0.5 * (L_real + L_fake) + lambda_gp * GP

---

## VGGPerceptualLoss

Uses frozen VGG19 feature maps to compare generated and target images at feature level.

Constructor:

- resize_to: optional size for bilinear resizing before VGG extraction

Internal slices:

- slice1: early VGG features (relu1_2 region)
- slice2: mid VGG features (relu2_2 region)
- slice3: deeper VGG features (relu3_4 region)

Normalization:

- ImageNet mean/std buffers are registered and applied before feature extraction

Forward behavior:

1. expand grayscale to 3 channels if needed
2. optionally resize both inputs
3. normalize with ImageNet stats
4. extract three feature levels
5. sum L1 distances across feature levels

All VGG parameters are frozen (requires_grad=False).

---

## CycleGANLoss

Constructor parameters:

- lambda_cycle
- lambda_identity
- lambda_cycle_perceptual
- lambda_identity_perceptual
- lambda_gp
- perceptual_resize
- device

Internal criteria:

- criterion_GAN: MSELoss (LSGAN style)
- criterion_cycle: L1Loss
- criterion_identity: L1Loss
- criterion_perceptual: VGGPerceptualLoss

Replay buffers:

- fake_A_buffer
- fake_B_buffer

These reduce discriminator oscillation by mixing historical generated samples.

---

## Identity Weight Schedule

### get_identity_lambda(epoch, total_epochs)

Behavior:

- first half of training: returns lambda_identity unchanged
- second half: exponential decay with factor 0.997 per epoch offset from midpoint

This keeps identity regularization strong early, then relaxes it later.

---

## Gradient Penalty

### gradient_penalty(D, real, fake)

Computes WGAN-GP style two-sided penalty on random interpolations between real and fake samples.

Implementation notes:

- interpolation coefficient epsilon sampled per batch item
- computation forced to float32 inside autocast(enabled=False)
- uses autograd.grad on discriminator output wrt interpolated input
- penalty term: (||grad||_2 - 1)^2 mean

Returned value is scalar >= 0.

---

## Generator Loss API

### generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, total_epochs)

Computes and returns:

- loss_G: total generator objective
- fake_A: generated domain-A images from real_B
- fake_B: generated domain-B images from real_A

Included terms:

- identity losses for both domains (with scheduled identity lambda)
- GAN losses against D_A and D_B with target ones
- cycle reconstruction losses
- perceptual cycle losses
- perceptual identity losses

fake_A and fake_B are returned for discriminator updates in the same iteration.

---

## Discriminator Loss API

### discriminator_loss(D, real, fake, replay_buffer=None)

Computes per-domain discriminator objective.

Steps:

1. real prediction loss using one-sided label smoothing target (0.97)
2. fake prediction loss using buffered or fresh fake samples against target 0
3. average real/fake losses
4. optionally add lambda_gp * gradient_penalty

Returns scalar discriminator loss for that domain.

---

## Integration in v1 Training

model_v1/training_loop.py calls:

- generator_loss(...) once per batch
- discriminator_loss(...) separately for D_A and D_B

with AMP enabled when CUDA is available.
