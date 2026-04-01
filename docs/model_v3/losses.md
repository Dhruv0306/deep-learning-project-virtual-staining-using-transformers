# model_v3/losses.py - v3 Loss Helpers

Source of truth: ../../model_v3/losses.py

Role: Computes v3 diffusion loss with optional perceptual term, plus GAN adversarial objectives, cycle consistency, and identity constraints.

---

## Component Structure

1. `_lsgan_gen_loss` — Generator Least-Squares GAN loss
2. `_lsgan_disc_loss` — Discriminator Least-Squares GAN loss  
3. `_r1_penalty_loss` — R1 gradient regularization for discriminator
4. `_ddim_shortcut_from_xt` — Short DDIM denoising path (Phase 2 cycle)
5. `_compute_cycle_loss` — Cycle consistency L1 reconstruction loss
6. `_compute_identity_loss` — Identity L1 loss (no domain change at t=0)
7. `_compute_identity_weight` — Identity loss weight schedule (linear decay)
8. `compute_diffusion_loss` — Primary v3 training objective (diffusion + optional perceptual)

---

## 1) `_lsgan_gen_loss`

**Purpose**: Generator objective to fool discriminators.

**Inputs**:
- `fake_outputs`: discriminator output on generated images (tensor or list of tensors)

**Formula**:
```
L_gen = MSE(D_fake, 1)
```

**Output**: scalar tensor

---

## 2) `_lsgan_disc_loss`

**Purpose**: Discriminator objective to classify real vs fake.

**Inputs**:
- `real_outputs`: discriminator output on real images
- `fake_outputs`: discriminator output on generated images

**Formula**:
```
L_disc = 0.5 * (MSE(D_real, 1) + MSE(D_fake, 0))
```

**Output**: scalar tensor

---

## 3) `_r1_penalty_loss`

**Purpose**: Encourage small discriminator gradients on real images for training stability.

**Inputs**:
- `discriminator`: torch.nn.Module
- `real_images`: (N,3,256,256) or (N,4,32,32)
- `gamma`: penalty weight

**Formula**:
```
L_r1 = 0.5 * gamma * mean(||∇_x D(x)||^2)
```

**Output**: scalar tensor

---

## 4) `_ddim_shortcut_from_xt`

**Purpose**: Fast deterministic DDIM denoising from z_t start point (used in Phase 2 cycle).

**Inputs**:
- `model`: DiT generator
- `scheduler`: DDPMScheduler
- `z_t_start`: (N,4,32,32) noisy latent at timestep t_start
- `t_start`: (N,) timestep indices
- `condition`: (N,3,256,256) or encoded condition
- `target_domain`: 0 or 1 (A or B)
- `prediction_type`: "v" or "eps"
- `num_steps`: number of denoising steps
- `eta`: stochasticity parameter (0=deterministic, 1=full variance)

**Dataflow**:
1. Loop from `t_start` down to 0 in `num_steps` steps
2. Each step: model prediction → z0 pred → update z_t with DDIM formula
3. Optional noise injection when `eta > 0`

**Output**: z0: (N,4,32,32) denoised sample

---

## 5) `_compute_cycle_loss`

**Purpose**: Enforce unpaired-to-paired symmetry: A → B → A, B → A → B.

**Inputs**:
- `z0_fake_B`, `z0_fake_A`: latent samples from primary A→B, B→A passes
- `noise_A`, `noise_B`, `t_A`, `t_B`: **shared** noise and timesteps
- `fake_B_img`, `fake_A_img`: decoded intermediate images
- `real_A`, `real_B`: Ground truth images
- (model, scheduler, vae, prediction_type, ddim params)

**Formula**:
```
z_t_rec_A = add_noise(z0_fake_B, noise_A, t_A)  # Reuse shared noise
z0_rec_A = DDIM_shortcut(z_t_rec_A, ..., domain=A)
rec_A = VAE_decode(z0_rec_A)
L_cycle = ||rec_A - real_A||_1 + ||rec_B - real_B||_1
```

**Key Note**: Cycle reconstructs by **reusing the same noise and timestep** from the primary pass.

**Output**: scalar loss

---

## 6) `_compute_identity_loss`

**Purpose**: Preserve image identity in same domain (no translation at t=0).

**Inputs**:
- `z0_A`, `z0_B`: latents from real images at **t=0, epsilon=0**
- `real_A`, `real_B`: Ground truth images
- (model, scheduler, vae, device, prediction_type)

**Formula**:
```
t_idt = [0, 0, ..., 0]  # Zero timestep
z0_idt_A = model(z0_A, t=0, cond=real_A, target_domain=A)
idt_A = VAE_decode(z0_idt_A)
L_id = ||idt_A - real_A||_1 + ||idt_B - real_B||_1
```

**Output**: scalar loss

---

## 7) `_compute_identity_weight`

**Purpose**: Schedule identity loss weight from `l_start` → `l_end` over training.

**Formula**:
```
decay_end_epoch = num_epochs * decay_ratio
if epoch < decay_end_epoch:
    return l_start + (l_end - l_start) * (epoch / decay_end_epoch)
else:
    return l_end
```

**Use in training**: `weight = _compute_identity_weight(epoch, ...)` applied to `loss_idt` before backprop.

---

## 8) `compute_diffusion_loss`

**Purpose**: Primary v3 diffusion objective with Min-SNR weighting and optional perceptual term.

**Inputs**:
- `z0`, `z_t`, `t`: latent, noisy latent, timesteps
- `noise`: ground-truth noise (ε)
- `model_pred`: network prediction (v or ε)
- `real_B`: (N,3,256,256) real image for perceptual term
- `scheduler`, `vae`: diffusion infrastructure
- `perceptual_loss`: VGG19 module or None
- `lambda_perc`: perceptual weight
- `prediction_type`: "v" or "eps"
- `min_snr_gamma`: Min-SNR threshold (0 = disabled, e.g., 5.0 = enabled)
- `global_step`: current training step (for perceptual scheduling)
- `perceptual_every_n_steps`: run perceptual term every N steps
- `perceptual_batch_fraction`: fraction of batch for perceptual loss

**Dataflow**:

### Step 1: MSE Prediction Loss
```
if prediction_type == "v":
    v_target = scheduler.get_v_target(z0, noise, t)
    loss_mse = MSE(v_pred, v_target)
    x0_pred = scheduler.predict_x0_from_v(z_t, v_pred, t)
elif prediction_type == "eps":
    loss_mse = MSE(eps_pred, noise)
    x0_pred = scheduler.predict_x0(z_t, eps_pred, t)
```

### Step 2: Optional Min-SNR Weighting
```
if min_snr_gamma > 0:
    SNR = alpha_bar / (1 - alpha_bar)
    w = min(SNR, min_snr_gamma) / SNR  # v mode
    loss_mse = loss_mse * w
```

### Step 3: Optional Perceptual Term
```
if step % perceptual_every_n_steps == 0:
    n_perc = round(batch_size * perceptual_batch_fraction)
    fake_B_pred = vae.decode(x0_pred[:n_perc])
    loss_perc = perceptual_loss(fake_B_pred, real_B[:n_perc])
    total_loss = loss_mse + lambda_perc * loss_perc
else:
    total_loss = loss_mse
```

**Outputs**:
- `loss`: scalar tensor (total)
- `loss_simple`: scalar tensor (MSE-only, for logging)
- `loss_perc_val`: python float (perceptual scalar, 0.0 if disabled)

---

## Shape Summary

All loss functions preserve batch dimension:
- Latent tensors: (N, 4, 32, 32)
- Image tensors: (N, 3, 256, 256)
- Returned losses: scalar tensors (or float mirrors for logging)
