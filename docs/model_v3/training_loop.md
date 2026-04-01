# model_v3/training_loop.py - v3 Training Loop

Source of truth: ../../model_v3/training_loop.py

Role: Trains DiT diffusion model with adversarial losses, cycle consistency, and identity constraints. Includes EMA, gradient accumulation, AMP, and per-epoch validation with early stopping.

---

## Component Structure

1. `_make_cosine_warmup_lambda` — LR scheduler helper
2. `_global_grad_norm` — Gradient norm tracking
3. `_run_validation_v3` — Per-epoch validation and metrics
4. `train_v3` — Main training loop with phases

---

## 1) `_make_cosine_warmup_lambda`

**Purpose**: Create LR schedule with linear warmup → cosine decay.

**Formula**:
```
if epoch < warmup:
    return (epoch + 1) / warmup
else:
    progress = (epoch - warmup) / (total - warmup)
    cosine = 0.5 * (1 + cos(π * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cosine
```

**Use**: Passed to `torch.optim.lr_scheduler.LambdaLR` for generator and discriminators.

---

## 2) `_global_grad_norm`

**Purpose**: Compute L2 norm of all gradients for stability monitoring.

**Formula**:
```
total_grad_norm = ||[grad_1, grad_2, ..., grad_n]||_2
```

**Output**: Python float for TensorBoard logging

---

## 3) `_run_validation_v3`

**Purpose**: Run per-epoch validation on test set, compute metrics, save comparison images.

**Inputs**:
- `epoch`: current epoch number
- `ema_model`: EMA-averaged generator (uses `v_pred` output)
- `vae`: VAE decoder for latent → image conversion
- `sampler`: DDIMSampler for conditional image sampling
- `test_loader`: test dataloader with {A, B} pairs
- `device`: GPU/CPU
- `save_dir`: directory for validation image comparisons
- `calculator`: MetricsCalculator (SSIM, PSNR, FID)
- `num_steps`: DDIM denoising steps
- `writer`: TensorBoard SummaryWriter
- `max_batches`, `num_samples`, `fid_max_samples`: validation config

**Per-Batch Dataflow**:

1. Sample condition images:
   ```
   real_A: (B, 3, 256, 256)
   ```

2. Condition encode (via VAE):
   ```
   c = encode(real_A)  # Extract condition features
   ```

3. Latent-space DDIM sampling:
   ```
   z0 = sampler.sample(
       shape=(B, 4, 32, 32),
       condition=c,
       target_domain=1,  # Generate domain B
       num_steps=num_steps,
       eta=1.0
   )
   ```

4. Decode to image space:
   ```
   fake_B = vae.decode(z0)  # (B, 3, 256, 256)
   ```

5. Compute metrics:
   ```
   SSIM_B = ssim(fake_B, real_B)
   PSNR_B = psnr(fake_B, real_B)
   FID_B = fid_inception_distance(fake_B, real_B)
   ```

6. Save comparison image row for preview

**Outputs**:
- metrics_dict: {ssim_B, psnr_B, fid_B, ...}
- Saved PNG images to `save_dir`

---

## 4) `train_v3` — Main Training Loop

### 4A) Setup Phase

**Models**:
- `vae`: Frozen SD VAE for latent encoding/decoding
- `dit_model`: Trainable DiT generator
- `ema_model`: EMA copy of dit_model (for validation/testing)
- `D_A`: Domain A discriminator (multi-scale spectral-norm PatchGAN)
- `D_B`: Domain B discriminator (multi-scale spectral-norm PatchGAN)

**Optimizers**:
- Generator: AdamW, lr=1e-4, standard DCGAN-style schedule
- D_A, D_B: Adam, lr from config, beta1/beta2 from config

**Learning Rate Schedule**:
- Cosine warmup: 0 → 1e-4 over `warmup_epochs`
- Cosine decay: 1e-4 → 1e-10 over remaining epochs

**Infrastructure**:
- `scheduler`: DDPMScheduler (DDPM diffusion schedule)
- `sampler`: DDIMSampler for fast conditional sampling
- `scaler`: GradScaler for AMP (if enabled)
- `replay_A`, `replay_B`: ReplayBuffer for discriminator stability
- `perceptual_loss`: Optional VGG19 perceptual term

**Config Overrides**:
```python
train_v3(
    epoch_size=1024,           # Override
    num_epochs=120,            # Override
    model_dir="./models_v3",   # Override
    cfg=get_dit_8gb_config()   # Provide full config
)
```

### 4B) Per-Epoch Training Dataflow

**For each batch in `train_loader`**:

#### **1. Latent Encoding**

```
real_A, real_B: (N, 3, 256, 256) [from loader]
↓ VAE encoding
z0_A, z0_B: (N, 4, 32, 32) [latent at t=0]
↓ Add noise
t_A, t_B: (N,) [random timestep indices]
noise_A, noise_B: (N, 4, 32, 32) [random Gaussian]
↓ Noise addition
z_t_A, z_t_B: (N, 4, 32, 32) [noisy latents at t]
```

---

#### **2. Generator Step (Phase 1: Diffusion + Adversarial)**

**Forward pass** (both directions):

```
out_A2B = model(z_t_A, t_A, real_A, target_domain=1)
    → v_pred_A2B: (N, 4, 32, 32)
    → x0_pred_A2B: (N, 4, 32, 32) [x0 reconstruction]

out_B2A = model(z_t_B, t_B, real_B, target_domain=0)
    → v_pred_B2A: (N, 4, 32, 32)
    → x0_pred_B2A: (N, 4, 32, 32)
```

**Diffusion Loss** (for each direction):

```
loss_A2B = compute_diffusion_loss(
    z0=z0_A, z_t=z_t_A, t=t_A, noise=noise_A,
    model_pred=v_pred_A2B, real_B=real_B,
    lambda_perc=dcfg.lambda_perceptual_v3,
    min_snr_gamma=dcfg.min_snr_gamma
)
→ (loss, loss_simple, loss_perc_val)
```
Same for B→A.

**Adversarial Loss** (Generator):

```
z0_fake_B = x0_pred_A2B
fake_B_img = vae.decode(z0_fake_B).clamp(-1, 1)
D_B_output = D_B(fake_B_img)
loss_adv_G_B = LSGAN_gen_loss(D_B_output)
```
Same for domain A.

**Total Generator Loss**:

```
loss_G = (
    lambda_denoising * denoise_loss  # Main diffusion term
    + lambda_adv_curr * loss_adv_G   # Warmup-scaled adversarial term
    + lambda_cycle_v3 * loss_cyc     # Phase 2 cycle consistency
    + lambda_id_curr * loss_id       # Phase 2 identity loss
)
```

---

#### **3. Phase 2: Cycle & Identity Losses**

**Cycle Consistency**:

```
z0_fake_B, z0_fake_A from primary passes above
↓ Reuse same noise/timestep
z_t_rec_A = add_noise(z0_fake_B, noise_A, t_A)
z_t_rec_B = add_noise(z0_fake_A, noise_B, t_B)
↓ Short DDIM denoising (e.g., 4-10 steps)
z0_rec_A = DDIM_shortcut(..., z_t_rec_A, ...)
z0_rec_B = DDIM_shortcut(..., z_t_rec_B, ...)
↓ Decode
rec_A = vae.decode(z0_rec_A)
rec_B = vae.decode(z0_rec_B)
↓ L1 reconstruction
loss_cyc = ||rec_A - real_A||_1 + ||rec_B - real_B||_1
```

**Identity Loss** (only if `lambda_id_curr > 0`):

```
t_idt = [0, 0, ..., 0]  [zero timestep]
z0_idt_A = model(z0_A, t=0, cond=real_A, domain=A)["x0_pred"]
z0_idt_B = model(z0_B, t=0, cond=real_B, domain=B)["x0_pred"]
↓ Decode
idt_A = vae.decode(z0_idt_A)
idt_B = vae.decode(z0_idt_B)
↓ L1 constraint
loss_id = ||idt_A - real_A||_1 + ||idt_B - real_B||_1
```

**Identity Weight Schedule**:

```
lambda_id_curr = linear_decay(
    epoch,
    start=lambda_identity_v3_start,
    end=lambda_identity_v3_end,
    decay_ratio=identity_decay_end_ratio
)
```

---

#### **4. Discriminator Steps**

**Replay Buffer**:

```
fake_A_buffer = replay_A.push_and_pop(fake_A_img.detach())
fake_B_buffer = replay_B.push_and_pop(fake_B_img.detach())
# Returns mix of current-batch fakes + older fakes from history
```

**Discriminator A Loss**:

```
loss_D_A = LSGAN_disc_loss(
    D_A(real_A),         # discriminator real
    D_A(fake_A_buffer)   # discriminator fake
)

# Optional: R1 penalty (every r1_interval steps)
if use_r1_penalty and global_step % r1_interval == 0:
    r1_A = R1_penalty_loss(D_A, real_A, r1_gamma=100)
    loss_D_A += r1_A
```

Same for discriminator B.

**Adaptive Discriminator Update**:

```
if not (adaptive_d_update and loss_D_A < adaptive_d_loss_threshold):
    # Standard backward + step
    scaler.scale(loss_D_A).backward()
    scaler.step(optimizer_D_A)
else:
    # Skip update if discriminator loss is already low
    pass
```

---

### 4C) Gradient Accumulation & AMP

**Accumulation** (when `accumulate_grads > 1`):

```
effective_batch = accumulate_grads * batch_size
loss_scaled = loss / accumulate_grads
scaler.scale(loss_scaled).backward()
accum_count += 1

if accum_count == accumulate_grads:
    [clip gradients if grad_clip > 0]
    scaler.step(optimizer)
    scaler.update()
```

---

### 4D) EMA Update

**After generator step**:

```python
for ema_p, p in zip(ema_model.parameters(), dit_model.parameters()):
    ema_p.data = 0.9999 * ema_p.data + (1 - 0.9999) * p.data
    # Exponential moving average decay: 0.9999
```

---

### 4E) Logging & Checkpointing

**Per-batch logging** (every 50 batches + first & last):

```
Loss_DiT_A2B, Loss_DiT_B2A, Loss_DiT [avg]
Loss_G_Adv [generator adversarial]
Loss_Cyc [cycle consistency]
Loss_Id [identity, may be 0 if lambda=0]
Loss_D_A, Loss_D_B [discriminator losses]
Lambda_Adv, Lambda_Id [current schedule values]
Loss_Perceptual
Loss Total [weighted sum]
GradNorm
```

**Per-epoch logging** (TensorBoard):

```
Epoch, avg Loss_DiT, avg Loss_Perceptual, avg GradNorm
Metrics from validation run (SSIM, PSNR, FID)
LR values for all optimizers
```

**Checkpoint Saving** (every `checkpoint_interval` epochs):

```python
torch.save({
    'dit_state_dict': dit_model.state_dict(),
    'ema_state_dict': ema_model.state_dict(),
    'D_A_state_dict': D_A.state_dict(),
    'D_B_state_dict': D_B.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
    'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
    'lr_scheduler_G_state_dict': lr_scheduler_G.state_dict(),
    'lr_scheduler_D_A_state_dict': lr_scheduler_D_A.state_dict(),
    'lr_scheduler_D_B_state_dict': lr_scheduler_D_B.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'epoch': epoch,
    ...
}, checkpoint_path)
```

---

### 4F) Validation & Early Stopping

**Per-epoch validation** (every `validation_interval` epochs after `validation_warmup_epochs`):

```
test_metrics = _run_validation_v3(...)
history update with {ssim_B, psnr_B, fid_B}
```

**Early Stopping Criteria**:

1. **SSIM Plateau**: no improvement > `min_delta` for `patience` epochs
2. **Loss Divergence**: generator loss > `divergence_threshold` for `divergence_patience` epochs

**Resume Training**:

```python
train_v3(resume_checkpoint="models_v3/.../checkpoint_epoch_60.pth")
# Loads all model states, optimizer states, schedulers, scaler, epoch number
# Continues from next epoch
```

---

## Key Hyperparameters (from config)

| Parameter | Role | Default |
|-----------|------|---------|
| `lambda_denoising` | Diffusion loss weight | 1.0 |
| `lambda_adv_v3` | Generator adversarial weight | 0.1 |
| `lambda_adv_warmup_steps` | Ramp-up diffusion → adversarial | 500 |
| `lambda_cycle_v3` | Cycle consistency weight | 1.0 |
| `lambda_identity_v3_start` | Initial identity weight | 0.5 |
| `lambda_identity_v3_end` | Final identity weight | 0.1 |
| `identity_decay_end_ratio` | Decay over first N% of training | 0.5 |
| `lambda_perceptual_v3` | Perceptual loss weight | 0.01 |
| `use_r1_penalty` | Enable R1 discriminator regularization | true |
| `r1_gamma` | R1 penalty strength | 100 |
| `r1_interval` | Run R1 every N steps | 16 |
| `cycle_ddim_steps` | DDIM steps for Phase 2 cycle | 5 |
| `min_snr_gamma` | Min-SNR weighting threshold | 5.0 |

---

## Returns

```python
history, dit_model, ema_model, cond_tokenizer = train_v3(...)
```

- `history`: dict of epoch → metrics
- `dit_model`: final trained generator (before EMA)
- `ema_model`: EMA-averaged generator (recommended for inference)
- fourth return: currently `None` (reserved compatibility slot)
- real_B: (N,3,256,256)

Step 1: encode target image to latent
- z0 = vae.encode(real_B): (N,4,32,32)

Step 2: sample timestep and noise
- t: (N,)
- noise: (N,4,32,32)

Step 3: create noisy latent
- z_t = scheduler.add_noise(z0, noise, t): (N,4,32,32)

Step 4: condition and prediction
- c = cond_encoder(real_A): (N,Hd)
- eps_pred = dit_model(z_t, t, c): (N,4,32,32)

Step 5: loss
- compute_diffusion_loss returns scalar total and scalar components

Step 6: backward/step
- optional gradient accumulation
- grad clipping on dit_model + cond_encoder params
- optimizer step via GradScaler

Step 7: EMA update
- ema_model params updated from dit_model params

### 4C) Epoch-end dataflow

1. aggregate scalar means
2. log losses, grad norm, LR
3. periodic history CSV append
4. periodic checkpoint save

Checkpoint content includes:
- dit_state_dict
- cond_encoder_state_dict
- ema_state_dict
- optimizer_state_dict
- diffusion config

### 4D) Validation and stopping

1. run _run_validation_v3 after validation warmup
2. every early-stopping interval:
   - use ssim_B scalar
   - call EarlyStopping with loss dictionary
3. possibly break early

### 4E) Finalization

1. save final checkpoint
2. run final test export via _run_validation_v3 (is_test=True)
3. append and reload history
4. close writer

---

## Batch Shape Summary

With train batch size N:
- real_A, real_B: (N,3,256,256)
- z0, z_t, noise, eps_pred: (N,4,32,32)
- c: (N,Hd)
- decoded fake_B: (N,3,256,256)
