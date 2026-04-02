# model_v3/training_loop.py - v3 Training Loop

Source of truth: ../../model_v3/training_loop.py

This module trains the v3 CycleDiT pipeline with two-stage generator updates,
dual discriminators, EMA, replay buffers, AMP, gradient accumulation, periodic
validation, and resume support.

## Public Components

1. `_make_cosine_warmup_lambda`
2. `_global_grad_norm`
3. `_set_requires_grad`
4. `_run_validation_v3`
5. `train_v3`

## `_make_cosine_warmup_lambda`

Creates a learning-rate multiplier with linear warmup followed by cosine decay
to a small minimum ratio.

## `_global_grad_norm`

Returns the L2 norm of all available gradients as a Python float for logging.

## `_run_validation_v3`

Runs validation or test-time export on the A/B paired test loader.

Behavior:

1. use the EMA generator when available
2. sample A→B and B→A translations with DDIM
3. decode latents through the VAE
4. compute SSIM, PSNR, and optionally FID for both domains
5. save comparison grids to the output directory

The validation path now tracks both domains instead of only the B domain.

## `train_v3`

Signature highlights:

- `resume_checkpoint` is supported
- `cfg` defaults to `get_dit_8gb_config()` when omitted
- return value is `(history, dit_model, ema_model, None)`

### Setup

The loop constructs:

- frozen VAE wrapper
- trainable DiT generator
- EMA generator copy
- dual `ProjectionDiscriminator` instances
- DDPM scheduler and DDIM sampler
- replay buffers
- optional VGG perceptual loss

### Resume Support

When `resume_checkpoint` is provided, the loop restores:

- model weights
- EMA weights
- discriminator weights
- optimizer state
- scheduler state
- AMP scaler state when available
- starting epoch

### Per-Batch Flow

For each batch the code now performs:

1. encode real images to VAE latents
2. sample timesteps and noise
3. run diffusion-only generator loss first
4. run a fresh forward pass for adversarial, cycle, and identity losses
5. update the EMA model after generator stepping
6. update discriminators with replay-buffer fakes and optional R1 penalty

This ordering matches the current implementation and is intended to reduce
peak VRAM by freeing diffusion activations before the auxiliary graph is built.

### Loss Terms

- diffusion loss from `compute_diffusion_loss`
- adversarial loss from `_lsgan_gen_loss`
- cycle loss from `_compute_cycle_loss`
- identity loss from `_compute_identity_loss`
- discriminator losses from `_lsgan_disc_loss`
- optional R1 penalty from `_r1_penalty_loss`

### Logging and Checkpointing

Per-batch history now stores:

- `Loss_DiT_A2B`
- `Loss_DiT_B2A`
- `Loss_DiT`
- `Loss_G_Adv`
- `Loss_Cyc`
- `Loss_Id`
- `Loss_D_A`
- `Loss_D_B`
- `Lambda_Adv`
- `Lambda_Id`
- `Loss_Perceptual`
- `Loss Total`
- `GradNorm`

The history is written to CSV and visualized through `model_v3/history_utils.py`.

### Validation and Early Stopping

Validation runs after the configured warmup and uses the averaged SSIM score
from both domains for early-stopping decisions.

### Returned Values

- `history`: nested epoch -> batch loss dictionary
- `dit_model`: raw generator weights
- `ema_model`: EMA generator weights
- `None`: placeholder to preserve compatibility with the v1/v2/v4-style tuple

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
