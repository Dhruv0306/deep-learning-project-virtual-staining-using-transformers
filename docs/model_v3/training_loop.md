# model_v3/training_loop.py - v3 Training Loop

Source of truth: ../../model_v3/training_loop.py

Role: Trains DiT diffusion model in latent space with EMA, optional perceptual term, validation, and test export.

---

## Component Structure

1. _make_cosine_warmup_lambda
2. _global_grad_norm
3. _run_validation_v3
4. train_v3

---

## 1) _make_cosine_warmup_lambda

Input:
- warmup, total, lr_min_ratio

Dataflow:
1. warmup region -> linear ramp
2. post-warmup -> cosine decay to lr_min_ratio

Output:
- scalar LR multiplier

---

## 2) _global_grad_norm

Input:
- parameter iterable

Dataflow:
1. collect non-null grads
2. per-grad L2
3. stacked global L2

Output:
- scalar float

---

## 3) _run_validation_v3

Inputs:
- ema_model, cond_encoder, vae, sampler
- test_loader
- num_steps, max_batches, num_samples

Per-batch shape dataflow:

1. batch from loader:
   - real_A: (B,3,256,256)
   - real_B: (B,3,256,256)
2. condition encode:
   - c = cond_encoder(real_A): (B,Hd)
3. latent sample:
   - z0 = sampler.sample(..., shape=(B,4,32,32)): (B,4,32,32)
4. decode:
   - fake_B = vae.decode(z0): (B,3,256,256)
5. metric inputs:
   - SSIM/PSNR compare fake_B vs real_B
6. optional save row image tensor for preview:
   - concat 4 images along batch dim -> (4,3,256,256)

Outputs:
- avg_metrics dict with scalar values (ssim_B, psnr_B, optional fid)

---

## 4) train_v3

Main return:
- history
- dit_model
- ema_model
- cond_encoder

### 4A) Setup dataflow

1. resolve config and overrides
2. create train/test loaders
3. instantiate modules:
   - vae (frozen)
   - cond_encoder
   - dit_model
   - ema_model
   - scheduler + sampler
4. optimizer/scheduler/scaler setup
5. optional perceptual module
6. writer and output paths

### 4B) Per-batch training dataflow

Input batch:
- real_A: (N,3,256,256)
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
