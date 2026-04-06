# model_v2/training_loop.py - v2 Training Loop

Source of truth: ../../model_v2/training_loop.py

Model: True UVCGAN v2
Entrypoint: train_v2(...)

---

## Component Structure

1. _make_lr_lambda
2. _global_grad_norm
3. _snapshot_module_to_cpu
4. train_v2

Within train_v2:
- setup phase
- epoch loop
- per-batch defensive checks
- discriminator stage
- generator stage (with rollback)
- epoch logging and scheduler stage
- validation and stopping stage
- finalization stage

---

## 1) _make_lr_lambda

Input:
- warmup, decay_start, total (integers)

Dataflow:
1. if epoch < warmup:
   - return max(1e-8, epoch/warmup)  [floor prevents exactly-zero LR]
2. else if epoch < decay_start:
   - return 1.0
3. else:
   - return max(0.0, 1.0 - (epoch - decay_start) / (total - decay_start))

Output:
- scalar multiplier for LambdaLR scheduler

---

## 2) _global_grad_norm

Input:
- iterable of parameters

Dataflow:
1. collect non-None .grad tensors, cast to float
2. compute per-tensor L2 norm
3. stack norms and compute global L2

Output:
- scalar float global grad norm (0.0 if no gradients)

---

## 3) _snapshot_module_to_cpu

Creates a CPU copy of a module's state_dict for lightweight rollback.

Input:
- module: nn.Module

Dataflow:
1. iterate state_dict items
2. detach and clone each tensor to CPU

Output:
- dict mapping parameter names to CPU tensors

Notes:
- used to save last-known-good G_AB and G_BA states
- avoids pinning extra GPU memory between training steps

---

## 4) train_v2

### 4A) Setup Phase

Input:
- cfg (UVCGANConfig) and optional scalar overrides

Dataflow:
1. resolve cfg (create default v2 config if None)
2. apply argument overrides (epoch_size, num_epochs, model_dir, val_dir, test_size)
3. enable cudnn.benchmark and TF32
4. load train/test dataloaders
5. instantiate G_AB, G_BA via getGeneratorsV2
6. instantiate D_A, D_B via getDiscriminatorsV2
7. instantiate UVCGANLoss
8. configure AMP GradScaler (disabled when not CUDA)
9. configure MetricsCalculator and EarlyStopping
10. configure Adam optimisers for G, D_A, D_B
11. configure LambdaLR schedulers for G, D_A, D_B
12. create output dirs, TensorBoard writer, history CSV
13. initialise last-known-good G snapshots via _snapshot_module_to_cpu

Initial model I/O shapes expected:
- real_A: (N, 3, H, W)
- real_B: (N, 3, H, W)
- fake_A / fake_B: (N, 3, H, W)
- discriminator outputs: list of per-scale logit maps

### 4B) Epoch Loop

Per epoch:
1. set G_AB, G_BA, D_A, D_B to train mode
2. reset scalar accumulators (loss_G, loss_D_A, loss_D_B, grad_norm_G)
3. reset accum_count and warn_near_uniform counters
4. iterate batches

### 4C) Per-batch Defensive Checks

Input:
- real_A, real_B: (N, 3, H, W) from dataloader

Checks:
- finite check: skip batch if any non-finite value
- std check over dims [1,2,3] -> (N,): warn if std < 1e-4 (near-uniform patch)

### 4D) Discriminator Stage

Inputs:
- real_A: (N, 3, H, W)
- real_B: (N, 3, H, W)

Dataflow:
1. freeze G (requires_grad=False), unfreeze D
2. no_grad generation:
   - fake_B_d = G_AB(real_A): (N, 3, H, W)
   - fake_A_d = G_BA(real_B): (N, 3, H, W)
3. zero_grad D_A, D_B
4. loss_D_A = discriminator_loss(D_A, real_A, fake_A_d, fake_A_buffer)
5. loss_D_B = discriminator_loss(D_B, real_B, fake_B_d, fake_B_buffer)
   - GP always computed in float32 (autocast disabled inside UVCGANLoss)
6. skip batch if either loss is non-finite
7. backward + grad_clip + optimizer step for D_A, then D_B

Discriminator output shapes in loss path:
- multi-scale: list of (N, 1, h, w) tensors per scale

Stage output:
- scalar loss_D_A_val and loss_D_B_val per batch

### 4E) Generator Stage (with rollback)

Inputs:
- same real_A and real_B

Dataflow:
1. unfreeze G, freeze D
2. zero_grad optimizer_G only at start of accumulation window (accum_count == 0)
3. forward under autocast:
   - loss_G, fake_A, fake_B = generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs)
   - fake_A, fake_B each (N, 3, H, W)
4. non-finite loss_G or non-finite fake outputs:
   - load last_good_G_AB_state / last_good_G_BA_state back into G_AB / G_BA
   - zero_grad, reset accum_count
   - halve AMP scaler scale if use_amp
   - increment nonfinite_g_streak counter and continue
5. scaler.scale(loss_G / accumulate).backward()
6. increment accum_count
7. at accumulation boundary (accum_count == accumulate or last batch):
   - scaler.unscale_ + grad_clip on G parameters
   - compute grad_norm_G via _global_grad_norm
   - scaler.step(optimizer_G) + scaler.update()
   - warn if AMP scaler scale drops below 1.0
   - refresh last_good_G_AB_state and last_good_G_BA_state snapshots
   - reset nonfinite_g_streak and accum_count

Stage outputs:
- scalar loss_G_val
- scalar grad_norm_G when stepping (0.0 otherwise)

### 4F) Epoch Logging and Scheduler Stage

Dataflow:
1. aggregate per-batch scalars into epoch averages
2. write TensorBoard: Loss/Generator, Loss/Discriminator_A, Loss/Discriminator_B, Diagnostics/GradNorm_G
3. step LR schedulers for G, D_A, D_B
4. write TensorBoard: LR/Generator, LR/Discriminator_A, LR/Discriminator_B
5. flush history to CSV every 5 epochs
6. save checkpoint every 20 epochs

Checkpoint payload:
- epoch, G_AB, G_BA, D_A, D_B state_dicts
- optimizer_G, optimizer_D_A, optimizer_D_B state_dicts

### 4G) Validation and Stopping Stage

Validation image path:
- run_validation called every epoch after validation_warmup_epochs

Metrics path:
1. calculate_metrics every early_stopping_interval epochs after early_stopping_warmup
2. avg_ssim = (ssim_A + ssim_B) / 2
3. EarlyStopping(ssim=avg_ssim, losses={G, D_A, D_B})
   - LSGAN losses are always >= 0; divergence check works without sign correction
4. write TensorBoard: EarlyStopping/ssim, EarlyStopping/counter, EarlyStopping/divergence_counter
5. break if early stopping triggered

### 4H) Finalization Stage

Dataflow:
1. final calculate_metrics call on test set
2. run_testing image export to test_images/
3. save final_checkpoint_epoch_N.pth
4. append remaining history to CSV, reload full history
5. close TensorBoard writer

Return:
- (history, G_AB, G_BA, D_A, D_B)

---

## Batch-level Shape Summary

Given batch size N and image size H × W:

- real_A: (N, 3, H, W)
- real_B: (N, 3, H, W)
- fake_B from G_AB: (N, 3, H, W)
- fake_A from G_BA: (N, 3, H, W)
- D outputs: list of per-scale weighted logit maps
  - example for H=W=256 and 3 scales:
    - (N, 1, 30, 30)
    - (N, 1, 14, 14)
    - (N, 1, 6, 6)

All scalar losses are reduced from these tensors before optimiser updates.

---

## Artifacts

Written under model_dir:
- tensorboard_logs/
- training_history.csv
- checkpoint_epoch_N.pth  (every 20 epochs)
- final_checkpoint_epoch_N.pth
- validation_images/
- test_images/
