# model_v2/training_loop.py - v2 Training Loop

Source of truth: ../../model_v2/training_loop.py

Model: True UVCGAN v2
Entrypoint: train_v2(...)

---

## Component Structure

1. _make_lr_lambda
2. _global_grad_norm
3. train_v2

Within train_v2:
- setup phase
- epoch loop
- per-batch checks
- discriminator stage
- generator stage
- epoch logging and scheduler stage
- validation and stopping stage
- finalization stage

---

## 1) _make_lr_lambda

Input:
- warmup, decay_start, total (integers)

Dataflow:
1. if epoch < warmup:
   - return max(1e-8, epoch/warmup)
2. else if epoch < decay_start:
   - return 1.0
3. else:
   - return linear decay factor to zero

Output:
- scalar multiplier for LR scheduler

---

## 2) _global_grad_norm

Input:
- iterable of parameters

Dataflow:
1. collect non-None grads
2. each grad converted to float and L2 norm computed
3. stack norms and compute global L2

Output:
- scalar float grad norm

---

## 3) train_v2

### 3A) Setup Phase

Input data objects:
- cfg and optional override args

Dataflow:
1. resolve cfg
2. load train/test dataloaders
3. instantiate G_AB, G_BA, D_A, D_B
4. instantiate loss module
5. configure optimizers/schedulers/scaler/writer

Initial model I/O shapes expected:
- real_A: (N,3,H,W)
- real_B: (N,3,H,W)
- fake_A/fake_B: (N,3,H,W)
- discriminator outputs: tensor or list by scale

### 3B) Epoch Loop

Per epoch dataflow:
1. set train mode
2. reset scalar accumulators
3. iterate batch data

### 3C) Per-batch Defensive Checks

Input:
- real_A, real_B from dataloader

Checks with shapes:
- finite check on (N,3,H,W)
- std check over dims [1,2,3] producing (N,) for each domain

### 3D) Discriminator Stage

Inputs:
- real_A: (N,3,H,W)
- real_B: (N,3,H,W)

Dataflow:
1. freeze G, unfreeze D
2. no_grad generation:
   - fake_B_d = G_AB(real_A): (N,3,H,W)
   - fake_A_d = G_BA(real_B): (N,3,H,W)
3. compute loss_D_A via discriminator_loss(D_A, real_A, fake_A_d, buffer)
4. compute loss_D_B via discriminator_loss(D_B, real_B, fake_B_d, buffer)
5. backward and optimizer steps for D_A, D_B

Discriminator output shapes in loss path:
- single-scale: (N,1,h,w)
- multi-scale: list of such tensors

Stage output:
- scalar loss_D_A_val and loss_D_B_val per batch

### 3E) Generator Stage

Inputs:
- same real_A and real_B

Dataflow:
1. unfreeze G, freeze D
2. optional accumulation window zero_grad
3. forward under autocast:
   - loss_G, fake_A, fake_B = generator_loss(...)
   - fake_A, fake_B each (N,3,H,W)
4. scale by accumulate_grads
5. backward via scaler
6. step optimizer_G at accumulation boundary

Stage outputs:
- scalar loss_G_val
- optional scalar grad_norm_G when stepping

### 3F) Epoch logging and scheduler stage

Dataflow:
1. aggregate scalar averages over batches
2. write TensorBoard scalars
3. step LR schedulers
4. write LR scalars
5. periodic CSV flush
6. periodic checkpoint save

Checkpoint payload shapes:
- model state dict tensors with model-defined parameter shapes
- optimizer state tensors with matching parameter groups

### 3G) Validation and stopping stage

Validation image path:
- run_validation after warmup

Metrics path:
1. calculate_metrics -> scalar metrics dictionary
2. avg_ssim computed from domain A and B SSIM scalars
3. EarlyStopping call with scalar ssim and scalar losses
4. possible break

### 3H) Finalization stage

Dataflow:
1. final calculate_metrics call
2. run_testing image export
3. final checkpoint save
4. append and reload history
5. close writer

Return:
- history dictionary
- G_AB, G_BA, D_A, D_B modules

---

## Batch-level Shape Summary

Given input batch size N and image size H by W:

- real_A: (N,3,H,W)
- real_B: (N,3,H,W)
- fake_B from G_AB: (N,3,H,W)
- fake_A from G_BA: (N,3,H,W)
- D outputs: list of per-scale maps
  - example for H=W=256 and 3 scales:
    - (N,1,30,30)
    - (N,1,14,14)
    - (N,1,6,6)

All scalar losses are reduced from these tensors before optimizer updates.

---

## Artifacts

Written under model_dir:
- tensorboard_logs
- training_history.csv
- checkpoint_epoch_*.pth
- final_checkpoint_epoch_*.pth
- validation_images
- test_images
