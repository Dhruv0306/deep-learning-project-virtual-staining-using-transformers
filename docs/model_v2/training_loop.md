# model_v2/training_loop.py — v2 Training Loop

Source: `../../model_v2/training_loop.py`  
Model: True UVCGAN v2 (Prokopenko et al., 2023)  
Entry point: `train_v2(...)`

---

## Component Structure

1. `_make_lr_lambda`
2. `_global_grad_norm`
3. `_snapshot_module_to_cpu`
4. `_snapshot_optimizer_state`
5. `_snapshot_scaler_state`
6. `train_v2`
   - A) Setup
   - B) Epoch loop
   - C) Per-batch defensive checks
   - D) Discriminator stage
   - E) Generator stage (with rollback)
   - F) Epoch logging and scheduler
   - G) Validation and early stopping
   - H) Finalization

---

## 1) _make_lr_lambda

Input: `warmup`, `decay_start`, `total` (int)

Three-phase schedule:
1. `epoch < warmup` → `max(1e-8, epoch / warmup)` (linear ramp; floor prevents zero LR)
2. `warmup <= epoch < decay_start` → `1.0` (constant plateau)
3. `epoch >= decay_start` → `max(0.0, 1.0 - (epoch - decay_start) / (total - decay_start))` (linear decay)

Output: `Callable[[int], float]` — multiplicative factor for `LambdaLR`.

---

## 2) _global_grad_norm

Input: iterable of `nn.Parameter`

Dataflow:
1. Collect non-None `.grad` tensors, cast to float.
2. Compute per-tensor L2 norm.
3. Stack norms and compute global L2.

Output: `float` — global grad norm (`0.0` if no gradients).

---

## 3) _snapshot_module_to_cpu

Input: `module: nn.Module`

Dataflow:
1. Iterate `state_dict` items.
2. Detach and clone each tensor to CPU.

Output: `dict` mapping parameter names → CPU tensors.

Used to save last-known-good `G_AB` / `G_BA` states for non-finite loss rollback.  
CPU placement avoids pinning extra GPU memory between training steps.

---

## 4) _snapshot_optimizer_state

Input: `optimizer: torch.optim.Optimizer`

Deep-copies the optimizer's `state_dict` so the rollback copy is independent of
live mutable tensors.

Output: `dict` — deep-copied optimizer state.

---

## 5) _snapshot_scaler_state

Input: `scaler: GradScaler`

Deep-copies the AMP `GradScaler` state for rollback alongside the generator snapshot.

Output: `dict` — deep-copied scaler state.

---

## 6) train_v2

### A) Setup

Input: `cfg: UVCGANConfig` and optional scalar overrides.

Dataflow:
1. Resolve `cfg` (create `get_default_config(model_version=2)` if `None`).
2. Apply argument overrides: `epoch_size`, `num_epochs`, `model_dir`, `val_dir`, `test_size`.
3. Enable `cudnn.benchmark` and TF32.
4. Build `train_loader` / `test_loader` via `getDataLoader`.
5. Instantiate `G_AB`, `G_BA` via `getGeneratorsV2`; move to device.
6. Instantiate `D_A`, `D_B` via `getDiscriminatorsV2`; move to device.
7. Instantiate `UVCGANLoss`.
8. Configure `GradScaler` (disabled when not CUDA).
9. Configure `MetricsCalculator` and `EarlyStopping`.
10. Configure AdamW optimizer for G (Transformer bottleneck) and Adam for D_A, D_B.
11. Configure `LambdaLR` schedulers for G, D_A, D_B.
12. Create output dirs, `SummaryWriter`, history CSV.
13. Initialize last-known-good G snapshots via `_snapshot_module_to_cpu`,
    `_snapshot_optimizer_state`, `_snapshot_scaler_state`.

Expected I/O shapes:
- `real_A`, `real_B`: `(N, 3, H, W)`
- `fake_A`, `fake_B`: `(N, 3, H, W)`
- Discriminator outputs: list of per-scale logit maps `(N, 1, h, w)`

### B) Epoch Loop

Per epoch:
1. Set all four models to `train()`.
2. Reset scalar accumulators (`loss_G`, `loss_D_A`, `loss_D_B`, `grad_norm_G`).
3. Reset `accum_count` and `warn_near_uniform` counters.
4. Iterate batches.

### C) Per-batch Defensive Checks

Input: `real_A`, `real_B` — `(N, 3, H, W)` from dataloader.

- Finite check: skip batch if any non-finite value.
- Std check over dims `[1,2,3]` → `(N,)`: warn (throttled to every 100 hits) if std < 1e-4.

### D) Discriminator Stage

Dataflow:
1. Freeze G (`requires_grad=False`), unfreeze D.
2. `torch.no_grad()` generation:
   - `fake_B_d = G_AB(real_A)` — `(N, 3, H, W)`
   - `fake_A_d = G_BA(real_B)` — `(N, 3, H, W)`
3. Zero-grad D_A, D_B.
4. `loss_D_A = discriminator_loss(D_A, real_A, fake_A_d, fake_A_buffer)`
5. `loss_D_B = discriminator_loss(D_B, real_B, fake_B_d, fake_B_buffer)`
   - GP always computed in float32 (`autocast` disabled inside `UVCGANLoss`).
6. Skip batch if either loss is non-finite.
7. `backward` + grad-clip + optimizer step for D_A, then D_B.

Outputs: scalar `loss_D_A_val`, `loss_D_B_val` per batch.

### E) Generator Stage (with rollback)

Dataflow:
1. Unfreeze G, freeze D.
2. Zero-grad `optimizer_G` only at `accum_count == 0`.
3. Forward under `autocast`:
   - `loss_G, fake_A, fake_B = generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs)`
   - `loss_G_scaled = loss_G / accumulate`
4. Non-finite `loss_G` or non-finite generator outputs:
   - Restore `G_AB`, `G_BA`, `optimizer_G`, `scaler` from last-known-good snapshots.
   - Zero-grad, reset `accum_count`.
   - Halve AMP scaler scale if `use_amp`.
   - Increment `nonfinite_g_streak`; continue.
5. `scaler.scale(loss_G_scaled).backward()`; increment `accum_count`.
6. At accumulation boundary (`accum_count == accumulate` or last batch):
   - `scaler.unscale_` + grad-clip on G parameters.
   - Compute `grad_norm_G` via `_global_grad_norm`.
   - `scaler.step(optimizer_G)` + `scaler.update()`.
   - Warn if AMP scaler scale drops below 1.0.
   - Refresh all last-known-good snapshots.
   - Reset `nonfinite_g_streak` and `accum_count`.

Outputs: scalar `loss_G_val`; `grad_norm_G` when stepping (`0.0` otherwise).

### F) Epoch Logging and Scheduler

Dataflow:
1. Aggregate per-batch scalars into epoch averages.
2. TensorBoard: `Loss/Generator`, `Loss/Discriminator_A`, `Loss/Discriminator_B`, `Diagnostics/GradNorm_G`.
3. Step LR schedulers for G, D_A, D_B.
4. TensorBoard: `LR/Generator`, `LR/Discriminator_A`, `LR/Discriminator_B`.
5. Flush history to CSV every 5 epochs.
6. Save checkpoint every 20 epochs.

Checkpoint payload: `epoch`, `G_AB`, `G_BA`, `D_A`, `D_B` state_dicts + optimizer state_dicts.

### G) Validation and Early Stopping

- `run_validation` called every epoch after `validation_warmup_epochs`.
- `calculate_metrics` called every `early_stopping_interval` epochs after `early_stopping_warmup`.
- `avg_ssim = (ssim_A + ssim_B) / 2`.
- `EarlyStopping(ssim=avg_ssim, losses={G, D_A, D_B})`.
  - LSGAN losses are always >= 0; divergence check needs no sign correction.
- TensorBoard: `EarlyStopping/ssim`, `EarlyStopping/counter`, `EarlyStopping/divergence_counter`.
- Break if early stopping triggered.

### H) Finalization

Dataflow:
1. Final `calculate_metrics` on test set.
2. `run_testing` image export to `test_images/`.
3. Save `final_checkpoint_epoch_N.pth`.
4. Append remaining history to CSV; reload full history.
5. Close `SummaryWriter`.

Returns: `(history, G_AB, G_BA, D_A, D_B)`

---

## Batch-level Shape Summary

Given batch size N and image size H × W:

| Tensor | Shape |
|---|---|
| `real_A`, `real_B` | `(N, 3, H, W)` |
| `fake_A`, `fake_B` | `(N, 3, H, W)` |
| D output (per scale, H=W=256, 3 scales) | `(N,1,30,30)`, `(N,1,14,14)`, `(N,1,6,6)` |

All scalar losses are reduced from these tensors before optimizer updates.

---

## Artifacts Written Under `model_dir`

```
tensorboard_logs/
training_history.csv
checkpoint_epoch_N.pth        (every 20 epochs)
final_checkpoint_epoch_N.pth
validation_images/
test_images/
```
