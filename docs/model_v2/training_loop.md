# model_v2/training_loop.py — Detailed Reference

Source: `../../model_v2/training_loop.py`  
Model: True UVCGAN v2 (Prokopenko et al., 2023)  
Primary entry point: `train_v2(...)`

---

## Table of Contents

1. [Module-level Data Flow](#1-module-level-data-flow)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Helper Function: `_load_checkpoint_compat`](#3-helper-function-_load_checkpoint_compat)
4. [Helper Function: `_load_state_dict_with_compat`](#4-helper-function-_load_state_dict_with_compat)
5. [Helper Function: `_make_lr_lambda`](#5-helper-function-_make_lr_lambda)
6. [Helper Function: `_global_grad_norm`](#6-helper-function-_global_grad_norm)
7. [Helper Function: Snapshot/Rollback Utilities](#7-helper-function-snapshotrollback-utilities)
8. [Function: `train_v2`](#8-function-train_v2)
   - [8A Signature and Return](#8a-signature-and-return)
   - [8B Parameters](#8b-parameters)
   - [8C Config and Override Resolution](#8c-config-and-override-resolution)
   - [8D Resume Phase](#8d-resume-phase)
   - [8E Setup Phase](#8e-setup-phase)
   - [8F Per-epoch Loop](#8f-per-epoch-loop)
   - [8G Per-batch Update Order](#8g-per-batch-update-order)
   - [8H Epoch-end Logging and Scheduler](#8h-epoch-end-logging-and-scheduler)
   - [8I Validation and Early Stopping](#8i-validation-and-early-stopping)
   - [8J Finalization](#8j-finalization)
9. [Tensor Shape Reference](#9-tensor-shape-reference)
10. [Loss and Stability Notes](#10-loss-and-stability-notes)
11. [Optimizer and LR Schedule Reference](#11-optimizer-and-lr-schedule-reference)
12. [Checkpoint Payload Reference](#12-checkpoint-payload-reference)
13. [TensorBoard Scalar Reference](#13-tensorboard-scalar-reference)
14. [Artifact Layout](#14-artifact-layout)

---

## 1. Module-level Data Flow

```text
getDataLoader(epoch_size, image_size, batch_size, num_workers)
  -> train_loader
  -> test_loader

getGeneratorsV2(...) -> G_AB, G_BA
getDiscriminatorsV2(...) -> D_A, D_B
UVCGANLoss(...) -> loss_fn

---- per epoch ---------------------------------------------------------------
for batch in train_loader:
  real_A, real_B: (N,3,H,W)

  1) Discriminator stage (float32-safe GP inside loss)
     fake_B_d = G_AB(real_A) [no_grad]
     fake_A_d = G_BA(real_B) [no_grad]
     loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A_d, fake_A_buffer)
     loss_D_B = loss_fn.discriminator_loss(D_B, real_B, fake_B_d, fake_B_buffer)
     backward + clip + step D_A/D_B

  2) Generator stage (AMP + gradient accumulation)
     loss_G, fake_A, fake_B = loss_fn.generator_loss(...)
     scale(loss_G / accumulate).backward()
     if accumulation boundary:
       unscale + clip + optimizer_G.step + scaler.update

  3) Defensive rollback paths
     - if non-finite D loss/params -> restore last-good D snapshots
     - if non-finite G loss/outputs/params -> restore last-good G snapshots

---- epoch end ---------------------------------------------------------------
aggregate averages
TensorBoard scalars
LR schedulers step
CSV flush every 5 epochs
checkpoint every save_checkpoint_every
validation + periodic metric evaluation + early stopping

---- finalization ------------------------------------------------------------
calculate_metrics(...)
run_testing(...) -> test_images/
save final checkpoint
flush + reload full history CSV
return history, G_AB, G_BA, D_A, D_B
```

---

## 2. Imports and Dependencies

| Symbol | Source | Purpose |
|---|---|---|
| `UVCGANLoss` | `model_v2.losses` | Composite v2 objective (GAN + cycle + identity + perceptual + penalties) |
| `UVCGANConfig`, `get_default_config` | `config` | v2 runtime and training config |
| `getDataLoader` | `shared.data_loader` | Unpaired dataloader for domains A/B |
| `EarlyStopping` | `shared.EarlyStopping` | SSIM + divergence-based early-stop logic |
| `append_history_to_csv`, `load_history_from_csv` | `shared.history_utils` | Incremental and full history persistence |
| `MetricsCalculator` | `shared.metrics` | SSIM/PSNR/FID metrics |
| `getDiscriminatorsV2` | `model_v2.discriminator` | Multi-scale discriminators |
| `getGeneratorsV2` | `model_v2.generator` | Cross-domain-aware v2 generators |
| `run_validation`, `calculate_metrics` | `shared.validation` | Qualitative and quantitative validation |
| `run_testing` | `shared.testing` | Final test image export |

---

## 3. Helper Function: `_load_checkpoint_compat`

### Signature

```python
_load_checkpoint_compat(checkpoint_path: str, map_location) -> dict
```

### Purpose

Loads local checkpoints safely across PyTorch versions, including PyTorch 2.6+
behavior where `torch.load` defaults to `weights_only=True`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | Path to `.pth` file |
| `map_location` | `str | torch.device` | Device remapping (`"cpu"`, `"cuda"`, etc.) |

### Return

`dict` checkpoint payload.

### Sample I/O

- Input: `("checkpoint_epoch_80.pth", "cpu")`
- Output keys (example): `{epoch, config, G_AB, G_BA, D_A, D_B, optimizer_G, ...}`

---

## 4. Helper Function: `_load_state_dict_with_compat`

### Signature

```python
_load_state_dict_with_compat(module: nn.Module, state_dict: dict, tag: str) -> None
```

### Purpose

Attempts strict loading first; if strict load fails, loads only key/shape-compatible tensors.
Useful for resuming from older checkpoints after architecture changes.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `module` | `nn.Module` | Target model to load into |
| `state_dict` | `dict` | Saved parameter tensors |
| `tag` | `str` | Human-readable module label for warning logs |

### Sample I/O

- Input: `module=G_AB`, partial `state_dict` from older run
- Output: module updated with compatible tensors; warnings printed if partial

---

## 5. Helper Function: `_make_lr_lambda`

### Signature

```python
_make_lr_lambda(warmup: int, decay_start: int, total: int) -> Callable[[int], float]
```

### Purpose

Creates a three-phase LR multiplier function for `LambdaLR`:

1. Warm-up: linear ramp near `0 -> 1`
2. Plateau: constant `1`
3. Decay: linear `1 -> 0`

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `warmup` | `int` | Warm-up epoch count |
| `decay_start` | `int` | Epoch index where linear decay starts |
| `total` | `int` | Total epochs |

### Return

Callable `lr_lambda(epoch: int) -> float`.

### Sample I/O

- Input: `(warmup=10, decay_start=100, total=200)`
- Example output: `lr_lambda(0)=1e-8`, `lr_lambda(10)=1.0`, `lr_lambda(150)=0.5`, `lr_lambda(200)=0.0`

---

## 6. Helper Function: `_global_grad_norm`

### Signature

```python
_global_grad_norm(parameters) -> float
```

### Purpose

Returns global L2 norm of currently available gradients.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `parameters` | `Iterable[nn.Parameter]` | Parameters to inspect |

### Return

`float` gradient norm. Returns `0.0` if no gradients are present.

### Sample I/O

- Input: parameters from `G_AB` and `G_BA`
- Output: e.g., `3.71`

---

## 7. Helper Function: Snapshot/Rollback Utilities

### `_snapshot_module_to_cpu(module)`

Creates detached CPU clone of `state_dict` for safe rollback.

### `_snapshot_optimizer_state(optimizer)`

Deep-copies optimizer state (no tensor aliasing).

### `_snapshot_scaler_state(scaler)`

Deep-copies AMP `GradScaler` state.

### `_module_parameters_are_finite(module)`

Returns `True` only if every parameter tensor is finite.

### Why They Exist

These utilities implement last-known-good rollback for both generators and
discriminators when non-finite values appear in loss or parameters.

---

## 8. Function: `train_v2`

### 8A Signature and Return

```python
train_v2(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint=None,
    cfg: Optional[UVCGANConfig] = None,
) -> tuple[dict, nn.Module, nn.Module, nn.Module, nn.Module]
```

Returns `(history, G_AB, G_BA, D_A, D_B)`.

### 8B Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | `int | None` | `cfg.training.epoch_size` | Max samples drawn each epoch |
| `num_epochs` | `int | None` | `cfg.training.num_epochs` | Total epochs |
| `model_dir` | `str | None` | `cfg.model_dir` | Output root: checkpoints/logs/history |
| `val_dir` | `str | None` | `cfg.val_dir` | Validation image output directory |
| `test_size` | `int | None` | `cfg.training.test_size` | Test sample count for final export |
| `resume_checkpoint` | `str | None` | `None` | Optional resume checkpoint path |
| `cfg` | `UVCGANConfig | None` | `get_default_config(model_version=2)` | Full v2 config object |

### 8C Config and Override Resolution

- Initializes `cfg` via `get_default_config(model_version=2)` when absent.
- Applies explicit function arguments as top-priority overrides.
- If resuming and checkpoint contains a full `UVCGANConfig`, replaces config from checkpoint, then reapplies explicit overrides.

### 8D Resume Phase

When `resume_checkpoint` is provided:

1. Validates file exists.
2. Loads checkpoint with `_load_checkpoint_compat`.
3. Restores model weights with backward-compatible keys:
   - Generators: `G_AB` or `G_AB_state_dict`, `G_BA` or `G_BA_state_dict`
   - Discriminators: `D_A` / `D_A_state_dict`, `D_B` / `D_B_state_dict`
4. Restores optimizer states when compatible.
5. Restores scheduler/scaler/early-stopping states when present.
6. Sets `start_epoch = checkpoint["epoch"]`.
7. Guards that `start_epoch < num_epochs`.

### 8E Setup Phase

- Enables `cudnn.benchmark`, TF32 matmul and cuDNN.
- Builds loaders via `getDataLoader(...)`.
- Builds models via `getGeneratorsV2(...)`, `getDiscriminatorsV2(...)`.
- Instantiates `UVCGANLoss(...)` with config-provided lambdas.
- Configures AMP `GradScaler` (`enabled` only for CUDA + `tcfg.use_amp`).
- Builds `MetricsCalculator` and `EarlyStopping`.
- Configures optimizers:
  - `optimizer_G`: AdamW over both generators
  - `optimizer_D_A`, `optimizer_D_B`: Adam
- Configures three `LambdaLR` schedulers.
- Creates output dirs, `SummaryWriter`, and history CSV path.
- Initializes last-known-good snapshots for G, D, optimizer, scaler.

### 8F Per-epoch Loop

For each epoch:

1. `train()` mode for all 4 models.
2. Initializes scalar accumulators and per-batch `epoch_step`.
3. Iterates over `train_loader` batches.
4. Runs D stage then G stage.
5. Applies accumulation and rollback safety.
6. Stores per-batch losses in `history[epoch][batch]`.

### 8G Per-batch Update Order

#### Step 0: Input validation

- Loads `real_A`, `real_B` from batch dict.
- Skips batch if any non-finite input values.
- Tracks near-uniform patch warning (`std < 1e-4`).

#### Step 1: Discriminator updates

- Freezes generators, unfreezes discriminators.
- Generates detached fakes:
  - `fake_B_d = G_AB(real_A)`
  - `fake_A_d = G_BA(real_B)`
- Computes losses:
  - `loss_D_A = discriminator_loss(D_A, real_A, fake_A_d, fake_A_buffer)`
  - `loss_D_B = discriminator_loss(D_B, real_B, fake_B_d, fake_B_buffer)`
- Non-finite loss/params triggers D rollback from snapshots.
- Otherwise backward + grad clip + optimizer step for D_A and D_B.

#### Step 2: Generator update

- Unfreezes generators, freezes discriminators.
- Zeros `optimizer_G` at accumulation-window start.
- Under autocast:
  - `loss_G, fake_A, fake_B = generator_loss(...)`
  - scales by accumulation factor: `loss_G_scaled = loss_G / accumulate`
- Non-finite `loss_G` or non-finite fake outputs triggers G rollback and AMP scale backoff.
- At accumulation boundary:
  - unscale
  - clip
  - `optimizer_G.step`
  - `scaler.update`
  - snapshot refresh if finite

#### Step 3: Per-batch history entry

```python
{
  "Batch": i,
  "Loss_G": float,
  "Loss_D_A": float,
  "Loss_D_B": float,
}
```

### 8H Epoch-end Logging and Scheduler

- Computes epoch averages:
  - `avg_loss_G`, `avg_loss_D_A`, `avg_loss_D_B`, `avg_grad_norm_G`
- Logs TensorBoard loss + grad norm scalars.
- Steps all LR schedulers.
- Logs current LR scalars.
- Every 5 epochs:
  - append in-memory history to CSV
  - clear in-memory history chunk
- Saves periodic checkpoint every `save_checkpoint_every` epochs.

### 8I Validation and Early Stopping

- After `validation_warmup_epochs`, runs `run_validation(...)` every epoch and writes image grids.
- Every `early_stopping_interval` epochs after warmup:
  - runs `calculate_metrics(...)`
  - computes `avg_ssim = 0.5 * (ssim_A + ssim_B)`
  - calls `early_stopping(ssim=avg_ssim, losses={G,D_A,D_B})`
- Logs early-stopping counters to TensorBoard.
- Breaks training loop when `should_stop` is `True`.

### 8J Finalization

1. Runs final `calculate_metrics(...)`.
2. Runs `run_testing(...)` into `test_images/`.
3. Saves final checkpoint `final_checkpoint_epoch_{stopped_epoch}.pth`.
4. Flushes remaining history to CSV.
5. Reloads full CSV to return complete run history.
6. Closes TensorBoard writer.

---

## 9. Tensor Shape Reference

| Tensor | Shape | Notes |
|---|---|---|
| `real_A`, `real_B` | `(N,3,H,W)` | Input domains A/B |
| `fake_A`, `fake_B` | `(N,3,H,W)` | Generator outputs |
| `D_A` output | list of maps | Multi-scale logits per scale |
| `D_B` output | list of maps | Multi-scale logits per scale |
| Typical v2 for 256x256 | `(N,1,30,30)`, `(N,1,14,14)`, `(N,1,6,6)` | 3 scales |
| `loss_G`, `loss_D_A`, `loss_D_B` | scalar | Per-batch scalar losses |

---

## 10. Loss and Stability Notes

- Objective family is LSGAN, so discriminator losses are non-negative.
- Discriminator loss path disables autocast inside loss for safer GP numerics.
- Replay buffers reduce discriminator overfitting to latest fake distribution.
- Rollback mechanism handles numerical instability by restoring last finite state.
- AMP overflow symptoms are monitored (`GradScaler` scale warnings).

---

## 11. Optimizer and LR Schedule Reference

| Optimizer | Params | Type | LR | Betas | Weight Decay |
|---|---|---|---|---|---|
| `optimizer_G` | `G_AB + G_BA` | AdamW | `tcfg.lr` | `(beta1,beta2)` | `0.01` |
| `optimizer_D_A` | `D_A` | Adam | `tcfg.lr` | `(beta1,beta2)` | — |
| `optimizer_D_B` | `D_B` | Adam | `tcfg.lr` | `(beta1,beta2)` | — |

LR schedule uses `_make_lr_lambda(warmup, decay_start, total)` for all 3 optimizers.

---

## 12. Checkpoint Payload Reference

Periodic and final checkpoints include:

| Key | Description |
|---|---|
| `epoch` | Epoch number |
| `config` | Full v2 config object |
| `G_AB`, `G_BA` | Generator weights |
| `D_A`, `D_B` | Discriminator weights |
| `optimizer_G`, `optimizer_D_A`, `optimizer_D_B` | Optimizer states |
| `lr_scheduler_G_state_dict`, `lr_scheduler_D_A_state_dict`, `lr_scheduler_D_B_state_dict` | Scheduler states |
| `scaler_state_dict` | AMP scaler state (or `None` on CPU/no AMP) |
| `early_stopping_state` | EarlyStopping state |

Resume also accepts backward-compatible naming variants where present.

---

## 13. TensorBoard Scalar Reference

| Scalar | Frequency |
|---|---|
| `Epoch` | Every epoch |
| `Loss/Generator` | Every epoch |
| `Loss/Discriminator_A` | Every epoch |
| `Loss/Discriminator_B` | Every epoch |
| `Diagnostics/GradNorm_G` | Every epoch |
| `LR/Generator` | Every epoch |
| `LR/Discriminator_A` | Every epoch |
| `LR/Discriminator_B` | Every epoch |
| `EarlyStopping/ssim` | Early-stopping evaluation epochs |
| `EarlyStopping/counter` | Early-stopping evaluation epochs |
| `EarlyStopping/divergence_counter` | Early-stopping evaluation epochs |
| `Testing Started` | Finalization |
| `Training Completed` | Finalization |

---

## 14. Artifact Layout

```text
model_dir/
  tensorboard_logs/
  validation_images/
    epoch_*/
  test_images/
  training_history.csv
  checkpoint_epoch_*.pth
  final_checkpoint_epoch_*.pth
```
