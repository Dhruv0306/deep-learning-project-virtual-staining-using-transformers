# model_v4/training_loop.py — Detailed Reference

Source: `../../model_v4/training_loop.py`  
Model: CUT-style GAN + Transformer + PatchNCE (v4)  
Primary entry point: `train_v4(...)`

---

## Table of Contents

1. [Module-level Data Flow](#1-module-level-data-flow)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Helper Function: `_load_checkpoint_compat`](#3-helper-function-_load_checkpoint_compat)
4. [Helper Function: `_build_v4_model_config_from_checkpoint`](#4-helper-function-_build_v4_model_config_from_checkpoint)
5. [Helper Function: `_load_state_dict_with_compat`](#5-helper-function-_load_state_dict_with_compat)
6. [Helper Function: `_set_requires_grad`](#6-helper-function-_set_requires_grad)
7. [Helper Function: `_global_grad_norm`](#7-helper-function-_global_grad_norm)
8. [Helper Function: `_make_lr_lambda`](#8-helper-function-_make_lr_lambda)
9. [Helper Loss Functions: `_lsgan_gen_loss`, `_lsgan_disc_loss`](#9-helper-loss-functions-_lsgan_gen_loss-_lsgan_disc_loss)
10. [Function: `_run_validation_v4`](#10-function-_run_validation_v4)
    - [10A Signature](#10a-signature)
    - [10B Parameters](#10b-parameters)
    - [10C Validation Flow](#10c-validation-flow)
    - [10D Return Value](#10d-return-value)
11. [Function: `train_v4`](#11-function-train_v4)
    - [11A Signature and Return](#11a-signature-and-return)
    - [11B Parameters](#11b-parameters)
    - [11C Setup and Configuration](#11c-setup-and-configuration)
    - [11D Resume Phase](#11d-resume-phase)
    - [11E Per-batch Update Order](#11e-per-batch-update-order)
    - [11F Epoch-end Logging and Scheduler](#11f-epoch-end-logging-and-scheduler)
    - [11G Validation and Early Stopping](#11g-validation-and-early-stopping)
    - [11H Finalization](#11h-finalization)
12. [Tensor Shape Reference](#12-tensor-shape-reference)
13. [Loss Composition Reference](#13-loss-composition-reference)
14. [Optimizer and LR Schedule Reference](#14-optimizer-and-lr-schedule-reference)
15. [Checkpoint Payload Reference](#15-checkpoint-payload-reference)
16. [TensorBoard Scalar Reference](#16-tensorboard-scalar-reference)
17. [Artifact Layout](#17-artifact-layout)

---

## 1. Module-level Data Flow

```text
getDataLoader(...) -> train_loader, test_loader
getGeneratorV4(...) -> G_AB, G_BA
getDiscriminatorV4(...) -> D_A, D_B
PatchSampler + PatchNCELoss
(optional) EMA copies: ema_G_AB, ema_G_BA

---- per batch ---------------------------------------------------------------
real_A, real_B: (N,3,H,W)

1) Discriminator stage
   fake_B = G_AB(real_A) [no_grad]
   fake_A = G_BA(real_B) [no_grad]
   loss_D_A = LSGAN(D_A(real_A), D_A(fake_A_detached))
   loss_D_B = LSGAN(D_B(real_B), D_B(fake_B_detached))
   backward + clip + step D_A/D_B

2) Generator stage
   fake_B, feats_real_A = G_AB(real_A, return_features=True)
   fake_A, feats_real_B = G_BA(real_B, return_features=True)

   loss_G_gan = LSGAN_gen(D_B(fake_B)) + LSGAN_gen(D_A(fake_A))
   loss_nce   = PatchNCE(real-vs-fake sampled patches, both directions)
   loss_id    = L1(G_AB(real_B), real_B) + L1(G_BA(real_A), real_A)

   loss_G = lambda_gan * loss_G_gan
          + lambda_nce * loss_nce
          + lambda_identity * loss_id

   backward with accumulation
   clip + optimizer_G.step at accumulation boundary
   EMA update after successful G step

---- epoch end ---------------------------------------------------------------
aggregate and log losses/LR
periodic checkpoint
validation via _run_validation_v4 (EMA models optional)
early stopping on mean SSIM

---- finalization ------------------------------------------------------------
save final checkpoint
test export via _run_validation_v4(is_test=True)
return history, G_AB, G_BA, D_A, D_B
```

---

## 2. Imports and Dependencies

| Symbol | Source | Purpose |
|---|---|---|
| `V4Config` | `config` | Full v4 runtime/training/model config |
| `getDataLoader` | `shared.data_loader` | Unpaired loader |
| `getGeneratorV4` | `model_v4.generator` | v4 generator constructor |
| `getDiscriminatorV4` | `model_v4.discriminator` | v4 discriminator constructor |
| `PatchSampler` | `model_v4.patch_sampler` | Shared-index patch extraction |
| `PatchNCELoss` | `model_v4.nce_loss` | Patch contrastive objective |
| `ReplayBuffer` | `shared.replay_buffer` | Optional fake replay for D stability |
| `MetricsCalculator` | `shared.metrics` | SSIM/PSNR/FID |
| `save_images_with_title` | `shared.validation` | Grid image writer |
| `EarlyStopping` | `shared.EarlyStopping` | SSIM/divergence stopping policy |

---

## 3. Helper Function: `_load_checkpoint_compat`

### Signature

```python
_load_checkpoint_compat(checkpoint_path: str, map_location) -> dict
```

### Purpose

Checkpoint loader resilient to PyTorch version-specific serialization defaults.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | `.pth` path |
| `map_location` | `str | torch.device` | Tensor remapping target |

### Sample I/O

- Input: `checkpoint_path=".../checkpoint_epoch_80.pth"`, `map_location="cpu"`
- Output (example keys): `epoch`, `config`, `G_AB_state_dict`, `optimizer_G_state_dict`, `early_stopping_state`

---

## 4. Helper Function: `_build_v4_model_config_from_checkpoint`

### Signature

```python
_build_v4_model_config_from_checkpoint(checkpoint: dict) -> V4ModelConfig | None
```

### Purpose

Reconstructs `V4ModelConfig` from checkpoint `config` field.
Accepts either dataclass instance or plain dict.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint` | `dict` | Loaded checkpoint payload |

### Return

- `V4ModelConfig` when reconstruction succeeds
- `None` when field missing or incompatible

### Sample I/O

- Input: checkpoint containing `config` as dict of model fields
- Output: reconstructed `V4ModelConfig(...)` used to rebuild architecture on resume

---

## 5. Helper Function: `_load_state_dict_with_compat`

### Signature

```python
_load_state_dict_with_compat(module: nn.Module, state_dict: dict, tag: str) -> None
```

### Purpose

Strict state-dict load first; if failed, loads only shape-compatible keys and logs warning.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `module` | `nn.Module` | Model receiving loaded weights |
| `state_dict` | `dict` | Serialized parameter tensors |
| `tag` | `str` | Label used in warning logs |

### Sample I/O

- Input: `module=G_AB`, older checkpoint state dict with partial key overlap
- Output: compatible tensors loaded; warning emitted for strict-load mismatch

---

## 6. Helper Function: `_set_requires_grad`

### Signature

```python
_set_requires_grad(module: nn.Module, flag: bool) -> None
```

### Purpose

Toggles trainability for all module parameters.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `module` | `nn.Module` | Target module |
| `flag` | `bool` | `True` to enable grads, `False` to freeze |

### Sample I/O

- Input: `_set_requires_grad(D_A, False)`
- Output: no return; all params in `D_A` frozen for current stage

---

## 7. Helper Function: `_global_grad_norm`

### Signature

```python
_global_grad_norm(parameters) -> float
```

### Purpose

Computes global L2 norm of available gradients.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `parameters` | `Iterable[nn.Parameter]` | Parameter list/iterator with potential gradients |

### Sample I/O

- Input: `list(G_AB.parameters()) + list(G_BA.parameters())`
- Output: scalar grad norm (example `4.17`)

---

## 8. Helper Function: `_make_lr_lambda`

### Signature

```python
_make_lr_lambda(warmup: int, decay_start: int, total: int) -> Callable[[int], float]
```

### Purpose

Builds linear warmup + plateau + linear decay LR multiplier for `LambdaLR`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `warmup` | `int` | Number of warmup epochs |
| `decay_start` | `int` | Epoch where decay starts |
| `total` | `int` | Total epochs |

### Sample I/O

- Example: `warmup=5, decay_start=100, total=200`
- `epoch 0 -> ~0`, `epoch 5 -> 1`, `epoch 150 -> 0.5`, `epoch 200 -> 0`

---

## 9. Helper Loss Functions: `_lsgan_gen_loss`, `_lsgan_disc_loss`

### `_lsgan_gen_loss(pred_fake)`

- Formula: `E[(D(fake) - 1)^2]`
- Parameter: `pred_fake` is discriminator output map `(N,1,h,w)`
- Input shape: discriminator map `(N,1,h,w)`
- Output: scalar tensor

Sample I/O:
- Input: `pred_fake` shape `(8,1,30,30)`
- Output: scalar tensor such as `0.83`

### `_lsgan_disc_loss(pred_real, pred_fake)`

- Formula: `E[(D(real) - 1)^2] + E[D(fake)^2]`
- Inputs: two discriminator maps `(N,1,h,w)`
- Output: scalar tensor

Parameters:
- `pred_real`: discriminator map on real images `(N,1,h,w)`
- `pred_fake`: discriminator map on fake images `(N,1,h,w)`

Sample I/O:
- Input: `pred_real` `(8,1,30,30)`, `pred_fake` `(8,1,30,30)`
- Output: scalar tensor such as `1.42`

---

## 10. Function: `_run_validation_v4`

### 10A Signature

```python
_run_validation_v4(
    epoch,
    G_AB,
    G_BA,
    test_loader,
    device,
    save_dir,
    num_samples,
    calculator,
    max_batches=50,
    fid_max_samples=200,
    fid_min_samples=50,
    writer=None,
    is_test=False,
) -> dict[str, float]
```

### 10B Parameters

| Parameter | Type | Description |
|---|---|---|
| `epoch` | `int` | Step index for logging |
| `G_AB`, `G_BA` | `nn.Module` | Evaluation generators |
| `test_loader` | `DataLoader` | Batches with `A`, `B` |
| `device` | `torch.device` | Inference device |
| `save_dir` | `str` | Output image-grid directory |
| `num_samples` | `int` | Number of saved qualitative samples |
| `calculator` | `MetricsCalculator` | Metrics backend |
| `max_batches` | `int` | Maximum evaluated batches |
| `fid_max_samples` | `int` | Cap for FID sample count |
| `fid_min_samples` | `int` | Minimum count before FID |
| `writer` | `SummaryWriter | None` | TensorBoard writer |
| `is_test` | `bool` | Use `Testing/` prefix instead of `Validation/` |

### 10C Validation Flow

1. `eval()` mode for both generators.
2. For each batch up to `max_batches`:
   - generate `fake_B`, `fake_A`, `rec_A`, `rec_B`
   - compute batch metrics with `calculator.evaluate_batch(...)`
   - cache CPU tensors for optional FID
   - save first `num_samples` as two 4-panel rows
3. Compute epoch averages for SSIM/PSNR.
4. Optionally compute FID for domains A and B.
5. Log scalars to TensorBoard.
6. Restore `train()` mode and return metrics.

### 10D Return Value

Dictionary containing:

- always: `ssim_A`, `psnr_A`, `ssim_B`, `psnr_B`
- optional: `fid_A`, `fid_B`

---

## 11. Function: `train_v4`

### 11A Signature and Return

```python
train_v4(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    resume_checkpoint=None,
    cfg=None,
) -> tuple[dict, nn.Module, nn.Module, nn.Module, nn.Module]
```

Returns `(history, G_AB, G_BA, D_A, D_B)`.

### 11B Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | `int | None` | from `cfg.training` | Samples per epoch |
| `num_epochs` | `int | None` | from `cfg.training` | Total epochs |
| `model_dir` | `str | None` | from `cfg` | Root output directory |
| `val_dir` | `str | None` | from `cfg` | Validation image root |
| `resume_checkpoint` | `str | None` | `None` | Optional checkpoint path |
| `cfg` | `V4Config | None` | `V4Config()` | Full v4 config |

### 11C Setup and Configuration

- Creates/overrides runtime config.
- Builds dataloaders.
- Builds generators and discriminators from `cfg.model`.
- Prints model parameter counts.
- Configures optimizers:
  - generator: AdamW if transformer encoder enabled; else Adam
  - discriminators: Adam
- Optional LR schedulers if `use_lr_schedule=True`.
- AMP scaler setup.
- Optional EMA generator copies.
- EarlyStopping setup using interval-aware patience conversion.
- Initializes PatchNCE stack (`PatchSampler`, `PatchNCELoss`) and NCE layer filtering.
- Optional replay buffers.
- Creates output dirs + TensorBoard writer.

### 11D Resume Phase

On resume:

1. Loads checkpoint with compatibility helper.
2. Attempts to reconstruct model config from checkpoint and replace runtime model config.
3. Restores generator/discriminator weights with compatibility loader.
4. Restores EMA weights when enabled and available.
5. Restores optimizer/scheduler/scaler/early-stopping states if compatible.
6. Sets `start_epoch` from checkpoint `epoch`.

### 11E Per-batch Update Order

#### Step 1: Input guard

- Loads `real_A`, `real_B`.
- Skips non-finite input batches.

#### Step 2: Discriminator updates

- Freeze G; unfreeze D.
- Generate detached fakes under `no_grad`.
- Optional replay-buffer substitution.
- Compute LSGAN D losses in float32 (autocast disabled).
- Backprop, clip, step each discriminator.

#### Step 3: Generator update

- Freeze D; unfreeze G.
- At accumulation-window start, zero G grads.
- Forward with feature return:
  - `fake_B, feats_real_A = G_AB(real_A, return_features=True, nce_layers=...)`
  - `fake_A, feats_real_B = G_BA(real_B, return_features=True, nce_layers=...)`
- GAN losses from discriminator predictions on fakes.
- NCE losses:
  - encode fake features
  - sample real/fake patches at shared indices
  - compute PatchNCELoss in both directions and average
- Identity loss if enabled.
- Combine weighted total generator loss.
- Backprop scaled by accumulation factor.
- At accumulation boundary:
  - unscale (if AMP)
  - clip
  - optimizer step
  - scaler update (AMP)
  - EMA update

#### Step 4: Per-batch history entry

```python
{
  "Batch": i,
  "Loss_G": ...,
  "Loss_G_GAN": ...,
  "Loss_NCE": ...,
  "Loss_NCE_AB": ...,
  "Loss_NCE_BA": ...,
  "Loss_Id": ...,
  "Loss_G_AB": ...,
  "Loss_G_BA": ...,
  "Loss_D_A": ...,
  "Loss_D_B": ...,
  "GradNorm_G": ...,
}
```

### 11F Epoch-end Logging and Scheduler

- Averages and logs all major losses + grad norm.
- Logs LR scalars for each optimizer when schedulers are enabled.
- Saves periodic checkpoint every `save_checkpoint_every` epochs.

### 11G Validation and Early Stopping

- Runs `_run_validation_v4` after `validation_every`.
- Uses EMA generators for validation/test if `use_ema=True`.
- At `early_stopping_interval` epochs after warmup:
  - computes score `avg_ssim = 0.5 * (ssim_A + ssim_B)`
  - calls `EarlyStopping(ssim=avg_ssim, losses={G,D_A,D_B})`
- Logs early-stopping counters.
- Stops when criterion is met.

### 11H Finalization

1. Saves `final_checkpoint_{stopped_epoch}.pth`.
2. Runs `_run_validation_v4(..., is_test=True)` into `test_images/`.
3. Logs completion scalars and closes writer.
4. Returns trained modules and in-memory history.

---

## 12. Tensor Shape Reference

| Tensor | Shape | Notes |
|---|---|---|
| `real_A`, `real_B` | `(N,3,H,W)` | Inputs |
| `fake_A`, `fake_B` | `(N,3,H,W)` | Generator outputs |
| `rec_A`, `rec_B` | `(N,3,H,W)` | Cycle reconstructions in validation |
| `D_A`, `D_B` outputs | `(N,1,h,w)` | Patch logits |
| Encoder features list | list of `(N,C_l,H_l,W_l)` | Used by PatchNCE |
| Sampled patch tensors | `(N, num_patches, proj_dim)` (conceptual) | Internal PatchNCE representation |
| Loss scalars | scalar | Per batch/epoch |

---

## 13. Loss Composition Reference

Generator total:

```text
loss_G = lambda_gan      * loss_G_gan
       + lambda_nce      * loss_nce
       + lambda_identity * loss_id
```

Where:

- `loss_G_gan = _lsgan_gen_loss(D_B(fake_B)) + _lsgan_gen_loss(D_A(fake_A))`
- `loss_nce = 0.5 * (loss_nce_AB + loss_nce_BA)`
- `loss_id = L1(G_AB(real_B), real_B) + L1(G_BA(real_A), real_A)`

Discriminator per domain:

```text
loss_D = E[(D(real)-1)^2] + E[D(fake)^2]
```

---

## 14. Optimizer and LR Schedule Reference

| Optimizer | Params | Type | LR |
|---|---|---|---|
| `optimizer_G` | `G_AB + G_BA` | AdamW or Adam | `tcfg.lr` |
| `optimizer_D_A` | `D_A` | Adam | `tcfg.lr` |
| `optimizer_D_B` | `D_B` | Adam | `tcfg.lr` |

Scheduler:

- Optional `LambdaLR` with warmup -> plateau -> decay.

---

## 15. Checkpoint Payload Reference

### Periodic (`checkpoint_epoch_{N}.pth`)

| Key | Description |
|---|---|
| `epoch` | Completed epoch |
| `config` | `V4ModelConfig` snapshot |
| `G_AB_state_dict`, `G_BA_state_dict` | Generator weights |
| `D_A_state_dict`, `D_B_state_dict` | Discriminator weights |
| `ema_G_AB_state_dict`, `ema_G_BA_state_dict` | EMA weights (or `None`) |
| `optimizer_G_state_dict`, `optimizer_D_A_state_dict`, `optimizer_D_B_state_dict` | Optimizer states |
| `lr_scheduler_G_state_dict`, `lr_scheduler_D_A_state_dict`, `lr_scheduler_D_B_state_dict` | Scheduler states (or `None`) |
| `scaler_state_dict` | AMP scaler state |
| `early_stopping_state` | EarlyStopping state |

### Final (`final_checkpoint_{stopped_epoch}.pth`)

Contains same training states except no `config` key in final payload.

---

## 16. TensorBoard Scalar Reference

| Scalar | Frequency |
|---|---|
| `Epoch` | Every epoch |
| `Loss/Generator` | Every epoch |
| `Loss/GAN` | Every epoch |
| `Loss/NCE` | Every epoch |
| `Loss/Identity` | Every epoch |
| `Loss/Generator_AB` | Every epoch |
| `Loss/Generator_BA` | Every epoch |
| `Loss/Discriminator_A` | Every epoch |
| `Loss/Discriminator_B` | Every epoch |
| `Diagnostics/GradNorm_G` | Every epoch |
| `LR/Generator` | Every epoch when scheduler enabled |
| `LR/Discriminator_A` | Every epoch when scheduler enabled |
| `LR/Discriminator_B` | Every epoch when scheduler enabled |
| `Validation/*` | Validation runs |
| `Testing/*` | Final test run |
| `EarlyStopping/ssim`, `EarlyStopping/counter`, `EarlyStopping/divergence_counter` | Early-stopping intervals |
| `Testing Started`, `Training Completed` | Finalization |

---

## 17. Artifact Layout

```text
model_dir/
  tensorboard_logs/
  validation_images/
    epoch_*/
  test_images/
  checkpoint_epoch_*.pth
  final_checkpoint_*.pth
```
