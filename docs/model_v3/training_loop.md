# model_v3/training_loop.py — Detailed Reference

Source: `../../model_v3/training_loop.py`  
Model: CycleDiT latent diffusion (v3)  
Primary entry point: `train_v3(...)`

---

## Table of Contents

1. [Module-level Data Flow](#1-module-level-data-flow)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Helper Function: `_load_checkpoint_compat`](#3-helper-function-_load_checkpoint_compat)
4. [Helper Function: `_make_cosine_warmup_lambda`](#4-helper-function-_make_cosine_warmup_lambda)
5. [Helper Function: `_global_grad_norm`](#5-helper-function-_global_grad_norm)
6. [Helper Function: `_set_requires_grad`](#6-helper-function-_set_requires_grad)
7. [Function: `_run_validation_v3`](#7-function-_run_validation_v3)
   - [7A Signature](#7a-signature)
   - [7B Parameters](#7b-parameters)
   - [7C Validation Flow](#7c-validation-flow)
   - [7D Return Value](#7d-return-value)
8. [Function: `train_v3`](#8-function-train_v3)
   - [8A Signature and Return](#8a-signature-and-return)
   - [8B Parameters](#8b-parameters)
   - [8C Setup Phase](#8c-setup-phase)
   - [8D Resume Phase](#8d-resume-phase)
   - [8E Per-epoch Loop](#8e-per-epoch-loop)
   - [8F Two-stage Generator Update](#8f-two-stage-generator-update)
   - [8G Discriminator Update](#8g-discriminator-update)
   - [8H Epoch-end Logging and Checkpointing](#8h-epoch-end-logging-and-checkpointing)
   - [8I Validation and Early Stopping](#8i-validation-and-early-stopping)
   - [8J Finalization](#8j-finalization)
9. [Tensor Shape Reference](#9-tensor-shape-reference)
10. [Loss Components and Schedules](#10-loss-components-and-schedules)
11. [Optimizer and LR Schedule Reference](#11-optimizer-and-lr-schedule-reference)
12. [Checkpoint Payload Reference](#12-checkpoint-payload-reference)
13. [TensorBoard Scalar Reference](#13-tensorboard-scalar-reference)
14. [Artifact Layout](#14-artifact-layout)

---

## 1. Module-level Data Flow

```text
getDataLoader(...) -> train_loader, test_loader
VAEWrapper(...) -> frozen VAE
getGeneratorV3(...) -> dit_model
copy.deepcopy(dit_model) -> ema_model
getDiscriminatorsV3(...) -> D_A, D_B
DDPMScheduler + DDIMSampler

---- per batch ---------------------------------------------------------------
real_A, real_B: (N,3,256,256)
  -> vae.encode -> z0_A, z0_B: (N,4,32,32)
  -> scheduler.add_noise -> z_t_A, z_t_B

Stage 1 (diffusion objective):
  dit_model(z_t_A, t_A, cond=real_A, target_domain=1) -> out_A2B
  dit_model(z_t_B, t_B, cond=real_B, target_domain=0) -> out_B2A
  compute_diffusion_loss(...) for both directions

Stage 2 (auxiliary objectives, fresh forward):
  recompute out_A2B/out_B2A
  decode x0_pred via VAE -> fake_B_img, fake_A_img
  adversarial + cycle + identity

Discriminator stage:
  replay-buffer fake_A/fake_B
  _lsgan_disc_loss(D_A(real_A), D_A(fake_A_buffer))
  _lsgan_disc_loss(D_B(real_B), D_B(fake_B_buffer))
  optional R1 penalty

---- epoch end ---------------------------------------------------------------
log losses/weights/LR/diagnostics
periodic checkpoint
validation pass with ema_model via DDIM
periodic early-stopping checks on mean SSIM

---- finalization ------------------------------------------------------------
save final checkpoint
run test-time validation/export using ema_model
flush + reload history CSV
return history, dit_model, ema_model, None
```

---

## 2. Imports and Dependencies

| Symbol | Source | Purpose |
|---|---|---|
| `get_dit_8gb_config` | `config` | Default v3 config |
| `getDataLoader` | `shared.data_loader` | Unpaired data pipeline |
| `VAEWrapper` | `model_v3.vae_wrapper` | Frozen latent encoder/decoder |
| `getGeneratorV3` | `model_v3.generator` | CycleDiT generator |
| `getDiscriminatorsV3` | `model_v3.discriminator` | Projection discriminators |
| `DDPMScheduler`, `DDIMSampler` | `model_v3.noise_scheduler` | Noise process + deterministic sampler |
| `compute_diffusion_loss`, `_compute_cycle_loss`, `_compute_identity_loss`, etc. | `model_v3.losses` | Core diffusion and GAN/cycle/id losses |
| `VGGPerceptualLossV2` | `model_v2.losses` | Optional perceptual loss |
| `ReplayBuffer` | `shared.replay_buffer` | Discriminator stabilization |
| `MetricsCalculator` | `shared.metrics` | SSIM/PSNR/FID |
| `save_images_with_title` | `shared.validation` | Validation grid export |
| `append_history_to_csv_v3`, `load_history_from_csv_v3` | `model_v3.history_utils` | v3 history persistence |
| `EarlyStopping` | `shared.EarlyStopping` | Stopping on SSIM + divergence |

---

## 3. Helper Function: `_load_checkpoint_compat`

### Signature

```python
_load_checkpoint_compat(checkpoint_path: str, map_location) -> dict
```

### Purpose

Cross-version-safe checkpoint loading with fallback for PyTorch 2.6+ `weights_only` behavior.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | Path to `.pth` file |
| `map_location` | `str | torch.device` | Device remapping |

### Return

Checkpoint dict.

### Sample I/O

- Input: `checkpoint_path=".../checkpoint_epoch_120.pth"`, `map_location="cpu"`
- Output (example keys): `epoch`, `dit_state_dict`, `ema_state_dict`, `optimizer_G_state_dict`, `scaler_state_dict`

---

## 4. Helper Function: `_make_cosine_warmup_lambda`

### Signature

```python
_make_cosine_warmup_lambda(warmup: int, total: int, lr_min_ratio: float) -> Callable[[int], float]
```

### Purpose

Builds LR multiplier with:

1. Linear warmup in `[0, warmup)`
2. Cosine decay in `[warmup, total)` down to `lr_min_ratio`

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `warmup` | `int` | Warmup epochs |
| `total` | `int` | Total epochs |
| `lr_min_ratio` | `float` | Minimum LR as ratio of base LR |

### Sample I/O

- Input: `(warmup=10, total=200, lr_min_ratio=1e-2)`
- Output examples: `epoch=0 -> ~0.1`, `epoch=10 -> 1.0`, late epochs approach `0.01`

---

## 5. Helper Function: `_global_grad_norm`

### Signature

```python
_global_grad_norm(parameters) -> float
```

### Purpose

Computes global L2 norm of existing gradients for monitoring.

### Return

`float`, returns `0.0` when no gradients exist.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `parameters` | `Iterable[nn.Parameter]` | Parameter iterator whose `.grad` fields are inspected |

### Sample I/O

- Input: parameters from `dit_model`
- Output: scalar grad norm (example `2.84`)

---

## 6. Helper Function: `_set_requires_grad`

### Signature

```python
_set_requires_grad(module: nn.Module, flag: bool) -> None
```

### Purpose

Bulk enables/disables gradients to control which networks are trainable during each stage.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `module` | `nn.Module` | Target module |
| `flag` | `bool` | `True` enable gradients, `False` freeze |

### Sample I/O

- Input: `module=D_A`, `flag=False`
- Effect: all discriminator parameters in `D_A` are frozen for generator stage

---

## 7. Function: `_run_validation_v3`

### 7A Signature

```python
_run_validation_v3(
    epoch,
    ema_model,
    vae,
    sampler,
    test_loader,
    device,
    save_dir,
    calculator,
    num_steps,
    writer,
    max_batches=50,
    num_samples=6,
    fid_max_samples=200,
    fid_min_samples=50,
    prediction_type="v",
    cfg_scale=1.0,
    is_test=False,
) -> dict
```

### 7B Parameters

| Parameter | Type | Description |
|---|---|---|
| `epoch` | `int` | Logging and file naming step |
| `ema_model` | `nn.Module` | EMA CycleDiT model used for inference |
| `vae` | `VAEWrapper` | Frozen latent decoder |
| `sampler` | `DDIMSampler` | DDIM generation sampler |
| `test_loader` | `DataLoader` | Batch dicts with keys `A`, `B` |
| `device` | `torch.device` | Inference device |
| `save_dir` | `str` | Output dir for image grids |
| `calculator` | `MetricsCalculator` | SSIM/PSNR/FID helper |
| `num_steps` | `int` | DDIM denoising steps |
| `writer` | `SummaryWriter` | TensorBoard writer |
| `max_batches` | `int` | Maximum evaluated batches |
| `num_samples` | `int` | Number of saved qualitative grids |
| `fid_max_samples` | `int` | FID upper sample cap |
| `fid_min_samples` | `int` | FID minimum sample threshold |
| `prediction_type` | `str` | Diffusion parameterization (`v` or `eps`) |
| `cfg_scale` | `float` | CFG guidance scale |
| `is_test` | `bool` | Switches TensorBoard prefix to `Testing` |

### 7C Validation Flow

1. Runs A->B generation across up to `max_batches`.
2. Computes SSIM/PSNR for domain B every evaluated batch.
3. For first `num_samples` batches, additionally:
   - runs B->A generation
   - runs both cycle reconstructions
   - writes 4-panel image grids:
     - row A: `Real A | Fake B | Rec A | Real B`
     - row B: `Real B | Fake A | Rec B | Real A`
4. Optionally computes FID for both directions if sample thresholds allow.
5. Logs metrics into TensorBoard under `Validation/` or `Testing/`.

### 7D Return Value

Dictionary with averaged metrics:

- always: `ssim_A`, `psnr_A`, `ssim_B`, `psnr_B`
- optional: `fid_A`, `fid_B`

---

## 8. Function: `train_v3`

### 8A Signature and Return

```python
train_v3(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint=None,
    cfg=None,
) -> tuple[dict, nn.Module, nn.Module, None]
```

Returns `(history, dit_model, ema_model, None)`.

### 8B Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | `int | None` | from cfg | Samples per epoch |
| `num_epochs` | `int | None` | from cfg | Total epochs |
| `model_dir` | `str | None` | from cfg | Output root |
| `val_dir` | `str | None` | from cfg | Validation images root |
| `test_size` | `int | None` | from cfg | Number of test batches/images in final run |
| `resume_checkpoint` | `str | None` | `None` | Resume path |
| `cfg` | `UVCGANConfig | None` | `get_dit_8gb_config()` | v3 config object |

### 8C Setup Phase

- Applies argument overrides to config.
- Asserts required fields (`num_epochs`, `epoch_size`, `validation_size`, warmup consistency).
- Creates dataloaders (`train_loader`, `test_loader`).
- Instantiates:
  - frozen `VAEWrapper`
  - trainable `dit_model`
  - non-trainable EMA `ema_model`
  - discriminators `D_A`, `D_B`
  - replay buffers
  - `DDPMScheduler` + `DDIMSampler`
- Configures optimizers:
  - `AdamW` for DiT
  - `Adam` for both discriminators
- Configures cosine-warmup LR schedulers.
- Configures AMP scaler.
- Configures `MetricsCalculator`, `EarlyStopping`, optional `VGGPerceptualLossV2`.
- Creates output directories + TensorBoard writer + history CSV path.

### 8D Resume Phase

On resume:

1. Loads checkpoint and validates presence of `dit_state_dict`.
2. Restores `dit_model`, `ema_model`, `D_A`, `D_B`.
3. Restores optimizer/scheduler/scaler/early-stopping states.
4. Supports legacy `optimizer_state_dict` alias for generator optimizer.
5. Sets `start_epoch` from checkpoint `epoch`.

### 8E Per-epoch Loop

Each epoch:

1. Sets `dit_model.train()`.
2. Resets accumulators and `accum_count`.
3. Iterates batches; skips non-finite input batches.
4. Performs two-stage generator update then discriminator updates.
5. Stores per-batch metrics in `history`.

### 8F Two-stage Generator Update

#### Stage 1: Diffusion denoising

- Encodes images to latents (`z0_A`, `z0_B`).
- Samples timesteps `t_A`, `t_B` and noises.
- Generates noisy latents `z_t_A`, `z_t_B`.
- Runs DiT forward for both directions.
- Computes diffusion loss via `compute_diffusion_loss` for A->B and B->A.
- Backpropagates `loss_denoise / accumulate`.
- Frees stage-1 forward outputs.

#### Stage 2: Adversarial + cycle + identity

- Re-runs DiT forward to build fresh computation graph.
- Decodes predicted clean latents to pixel images.
- Computes:
  - adversarial generator loss (`_lsgan_gen_loss`)
  - cycle-consistency loss (`_compute_cycle_loss`)
  - identity loss (`_compute_identity_loss`) if enabled
- Combines with scheduled weights:
  - `lambda_adv_curr`
  - `lambda_cycle_v3`
  - `lambda_identity_curr`
- Backpropagates `aux_loss / accumulate`.
- At accumulation boundary:
  - gradient clipping
  - optimizer step
  - EMA update

### 8G Discriminator Update

After generator stage:

- Unfreezes discriminators.
- Uses replay-buffer detached fakes.
- Computes LSGAN discriminator losses for `D_A`, `D_B`.
- Optional R1 penalty every `r1_interval` steps.
- Adaptive update option skips D backward when loss is below threshold.
- Calls `scaler.step` for performed steps and `scaler.update` once if any optimizer stepped.

### 8H Epoch-end Logging and Checkpointing

- Aggregates epoch means (`Loss/DiT`, `Loss/Perceptual`, grad norms).
- Logs last-batch GAN/cycle/identity losses and lambda weights.
- Logs timestep distribution diagnostics (`TimestepMean`, `TimestepStd`).
- Steps all LR schedulers.
- Flushes history CSV every 5 epochs.
- Saves checkpoint every `save_checkpoint_every` epochs.

### 8I Validation and Early Stopping

- Runs `_run_validation_v3(...)` after `validation_warmup_epochs`.
- On every `early_stopping_interval` after warmup:
  - score = `0.5 * (ssim_A + ssim_B)`
  - `EarlyStopping(ssim=score, losses={"DiT": avg_loss})`
- Logs early stopping counters.
- Stops when criteria are met.

### 8J Finalization

1. Saves `final_checkpoint_epoch_{stopped_epoch}.pth`.
2. Runs test-time `_run_validation_v3(..., is_test=True)` into `test_images/`.
3. Logs final TensorBoard markers.
4. Flushes and reloads full CSV history.
5. Closes writer.

---

## 9. Tensor Shape Reference

| Tensor | Shape | Notes |
|---|---|---|
| `real_A`, `real_B` | `(N,3,256,256)` | Input images |
| `z0_A`, `z0_B` | `(N,4,32,32)` | VAE latent |
| `z_t_A`, `z_t_B` | `(N,4,32,32)` | Noisy latent |
| `out["v_pred"]` | `(N,4,32,32)` | DiT output for diffusion objective |
| `z0_fake_A`, `z0_fake_B` | `(N,4,32,32)` | Predicted clean latent |
| `fake_A_img`, `fake_B_img` | `(N,3,256,256)` | Decoded pixel-space fakes |
| `loss_*` | scalar | Loss terms used in optimization |

---

## 10. Loss Components and Schedules

| Component | Symbol | Notes |
|---|---|---|
| Diffusion denoising | `loss_denoise` | Weighted MSE + optional Min-SNR + optional perceptual |
| Adversarial | `loss_adv_G` | LSGAN generator loss |
| Cycle | `loss_cyc` | DDIM-shortcut-based cycle penalty |
| Identity | `loss_id` | Optional identity consistency |
| Total Stage 2 | `aux_loss` | `lambda_adv_curr * adv + lambda_cycle * cyc + lambda_id_curr * id` |

Schedules:

- `lambda_adv_curr`: linear warmup by global step
- `lambda_identity_curr`: epoch-based decay from start to end value
- LR: warmup + cosine decay via `_make_cosine_warmup_lambda`

---

## 11. Optimizer and LR Schedule Reference

| Optimizer | Params | Type | LR |
|---|---|---|---|
| `optimizer_G` | `dit_model` | AdamW | `1e-4` |
| `optimizer_D_A` | `D_A` | Adam | `tcfg.lr` |
| `optimizer_D_B` | `D_B` | Adam | `tcfg.lr` |

All 3 use LambdaLR cosine-warmup schedule.

---

## 12. Checkpoint Payload Reference

Periodic and final checkpoint keys:

| Key | Description |
|---|---|
| `checkpoint_format_version` | Format marker (`2`) |
| `epoch` | Completed epoch |
| `dit_state_dict` | DiT model weights |
| `ema_state_dict` | EMA model weights |
| `D_A_state_dict`, `D_B_state_dict` | Discriminator weights |
| `optimizer_state_dict` | Legacy alias of generator optimizer |
| `optimizer_G_state_dict` | Generator optimizer |
| `optimizer_D_A_state_dict`, `optimizer_D_B_state_dict` | Discriminator optimizers |
| `lr_scheduler_G_state_dict`, `lr_scheduler_D_A_state_dict`, `lr_scheduler_D_B_state_dict` | Scheduler states |
| `scaler_state_dict` | AMP scaler state |
| `early_stopping_state` | EarlyStopping state |
| `config` | Diffusion config snapshot (`dcfg`) |

---

## 13. TensorBoard Scalar Reference

| Scalar | Frequency |
|---|---|
| `Epoch` | Every epoch |
| `Loss/DiT` | Every epoch |
| `Loss/Perceptual` | Every epoch |
| `Loss/G_adv`, `Loss/D_A`, `Loss/D_B` | Every epoch (from last batch snapshot) |
| `Loss/Cycle`, `Loss/Identity` | Every epoch (from last batch snapshot) |
| `Weights/lambda_adv_current`, `Weights/lambda_adv_epoch_avg` | Every epoch |
| `Weights/lambda_identity_current`, `Weights/lambda_identity_epoch_avg` | Every epoch |
| `Diagnostics/GradNorm` | Every epoch |
| `Diagnostics/TimestepMean`, `Diagnostics/TimestepStd` | Every epoch |
| `LR/DiT`, `LR/D_A`, `LR/D_B` | Every epoch |
| `Validation/*` | Validation runs |
| `Testing/*` | Final test run |
| `EarlyStopping/ssim`, `EarlyStopping/counter`, `EarlyStopping/divergence_counter` | Early-stopping intervals |
| `Testing Started`, `Training Completed` | Finalization |

---

## 14. Artifact Layout

```text
model_dir/
  tensorboard_logs/
  training_history.csv
  checkpoint_epoch_*.pth
  final_checkpoint_epoch_*.pth
  validation_images/
    epoch_*/
  test_images/
```
