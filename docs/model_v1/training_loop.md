# model_v1/training_loop.py — Detailed Reference

Source: `../../model_v1/training_loop.py`

Model: Hybrid CycleGAN / UVCGAN v1  
Primary entry point: `train(...)`  
Compatibility alias: `train_v1(...)`

---

## Table of Contents

1. [Module-level Data Flow](#1-module-level-data-flow)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Function: `_load_checkpoint_compat`](#3-function-_load_checkpoint_compat)
4. [Function: `train`](#4-function-train)
   - [4A Parameters](#4a-parameters)
   - [4B Setup Phase](#4b-setup-phase)
   - [4C Resume Phase](#4c-resume-phase)
   - [4D Per-Epoch Loop](#4d-per-epoch-loop)
   - [4E Per-Batch Update Order](#4e-per-batch-update-order)
   - [4F Epoch-end Logging and Checkpointing](#4f-epoch-end-logging-and-checkpointing)
   - [4G Validation and Early Stopping](#4g-validation-and-early-stopping)
   - [4H Finalization](#4h-finalization)
5. [Function: `train_v1`](#5-function-train_v1)
6. [Tensor Shape Reference](#6-tensor-shape-reference)
7. [Loss Term Reference](#7-loss-term-reference)
8. [Optimizer and Scheduler Reference](#8-optimizer-and-scheduler-reference)
9. [Checkpoint Payload Reference](#9-checkpoint-payload-reference)
10. [TensorBoard Scalar Reference](#10-tensorboard-scalar-reference)
11. [Artifact Layout](#11-artifact-layout)

---

## 1. Module-level Data Flow

```
getDataLoader(epoch_size)
  → train_loader  (batches of {"A": (N,3,256,256), "B": (N,3,256,256)})
  → test_loader

getGenerators()   → G_AB, G_BA   (ViTUNetGenerator)
getDiscriminators() → D_A, D_B   (PatchDiscriminator)

CycleGANLoss(...)  → loss_fn

─── per epoch ───────────────────────────────────────────────────────────
  for batch in train_loader:
    real_A (N,3,256,256), real_B (N,3,256,256)
      │
      ├─ Generator step
      │    loss_fn.generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs)
      │      → loss_G (scalar), fake_A (N,3,256,256), fake_B (N,3,256,256)
      │    AMP backward → optimizer_G.step()
      │
      ├─ Discriminator A step
      │    loss_fn.discriminator_loss(D_A, real_A, fake_A, fake_A_buffer)
      │      → loss_D_A (scalar)
      │    AMP backward → optimizer_D_A.step()
      │
      └─ Discriminator B step
           loss_fn.discriminator_loss(D_B, real_B, fake_B, fake_B_buffer)
             → loss_D_B (scalar)
           AMP backward → optimizer_D_B.step()

  lr_scheduler_G/D_A/D_B .step()
  run_validation(...)   → val_dir/epoch_N/*.png
  every 10 epochs:
    calculate_metrics(...)  → {ssim_A, ssim_B, psnr_A, psnr_B}
    EarlyStopping(avg_ssim, {G, D_A, D_B})

─── finalization ────────────────────────────────────────────────────────
  calculate_metrics(...)
  run_testing(...)  → test_images/*.png
  torch.save(final_checkpoint)
  append_history_to_csv / load_history_from_csv
  writer.close()
  return history, G_AB, G_BA, D_A, D_B
```

---

## 2. Imports and Dependencies

| Symbol | Source | Purpose |
|---|---|---|
| `getDataLoader` | `shared.data_loader` | Unpaired dataset loader |
| `getDiscriminators` | `model_v1.discriminator` | PatchGAN discriminator factory |
| `EarlyStopping` | `shared.EarlyStopping` | SSIM-based early stopping |
| `getGenerators` | `model_v1.generator` | ViTUNetGenerator factory |
| `append_history_to_csv` | `shared.history_utils` | Incremental CSV flush |
| `load_history_from_csv` | `shared.history_utils` | Full history reload |
| `CycleGANLoss` | `model_v1.losses` | Composite GAN + cycle + perceptual loss |
| `MetricsCalculator` | `shared.metrics` | SSIM / PSNR / FID computation |
| `run_testing` | `shared.testing` | Final test-set image export |
| `calculate_metrics` | `shared.validation` | Metric computation over test loader |
| `run_validation` | `shared.validation` | Per-epoch comparison image export |
| `get_default_config` | `config` | Default UVCGANConfig for v1 |

---

## 3. Function: `_load_checkpoint_compat`

### Signature

```python
_load_checkpoint_compat(checkpoint_path: str, map_location) -> dict
```

### Purpose

Loads a `.pth` checkpoint file safely across PyTorch versions.  PyTorch 2.6+
changed the default of `torch.load` to `weights_only=True`, which raises when
the checkpoint contains pickled Python objects such as config dataclasses.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | Path to the `.pth` file |
| `map_location` | `str` or `torch.device` | Device to map tensors onto (e.g. `"cpu"`, `device`) |

### Logic

```
try:
    torch.load(..., weights_only=False)   ← PyTorch 2.6+ path
except TypeError:
    torch.load(...)                       ← PyTorch < 2.6 fallback
except pickle.UnpicklingError:
    torch.load(..., weights_only=False)   ← retry on custom serialization
```

### Return Value

`dict` — the full checkpoint dictionary as saved by `torch.save`.

### Example

```python
ckpt = _load_checkpoint_compat("checkpoint_epoch_40.pth", map_location="cpu")
# ckpt.keys() → {"epoch", "G_AB", "G_BA", "D_A", "D_B",
#                "optimizer_G", "optimizer_D_A", "optimizer_D_B",
#                "lr_scheduler_G_state_dict", ..., "early_stopping_state"}
```

---

## 4. Function: `train`

### Signature

```python
train(
    epoch_size=None,
    num_epochs=None,
    model_dir=None,
    val_dir=None,
    test_size=None,
    resume_checkpoint=None,
    cfg=None,
) -> tuple[dict, nn.Module, nn.Module, nn.Module, nn.Module]
```

---

### 4A. Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | `int \| None` | `3000` | Maximum number of samples drawn per epoch from the dataset. Controls `train_loader` length. |
| `num_epochs` | `int \| None` | `200` | Total number of training epochs. On resume, must exceed the epoch stored in the checkpoint. |
| `model_dir` | `str \| None` | `data/.../models` | Root output directory. Checkpoints, TensorBoard logs, and history CSV are written here. |
| `val_dir` | `str \| None` | `model_dir/validation_images` | Directory for per-epoch comparison image grids. Created automatically if absent. |
| `test_size` | `int \| None` | `200` | Number of test-set images exported during final inference. Cast to `int` before use. |
| `resume_checkpoint` | `str \| None` | `None` | Path to a v1 `.pth` checkpoint. When set, model/optimizer/scheduler/scaler/early-stopping states are restored and training continues from the saved epoch. |
| `cfg` | `UVCGANConfig \| None` | `get_default_config(1)` | Full configuration object. All v1 loss lambdas are hardcoded inside `train()` and do not read from `cfg`; only `cfg.training.save_checkpoint_every` is used. |

---

### 4B. Setup Phase

**Backend flags**

```python
torch.backends.cudnn.benchmark = True        # fastest conv algorithm selection
torch.backends.cuda.matmul.allow_tf32 = True # TF32 for matmul
torch.backends.cudnn.allow_tf32 = True       # TF32 for cuDNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # deterministic numerics
```

**Data loaders**

```python
train_loader, test_loader = getDataLoader(epoch_size=epoch_size or 3000)
# train_loader: batches of {"A": Tensor(N,3,256,256), "B": Tensor(N,3,256,256)}
# test_loader:  same structure, fixed split
```

**Models**

```python
G_AB, G_BA = getGenerators()
# G_AB: ViTUNetGenerator  (unstained → stained)
# G_BA: ViTUNetGenerator  (stained   → unstained)
# Both: input (N,3,256,256) → output (N,3,256,256)

D_A, D_B = getDiscriminators()
# D_A: PatchDiscriminator  (judges domain A patches)
# D_B: PatchDiscriminator  (judges domain B patches)
# Both: input (N,3,256,256) → output (N,1,30,30)
```

**Loss function**

```python
loss_fn = CycleGANLoss(
    lambda_cycle=10.0,           # cycle-consistency L1 weight
    lambda_identity=5.0,         # identity L1 weight (decays after 50% of training)
    lambda_cycle_perceptual=0.2, # VGG19 perceptual on cycle images
    lambda_identity_perceptual=0.1, # VGG19 perceptual on identity images
    lambda_gp=10.0,              # two-sided gradient penalty weight
    perceptual_resize=160,       # resize images to 160×160 before VGG19
    device=device,
)
```

**AMP**

```python
use_amp = device.type == "cuda"
scaler = GradScaler("cuda", enabled=use_amp)
```

**Early stopping**

```python
early_stopping_check_interval = 10   # check every 10 epochs
early_stopping_patience_epochs = 40  # stop after 40 epochs without improvement
early_stopping_warmup_epochs = 80    # no stopping before epoch 80
# patience in checks = ceil(40 / 10) = 4
early_stopping = EarlyStopping(patience=4, min_delta=1e-5,
                                divergence_threshold=5.0, divergence_patience=2)
```

**Optimizers**

```python
optimizer_G = AdamW(G_AB.params + G_BA.params,
                    lr=2e-4, betas=(0.5, 0.999), weight_decay=0.01)
optimizer_D_A = Adam(D_A.params, lr=2e-4, betas=(0.5, 0.999))
optimizer_D_B = Adam(D_B.params, lr=2e-4, betas=(0.5, 0.999))
```

**LR schedulers**

```python
# factor = 1.0 - max(0, epoch - 100) / 100
# epochs 0–100: factor = 1.0  (constant)
# epoch 101:    factor = 0.99
# epoch 200:    factor = 0.0
lr_scheduler_G   = LambdaLR(optimizer_G,   lr_lambda)
lr_scheduler_D_A = LambdaLR(optimizer_D_A, lr_lambda)
lr_scheduler_D_B = LambdaLR(optimizer_D_B, lr_lambda)
```

---

### 4C. Resume Phase

Triggered when `resume_checkpoint` is not `None`.

```
_load_checkpoint_compat(resume_checkpoint, map_location=device)
  → checkpoint dict

Restore (with backward-compatible key lookup):
  G_AB  ← checkpoint["G_AB"]  or  checkpoint["G_AB_state_dict"]
  G_BA  ← checkpoint["G_BA"]  or  checkpoint["G_BA_state_dict"]
  D_A   ← checkpoint["D_A"]   or  checkpoint["D_A_state_dict"]   (optional)
  D_B   ← checkpoint["D_B"]   or  checkpoint["D_B_state_dict"]   (optional)
  optimizer_G   ← checkpoint["optimizer_G"]   (optional)
  optimizer_D_A ← checkpoint["optimizer_D_A"] (optional)
  optimizer_D_B ← checkpoint["optimizer_D_B"] (optional)
  lr_scheduler_G   ← checkpoint["lr_scheduler_G_state_dict"]
                   or set last_epoch = resume_epoch - 1
  lr_scheduler_D_A ← checkpoint["lr_scheduler_D_A_state_dict"]
                   or set last_epoch = resume_epoch - 1
  lr_scheduler_D_B ← checkpoint["lr_scheduler_D_B_state_dict"]
                   or set last_epoch = resume_epoch - 1
  scaler           ← checkpoint["scaler_state_dict"]  (if use_amp)
  early_stopping   ← checkpoint["early_stopping_state"]

start_epoch = checkpoint["epoch"]
Guard: start_epoch < num_epochs  (raises ValueError otherwise)
```

---

### 4D. Per-Epoch Loop

```
for epoch in range(start_epoch, num_epochs):
  G_AB.train(), G_BA.train(), D_A.train(), D_B.train()
  epoch_loss_G = epoch_loss_D_A = epoch_loss_D_B = 0.0

  for i, batch in enumerate(train_loader):
    real_A = batch["A"]  # (N, 3, 256, 256)
    real_B = batch["B"]  # (N, 3, 256, 256)
    [per-batch update — see 4E]

  [epoch-end logging — see 4F]
  [validation + early stopping — see 4G]
```

---

### 4E. Per-Batch Update Order

#### Step 1 — Generator

```
Freeze D_A, D_B (requires_grad=False)
optimizer_G.zero_grad(set_to_none=True)

with autocast("cuda", enabled=use_amp):
    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A,    # (N, 3, 256, 256)
        real_B,    # (N, 3, 256, 256)
        G_AB, G_BA, D_A, D_B,
        epoch, num_epochs
    )
    # loss_G: scalar (LSGAN + cycle + identity + perceptual + GP)
    # fake_A: (N, 3, 256, 256)  G_BA(real_B)
    # fake_B: (N, 3, 256, 256)  G_AB(real_A)

scaler.scale(loss_G).backward()
scaler.step(optimizer_G)
scaler.update()
```

#### Step 2 — Discriminator A

```
Unfreeze D_A, D_B (requires_grad=True)
optimizer_D_A.zero_grad(set_to_none=True)

with autocast("cuda", enabled=use_amp):
    loss_D_A = loss_fn.discriminator_loss(
        D_A,
        real_A,              # (N, 3, 256, 256)  real samples
        fake_A,              # (N, 3, 256, 256)  from G_BA(real_B)
        loss_fn.fake_A_buffer  # replay buffer
    )
    # loss_D_A: scalar  0.5*(LSGAN_real + LSGAN_fake) + GP

scaler.scale(loss_D_A).backward()
scaler.step(optimizer_D_A)
scaler.update()
```

#### Step 3 — Discriminator B

```
optimizer_D_B.zero_grad(set_to_none=True)

with autocast("cuda", enabled=use_amp):
    loss_D_B = loss_fn.discriminator_loss(
        D_B,
        real_B,              # (N, 3, 256, 256)
        fake_B,              # (N, 3, 256, 256)  from G_AB(real_A)
        loss_fn.fake_B_buffer
    )

scaler.scale(loss_D_B).backward()
scaler.step(optimizer_D_B)
scaler.update()
```

#### Per-batch history entry

```python
epoch_step[i] = {
    "Batch":   i,           # int  batch index (1-based)
    "Loss_G":  float,       # total generator loss
    "Loss_D_A": float,      # discriminator A loss
    "Loss_D_B": float,      # discriminator B loss
}
```

---

### 4F. Epoch-end Logging and Checkpointing

```
# TensorBoard epoch-average scalars
writer.add_scalar("Loss/Generator",      epoch_loss_G   / n_batches, epoch+1)
writer.add_scalar("Loss/Discriminator_A", epoch_loss_D_A / n_batches, epoch+1)
writer.add_scalar("Loss/Discriminator_B", epoch_loss_D_B / n_batches, epoch+1)

# CSV flush every 5 epochs
if (epoch + 1) % 5 == 0:
    append_history_to_csv(history, history_csv_path)
    history.clear()

# Checkpoint every save_checkpoint_every epochs (default 20)
# Payload: see Section 9

# LR scheduler step
lr_scheduler_G.step()
lr_scheduler_D_A.step()
lr_scheduler_D_B.step()

# TensorBoard LR scalars
writer.add_scalar("Learning Rate/Generator",      lr_G,  epoch+1)
writer.add_scalar("Learning Rate/Discriminator_A", lr_DA, epoch+1)
writer.add_scalar("Learning Rate/Discriminator_B", lr_DB, epoch+1)
```

---

### 4G. Validation and Early Stopping

**Every epoch — qualitative validation**

```python
run_validation(
    epoch=epoch+1,
    G_AB=G_AB, G_BA=G_BA,
    test_loader=test_loader,
    device=device,
    save_dir=val_dir/epoch_{epoch+1},
    num_samples=10,
    writer=writer,
)
# Saves 10 × 4-panel PNG grids:
#   [Real A | Fake B | Rec A | Real B]
#   [Real B | Fake A | Rec B | Real A]
```

**Every 10 epochs — metric computation**

```python
avg_metrics = calculate_metrics(
    calculator=metrics_calculator,
    G_AB=G_AB, G_BA=G_BA,
    test_loader=test_loader,
    device=device,
    writer=writer,
    epoch=epoch+1,
)
# avg_metrics: {"ssim_A": float, "ssim_B": float,
#               "psnr_A": float, "psnr_B": float}

avg_ssim = (avg_metrics["ssim_A"] + avg_metrics["ssim_B"]) / 2
# range: [0, 1]; higher is better
```

**Early stopping check (after warmup epoch 80)**

```python
should_stop = early_stopping(
    ssim=avg_ssim,
    losses={"G": avg_loss_G, "D_A": avg_loss_D_A, "D_B": avg_loss_D_B}
)
```

EarlyStopping configuration:

| Parameter | Value | Meaning |
|---|---|---|
| `patience` | 4 checks | Stop after 4 consecutive checks (= 40 epochs) without SSIM improvement |
| `min_delta` | 1e-5 | Minimum SSIM improvement to reset counter |
| `divergence_threshold` | 5.0 | Loss ratio above which divergence is flagged |
| `divergence_patience` | 2 | Consecutive divergence checks before stopping |
| Warmup | epoch 80 | No stopping before this epoch |

**TensorBoard early-stopping scalars**

```
EarlyStopping/avg_ssim
EarlyStopping/loss_G
EarlyStopping/loss_D_A
EarlyStopping/loss_D_B
EarlyStopping/best_ssim
EarlyStopping/best_loss_G
EarlyStopping/best_loss_D_A
EarlyStopping/best_loss_D_B
EarlyStopping/counter
EarlyStopping/divergence_counter
```

---

### 4H. Finalization

```
calculate_metrics(...)          # final metrics on test set
run_testing(
    G_AB, G_BA, test_loader, device,
    save_dir=model_dir/test_images,
    num_samples=test_size,      # default 200
)
torch.save(final_checkpoint, model_dir/final_checkpoint_epoch_N.pth)
append_history_to_csv(history, history_csv_path)
history = load_history_from_csv(history_csv_path)  # reload full run
writer.close()
return history, G_AB, G_BA, D_A, D_B
```

---

## 5. Function: `train_v1`

### Signature

```python
train_v1(*args, **kwargs) -> tuple
```

Thin alias that forwards all arguments to `train(...)`.  Exists so that
`trainModel.py` can import `train_v1` with the same pattern used for
`train_v2`, `train_v3`, and `train_v4`.

---

## 6. Tensor Shape Reference

| Tensor | Shape | Notes |
|---|---|---|
| `real_A`, `real_B` | `(N, 3, 256, 256)` | Normalised to `[-1, 1]` |
| `fake_A`, `fake_B` | `(N, 3, 256, 256)` | Generator output, `tanh` activated |
| `rec_A` = G_BA(fake_B) | `(N, 3, 256, 256)` | Cycle reconstruction |
| `rec_B` = G_AB(fake_A) | `(N, 3, 256, 256)` | Cycle reconstruction |
| `idt_A` = G_BA(real_A) | `(N, 3, 256, 256)` | Identity output |
| `idt_B` = G_AB(real_B) | `(N, 3, 256, 256)` | Identity output |
| D output | `(N, 1, 30, 30)` | 70×70 PatchGAN receptive field on 256×256 input |
| `loss_G` | scalar | Weighted sum of all generator loss terms |
| `loss_D_A`, `loss_D_B` | scalar | LSGAN + GP per discriminator |

---

## 7. Loss Term Reference

All weights are hardcoded in `train()`, not read from `cfg`.

| Term | Formula | Weight | Notes |
|---|---|---|---|
| LSGAN generator | `E[(D(fake)-1)²]` | 1.0 | Per direction, summed |
| Cycle-consistency | `L1(G_BA(G_AB(A)), A) + L1(G_AB(G_BA(B)), B)` | λ=10.0 | |
| Identity | `L1(G_AB(B), B) + L1(G_BA(A), A)` | λ=5.0 | Decays to 0 after 50% of training |
| Perceptual cycle | VGG19 relu1_2/2_2/3_4 on cycle images | λ=0.2 | Images resized to 160×160 |
| Perceptual identity | VGG19 on identity images | λ=0.1 | Images resized to 160×160 |
| Two-sided GP | `E[(‖∇D(x̂)‖₂ - 1)²]` | λ=10.0 | x̂ interpolated between real and fake |
| LSGAN discriminator | `0.5*(E[(D(real)-1)²] + E[D(fake)²])` | 1.0 | Per discriminator |

---

## 8. Optimizer and Scheduler Reference

| Optimizer | Module | Type | lr | betas | weight_decay |
|---|---|---|---|---|---|
| `optimizer_G` | G_AB + G_BA | AdamW | 2e-4 | (0.5, 0.999) | 0.01 |
| `optimizer_D_A` | D_A | Adam | 2e-4 | (0.5, 0.999) | — |
| `optimizer_D_B` | D_B | Adam | 2e-4 | (0.5, 0.999) | — |

LR schedule (same lambda for all three):

| Epoch range | Multiplier | Effective LR |
|---|---|---|
| 0 – 100 | 1.0 | 2e-4 |
| 101 | 0.99 | 1.98e-4 |
| 150 | 0.50 | 1.0e-4 |
| 200 | 0.0 | 0.0 |

---

## 9. Checkpoint Payload Reference

Saved every `save_checkpoint_every` epochs (default 20) and at end of training.

| Key | Type | Description |
|---|---|---|
| `"epoch"` | `int` | Epoch number just completed |
| `"config"` | `UVCGANConfig` | Full config dataclass |
| `"G_AB"` | `state_dict` | Generator A→B weights |
| `"G_BA"` | `state_dict` | Generator B→A weights |
| `"D_A"` | `state_dict` | Discriminator A weights |
| `"D_B"` | `state_dict` | Discriminator B weights |
| `"optimizer_G"` | `state_dict` | AdamW optimizer state |
| `"optimizer_D_A"` | `state_dict` | Adam optimizer state |
| `"optimizer_D_B"` | `state_dict` | Adam optimizer state |
| `"lr_scheduler_G_state_dict"` | `state_dict` | LambdaLR state |
| `"lr_scheduler_D_A_state_dict"` | `state_dict` | LambdaLR state |
| `"lr_scheduler_D_B_state_dict"` | `state_dict` | LambdaLR state |
| `"scaler_state_dict"` | `state_dict \| None` | AMP GradScaler state; `None` on CPU |
| `"early_stopping_state"` | `dict` | EarlyStopping counters and best values |

---

## 10. TensorBoard Scalar Reference

| Tag | Logged at | Value |
|---|---|---|
| `Epoch:` | Every epoch | Current epoch number |
| `Loss/Generator` | Every epoch | `epoch_loss_G / n_batches` |
| `Loss/Discriminator_A` | Every epoch | `epoch_loss_D_A / n_batches` |
| `Loss/Discriminator_B` | Every epoch | `epoch_loss_D_B / n_batches` |
| `Learning Rate/Generator` | Every epoch | Current LR after scheduler step |
| `Learning Rate/Discriminator_A` | Every epoch | Current LR |
| `Learning Rate/Discriminator_B` | Every epoch | Current LR |
| `Validation Started` | Every epoch | Epoch number |
| `Checkpoint saved` | Every 20 epochs | Epoch number |
| `EarlyStopping/avg_ssim` | Every 10 epochs | Mean SSIM both domains |
| `EarlyStopping/counter` | Every 10 epochs | Patience counter |
| `EarlyStopping/divergence_counter` | Every 10 epochs | Divergence counter |
| `Testing Started` | End of training | `stopped_epoch` |
| `Training Completed` | End of training | `stopped_epoch` |

---

## 11. Artifact Layout

```
model_dir/
  tensorboard_logs/
  validation_images/
    epoch_1/
      image_1_A.png   ← [Real A | Fake B | Rec A | Real B]
      image_1_B.png   ← [Real B | Fake A | Rec B | Real A]
      ...
    epoch_2/ ...
  test_images/
    image_1_A.png
    ...
  training_history.csv
  checkpoint_epoch_20.pth
  checkpoint_epoch_40.pth
  ...
  final_checkpoint_epoch_N.pth
```
