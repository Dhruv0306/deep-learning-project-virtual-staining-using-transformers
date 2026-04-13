# trainModel.py — Detailed Reference

Source: `../trainModel.py`

Interactive training entry point for all four model variants.  Reads run-time
parameters from stdin, resolves output directories, dispatches to the selected
training loop, and persists training history as a PNG figure and a CSV file.

---

## Table of Contents

1. [Module-level Data Flow](#1-module-level-data-flow)
2. [Imports and Dependencies](#2-imports-and-dependencies)
3. [Function: `_parse_checkpoint_epoch`](#3-function-_parse_checkpoint_epoch)
4. [Function: `main`](#4-function-main)
   - [4A Prompt Sequence](#4a-prompt-sequence)
   - [4B Epoch / Resume Validation](#4b-epoch--resume-validation)
   - [4C Model Version Dispatch](#4c-model-version-dispatch)
   - [4D History Persistence](#4d-history-persistence)
5. [Return Value](#5-return-value)
6. [Artifact Layout](#6-artifact-layout)
7. [TensorBoard Scalars Written by Each Loop](#7-tensorboard-scalars-written-by-each-loop)

---

## 1. Module-level Data Flow

```
stdin
  │
  ├─ mode ('new' | 'resume')
  ├─ resume_checkpoint path  (resume only)
  ├─ epoch_size  (int)
  ├─ num_epochs  (int)
  ├─ test_size   (int)
  └─ model_version (1 | 2 | 3 | 4)
          │
          ▼
  _parse_checkpoint_epoch(resume_checkpoint)
          │  → resume_epoch (int | None)
          │
          ▼
  model_dir  ← parent(resume_checkpoint)  [resume]
             ← dataset_root / models_vN_TIMESTAMP  [new]
          │
          ▼
  cfg  ← get_default_config(1) | get_8gb_config() |
         get_dit_8gb_config()  | get_v4_8gb_config()
          │
          ▼
  train_v1 / train_v2 / train_v3 / train_v4
          │
          ▼
  (history, G_AB, G_BA, D_A, D_B)
          │
          ├─ history_visualizer(history, model_dir)  → training_history.png
          └─ history_saver(history, path)            → training_history.csv
```

---

## 2. Imports and Dependencies

| Symbol | Source | Purpose |
|---|---|---|
| `save_history_to_csv` | `shared.history_utils` | Persist history dict to CSV (v1/v2/v4) |
| `visualize_history` | `shared.history_utils` | Plot training curves (v1/v2/v4) |
| `train_v1` | `model_v1.training_loop` | v1 training loop |
| `get_default_config` | `config` | Config preset for v1 |
| `get_8gb_config` | `config` | Config preset for v2 |
| `get_v4_8gb_config` | `config` | Config preset for v4 |
| `train_v2` | `model_v2.training_loop` | v2 training loop |
| `train_v4` | `model_v4.training_loop` | v4 training loop |
| `get_dit_8gb_config` | `config` (lazy import) | Config preset for v3 |
| `train_v3` | `model_v3.training_loop` (lazy import) | v3 training loop |
| `visualize_history_v3` | `model_v3.history_utils` (lazy import) | v3-specific plot |
| `save_history_to_csv_v3` | `model_v3.history_utils` (lazy import) | v3-specific CSV |

v3 imports are deferred inside the `elif model_version == 3` branch to avoid
loading heavy diffusion dependencies when running other model versions.

---

## 3. Function: `_parse_checkpoint_epoch`

### Signature

```python
_parse_checkpoint_epoch(checkpoint_path: str) -> int | None
```

### Purpose

Extracts the integer epoch number embedded in a checkpoint filename so that
`main()` can validate that `num_epochs > resume_epoch` before dispatching.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `checkpoint_path` | `str` | Absolute or relative path to a `.pth` checkpoint file |

### Regex Pattern

```
(?:final_)?checkpoint_epoch_(\d+)\.pth$
```

Matches both periodic and final checkpoint naming conventions.

### Supported Filename Patterns

| Filename | Extracted epoch |
|---|---|
| `checkpoint_epoch_40.pth` | `40` |
| `checkpoint_epoch_120.pth` | `120` |
| `final_checkpoint_epoch_200.pth` | `200` |
| `my_model.pth` | `None` |
| `final_checkpoint_200.pth` | `None` (no `epoch_` prefix) |

### Return Value

| Condition | Return |
|---|---|
| Pattern matched | `int` — the epoch number |
| Pattern not matched | `None` |

### Example

```python
_parse_checkpoint_epoch("models_v2_2024_01_01/checkpoint_epoch_80.pth")
# → 80

_parse_checkpoint_epoch("models_v2_2024_01_01/final_checkpoint_epoch_200.pth")
# → 200

_parse_checkpoint_epoch("models_v2_2024_01_01/my_weights.pth")
# → None
```

---

## 4. Function: `main`

### Signature

```python
main() -> tuple[dict, nn.Module, nn.Module, nn.Module | None, nn.Module | None]
```

### Purpose

Orchestrates the full training launch: reads stdin, validates inputs, resolves
directories, selects and calls the appropriate training loop, and saves history.

---

### 4A. Prompt Sequence

Prompts are issued in this exact order:

| # | Prompt text | Type | Validation |
|---|---|---|---|
| 1 | `"Start new training or resume? Enter 'new' or 'resume': "` | `str` | Must be `'new'` or `'resume'`; raises `ValueError` otherwise |
| 2 | `"Enter resume checkpoint path (.pth): "` | `str` | Resume only; stripped of surrounding quotes; raises `ValueError` if empty, `FileNotFoundError` if path does not exist |
| 3 | `"Enter Epoch Size: "` | `int` | Cast with `int()`; no range check |
| 4 | `"Enter Number of Epochs: "` | `int` | Must exceed checkpoint epoch on resume; raises `ValueError` if not |
| 5 | `"Enter Test Size: "` | `int` | Cast with `int()`; passed to training loop |
| 6 | `"Enter model version you want 1 for Hybrid, 2 for true UVCGAN, 3 for DiT diffusion, 4 for v4 (Transformer + NCE): "` | `int` | Must be 1–4; raises `ValueError` otherwise |

---

### 4B. Epoch / Resume Validation

```
resume_epoch = _parse_checkpoint_epoch(resume_checkpoint)
if resume_epoch is not None and num_epochs <= resume_epoch:
    raise ValueError(...)
```

This check runs before any model is constructed.  If `_parse_checkpoint_epoch`
returns `None` (non-standard filename), the check is skipped and the training
loop itself will raise if the epoch is invalid.

---

### 4C. Model Version Dispatch

#### Directory Resolution

For all versions, `model_dir` is resolved as:

```python
# resume:
model_dir = os.path.dirname(resume_checkpoint)

# new:
model_dir = os.path.join(
    "data", "E_Staining_DermaRepo", "H_E-Staining_dataset",
    f"models_vN_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
)
```

`val_dir = os.path.join(model_dir, "validation_images")` is always created.

#### Per-version Details

**Version 1 — Hybrid CycleGAN/UVCGAN**

| Item | Value |
|---|---|
| Directory prefix | `models_` |
| Config | `get_default_config(model_version=1)` |
| Config fields set | `cfg.training.test_size = test_size` |
| `save_checkpoint_every` guard | Set to 20 if ≤ 0 |
| Training function | `train_v1(epoch_size, num_epochs, model_dir, val_dir, test_size, resume_checkpoint, cfg)` |

**Version 2 — True UVCGAN v2**

| Item | Value |
|---|---|
| Directory prefix | `models_v2_` |
| Config | `get_8gb_config()` |
| Config fields set | `save_checkpoint_every` guard only |
| Training function | `train_v2(epoch_size, num_epochs, model_dir, val_dir, test_size, resume_checkpoint, cfg)` |

**Version 3 — CycleDiT Latent Diffusion**

| Item | Value |
|---|---|
| Directory prefix | `models_v3_` |
| Config | `get_dit_8gb_config()` (lazy import) |
| Config fields set | `save_checkpoint_every` guard only |
| Training function | `train_v3(epoch_size, num_epochs, model_dir, val_dir, test_size, resume_checkpoint, cfg)` |
| Return remapping | `G_AB ← dit_model`, `G_BA ← ema_model`, `D_A ← cond_encoder`, `D_B ← None` |

**Version 4 — Transformer + PatchNCE**

| Item | Value |
|---|---|
| Directory prefix | `models_v4_` |
| Config | `get_v4_8gb_config()` |
| Config fields set | `cfg.training.test_size = test_size`, `save_checkpoint_every` guard |
| Training function | `train_v4(epoch_size, num_epochs, model_dir, val_dir, resume_checkpoint, cfg)` |

Note: `test_size` is not passed as a keyword argument to `train_v4`; it is
written into `cfg.training.test_size` instead.

---

### 4D. History Persistence

After the training loop returns:

```python
# v1 / v2 / v4:
visualize_history(history, model_dir=model_dir)
save_history_to_csv(history, os.path.join(model_dir, "training_history.csv"))

# v3 only:
visualize_history_v3(history, model_dir=model_dir)
save_history_to_csv_v3(history, os.path.join(model_dir, "training_history.csv"))
```

`history` at this point is the dict returned by the training loop, which has
already been reloaded from CSV (so it contains the full run, not just the
in-memory tail).

---

## 5. Return Value

`main()` returns `(history, G_AB, G_BA, D_A, D_B)`.

| Slot | v1 / v2 / v4 type | v3 type | Description |
|---|---|---|---|
| `history` | `dict[int, dict[int, dict]]` | same | Nested `{epoch: {batch: {loss_key: float}}}` |
| `G_AB` | `nn.Module` | `CycleDiTGenerator` | Generator A→B (unstained→stained) |
| `G_BA` | `nn.Module` | EMA copy of generator | Generator B→A (stained→unstained) |
| `D_A` | `nn.Module` | `None` (cond_encoder placeholder) | Discriminator for domain A |
| `D_B` | `nn.Module` | `None` | Discriminator for domain B |

History dict structure example (batch_size=4, epoch_size=100):

```python
{
  1: {                          # epoch 1
    1: {"Batch": 1, "Loss_G": 2.31, "Loss_D_A": 0.48, "Loss_D_B": 0.51},
    2: {"Batch": 2, "Loss_G": 2.18, "Loss_D_A": 0.45, "Loss_D_B": 0.49},
    ...
    25: {"Batch": 25, ...}      # 100 samples / batch_size=4 = 25 batches
  },
  2: { ... },
  ...
}
```

---

## 6. Artifact Layout

```
data/E_Staining_DermaRepo/H_E-Staining_dataset/
  models_TIMESTAMP/                  ← v1
  models_v2_TIMESTAMP/               ← v2
  models_v3_TIMESTAMP/               ← v3
  models_v4_TIMESTAMP/               ← v4
    ├── validation_images/
    │     └── epoch_N/
    │           ├── image_1_A.png    (Real A | Fake B | Rec A | Real B)
    │           └── image_1_B.png    (Real B | Fake A | Rec B | Real A)
    ├── test_images/
    ├── tensorboard_logs/
    ├── training_history.csv
    ├── training_history.png
    ├── checkpoint_epoch_20.pth
    ├── checkpoint_epoch_40.pth
    └── final_checkpoint_epoch_N.pth
```

---

## 7. TensorBoard Scalars Written by Each Loop

| Scalar key | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| `Loss/Generator` | ✓ | ✓ | — | ✓ |
| `Loss/Discriminator_A` | ✓ | ✓ | ✓ | ✓ |
| `Loss/Discriminator_B` | ✓ | ✓ | ✓ | ✓ |
| `Loss/DiT` | — | — | ✓ | — |
| `Loss/GAN` | — | — | — | ✓ |
| `Loss/NCE` | — | — | — | ✓ |
| `Loss/Identity` | — | — | ✓ | ✓ |
| `Loss/Cycle` | — | — | ✓ | — |
| `Loss/Perceptual` | — | — | ✓ | — |
| `LR/Generator` | ✓ | ✓ | ✓ | ✓ |
| `LR/Discriminator_A` | ✓ | ✓ | ✓ | ✓ |
| `LR/Discriminator_B` | ✓ | ✓ | ✓ | ✓ |
| `Diagnostics/GradNorm_G` | — | ✓ | ✓ | ✓ |
| `Validation/ssim_A` | ✓ | ✓ | ✓ | ✓ |
| `Validation/ssim_B` | ✓ | ✓ | ✓ | ✓ |
| `Validation/psnr_A` | ✓ | ✓ | ✓ | ✓ |
| `Validation/psnr_B` | ✓ | ✓ | ✓ | ✓ |
| `EarlyStopping/ssim` | ✓ | ✓ | ✓ | ✓ |
| `EarlyStopping/counter` | ✓ | ✓ | ✓ | ✓ |
| `EarlyStopping/divergence_counter` | ✓ | ✓ | ✓ | ✓ |
