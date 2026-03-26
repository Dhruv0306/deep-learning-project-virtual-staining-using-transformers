# `trainModel.py` — Training Entry Point

Source of truth: `../trainModel.py`

**Role:** Interactive command-line entry point for launching a training run. Prompts the user for key parameters, creates timestamped output directories, dispatches to the appropriate training loop (v1, v2, or v3), and saves training history on completion.

---

## Function: `main()`

### Interactive Prompts

| Prompt | Type | Description |
|---|---|---|
| `Enter Epoch Size:` | `int` | Number of training samples per epoch. Passed directly to `getDataLoader` via the training loop |
| `Enter Number of Epochs:` | `int` | Total number of training epochs |
| `Enter Test Size:` | `float` | Number of test samples to export during final evaluation |
| `Enter model version you want 1 for Hybrid, 2 for true UVCGAN, 3 for DiT diffusion:` | `int` | `1` launches `model_v1.training_loop.train_v1()`, `2` launches `model_v2.training_loop.train_v2()`, `3` launches `model_v3.training_loop.train_v3()` |

### Dispatching

**Model version 3 (DiT diffusion v3):**
- Creates a timestamped directory: `models_v3_{YYYY_MM_DD_HH_MM_SS}/`
- Loads config via `get_dit_8gb_config()` (VRAM-optimised for 8 GB GPUs)
- Calls `train_v3(epoch_size, num_epochs, model_dir, val_dir, test_size, cfg)`

**Model version 2 (True UVCGAN v2):**
- Creates a timestamped directory: `models_v2_{YYYY_MM_DD_HH_MM_SS}/`
- Loads config via `get_8gb_config()` (VRAM-optimised for 8 GB GPUs)
- Calls `train_v2(epoch_size, num_epochs, model_dir, val_dir, test_size, cfg)`

**Model version 1 (Hybrid UVCGAN + CycleGAN):**
- Creates a timestamped directory: `models_{YYYY_MM_DD_HH_MM_SS}/`
- Calls `train_v1(epoch_size, num_epochs, model_dir, val_dir, test_size)` with hyperparameters hardcoded inside `model_v1/training_loop.py`

All versions create a `validation_images/` subdirectory inside the model directory before launching.

### Post-training

After training completes:
1. `visualize_history(history, model_dir)` — saves a 2×2 loss plot as `training_history.png`
2. `save_history_to_csv(history, ...)` — saves the full per-batch history as `training_history.csv`

### Returns

`(history, G_AB, G_BA, D_A, D_B)` — the trained models and complete training history dict.

Note for v3: `trainModel.py` maps `(dit_model, ema_model, cond_encoder)` into `(G_AB, G_BA, D_A)` for a consistent return signature.

---

## Example Usage

```
$ python trainModel.py
Enter Epoch Size: 3000
Enter Number of Epochs: 200
Enter Test Size: 200
Enter model version you want 1 for Hybrid, 2 for true UVCGAN, 3 for DiT diffusion: 3
Model directory: data\...\models_v3_2024_01_15_10_30_00
Validation image directory: data\...\models_v3_...\validation_images
```

---

## Output Directory Structure

```
models_v3_{timestamp}/
    +-- validation_images/
    ¦   +-- image_*_A.png, image_*_B.png    ? per-epoch visual checks
    +-- checkpoint_epoch_*.pth             ? periodic checkpoints
    +-- final_checkpoint_epoch_*.pth       ? final trained weights
    +-- training_history.csv               ? per-batch loss log
    +-- training_history.png               ? loss curve plot
```
