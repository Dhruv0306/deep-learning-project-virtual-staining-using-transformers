# trainModel.py Training Entry Point

Source of truth: ../trainModel.py

This script is the interactive launcher for all model variants in the repo.

## What It Does

1. Prompts for:
   - epoch size
   - number of epochs
   - test size
   - model version (1, 2, 3, or 4)
2. Creates a timestamped model directory under:
   - `data/E_Staining_DermaRepo/H_E-Staining_dataset/`
3. Creates `validation_images/` in that run directory.
4. Dispatches to the selected training loop.
5. Saves training history as image and CSV.

## Model Version Dispatch

- `model_version = 1`
  - calls `model_v1.training_loop.train_v1`
  - directory prefix: `models_`

- `model_version = 2`
  - calls `model_v2.training_loop.train_v2`
  - uses `config.get_8gb_config()`
  - directory prefix: `models_v2_`

- `model_version = 3`
  - calls `model_v3.training_loop.train_v3`
  - uses `config.get_dit_8gb_config()`
  - directory prefix: `models_v3_`

- `model_version = 4`
  - calls `model_v4.training_loop.train_v4`
  - uses `config.get_v4_8gb_config()`
  - directory prefix: `models_v4_`

## Return Signature

`main()` returns:

`(history, G_AB, G_BA, D_A, D_B)`

For v3, this tuple is adapted for compatibility with downstream tooling:

- `G_AB <- dit_model`
- `G_BA <- ema_model`
- `D_A <- cond_encoder`
- `D_B <- None`

For v1, v2, and v4, these values are the trained generators and discriminators
returned by each training loop.

## Saved Artifacts

Each run directory includes at least:

- `validation_images/`
- `training_history.csv`
- training-history plot image
- periodic and/or final checkpoints from the selected loop

## Usage

```bash
python trainModel.py
```
