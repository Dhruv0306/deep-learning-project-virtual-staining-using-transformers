# trainModel.py Training Entry Point

Source of truth: ../trainModel.py

This script is the interactive launcher for all training variants in this repository.

## What It Does

1. Prompts for runtime parameters:
   - epoch size
   - number of epochs
   - test size
   - model version (1, 2, or 3)
2. Creates a timestamped model output directory under:
   - data/E_Staining_DermaRepo/H_E-Staining_dataset/
3. Creates a validation_images subfolder.
4. Dispatches to the selected training loop.
5. Saves training history as both plot and CSV.

## Model Version Dispatch

- model_version = 1
  - Calls model_v1.training_loop.train_v1
  - Model directory prefix: models_

- model_version = 2
  - Calls model_v2.training_loop.train_v2
  - Uses config.get_8gb_config()
  - Model directory prefix: models_v2_

- model_version = 3
  - Calls model_v3.training_loop.train_v3
  - Uses config.get_dit_8gb_config()
  - Model directory prefix: models_v3_

## Return Signature

main() returns a consistent tuple:

(history, G_AB, G_BA, D_A, D_B)

For model version 3, values are mapped to preserve compatibility with tooling that expects CycleGAN-style outputs:

- G_AB <- dit_model
- G_BA <- ema_model
- D_A <- cond_encoder
- D_B <- None

## Saved Artifacts

Each run directory contains at least:

- validation_images/
- training_history.csv
- training_history.png
- model checkpoints created by the selected training loop

## Example

python trainModel.py
