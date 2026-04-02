# model_v3/history_utils.py - v3 History Utilities

Source of truth: ../../model_v3/history_utils.py

Utilities for saving, loading, appending, and visualizing v3 training history.

## Public Functions

1. `save_history_to_csv_v3`
2. `append_history_to_csv_v3`
3. `load_history_from_csv_v3`
4. `visualize_history_v3`
5. `_flatten_history`

## History Schema

The current v3 history stores per-batch values for:

- `Loss_DiT_A2B`
- `Loss_DiT_B2A`
- `Loss_DiT`
- `Loss_G_Adv`
- `Loss_Cyc`
- `Loss_Id`
- `Loss_D_A`
- `Loss_D_B`
- `Lambda_Adv`
- `Lambda_Id`
- `Loss_Perceptual`
- `Loss Total`
- `GradNorm`

Each row in the flattened CSV also includes `Epoch` and `Batch`.

## `save_history_to_csv_v3`

Overwrites the target CSV with the full flattened history.

## `append_history_to_csv_v3`

Appends a history chunk to an existing CSV and writes a header if the file is
new or empty.

## `load_history_from_csv_v3`

Reconstructs the nested epoch -> batch dictionary from the CSV on disk.
Missing columns fall back to `0.0` for backward compatibility.

## `visualize_history_v3`

Builds a 3x3 training plot and saves it to `training_history.png`.

Plotted series include:

- denoising A2B, B2A, and combined
- adversarial, cycle, and identity losses
- discriminator A and B losses
- perceptual loss
- total loss
- gradient norm

## `_flatten_history`

Converts the nested dictionary structure into a flat list of row dictionaries
ready for pandas DataFrame construction.
