# `history_utils.py` — Training History Utilities

**Shared by:** Both v1 and v2  
**Role:** Provides helpers to visualize, save, append, and reload per-batch training history. Used at the end of training (or periodically during training) to persist logs and produce diagnostic plots.

---

## History Format

The training history is a nested dict with the structure:

```python
history = {
    epoch_number: {
        batch_number: {
            'Loss_G':   float,   # generator loss
            'Loss_D_A': float,   # discriminator A loss
            'Loss_D_B': float,   # discriminator B loss
        },
        ...
    },
    ...
}
```

Both training loops build this dict in memory during training and flush it to CSV periodically.

---

## Functions

### `visualize_history(history, model_dir=None)`

Generates a 2×2 matplotlib figure and saves it to `{model_dir}/training_history.png`.

| Parameter | Type | Description |
|---|---|---|
| `history` | `dict` | Nested epoch → batch → loss dict |
| `model_dir` | `str` or `None` | Save directory. Defaults to the dataset models folder |

**Subplots produced:**

| Position | Content |
|---|---|
| (0,0) | All three per-epoch average losses on one axis (G, D_A, D_B) |
| (0,1) | Generator loss only |
| (1,0) | Discriminator A vs Discriminator B comparison |
| (1,1) | Batch-level losses for the **final epoch** (shows within-epoch stability) |

Also prints a training summary to stdout (total epochs, final and average losses).

---

### `save_history_to_csv(history, filename)`

Flattens the nested history dict into a DataFrame and writes it to CSV.

| Parameter | Type | Description |
|---|---|---|
| `history` | `dict` | Training history |
| `filename` | `str` | Output CSV path |

**CSV columns:** `Epoch`, `Batch`, `Loss_G`, `Loss_D_A`, `Loss_D_B`

Overwrites the file if it already exists. Used at the end of training for a complete persistent record.

---

### `append_history_to_csv(history, filename)`

Appends one chunk of history to an existing CSV (or creates it if it does not exist). This is the function called periodically **during** training (every 5 epochs) so that history survives a crash.

| Parameter | Type | Description |
|---|---|---|
| `history` | `dict` | Partial history chunk (typically the last 5 epochs) |
| `filename` | `str` | CSV path to append to |

Writes the header row only if the file is new or empty. Existing rows are not modified.

---

### `load_history_from_csv(filename)`

Reads a previously saved CSV and reconstructs the nested history dict.

| Parameter | Type | Description |
|---|---|---|
| `filename` | `str` | CSV path to read |

Returns an empty `{}` if the file does not exist or is empty. Otherwise returns the same nested dict structure that the training loops produce.

Useful for resuming visualization or inspection after a completed or interrupted run.
