# `model_v3/history_utils.py` — v3 History Utilities

Source of truth: `../../model_v3/history_utils.py`

**Role:** Stores and visualizes v3 training history without discriminator terms.

## CSV Schema

```
Epoch, Batch, Loss_DiT, Loss_Perceptual, GradNorm
```

## Plots

`visualize_history_v3()` writes `training_history.png` with DiT loss, perceptual loss, and grad norm curves.
