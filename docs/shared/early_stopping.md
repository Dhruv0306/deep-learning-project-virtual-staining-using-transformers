# `shared/EarlyStopping.py` — Early Stopping

**Shared by:** Both v1 and v2  
**Role:** Monitors validation SSIM and loss divergence across training epochs and decides when to stop training to prevent wasted compute or model degradation.

---

## Class: `EarlyStopping`

Tracks two independent stopping criteria:

1. **SSIM plateau** — stops training if validation SSIM has not improved by at least `min_delta` for `patience` consecutive checks.
2. **Loss divergence** — stops training if all monitored losses simultaneously exceed `divergence_threshold × best_ever_value` for `divergence_patience` consecutive checks.

### `__init__(patience, min_delta, divergence_threshold, divergence_patience)`

| Parameter | Default | Description |
|---|---|---|
| `patience` | 10 | Number of validation checks without SSIM improvement before stopping. One check occurs every `early_stopping_interval` epochs in the training loop |
| `min_delta` | 0.001 | Minimum increase in SSIM to count as an improvement |
| `divergence_threshold` | 5.0 | A loss is considered diverging if its current value exceeds `best_value × divergence_threshold` |
| `divergence_patience` | 2 | Number of consecutive divergence checks before stopping |

**Internal state:**

| Attribute | Initial value | Description |
|---|---|---|
| `best_ssim` | `-inf` | Best SSIM seen so far |
| `counter` | 0 | Number of consecutive checks without SSIM improvement |
| `best_losses` | `{}` | Per-key best (lowest) loss value seen so far |
| `divergence_counter` | 0 | Consecutive checks where all losses diverged |

---

### `__call__(ssim, losses)`

Called once per validation check. Updates internal state and returns `True` if training should stop.

| Parameter | Type | Description |
|---|---|---|
| `ssim` | `float` | Current validation SSIM value |
| `losses` | `dict` or `float` | Current loss values. A plain float is wrapped as `{'loss': value}` |

**SSIM check:**
```
if ssim > best_ssim + min_delta:
    best_ssim = ssim
    counter   = 0          ← reset on improvement
else:
    counter  += 1
```

**Divergence check (per loss key):**
```
for each (name, value) in losses:
    if value > best_losses[name] × divergence_threshold:
        count as diverged

if ALL losses are diverged:
    divergence_counter += 1
else:
    divergence_counter  = 0    ← reset if any loss recovers
```

**Stop conditions:**
```
return True  if divergence_counter >= divergence_patience
return True  if counter >= patience
return False otherwise
```

---

## Integration in the Training Loop

```python
early_stopping = EarlyStopping(patience=40, divergence_patience=2)

# Called every early_stopping_interval epochs (after early_stopping_warmup):
should_stop = early_stopping(
    ssim=avg_metrics['ssim_B'],
    losses={'G': loss_G_epoch, 'D_A': loss_D_A_epoch, 'D_B': loss_D_B_epoch}
)

if should_stop:
    print("Early stopping triggered.")
    break
```

The training loop in both v1 and v2 enforces a `warmup` period before early stopping can trigger, regardless of what `EarlyStopping` returns. This prevents stopping before the model has had time to converge from its random initialisation.

