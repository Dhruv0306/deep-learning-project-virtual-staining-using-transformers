# shared/EarlyStopping.py

Source of truth: ../../shared/EarlyStopping.py

Shared early-stopping utility used by training loops to stop on either:

1. SSIM plateau.
2. Repeated multi-loss divergence.

## Constructor

EarlyStopping accepts:

- patience: checks without SSIM improvement before stopping.
- min_delta: minimum SSIM gain required to reset patience.
- divergence_threshold: ratio used to detect loss explosions.
- divergence_patience: consecutive all-loss divergence checks before hard stop.
- divergence_floor: minimum baseline used in divergence ratio checks.

The divergence floor prevents false positives when the best loss becomes very close to zero.

## Stop Logic

On each call, the class:

1. Updates SSIM best/counter.
2. Normalizes losses to a dict.
3. Tracks per-loss best (minimum) values.
4. Marks each loss as diverged when:

```python
value > max(abs(best), divergence_floor) * divergence_threshold
```

5. Increments divergence counter only when all tracked losses diverge in the same check.

Training should stop if either:

- divergence_counter >= divergence_patience, or
- counter >= patience.

## State Persistence

The class supports checkpoint persistence via:

- state_dict()
- load_state_dict(state)

Saved fields include:

- best_ssim
- counter
- best_losses
- divergence_counter
- divergence_floor
- threshold/patience/min_delta values

This allows resumed training to keep the same early-stopping history.

## Typical Integration

```python
should_stop = early_stopping(
    ssim=avg_ssim,
    losses={"G": avg_loss_G, "D_A": avg_loss_D_A, "D_B": avg_loss_D_B},
)
```

Warmup/interval gating is handled in each training loop, not inside this class.

