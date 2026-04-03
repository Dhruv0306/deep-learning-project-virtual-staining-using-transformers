"""
Early stopping utility for unpaired image translation training.

Tracks validation SSIM improvement and loss divergence to decide when to halt training.
"""


class EarlyStopping:
    """
    Early stopping based on SSIM improvement and loss divergence detection.

    Stops training when either:
    - SSIM has not improved by more than min_delta for ``patience`` consecutive checks, or
    - All tracked losses simultaneously exceed ``divergence_threshold`` × their best
      value for ``divergence_patience`` consecutive checks.

    Args:
        patience:             Checks without SSIM improvement before stopping.
        min_delta:            Minimum SSIM gain required to reset the patience counter.
        divergence_threshold: Loss is considered diverged when it exceeds
                              ``best * divergence_threshold``.
        divergence_patience:  Consecutive divergence checks required to trigger a stop.
        divergence_floor:     Minimum baseline for divergence ratio checks; prevents
                              false positives when best loss is near zero.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        divergence_threshold=5.0,
        divergence_patience=2,
        divergence_floor=1e-3,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.divergence_patience = divergence_patience
        self.divergence_floor = max(0.0, float(divergence_floor))
        self.best_ssim = -float("inf")
        self.counter = 0
        self.best_losses = {}
        self.divergence_counter = 0

    def state_dict(self):
        """Return a serializable snapshot of the early-stopping state."""
        return {
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "divergence_threshold": float(self.divergence_threshold),
            "divergence_patience": int(self.divergence_patience),
            "divergence_floor": float(self.divergence_floor),
            "best_ssim": float(self.best_ssim),
            "counter": int(self.counter),
            "best_losses": {k: float(v) for k, v in self.best_losses.items()},
            "divergence_counter": int(self.divergence_counter),
        }

    def load_state_dict(self, state):
        """Restore early-stopping state from :meth:`state_dict` output."""
        if not isinstance(state, dict):
            return
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.divergence_threshold = float(
            state.get("divergence_threshold", self.divergence_threshold)
        )
        self.divergence_patience = int(
            state.get("divergence_patience", self.divergence_patience)
        )
        self.divergence_floor = max(
            0.0, float(state.get("divergence_floor", self.divergence_floor))
        )
        self.best_ssim = float(state.get("best_ssim", self.best_ssim))
        self.counter = int(state.get("counter", self.counter))

        loaded_losses = state.get("best_losses", self.best_losses)
        if isinstance(loaded_losses, dict):
            self.best_losses = {k: float(v) for k, v in loaded_losses.items()}

        self.divergence_counter = int(
            state.get("divergence_counter", self.divergence_counter)
        )

    def __call__(self, ssim, losses):
        """
        Update state and return whether training should stop.

        Args:
            ssim (float):          Current validation SSIM (higher is better).
            losses (dict | float): Current loss values to monitor for divergence.
                                   A plain float is wrapped into {"loss": value}.

        Returns:
            bool: True if training should stop.
        """
        # Check for SSIM improvement.
        if ssim > self.best_ssim + self.min_delta:
            self.best_ssim = ssim
            self.counter = 0
        else:
            self.counter += 1

        # Normalize losses to a dict for per-key baseline tracking.
        if not isinstance(losses, dict):
            losses = {"loss": float(losses)}

        diverged_losses = 0
        for name, value in losses.items():
            value = float(value)
            best = self.best_losses.get(name, float("inf"))

            # Initialize per-loss baseline on first observation.
            if best == float("inf"):
                self.best_losses[name] = value
                continue

            # Apply a floor so near-zero best values don't cause false positives.
            divergence_baseline = max(abs(best), self.divergence_floor)

            # Count this loss as diverged if it exceeds the threshold.
            if value > divergence_baseline * self.divergence_threshold:
                diverged_losses += 1

            # Track the best (lowest) value per loss.
            if value < best:
                self.best_losses[name] = value

        # Increment divergence counter only when all tracked losses diverge simultaneously.
        if len(losses) > 0 and diverged_losses == len(losses):
            self.divergence_counter += 1
        else:
            self.divergence_counter = 0

        # Hard stop on repeated divergence.
        if self.divergence_counter >= self.divergence_patience:
            return True

        # Stop if SSIM has not improved for too long.
        return self.counter >= self.patience
