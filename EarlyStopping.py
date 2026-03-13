"""
Early stopping utility for CycleGAN training.

Tracks validation SSIM and loss divergence to decide when to halt training.
"""


class EarlyStopping:
    """
    Early stopping based on SSIM improvements and loss divergence.

    Args:
        patience (int): Number of checks without SSIM improvement before stopping.
        min_delta (float): Minimum SSIM improvement to reset patience.
        divergence_threshold (float): Multiplier for detecting loss divergence.
        divergence_patience (int): Number of consecutive divergence checks before stopping.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.001,
        divergence_threshold=5.0,
        divergence_patience=2,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.divergence_threshold = divergence_threshold
        self.divergence_patience = divergence_patience
        self.best_ssim = -float("inf")
        self.counter = 0
        self.best_losses = {}
        self.divergence_counter = 0

    def __call__(self, ssim, losses):
        """
        Update internal state and decide whether to stop.

        Args:
            ssim (float): Current validation SSIM (higher is better).
            losses (dict | float): Current losses to monitor for divergence.

        Returns:
            bool: True if training should stop.
        """
        # Check for SSIM improvement.
        if ssim > self.best_ssim + self.min_delta:
            self.best_ssim = ssim
            self.counter = 0
        else:
            self.counter += 1

        # Normalize losses to a dict so we can track per-key baselines.
        if not isinstance(losses, dict):
            losses = {"loss": float(losses)}

        diverged_losses = 0
        for name, value in losses.items():
            value = float(value)
            best = self.best_losses.get(name, float("inf"))

            # Initialize baseline per-loss before using divergence checks.
            if best == float("inf"):
                self.best_losses[name] = value
                continue

            # Count a divergence if this loss explodes relative to its best value.
            if value > best * self.divergence_threshold:
                diverged_losses += 1

            # Track the best (lowest) value per loss.
            if value < best:
                self.best_losses[name] = value

        # If all tracked losses diverge, increment divergence counter.
        if len(losses) > 0 and diverged_losses == len(losses):
            self.divergence_counter += 1
        else:
            self.divergence_counter = 0

        # Hard stop on repeated divergence.
        if self.divergence_counter >= self.divergence_patience:
            return True

        # Stop if SSIM has not improved for too long.
        return self.counter >= self.patience
