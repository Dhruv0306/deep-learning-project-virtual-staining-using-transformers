"""
Perceptual loss for v4 training.

Provides VGGPerceptualLoss for comparing high-level feature similarity
between generated and real images using pre-trained VGG19 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """
    Multi-level perceptual loss using VGG19 features.

    Computes L1 distance between VGG19 activations at four levels:
    relu1_2, relu2_2, relu3_4, and relu4_4.

    All parameters are frozen (no gradient flow to VGG).

    Args:
        resize_to: Spatial resolution to which images are resized before
            being passed to VGG. None skips resizing.
        weights: Per-level loss weights (w1, w2, w3, w4) for the four layers.
    """

    def __init__(
        self,
        resize_to: int = 128,
        weights: tuple = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.resize_to = resize_to
        self.weights = weights

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])  # relu4_4

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std normalization."""
        mean = self.get_buffer("mean").to(x.device).to(x.dtype)
        std = self.get_buffer("std").to(x.device).to(x.dtype)
        return (x - mean) / std

    def _extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-level VGG19 feature activations.

        Returns:
            tuple: (h1, h2, h3, h4) at relu1_2, relu2_2, relu3_4, relu4_4.
        """
        h1 = self.slice1(x)  # type: ignore
        h2 = self.slice2(h1)  # type: ignore
        h3 = self.slice3(h2)  # type: ignore
        h4 = self.slice4(h3)  # type: ignore
        return h1, h2, h3, h4

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-level VGG19 perceptual loss between two image batches.

        Args:
            x: Generated image batch (N, C, H, W) in [-1, 1] range.
            y: Real/target image batch (N, C, H, W) in [-1, 1] range.

        Returns:
            Scalar perceptual loss (weighted sum of L1 distances).
        """
        x = x.float()
        y = y.float()

        # Handle grayscale
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        # Resize if needed
        if self.resize_to is not None:
            h, w = x.shape[-2], x.shape[-1]
            if h != self.resize_to or w != self.resize_to:
                x = F.interpolate(
                    x,
                    size=(self.resize_to, self.resize_to),
                    mode="bilinear",
                    align_corners=False,
                )
                y = F.interpolate(
                    y,
                    size=(self.resize_to, self.resize_to),
                    mode="bilinear",
                    align_corners=False,
                )

        # De-normalize from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        y = (y + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        y = y.clamp(0.0, 1.0)

        # Apply ImageNet normalization
        x = self._normalize(x)
        y = self._normalize(y)

        # Extract features in float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = self._extract(x.float())
            yf = self._extract(y.float())

        # Weighted L1 distance
        loss: torch.Tensor = sum(
            (w * F.l1_loss(xfi, yfi) for w, xfi, yfi in zip(self.weights, xf, yf)),
            torch.tensor(0.0, device=x.device, dtype=x.dtype),
        )
        return loss


__all__ = ["VGGPerceptualLoss"]
