"""
model_v4/discriminator.py — PatchGAN discriminator for CUT + Transformer (v4).

Provides a standard N-layer 70×70 PatchGAN discriminator (PatchGANDiscriminator)
used alongside the baseline GAN.  Each output element of the final
conv layer covers a 70×70 receptive field of the 256×256 input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
    use_norm: bool = True,
) -> nn.Sequential:
    """
    Single conv → [InstanceNorm] → LeakyReLU building block.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        stride:       Convolution stride (2 to downsample, 1 to keep size).
        use_norm:     Whether to insert InstanceNorm2d after the conv.
                      Disabled for the first layer per PatchGAN convention.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=stride,
            padding=1,
            bias=not use_norm,
        )
    ]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGANDiscriminator(nn.Module):
    """
    N-layer PatchGAN discriminator.

    Produces a spatial map of real/fake scores rather than a single scalar.
    Each score covers a 70×70 receptive field of the 256×256 input when
    n_layers=3 and base_channels=64 (standard CycleGAN / CUT setting).

    Architecture (n_layers=3 example):
        Conv(stride=2, no norm) → Conv(stride=2, IN) → Conv(stride=2, IN)
        → Conv(stride=1, IN) → Conv(stride=1) → score map

    Args:
        input_nc:      Number of input image channels.
        base_channels: Feature-map width of the first conv layer.
        n_layers:      Number of strided downsampling conv layers.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            _conv_block(input_nc, base_channels, stride=2, use_norm=False)
        ]
        in_ch = base_channels
        for _ in range(1, n_layers):
            out_ch = min(in_ch * 2, base_channels * 8)
            layers.append(_conv_block(in_ch, out_ch, stride=2, use_norm=True))
            in_ch = out_ch

        out_ch = min(in_ch * 2, base_channels * 8)
        layers.append(_conv_block(in_ch, out_ch, stride=1, use_norm=True))
        in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def init_weights_v4(net: nn.Module) -> None:
    """
    Initialise network weights with GAN-friendly defaults.

    - Conv2d / ConvTranspose2d: Normal(0, 0.02)
    - InstanceNorm2d scale:     Normal(1, 0.02), bias = 0
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


def getDiscriminatorV4(
    input_nc: int = 3,
    base_channels: int = 64,
    n_layers: int = 3,
    device: torch.device | None = None,
    run_smoke_test: bool = True,
) -> PatchGANDiscriminator:
    """
    Build, initialise, and optionally smoke-test a PatchGANDiscriminator.

    Args:
        input_nc:      Input image channels.
        base_channels: Base feature-map width (default 64).
        n_layers:      Number of strided downsampling layers (default 3).
        device:        Target device; defaults to CUDA if available.
        run_smoke_test: If True, runs a single forward pass on a random
                        256×256 tensor and prints the output shape.

    Returns:
        Initialised PatchGANDiscriminator on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatchGANDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
    ).to(device)
    model.apply(init_weights_v4)

    if run_smoke_test:
        with torch.no_grad():
            x = torch.randn(1, input_nc, 256, 256, device=device)
            y = model(x)
            print(f"[getDiscriminatorV4] output shape: {y.shape}")

    return model


if __name__ == "__main__":
    _ = getDiscriminatorV4()
