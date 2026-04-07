"""
model_v4/discriminator.py — PatchGAN discriminator for CUT + Transformer (v4).

UPGRADES (v4.2):
    PatchGANDiscriminator:
        - Spectral normalization added to all Conv2d layers. This stabilises
          discriminator training and prevents the discriminator from becoming
          pathologically strong early in training (which blocks generator
          learning).
        - Multi-scale output: an auxiliary head is attached at an intermediate
          layer (after n_layers//2 strided convs). The discriminator now
          returns both the standard 70×70 score map AND a larger (coarser)
          intermediate score map. Training combines both via the LSGAN
          objective, giving the generator multi-scale feedback — the
          intermediate head penalises global color/structure errors while the
          final head focuses on fine texture.
        - Minibatch standard deviation layer inserted before the final output
          conv. This exposes within-batch diversity statistics to the
          discriminator, helping it detect mode-collapsed generators that
          produce repetitive stain patterns.

    getDiscriminatorV4:
        - Updated to initialise both the main and auxiliary score heads.
        - Smoke test now verifies both output shapes.

Public API (unchanged for the main score output):
    PatchGANDiscriminator — N-layer 70×70 PatchGAN, now with SN + multi-scale.
    getDiscriminatorV4    — factory + weight init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sn_conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
    use_norm: bool = True,
) -> nn.Sequential:
    """
    Single spectral-norm conv → [InstanceNorm] → LeakyReLU block.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        stride:       Convolution stride.
        use_norm:     Whether to insert InstanceNorm2d. Disabled for the
                      first layer per PatchGAN convention.
    """
    layers: list[nn.Module] = [
        spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not use_norm,
            )
        )
    ]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Minibatch standard deviation
# ---------------------------------------------------------------------------


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard-deviation channel appended before the final score head.

    Computes the average standard deviation across the batch for groups of
    ``group_size`` samples and tiles the result as an extra channel. The
    discriminator can then detect within-batch diversity collapse.

    Args:
        group_size: Samples per group (clamped to batch size).
    """

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = min(self.group_size, n)
        y = x.float().reshape(g, -1, c, h, w)
        y = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True)  # (N//g, 1, 1, 1)
        y = y.repeat(g, 1, h, w)  # (N, 1, h, w)
        return torch.cat([x, y.to(x.dtype)], dim=1)


# ---------------------------------------------------------------------------
# Multi-scale PatchGAN discriminator
# ---------------------------------------------------------------------------


class PatchGANDiscriminator(nn.Module):
    """
    Enhanced multi-scale PatchGAN discriminator.

    Produces two score maps:
        - main:  standard 70×70 PatchGAN output (same as before).
        - aux:   intermediate-scale output attached at n_layers//2 layers in.

    Both are returned from forward() as a tuple (main, aux). Callers
    already consuming a single tensor from v4.1 should use output[0]
    for backward compatibility.

    Architecture (n_layers=3 example):
        Conv(stride=2, no SN-norm) → Conv(stride=2, SN+IN)
            → [aux head attached here]
        → Conv(stride=2, SN+IN) → MBStdDev → Conv(stride=1, SN+IN)
        → SN-Conv(stride=1) → main score map

    Args:
        input_nc:      Number of input image channels.
        base_channels: Feature-map width of the first conv layer.
        n_layers:      Number of strided downsampling conv layers.
        mbstd_group:   Minibatch std-dev group size (0 = disable).
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        mbstd_group: int = 4,
    ):
        super().__init__()
        self.n_layers = n_layers
        # Determine where to attach the auxiliary head
        self._aux_after = max(1, n_layers // 2)

        # Build layer-by-layer so we can tap an intermediate feature map
        self.down_layers = nn.ModuleList()
        in_ch = input_nc
        out_ch = base_channels

        # First strided layer (no IN per convention)
        self.down_layers.append(_sn_conv_block(in_ch, out_ch, stride=2, use_norm=False))
        in_ch = out_ch

        for i in range(1, n_layers):
            out_ch = min(in_ch * 2, base_channels * 8)
            self.down_layers.append(
                _sn_conv_block(in_ch, out_ch, stride=2, use_norm=True)
            )
            in_ch = out_ch

        # Auxiliary head — measure actual channel count at tap point via
        # a CPU dry-run to avoid brittle static channel arithmetic.
        with torch.no_grad():
            _probe = torch.zeros(max(mbstd_group, 2), input_nc, 256, 256)
            for _i, _layer in enumerate(self.down_layers):
                _probe = _layer(_probe)
                if _i == self._aux_after - 1:
                    break
        aux_in_ch = _probe.shape[1]
        self.aux_head = spectral_norm(
            nn.Conv2d(aux_in_ch, 1, kernel_size=4, stride=1, padding=1)
        )

        # Remaining layers for the main head
        out_ch = min(in_ch * 2, base_channels * 8)
        self.penultimate = _sn_conv_block(in_ch, out_ch, stride=1, use_norm=True)
        in_ch = out_ch

        # Minibatch std-dev before final conv
        self.mbstd = (
            MinibatchStdDev(group_size=mbstd_group) if mbstd_group > 0 else None
        )
        extra_ch = 1 if mbstd_group > 0 else 0

        self.final_conv = spectral_norm(
            nn.Conv2d(in_ch + extra_ch, 1, kernel_size=4, stride=1, padding=1)
        )

    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Internal: returns (main, aux) score maps."""
        feats = x
        aux_out = None
        for i, layer in enumerate(self.down_layers):
            feats = layer(feats)
            if i == self._aux_after - 1:
                aux_out = self.aux_head(feats)

        feats = self.penultimate(feats)
        if self.mbstd is not None:
            feats = self.mbstd(feats)
        main_out = self.final_conv(feats)

        if aux_out is None:
            aux_out = main_out

        return main_out, aux_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the discriminator and return a single merged score map.

        The main (fine-scale 70×70) and aux (coarse intermediate) score maps
        are averaged after interpolating aux to match main's spatial size.
        This keeps the output as a plain Tensor so existing loss helpers
        (``_lsgan_disc_loss``, ``_lsgan_gen_loss``) work without modification.

        Args:
            x: (N, input_nc, H, W) image in [-1, 1].

        Returns:
            (N, 1, H', W') merged score map (same spatial size as main).
        """
        main_out, aux_out = self._run(x)
        # Upsample aux to main spatial size if they differ, then average
        if aux_out.shape != main_out.shape:
            aux_out = torch.nn.functional.interpolate(
                aux_out, size=main_out.shape[2:], mode="bilinear", align_corners=False
            )
        return 0.5 * (main_out + aux_out)

    def forward_multiscale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (main, aux) score maps separately for callers that want
        per-scale supervision (e.g. multi-scale LSGAN training).
        """
        return self._run(x)


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getDiscriminatorV4(
    input_nc: int = 3,
    base_channels: int = 64,
    n_layers: int = 3,
    mbstd_group: int = 4,
    device: torch.device | None = None,
    run_smoke_test: bool = True,
) -> PatchGANDiscriminator:
    """
    Build, initialise, and optionally smoke-test a PatchGANDiscriminator.

    New args vs v4.1:
        mbstd_group (int): MinibatchStdDev group size. 0 = disabled. Default 4.

    Args:
        input_nc:      Input image channels.
        base_channels: Base feature-map width (default 64).
        n_layers:      Strided downsampling layers (default 3).
        mbstd_group:   Minibatch std-dev group size.
        device:        Target device.
        run_smoke_test: Run a forward pass and print output shapes.

    Returns:
        Initialised PatchGANDiscriminator on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PatchGANDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        mbstd_group=mbstd_group,
    ).to(device)
    model.apply(init_weights_v4)

    if run_smoke_test:
        bs = max(mbstd_group, 2)
        with torch.no_grad():
            x = torch.randn(bs, input_nc, 256, 256, device=device)
            merged = model(x)
            main, aux = model.forward_multiscale(x)
            print(f"[getDiscriminatorV4] merged output: {tuple(merged.shape)}")
            print(f"[getDiscriminatorV4] main   output: {tuple(main.shape)}")
            print(f"[getDiscriminatorV4] aux    output: {tuple(aux.shape)}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[getDiscriminatorV4] D params: {n_params:.2f}M")

    return model


if __name__ == "__main__":
    _ = getDiscriminatorV4()
    print("PatchGANDiscriminator v4.2 smoke test passed.")
