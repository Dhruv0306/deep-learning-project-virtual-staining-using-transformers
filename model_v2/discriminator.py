"""
Spectral-normalised multi-scale discriminator for UVCGAN v2.

Spectral normalisation constrains the Lipschitz constant of each layer by
dividing every weight matrix by its largest singular value.  This prevents
the discriminator from growing too strong relative to the generator and is a
key stabilisation technique for modern GANs.

The multi-scale design runs independent discriminators on the original image
and progressively downsampled versions, enabling discrimination at coarse
and fine spatial frequencies.

Public API
----------
``SpectralNormDiscriminator``   – single-scale discriminator.
``MultiScaleDiscriminator``     – wraps N scales.
``getDiscriminatorsV2``         – factory mirroring ``getDiscriminators`` from v1.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from model_v1.generator import init_weights


# ---------------------------------------------------------------------------
# Single-scale spectral-norm discriminator
# ---------------------------------------------------------------------------


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
    use_norm: bool = True,
    use_spectral: bool = True,
) -> nn.Sequential:
    """
    Build one discriminator convolution block.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        use_norm: Apply InstanceNorm2d after the convolution.
        use_spectral: Wrap the convolution with spectral normalisation.

    Returns:
        ``nn.Sequential`` containing the conv (optionally spec-normed), optional
        norm, and LeakyReLU.
    """
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=not use_norm,
    )
    if use_spectral:
        conv = spectral_norm(conv)
    layers: list[nn.Module] = [conv]
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class SpectralNormDiscriminator(nn.Module):
    """
    PatchGAN discriminator with optional spectral normalisation.

    Produces a spatial map of real/fake logits where each element covers a
    receptive-field patch of the input image.

    Args:
        input_nc: Number of input channels (default 3 for RGB).
        base_channels: Feature channels after the first convolution.
        n_layers: Number of strided downsampling layers (excluding the final
            stride-1 layer and the output layer).
        use_spectral_norm: Apply spectral normalisation to every conv layer.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        """
        Initialize SpectralNormDiscriminator.

        Args:
            input_nc: Number of input channels (default 3 for RGB).
            base_channels: Feature channels after the first convolution.
            n_layers: Number of strided downsampling layers (excluding the
                final stride-1 layer and the output layer).
            use_spectral_norm: Apply spectral normalisation to every conv
                layer.
        """
        super().__init__()
        layers: list[nn.Module] = [
            # First layer – no normalisation on the input.
            _conv_block(
                input_nc,
                base_channels,
                stride=2,
                use_norm=False,
                use_spectral=use_spectral_norm,
            )
        ]

        in_ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(in_ch * 2, base_channels * 8)
            layers.append(
                _conv_block(
                    in_ch,
                    out_ch,
                    stride=2,
                    use_norm=True,
                    use_spectral=use_spectral_norm,
                )
            )
            in_ch = out_ch

        # Stride-1 layer to increase receptive field without halving resolution.
        out_ch = min(in_ch * 2, base_channels * 8)
        layers.append(
            _conv_block(
                in_ch,
                out_ch,
                stride=1,
                use_norm=True,
                use_spectral=use_spectral_norm,
            )
        )
        in_ch = out_ch

        # Output layer – no norm, no activation.
        out_conv = nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            out_conv = spectral_norm(out_conv)
        layers.append(out_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-level real/fake logits.

        Args:
            x: Input image ``(N, input_nc, H, W)``.

        Returns:
            Logit map ``(N, 1, H', W')``.
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Multi-scale discriminator
# ---------------------------------------------------------------------------


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates on the original and downsampled
    versions of the input.

    Discrimination at multiple scales helps capture both fine-grained texture
    artefacts (fine scale) and global structural inconsistencies (coarse scale).

    Args:
        input_nc: Number of input channels.
        base_channels: Feature channels at the finest scale.
        n_layers: Strided layers per individual discriminator.
        num_scales: Number of spatial scales to discriminate at.
        use_spectral_norm: Apply spectral normalisation.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        """
        Initialize MultiScaleDiscriminator.

        Args:
            input_nc: Number of input channels.
            base_channels: Feature channels at the finest scale.
            n_layers: Strided layers per individual discriminator.
            num_scales: Number of spatial scales to discriminate at.
            use_spectral_norm: Apply spectral normalisation.
        """
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList(
            [
                SpectralNormDiscriminator(
                    input_nc=input_nc,
                    base_channels=base_channels,
                    n_layers=n_layers,
                    use_spectral_norm=use_spectral_norm,
                )
                for _ in range(num_scales)
            ]
        )
        # 2× average-pool for coarser scales.
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, x: torch.Tensor):
        """
        Compute logit maps at all scales.

        Args:
            x: Input image ``(N, input_nc, H, W)``.

        Returns:
            list[torch.Tensor]: One logit map per scale (finest to coarsest).
        """
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getDiscriminatorsV2(
    input_nc: int = 3,
    base_channels: int = 64,
    n_layers: int = 3,
    num_scales: int = 3,
    use_spectral_norm: bool = True,
):
    """
    Create and initialise two multi-scale spectral-norm discriminators.

    Returns:
        tuple: ``(D_A, D_B)`` – both moved to the available device and
        weight-initialised with the standard :func:`~generator.init_weights`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        num_scales=num_scales,
    )
    D_A = MultiScaleDiscriminator(**kwargs, use_spectral_norm=use_spectral_norm).to(
        device
    )
    D_B = MultiScaleDiscriminator(**kwargs, use_spectral_norm=use_spectral_norm).to(
        device
    )

    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # Smoke test.
    x = torch.randn(1, 3, 256, 256, device=device)
    out_A = D_A(x)
    out_B = D_B(x)
    print(f"D_A (v2) output shapes: {[o.shape for o in out_A]}")
    print(f"D_B (v2) output shapes: {[o.shape for o in out_B]}")

    return D_A, D_B


if __name__ == "__main__":
    D_A, D_B = getDiscriminatorsV2()

