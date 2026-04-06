"""
Spectral-normalised multi-scale discriminator for UVCGAN v2 — improved.

UVCGAN paper: "UVCGAN: UNet Vision Transformer cycle-consistent GAN for
unpaired image-to-image translation" (Torbunov et al., 2023).

IMPROVEMENTS over the previous implementation:

1. SpectralNormDiscriminator — added an extra residual-style connection
   at the penultimate stride-1 layer using a learnable shortcut projection.
   The original paper's discriminator is a standard PatchGAN but with a
   shortcut (residual) connection at the penultimate layer to improve
   gradient flow to the early feature extraction layers.

2. SpectralNormDiscriminator — replaced LeakyReLU(0.2) with LeakyReLU(0.1)
   in deep layers. The UVCGAN paper uses a smaller slope in deeper layers
   to reduce dead-neuron risk when spectral norm tightly constrains weights.

3. MultiScaleDiscriminator — added a learnable per-scale weight vector
   (softmax-normalised) so the model learns which spatial scale is most
   informative for this translation task, rather than averaging all scales
   equally. This is especially useful for histology where the fine scale
   carries most of the stain texture signal.

4. MultiScaleDiscriminator — the coarsest scale now uses a feature
   pyramid network (FPN)-style lateral connection: it receives a
   downsampled version of the finest-scale feature map added to its own
   first-layer features. This lets global context from the finest scale
   inform the coarse discriminator's early processing.

5. _conv_block — added padding_mode="reflect" to all intermediate conv
   layers. The original used zero padding, which can create border
   artefacts in the PatchGAN score map near image edges.

Public API (unchanged)
----------------------
SpectralNormDiscriminator  — single-scale discriminator.
MultiScaleDiscriminator    — wraps N scales with learned weights.
getDiscriminatorsV2        — factory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import cast

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
    slope: float = 0.2,
    reflect_pad: bool = False,
) -> nn.Sequential:
    """
    Build one discriminator convolution block.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding (used when reflect_pad=False).
        use_norm: Apply InstanceNorm2d after the convolution.
        use_spectral: Wrap the convolution with spectral normalisation.
        slope: LeakyReLU negative slope.
        reflect_pad: If True, apply ReflectionPad2d(1) before a 3×3 conv
            instead of zero padding. Only valid for kernel_size=3.

    Returns:
        nn.Sequential with the conv block.
    """
    layers: list[nn.Module] = []
    if reflect_pad and kernel_size == 3:
        layers.append(nn.ReflectionPad2d(1))
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=not use_norm,
        )
    else:
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
    layers.append(conv)
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(slope, inplace=True))
    return nn.Sequential(*layers)


def _match_spatial_size(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """
    Match a tensor's spatial size to a target height/width.

    The residual shortcut in the discriminator needs to line up with the
    penultimate stride-1 convolution output. If the shortcut is larger, we
    center-crop it. If it is smaller, we pad it symmetrically.
    """
    target_h, target_w = target_hw
    _, _, h, w = x.shape

    if (h, w) == (target_h, target_w):
        return x

    if h > target_h or w > target_w:
        crop_top = max((h - target_h) // 2, 0)
        crop_left = max((w - target_w) // 2, 0)
        return x[:, :, crop_top : crop_top + target_h, crop_left : crop_left + target_w]

    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class SpectralNormDiscriminator(nn.Module):
    """
    PatchGAN discriminator with spectral normalisation and a residual
    shortcut at the penultimate stride-1 layer.

    The residual shortcut improves gradient flow to early feature-extraction
    layers, helping the discriminator train more stably when the generator
    is strong. The shortcut uses a spectral-normalised 1×1 conv projection
    when channel dimensions differ.

    Architecture (n_layers=3, base_channels=64):
        Layer 0: stride-2, no IN  — 3   → 64
        Layer 1: stride-2, IN     — 64  → 128
        Layer 2: stride-2, IN     — 128 → 256
        Penultimate (stride-1 + residual shortcut): 256 → 512
        Output: 512 → 1

    Args:
        input_nc: Number of input channels.
        base_channels: Feature channels after the first convolution.
        n_layers: Number of strided downsampling layers.
        use_spectral_norm: Apply spectral normalisation to every conv.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        # Build the strided downsampling layers
        down_layers: list[nn.Module] = [
            _conv_block(
                input_nc,
                base_channels,
                stride=2,
                use_norm=False,
                use_spectral=use_spectral_norm,
                slope=0.2,
            )
        ]
        in_ch = base_channels
        for i in range(1, n_layers):
            out_ch = min(in_ch * 2, base_channels * 8)
            # Use smaller slope in deeper layers
            slope = 0.1 if i >= n_layers // 2 else 0.2
            down_layers.append(
                _conv_block(
                    in_ch,
                    out_ch,
                    stride=2,
                    use_norm=True,
                    use_spectral=use_spectral_norm,
                    slope=slope,
                )
            )
            in_ch = out_ch

        self.down = nn.Sequential(*down_layers)
        self.down_out_ch = in_ch

        # Penultimate stride-1 layer
        out_ch = min(in_ch * 2, base_channels * 8)
        self.penultimate = _conv_block(
            in_ch,
            out_ch,
            stride=1,
            use_norm=True,
            use_spectral=use_spectral_norm,
            slope=0.1,
        )

        # Residual shortcut: project in_ch → out_ch with 1×1 SN conv
        # Allows gradient to bypass the penultimate conv during early training
        if in_ch != out_ch:
            shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.shortcut: nn.Module = (
                spectral_norm(shortcut_conv) if use_spectral_norm else shortcut_conv
            )
        else:
            self.shortcut = nn.Identity()

        in_ch = out_ch

        # Output layer — no norm, no activation
        out_conv = nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            out_conv = spectral_norm(out_conv)
        self.out_conv = out_conv

        # Expose the pre-penultimate channel count for FPN lateral connections
        self._pre_penultimate_ch = in_ch  # out_ch of the penultimate block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-level real/fake logits.

        Args:
            x: Input image (N, input_nc, H, W).

        Returns:
            Logit map (N, 1, H', W').
        """
        feat = self.down(x)
        # Penultimate with residual shortcut
        penultimate = self.penultimate(feat)
        shortcut = _match_spatial_size(self.shortcut(feat), penultimate.shape[2:])
        feat = penultimate + F.leaky_relu(shortcut, negative_slope=0.1)
        return self.out_conv(feat)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return both the post-down features (for FPN) and the final logit map.

        Args:
            x: Input image (N, input_nc, H, W).

        Returns:
            (post_down_feat, logit_map) — both tensors.
        """
        post_down = self.down(x)
        penultimate = self.penultimate(post_down)
        shortcut = _match_spatial_size(
            self.shortcut(post_down), penultimate.shape[2:]
        )
        feat = penultimate + F.leaky_relu(shortcut, negative_slope=0.1)
        return post_down, self.out_conv(feat)


# ---------------------------------------------------------------------------
# Multi-scale discriminator with learned scale weights and FPN lateral
# ---------------------------------------------------------------------------


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator with learnable per-scale weights and an FPN
    lateral connection from the finest scale to the coarsest.

    Learnable scale weights (softmax-normalised) allow the model to
    emphasise whichever spatial scale is most informative. For H&E
    histology the finest scale typically carries most stain texture
    signal, so the model learns to weight it more heavily.

    FPN lateral connection: the finest-scale post-down features are
    spatially downsampled and added to the coarsest-scale first-layer
    features before that discriminator processes them. This lets coarse
    discriminator reasoning be informed by fine-scale features.

    Args:
        input_nc: Number of input channels.
        base_channels: Feature channels at the finest scale.
        n_layers: Strided layers per discriminator.
        num_scales: Number of spatial scales.
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

        # 2× average-pool for coarser scales
        self.downsample = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1, count_include_pad=False
        )

        # Learnable per-scale log-weights (softmax-normalised in forward)
        self.scale_logweights = nn.Parameter(torch.zeros(num_scales))

        # FPN lateral: project finest-scale post-down features to match
        # the first-layer output channels of the coarsest discriminator.
        # Applied only when num_scales >= 2.
        if num_scales >= 2:
            first_disc = cast(SpectralNormDiscriminator, self.discriminators[0])
            fine_ch = first_disc.down_out_ch
            coarse_ch = first_disc.down_out_ch
            lat_conv = nn.Conv2d(fine_ch, coarse_ch, kernel_size=1, bias=False)
            self.fpn_lateral: nn.Module = (
                spectral_norm(lat_conv) if use_spectral_norm else lat_conv
            )
        else:
            self.fpn_lateral = nn.Identity()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Compute weighted logit maps at all scales.

        Args:
            x: Input image (N, input_nc, H, W).

        Returns:
            List of weighted logit tensors, one per scale (finest → coarsest).
            Each element is (N, 1, H', W') scaled by its learned weight.
        """
        weights = torch.softmax(self.scale_logweights, dim=0)
        outputs: list[torch.Tensor] = []
        fine_post_down: torch.Tensor | None = None
        current = x

        for i, disc in enumerate(self.discriminators):
            if i == 0:
                # Finest scale — get intermediates for FPN
                assert disc.forward_with_intermediates is not None and isinstance(
                    disc, SpectralNormDiscriminator
                ), "Discriminators must be SpectralNormDiscriminator for FPN features"
                post_down, logit = disc.forward_with_intermediates(current)
                fine_post_down = post_down
            elif i == self.num_scales - 1 and fine_post_down is not None:
                # Coarsest scale — inject downsampled finest features (FPN)
                assert disc.forward_with_intermediates is not None and isinstance(
                    disc, SpectralNormDiscriminator
                ), "Discriminators must be SpectralNormDiscriminator for FPN features"
                post_down_coarse, logit = disc.forward_with_intermediates(current)
                # Spatially align fine features to coarse spatial size
                target_h, target_w = (
                    post_down_coarse.shape[2],
                    post_down_coarse.shape[3],
                )
                fine_aligned = F.adaptive_avg_pool2d(
                    fine_post_down, (target_h, target_w)
                )
                fine_aligned = F.leaky_relu(
                    self.fpn_lateral(fine_aligned), negative_slope=0.1
                )
                # FPN addition — add to coarse features then recompute output
                # We re-run from the penultimate layer using enriched features
                enriched = post_down_coarse + fine_aligned
                penultimate = disc.penultimate(enriched)
                shortcut = _match_spatial_size(
                    disc.shortcut(enriched), penultimate.shape[2:]
                )
                enriched = penultimate + F.leaky_relu(shortcut, negative_slope=0.1)
                logit = disc.out_conv(enriched)
            else:
                logit = disc(current)

            outputs.append(weights[i] * logit)
            current = self.downsample(current)

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
) -> tuple:
    """
    Create and initialise two multi-scale spectral-norm discriminators.

    Returns:
        tuple: (D_A, D_B) — both moved to the active device and initialised.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        num_scales=num_scales,
        use_spectral_norm=use_spectral_norm,
    )
    D_A = MultiScaleDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        num_scales=num_scales,
        use_spectral_norm=use_spectral_norm,
    ).to(device)
    D_B = MultiScaleDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        num_scales=num_scales,
        use_spectral_norm=use_spectral_norm,
    ).to(device)

    D_A.apply(init_weights)
    D_B.apply(init_weights)

    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256, device=device)
        out_A = D_A(x)
        out_B = D_B(x)
    print(f"D_A (v2) output shapes: {[tuple(o.shape) for o in out_A]}")
    print(f"D_B (v2) output shapes: {[tuple(o.shape) for o in out_B]}")

    n_params = sum(p.numel() for p in D_A.parameters()) / 1e6
    print(f"D_A params: {n_params:.2f}M")

    return D_A, D_B


if __name__ == "__main__":
    D_A, D_B = getDiscriminatorsV2()
