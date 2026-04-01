"""
Projection discriminator for the v3 CycleDiT training pipeline.

Motivation:
- Local PatchGAN branch for texture and micro-structure.
- Global branch for whole-image structure and stain balance.
- FFT branch for periodic VAE decode artifacts.

Outputs are returned as a list of logit tensors and are compatible with the
LSGAN helpers in this repo (list inputs are averaged).

Public API:
    SpectralNormDiscriminator  -- re-exported from model_v2 for convenience.
    GlobalDiscriminatorBranch  -- full-image receptive field.
    FFTDiscriminatorBranch     -- frequency-domain branch.
    ProjectionDiscriminator    -- composite of all three branches.
    getDiscriminatorsV3        -- factory returning (D_A, D_B).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model_v2.discriminator import SpectralNormDiscriminator
from model_v1.generator import init_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sn_conv(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 4,
    stride: int = 2,
    padding: int = 1,
) -> nn.Module:
    """
    Spectral-normalized Conv2d without bias (common for discriminators).
    """
    return spectral_norm(
        nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
    )


# ---------------------------------------------------------------------------
# Branch 2 -- Global discriminator (100% receptive field)
# ---------------------------------------------------------------------------


class GlobalDiscriminatorBranch(nn.Module):
    """
    Global discriminator branch that reduces an image to a single logit.

    Shape flow for 256x256 inputs:
        (N, 3, 256, 256) -> (N, 64, 64, 64) -> (N, 128, 16, 16)
        -> (N, 256, 4, 4) -> (N, 1, 1, 1) -> (N, 1)

    Stride-4 downsampling keeps the parameter count low while preserving
    global structure. Output is returned as (N, 1).
    """

    def __init__(self, input_nc: int = 3, base_channels: int = 64):
        """
        Args:
            input_nc: Input image channels (3 for RGB).
            base_channels: Feature channels after first layer.
        """
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            # Layer 1: no IN on the input (standard PatchGAN practice)
            _sn_conv(input_nc, c, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            _sn_conv(c, c * 2, kernel_size=4, stride=4, padding=0),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            _sn_conv(c * 2, c * 4, kernel_size=4, stride=4, padding=0),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: collapse to scalar -- kernel covers the 4x4 feature map
            spectral_norm(
                nn.Conv2d(c * 4, 1, kernel_size=4, stride=1, padding=0, bias=True)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) image tensor in [-1, 1].

        Returns:
            (N, 1) scalar logit per image.
        """
        out = self.net(x)
        # out is (N, 1, 1, 1) -- flatten to (N, 1)
        return out.view(x.size(0), 1)


# ---------------------------------------------------------------------------
# Branch 3 -- FFT (frequency-domain) discriminator
# ---------------------------------------------------------------------------


class FFTDiscriminatorBranch(nn.Module):
    """
    Frequency-domain discriminator branch to catch periodic VAE artifacts.

    Steps:
        1) Convert to grayscale.
        2) rfft2 -> magnitude -> log1p.
        3) Normalize per-sample mean.
        4) Small CNN -> scalar logit.

    FFT runs in float32; the result is cast back to the module dtype
    before the CNN so AMP can be used safely.
    """

    def __init__(self, base_channels: int = 32):
        """
        Args:
            base_channels: Feature channels in the first CNN layer.
                Kept at 32 (half of the other branches) because the frequency
                map has less spatial information -- a lighter CNN is sufficient.
        """
        super().__init__()
        c = base_channels
        # Input to CNN: (N, 1, H, W//2+1)
        # The spectrum is symmetric so rfft2 output has width = W//2+1 = 129
        # for a 256x256 input.
        self.cnn = nn.Sequential(
            _sn_conv(1, c, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c, c * 2, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 2, c * 4, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 4, c * 4, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(c * 4, 1, bias=True))

    @staticmethod
    def _log_magnitude(x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude spectrum of x.

        Args:
            x: (N, 1, H, W) float32 image.

        Returns:
            (N, 1, H, W//2+1) log-magnitude map.
        """
        # rfft2 gives the non-redundant half of the spectrum
        spec = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(spec)  # (N, 1, H, W//2+1) complex->real
        return torch.log1p(mag)  # log(1 + |FFT|) -- avoids log(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, H, W) image tensor in [-1, 1].

        Returns:
            (N, 1) scalar logit per image.
        """
        # Convert to grayscale in float32 for stable FFT.
        native_dtype = x.dtype
        x_f32 = x.float()
        gray = (
            0.299 * x_f32[:, 0:1] + 0.587 * x_f32[:, 1:2] + 0.114 * x_f32[:, 2:3]
        )  # (N, 1, H, W)

        log_mag = self._log_magnitude(gray)  # (N, 1, H, W//2+1)

        # Normalize per-sample to be brightness-invariant.
        # Clamp min to avoid /0 for degenerate (all-same) inputs.
        mean_mag = log_mag.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        log_mag = log_mag / mean_mag

        # Cast back to native dtype before the CNN (supports AMP).
        log_mag = log_mag.to(native_dtype)

        feats = self.cnn(log_mag)  # (N, c*4, 8, 8) roughly
        feats = feats.mean(dim=[2, 3])  # global average pool -> (N, c*4)
        return self.head(feats)  # (N, 1)


# ---------------------------------------------------------------------------
# Composite discriminator
# ---------------------------------------------------------------------------


class ProjectionDiscriminator(nn.Module):
    """
    Three-branch discriminator suited for DiT-generated images.

    Branches:
        - Local PatchGAN: texture and local stain granularity.
        - Global branch: overall color and tissue layout.
        - FFT branch: periodic decode artifacts.

    Forward output is a list of logit tensors with mixed shapes that is
    compatible with the LSGAN helpers (list outputs are averaged).
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        global_base_channels: int = 64,
        fft_base_channels: int = 32,
        use_spectral_norm: bool = True,
        use_local: bool = True,
        use_global: bool = True,
        use_fft: bool = True,
    ):
        super().__init__()
        self.use_local = use_local
        self.use_global = use_global
        self.use_fft = use_fft

        if use_local:
            self.local_branch = SpectralNormDiscriminator(
                input_nc=input_nc,
                base_channels=base_channels,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
            )
        if use_global:
            self.global_branch = GlobalDiscriminatorBranch(
                input_nc=input_nc,
                base_channels=global_base_channels,
            )
        if use_fft:
            self.fft_branch = FFTDiscriminatorBranch(
                base_channels=fft_base_channels,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run all enabled branches and return their logit outputs as a list.

        Args:
            x: (N, 3, H, W) image in [-1, 1].

        Returns:
            List of logit tensors. Length = number of enabled branches.
            Each element is either (N, 1, H', W') or (N, 1).
        """
        outputs: list[torch.Tensor] = []
        if self.use_local:
            outputs.append(self.local_branch(x))
        if self.use_global:
            outputs.append(self.global_branch(x))
        if self.use_fft:
            outputs.append(self.fft_branch(x))
        return outputs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getDiscriminatorsV3(
    input_nc: int = 3,
    base_channels: int = 64,
    n_layers: int = 3,
    global_base_channels: int = 64,
    fft_base_channels: int = 32,
    use_spectral_norm: bool = True,
    use_local: bool = True,
    use_global: bool = True,
    use_fft: bool = True,
    device: torch.device | None = None,
) -> tuple[ProjectionDiscriminator, ProjectionDiscriminator]:
    """
    Create and initialize two ProjectionDiscriminator instances (D_A, D_B).

    Both discriminators share the same architecture but have independent
    weights. A short forward pass is run as a smoke test before returning.

    Args:
        input_nc: Input image channels.
        base_channels: Feature channels for the local PatchGAN branch.
        n_layers: Strided layers in the local branch.
        global_base_channels: Feature channels for the global branch.
        fft_base_channels: Feature channels for the spectral branch.
        use_spectral_norm: Spectral normalization on all Conv2d layers.
        use_local: Enable local PatchGAN branch.
        use_global: Enable global branch.
        use_fft: Enable spectral FFT branch.
        device: Target device. Auto-detected from CUDA availability if None.

    Returns:
        (D_A, D_B) initialized, moved to device, and set to train mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D_A = ProjectionDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        global_base_channels=global_base_channels,
        fft_base_channels=fft_base_channels,
        use_spectral_norm=use_spectral_norm,
        use_local=use_local,
        use_global=use_global,
        use_fft=use_fft,
    ).to(device)
    D_B = ProjectionDiscriminator(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        global_base_channels=global_base_channels,
        fft_base_channels=fft_base_channels,
        use_spectral_norm=use_spectral_norm,
        use_local=use_local,
        use_global=use_global,
        use_fft=use_fft,
    ).to(device)

    # init_weights handles Conv2d, Linear, and InstanceNorm2d.
    # Spectral-norm wrappers are transparent to the initializer.
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # Smoke test -- confirm shapes before returning.
    x = torch.randn(1, input_nc, 256, 256, device=device)
    with torch.no_grad():
        out_A = D_A(x)
        out_B = D_B(x)

    shapes_A = [tuple(o.shape) for o in out_A]
    shapes_B = [tuple(o.shape) for o in out_B]
    print(f"[getDiscriminatorsV3] D_A output shapes: {shapes_A}")
    print(f"[getDiscriminatorsV3] D_B output shapes: {shapes_B}")

    n_params_A = sum(p.numel() for p in D_A.parameters()) / 1e6
    print(f"[getDiscriminatorsV3] D_A params: {n_params_A:.2f}M (each)")
    n_params_B = sum(p.numel() for p in D_B.parameters()) / 1e6
    print(f"[getDiscriminatorsV3] D_B params: {n_params_B:.2f}M (each)")
    return D_A, D_B


if __name__ == "__main__":
    D_A, D_B = getDiscriminatorsV3()
    print("ProjectionDiscriminator smoke test passed.")
