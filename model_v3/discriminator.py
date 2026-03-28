"""
Discriminator for the v3 CycleGAN-DiT training pipeline.

Why not just reuse the v2 MultiScaleDiscriminator?
---------------------------------------------------
The v2 discriminator is a multi-scale PatchGAN, designed to judge CNN-generator
outputs. It works by stacking strided convolutions that build up a 70×70
receptive field at the finest scale. For DiT-generated images, three failure
modes escape this architecture:

1. Global structure inconsistency.
   DiT generators can produce outputs that look locally correct (each patch
   passes the PatchGAN) but are globally incoherent — wrong overall colour
   cast, mismatched stain distribution across the slide, or wrong tissue
   morphology at the macro scale. A 70×70 receptive field on a 256×256 image
   covers only 7% of the image area. Adding coarser PatchGAN scales helps,
   but a dedicated global branch is more parameter-efficient and direct.

2. VAE decode artifacts.
   The frozen SD VAE introduces mild high-frequency ringing at the boundary
   of its 8× compression tiles. These artifacts are spatially periodic at a
   fixed frequency determined by the VAE architecture, not by the content.
   Standard PatchGAN is blind to periodic spatial frequency patterns because
   it has no frequency-domain sensitivity. A lightweight spectral (FFT) branch
   catches these artifacts directly in the frequency domain.

3. Over-smoothness from early diffusion training.
   Diffusion models are notoriously over-smooth in early training (they
   minimise MSE in latent space, which tends toward blurry averages). A
   PatchGAN that mainly sees high-frequency texture will give weak gradient
   signal to the generator in this regime because blurry images fool it.
   The global branch provides a complementary low-frequency signal.

Architecture: ProjectionDiscriminator
--------------------------------------
Three parallel branches, each producing a scalar or spatial logit map.
Their outputs are summed before the final LSGAN loss.

Branch 1 — Local (PatchGAN with spectral norm):
    Reuses SpectralNormDiscriminator from model_v2.discriminator.
    Receptive field ~70×70 at the finest scale (n_layers=3).
    Captures local texture, stain granularity, fine-grained artifacts.
    Output: (N, 1, H', W') spatial logit map.

Branch 2 — Global (lightweight CNN → MLP):
    4-layer strided CNN reduces 256×256 to 4×4, then flattens to a single
    scalar logit per image. This gives a 256×256 = 100% receptive field,
    forcing the discriminator to judge the entire image globally.
    Uses spectral normalisation on all layers.
    Output: (N, 1) scalar logit.

Branch 3 — Spectral (FFT magnitude → MLP):
    Computes the 2D FFT magnitude of the input image (log-scaled),
    then runs a 4-layer CNN on the frequency map to detect periodic
    artifacts introduced by the VAE decode step.
    This branch operates on the magnitude spectrum, which is invariant to
    spatial shifts and isolates periodic patterns.
    Output: (N, 1) scalar logit.

All three branches are trained jointly under the same LSGAN objective.
The LSGAN loss function (already in model_v2/losses.py) handles list/scalar
outputs by averaging across elements — so the multi-output format is
compatible with the existing UVCGANLoss._lsgan_disc_loss and
_lsgan_gen_loss without any changes.

Integration with existing loss infrastructure:
    discriminator_loss(D, real, fake) in UVCGANLoss already handles
    multi-output discriminators (any Tensor or list of Tensors).
    No changes needed in losses.py.

VRAM note:
    The spectral branch runs in float32 unconditionally (FFT is numerical).
    This adds negligible VRAM — the frequency map is (N, 1, H, W/2+1) after
    rfft2 and is immediately downsampled by the CNN. Total discriminator VRAM
    is slightly higher than v2 MultiScaleDiscriminator with 2 scales but lower
    than 3 scales.

Public API
----------
    SpectralNormDiscriminator  -- re-exported from model_v2 for convenience.
    GlobalDiscriminatorBranch  -- receptive field = full image.
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
    Spectral-normalised Conv2d, no bias (standard for discriminators with norm).
    """
    return spectral_norm(
        nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
    )


# ---------------------------------------------------------------------------
# Branch 2 — Global discriminator (100% receptive field)
# ---------------------------------------------------------------------------


class GlobalDiscriminatorBranch(nn.Module):
    """
    Lightweight CNN that collapses a 256×256 image to a single scalar logit.

    Architecture:
        Input: (N, 3, 256, 256)
        Conv(3→64, stride=4)      → (N, 64,  64, 64)
        LeakyReLU
        Conv(64→128, stride=4)    → (N, 128, 16, 16)
        IN + LeakyReLU
        Conv(128→256, stride=4)   → (N, 256,  4,  4)
        IN + LeakyReLU
        Conv(256→1, stride=1, p=0)→ (N,   1,  1,  1)

    All convolutions are spectral-normalised. Strides of 4 are used to
    reach a 1×1 spatial output in fewer layers, keeping parameter count low.
    The aggressive downsampling is intentional: global structure (overall
    colour distribution, tissue layout) is preserved; local texture is not.
    The discriminator therefore forces the generator to match global
    statistics, complementing the local PatchGAN branch.

    Output shape: (N, 1) — a scalar logit per image (squeeze applied).
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
            # Output: collapse to scalar — kernel covers the 4×4 feature map
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
        # out is (N, 1, 1, 1) — flatten to (N, 1)
        return out.view(x.size(0), 1)


# ---------------------------------------------------------------------------
# Branch 3 — FFT (frequency-domain) discriminator
# ---------------------------------------------------------------------------


class FFTDiscriminatorBranch(nn.Module):
    """
    Discriminates in the frequency domain to catch periodic VAE decode artifacts.

    The VAE encodes images to a 32×32×4 latent and decodes them back,
    introducing mild periodic ringing at the spatial frequency corresponding
    to the 8× compression factor. These artifacts are subtle in pixel space
    but prominent in the magnitude spectrum.

    Forward pass:
        1. Convert input to grayscale (to reduce channels while preserving
           luminance structure, which contains most frequency information).
        2. Compute 2D real FFT: (N, 1, H, W) → complex (N, 1, H, W//2+1)
        3. Log-magnitude: log(1 + |FFT|) → real (N, 1, H, W//2+1)
        4. Normalise by mean (stabilises training across different image
           brightness levels).
        5. Treat the log-magnitude map as an image and run a small CNN
           to produce a scalar logit.

    The CNN in step 5 uses the same spectral-norm pattern as the other
    branches for Lipschitz continuity.

    Output shape: (N, 1) scalar logit.

    Note: FFT is always computed in float32 regardless of AMP state.
    The input is cast to float32 before the FFT and the result is cast
    back to the discriminator's native dtype before the CNN. This is
    consistent with the convention used for the gradient penalty in
    model_v2/losses.py.
    """

    def __init__(self, base_channels: int = 32):
        """
        Args:
            base_channels: Feature channels in the first CNN layer.
                Kept at 32 (half of the other branches) because the frequency
                map has less spatial information — a lighter CNN is sufficient.
        """
        super().__init__()
        c = base_channels
        # Input to CNN: (N, 1, H, W//2+1)
        # The spectrum is symmetric so rfft2 output has width = W//2+1 = 129
        # for a 256×256 input.
        self.cnn = nn.Sequential(
            _sn_conv(1, c, kernel_size=4, stride=2, padding=1),  # → 64×64
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c, c * 2, kernel_size=4, stride=2, padding=1),  # → 32×32
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 2, c * 4, kernel_size=4, stride=2, padding=1),  # → 16×16
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 4, c * 4, kernel_size=4, stride=2, padding=1),  # → 8×8
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
        mag = torch.abs(spec)  # (N, 1, H, W//2+1) complex→real
        return torch.log1p(mag)  # log(1 + |FFT|) — avoids log(0)

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

        # Normalise per-sample to be brightness-invariant.
        # Clamp min to avoid /0 for degenerate (all-same) inputs.
        mean_mag = log_mag.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        log_mag = log_mag / mean_mag

        # Cast back to native dtype before the CNN (supports AMP).
        log_mag = log_mag.to(native_dtype)

        feats = self.cnn(log_mag)  # (N, c*4, 8, 8) roughly
        feats = feats.mean(dim=[2, 3])  # global average pool → (N, c*4)
        return self.head(feats)  # (N, 1)


# ---------------------------------------------------------------------------
# Composite discriminator
# ---------------------------------------------------------------------------


class ProjectionDiscriminator(nn.Module):
    """
    Three-branch discriminator suited for DiT-generated images.

    The three branches are designed to be complementary:

    - Local branch (PatchGAN):   fine texture, local stain granularity
    - Global branch (scalar):    overall colour, global tissue layout
    - Spectral branch (FFT):     periodic VAE decode artifacts

    Each branch produces either a spatial logit map (local) or a scalar
    logit per image (global, spectral). The existing UVCGANLoss handles
    both formats correctly: _lsgan_disc_loss and _lsgan_gen_loss accept
    a list of tensors with mixed shapes by averaging per-tensor MSE losses.

    Forward output: list[Tensor] of length 3
        [local_map (N,1,H',W'), global_scalar (N,1), spectral_scalar (N,1)]

    The caller (UVCGANLoss.discriminator_loss and generator_loss) passes the
    output directly to _lsgan_disc_loss / _lsgan_gen_loss, which already
    handles list inputs by averaging MSE over all elements in the list.
    No changes needed in the loss module.

    Args:
        input_nc: Input image channels (default 3).
        base_channels: Feature channels for the local branch.
        n_layers: Strided layers in the local PatchGAN branch.
        global_base_channels: Feature channels for the global branch.
        fft_base_channels: Feature channels for the spectral branch.
        use_spectral_norm: Apply spectral norm to all branches.
        use_local: Enable the local PatchGAN branch (default True).
        use_global: Enable the global branch (default True).
        use_fft: Enable the spectral FFT branch (default True).
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

        The list format is compatible with UVCGANLoss._lsgan_disc_loss and
        _lsgan_gen_loss, which already iterate over list outputs and average
        the per-element MSE losses. This means zero changes are required in
        the existing loss module.

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
    Create and initialise two ProjectionDiscriminator instances (D_A and D_B).

    Both discriminators share the same architecture but have independent
    weights — D_A judges the unstained domain, D_B judges the stained domain.

    Weight initialisation uses the same ``init_weights`` function as v1/v2
    (Normal(0, 0.02) for Conv2d, Xavier for Linear, Normal(1, 0.02) for IN).
    The spectral-norm wrapper is applied *before* initialisation, which is
    the correct order: spectral_norm wraps the weight tensor and computes the
    normalisation on the initialised values.

    Smoke test:
        A single (1, 3, 256, 256) random tensor is passed through both
        discriminators to confirm output shapes before returning.

    Args:
        input_nc: Input image channels.
        base_channels: Feature channels for the local PatchGAN branch.
        n_layers: Strided layers in the local branch.
        global_base_channels: Feature channels for the global branch.
        fft_base_channels: Feature channels for the spectral branch.
        use_spectral_norm: Spectral normalisation on all Conv2d layers.
        use_local: Enable local PatchGAN branch.
        use_global: Enable global branch.
        use_fft: Enable spectral FFT branch.
        device: Target device. Auto-detected from CUDA availability if None.

    Returns:
        tuple: (D_A, D_B) — both initialised, moved to device, in train mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        input_nc=input_nc,
        base_channels=base_channels,
        n_layers=n_layers,
        global_base_channels=global_base_channels,
        fft_base_channels=fft_base_channels,
        use_spectral_norm=use_spectral_norm,
        use_local=use_local,
        use_global=use_global,
        use_fft=use_fft,
    )

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
    # Spectral-norm wrappers are transparent to the initialiser.
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # Smoke test — confirm shapes before returning.
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
