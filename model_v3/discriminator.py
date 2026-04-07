"""
Projection discriminator for the v3 CycleDiT training pipeline (v3.2).

Three complementary branches cover different frequency bands and spatial
scales of the H&E staining signal:

    LocalPatchBranchWithMBStd  -- spectral-norm PatchGAN + MinibatchStdDev;
                                  targets mid-frequency stain texture and
                                  penalises mode-collapsed generators.
    GlobalDiscriminatorBranch  -- stride-4 CNN + self-attention on 4×4 tokens;
                                  captures full-image color balance and tissue
                                  layout.
    FFTDiscriminatorBranch     -- log-magnitude spectra (gray + R/G/B);
                                  detects periodic decode artifacts and
                                  per-channel stain imbalance.

Public API:
    SpectralNormDiscriminator  -- re-exported from model_v2.
    MinibatchStdDev            -- ProGAN diversity layer.
    LocalPatchBranchWithMBStd  -- Branch 1.
    GlobalDiscriminatorBranch  -- Branch 2.
    FFTDiscriminatorBranch     -- Branch 3.
    ProjectionDiscriminator    -- composite discriminator with learnable
                                  softmax branch weights.
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
    """Spectral-normalized Conv2d without bias."""
    return spectral_norm(
        nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
    )


# ---------------------------------------------------------------------------
# Minibatch standard deviation (ProGAN / StyleGAN trick)
# ---------------------------------------------------------------------------


class MinibatchStdDev(nn.Module):
    """
    Appends a minibatch standard-deviation feature map to the input.

    Computes the mean std-dev across the batch for a group of ``group_size``
    samples, then tiles it as an extra channel. This gives the discriminator
    a signal about within-batch diversity, penalising mode-dropped generators.

    Args:
        group_size: Samples per group. Clamped to min(group_size, N).
        num_features: Number of summary statistics appended (1 in original).
    """

    def __init__(self, group_size: int = 4, num_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        g = min(self.group_size, n)  # actual group size (clamped to batch)
        f = self.num_features
        # Split batch into groups: (g, n//g, f, c//f, h, w)
        y = x.reshape(g, -1, f, c // f, h, w).float()
        # Std over the group dimension → (n//g, f, c//f, h, w)
        y = torch.sqrt(y.var(dim=0, unbiased=False) + 1e-8)
        # Collapse feature/spatial dims to one scalar per (group, f): (n//g, f, 1, 1)
        y = y.mean(dim=[2, 3, 4], keepdim=True).squeeze(2)
        y = y.repeat(g, 1, h, w)  # tile back to full batch: (n, f, h, w)
        return torch.cat([x, y.to(x.dtype)], dim=1)


# ---------------------------------------------------------------------------
# Branch 1 — enhanced local PatchGAN (wrapped via SpectralNormDiscriminator)
# is handled by model_v2.discriminator.SpectralNormDiscriminator directly.
# We only add MinibatchStdDev as a wrapper here.
# ---------------------------------------------------------------------------


class LocalPatchBranchWithMBStd(nn.Module):
    """
    Wraps SpectralNormDiscriminator with a MinibatchStdDev layer inserted
    before the final scoring convolution to penalise texture repetition.

    Args:
        input_nc:        Input channels.
        base_channels:   Feature width of the first conv.
        n_layers:        Strided downsampling conv layers.
        use_spectral_norm: Spectral norm on all convolutions.
        group_size:      Minibatch std-dev group size.
    """

    def __init__(
        self,
        input_nc: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        group_size: int = 4,
    ):
        super().__init__()
        self.body = SpectralNormDiscriminator(
            input_nc=input_nc,
            base_channels=base_channels,
            n_layers=n_layers,
            use_spectral_norm=use_spectral_norm,
        )
        self.mbstd = MinibatchStdDev(group_size=group_size)

        # Probe the body (down → penultimate) on CPU to get the true channel
        # count before the final conv, without brittle static arithmetic.
        # Compatible with both SpectralNormDiscriminator layouts
        # (legacy self.model Sequential and current down/penultimate split).
        with torch.inference_mode():
            probe = torch.zeros(max(group_size, 2), input_nc, 256, 256)
            probe = self.body.down(probe)
            probe = self.body.penultimate(probe)
        actual_ch = probe.shape[1]
        extra = 1  # MinibatchStdDev appends num_features=1 channel
        self.final_conv = spectral_norm(
            nn.Conv2d(
                actual_ch + extra, 1, kernel_size=4, stride=1, padding=1, bias=True
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # down → penultimate, bypassing body.out_conv so we can insert MBStd.
        # The shortcut branch is intentionally skipped: its spatial size differs
        # from penultimate (4×4 kernel shrinks by 1 px), causing a shape mismatch.
        feats = self.body.down(x)
        feats = self.body.penultimate(feats)
        feats = self.mbstd(feats)  # appends 1 diversity channel
        return self.final_conv(feats)  # → patch logit map


# ---------------------------------------------------------------------------
# Branch 2 — Global discriminator with self-attention on feature map
# ---------------------------------------------------------------------------


class GlobalDiscriminatorBranch(nn.Module):
    """
    Global discriminator branch: full-image receptive field.

    Enhancement: a lightweight dot-product self-attention layer is applied
    on the flattened 4×4 feature map (16 tokens) before the final score
    convolution. This allows the discriminator to reason about long-range
    spatial co-occurrences in the global feature map (e.g., stain balance
    across the tissue).

    Shape flow for 256×256 inputs::

        (N,   3, 256, 256)  stride-4 conv + LReLU
        (N,  64,  64,  64)  stride-4 conv + IN + LReLU
        (N, 128,  16,  16)  stride-4 conv + IN + LReLU
        (N, 256,   4,   4)  flatten → (N, 16, 256) tokens
                            MultiheadAttention(heads=4) + LayerNorm
                            reshape → (N, 256, 4, 4)
        (N,   1,   1,   1)  4×4 conv head  →  view  →  (N, 1)
    """

    def __init__(self, input_nc: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            _sn_conv(input_nc, c, kernel_size=4, stride=4, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c, c * 2, kernel_size=4, stride=4, padding=0),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 2, c * 4, kernel_size=4, stride=4, padding=0),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Self-attention over 4×4 = 16 spatial tokens (global layout reasoning)
        self.attn = nn.MultiheadAttention(c * 4, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(c * 4)
        # Final score head
        self.head = spectral_norm(
            nn.Conv2d(c * 4, 1, kernel_size=4, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)  # (N, C, 4, 4)
        n, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (N, 16, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)  # residual + LN
        feat = tokens.transpose(1, 2).reshape(n, c, h, w)  # (N, C, 4, 4)
        out = self.head(feat)  # (N, 1, 1, 1)
        return out.view(n, 1)


# ---------------------------------------------------------------------------
# Branch 3 — Color-aware FFT discriminator
# ---------------------------------------------------------------------------


class FFTDiscriminatorBranch(nn.Module):
    """
    Frequency-domain discriminator branch — now color-aware.

    Enhancement: In addition to the grayscale magnitude spectrum (original),
    we compute per-channel (R, G, B) FFT magnitudes and concatenate them.
    Staining artifacts from H&E processing often differ strongly between
    the hematoxylin (blue) and eosin (pink/red) channels, so a 4-channel
    (gray + 3 color) input gives the discriminator more discriminative power.

    Steps:
        1) Grayscale + per-channel rfft2 -> magnitude -> log1p (4 channels).
        2) Normalize per-sample.
        3) Lightweight CNN -> scalar logit.

    Args:
        base_channels: Feature channels (kept modest for efficiency).
    """

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        in_ch = 4  # grayscale + 3 color channels
        self.cnn = nn.Sequential(
            _sn_conv(in_ch, c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            _sn_conv(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(c * 4, 1, bias=True))

    @staticmethod
    def _log_magnitude(x: torch.Tensor) -> torch.Tensor:
        """
        Compute log-magnitude spectrum along the last two dims.

        Args:
            x: (N, C, H, W) float32.

        Returns:
            (N, C, H, W//2+1) log-magnitude.
        """
        spec = torch.fft.rfft2(x, norm="ortho")
        return torch.log1p(torch.abs(spec))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        native_dtype = x.dtype
        x_f32 = x.float()

        # Grayscale channel
        gray = 0.299 * x_f32[:, 0:1] + 0.587 * x_f32[:, 1:2] + 0.114 * x_f32[:, 2:3]
        # Per-channel FFT magnitudes
        log_gray = self._log_magnitude(gray)  # (N, 1, H, W//2+1)
        log_r = self._log_magnitude(x_f32[:, 0:1])  # (N, 1, H, W//2+1)
        log_g = self._log_magnitude(x_f32[:, 1:2])
        log_b = self._log_magnitude(x_f32[:, 2:3])

        log_mag = torch.cat([log_gray, log_r, log_g, log_b], dim=1)  # (N, 4, H, W//2+1)

        # Normalize per-sample
        mean_mag = log_mag.mean(dim=[2, 3], keepdim=True).clamp(min=1e-6)
        log_mag = (log_mag / mean_mag).to(native_dtype)

        feats = self.cnn(log_mag)  # (N, c*4, ~8, ~8)
        feats = feats.mean(dim=[2, 3])  # (N, c*4)
        return self.head(feats)  # (N, 1)


# ---------------------------------------------------------------------------
# Composite discriminator with learnable branch weights
# ---------------------------------------------------------------------------


class ProjectionDiscriminator(nn.Module):
    """
    Three-branch discriminator for DiT-generated H&E histology images.

    Branches (each individually toggleable):
        local  -- LocalPatchBranchWithMBStd: mid-frequency texture + diversity.
        global -- GlobalDiscriminatorBranch: full-image color balance + layout.
        fft    -- FFTDiscriminatorBranch: periodic artifacts + channel imbalance.

    ``branch_logweights`` (nn.Parameter, shape (n_branches,)) are
    softmax-normalised at forward time, letting training automatically
    up-weight the most informative branch.  All three are enabled by default.

    Returns a list of weighted logit tensors (one per enabled branch).
    The LSGAN helpers sum/average across the list, making this equivalent
    to a learned weighted-average adversarial loss.
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
        mbstd_group_size: int = 4,
    ):
        super().__init__()
        self.use_local = use_local
        self.use_global = use_global
        self.use_fft = use_fft

        # Conditionally instantiate branches so disabled ones add no parameters.
        if use_local:
            self.local_branch = LocalPatchBranchWithMBStd(
                input_nc=input_nc,
                base_channels=base_channels,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
                group_size=mbstd_group_size,
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

        # Learnable branch weights; zeros → equal softmax weighting at init.
        n_branches = int(use_local) + int(use_global) + int(use_fft)
        self.branch_logweights = nn.Parameter(torch.zeros(n_branches))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Run all enabled branches and return their logit outputs.

        The list is weighted by softmax(branch_logweights) — each logit
        tensor is scaled by its branch weight before being returned.
        The LSGAN helpers average across the list, so the weighting is
        equivalent to a weighted average loss.

        Args:
            x: (N, 3, H, W) image in [-1, 1].

        Returns:
            List of weighted logit tensors.
        """
        raw: list[torch.Tensor] = []
        if self.use_local:
            raw.append(self.local_branch(x))
        if self.use_global:
            raw.append(self.global_branch(x))
        if self.use_fft:
            raw.append(self.fft_branch(x))

        weights = torch.softmax(self.branch_logweights, dim=0)  # sums to 1
        return [w * out for w, out in zip(weights, raw)]


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
    mbstd_group_size: int = 4,
    device: torch.device | None = None,
) -> tuple[ProjectionDiscriminator, ProjectionDiscriminator]:
    """
    Create and initialise two ProjectionDiscriminator instances (D_A, D_B).

    Args:
        input_nc:            Input image channels.
        base_channels:       Feature channels for the local PatchGAN branch.
        n_layers:            Strided layers in the local branch.
        global_base_channels: Feature channels for the global branch.
        fft_base_channels:   Feature channels for the spectral branch.
        use_spectral_norm:   Spectral normalisation on all Conv2d layers.
        use_local:           Enable local PatchGAN + MinibatchStdDev branch.
        use_global:          Enable global branch with self-attention.
        use_fft:             Enable color-aware spectral FFT branch.
        mbstd_group_size:    Minibatch std-dev group size.
        device:              Target device.

    Returns:
        (D_A, D_B) initialised, moved to device, and set to train mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build():
        return ProjectionDiscriminator(
            input_nc=input_nc,
            base_channels=base_channels,
            n_layers=n_layers,
            global_base_channels=global_base_channels,
            fft_base_channels=fft_base_channels,
            use_spectral_norm=use_spectral_norm,
            use_local=use_local,
            use_global=use_global,
            use_fft=use_fft,
            mbstd_group_size=mbstd_group_size,
        ).to(device)

    D_A = _build()
    D_B = _build()

    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # Smoke test — verify shapes and catch construction errors early.
    x = torch.randn(max(mbstd_group_size, 2), input_nc, 256, 256, device=device)
    with torch.inference_mode():
        out_A = D_A(x)
        out_B = D_B(x)

    shapes_A = [tuple(o.shape) for o in out_A]
    shapes_B = [tuple(o.shape) for o in out_B]
    print(f"[getDiscriminatorsV3] D_A output shapes: {shapes_A}")
    print(f"[getDiscriminatorsV3] D_B output shapes: {shapes_B}")

    n_params_A = sum(p.numel() for p in D_A.parameters()) / 1e6
    print(f"[getDiscriminatorsV3] D_A params: {n_params_A:.2f}M")
    n_params_B = sum(p.numel() for p in D_B.parameters()) / 1e6
    print(f"[getDiscriminatorsV3] D_B params: {n_params_B:.2f}M")

    return D_A, D_B


if __name__ == "__main__":
    D_A, D_B = getDiscriminatorsV3()
    print("ProjectionDiscriminator v3.2 smoke test passed.")
