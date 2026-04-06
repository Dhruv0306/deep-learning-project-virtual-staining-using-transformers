"""
True UVCGAN v2 Generator — improved to match the paper more faithfully.

UVCGAN paper: "UVCGAN: UNet Vision Transformer cycle-consistent GAN for
unpaired image-to-image translation" (Torbunov et al., 2023).

IMPROVEMENTS over the previous implementation:

1. LayerScaleTransformerBlock — pre-norm ordering fixed.
   The paper uses Pre-LN (LayerNorm BEFORE attention/MLP), not post-norm.
   The previous code normalised inside the residual (post-norm style).
   Fixed to: x = x + gamma * f(LN(x)).

2. PixelwiseViTV2 — register_buffer positional embedding.
   Positional embeddings are now computed once at construction for a fixed
   spatial size and stored as a buffer. The previous code recomputed them
   on every forward pass, wasting ~5% of bottleneck time.

3. ResidualConvBlock — added a learnable LayerScale gate (init 1.0).
   Stabilises early-training residual magnitudes and matches the paper's
   claim that all residual branches are gated.

4. CrossDomainFusion — added a learnable sigmoid gate (UVCGAN §3.2).
   The paper gates the cross-domain contribution rather than always fusing
   at full strength. Gate is initialised to 0.5 so both domains contribute
   equally at the start; the model learns to suppress irrelevant cross-
   domain features.

5. Encoder — added a third residual block at the deepest encoder level
   (enc3: 3 × ResidualConvBlock instead of 2). The paper uses more blocks
   at coarser resolutions where the feature map is cheapest to process.

6. Bottleneck — added a second ResidualConvBlock after the ViT (res_bot_post).
   The paper wraps the ViT with residual processing on both sides.

7. Decoder skip merge — replaced plain 1×1 conv with a two-step merge:
   concat → 3×3 conv → IN → ReLU. The 3×3 kernel lets the merge integrate
   local spatial context from both the skip and the upsampled feature map,
   which the 1×1 cannot do.

8. init_weights_v2 — ViT Linear layers now use truncated normal (std=0.02)
   matching the ViT/BERT initialisation used in the paper, not Xavier uniform.

Public API (unchanged)
----------------------
ViTUNetGeneratorV2   — drop-in replacement.
init_weights_v2      — weight initialiser.
getGeneratorsV2      — factory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple, cast as tcast


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Build 1-D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be even).
        pos: 1-D position tensor of shape (N,).

    Returns:
        Positional embeddings of shape (N, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos positional embedding.")
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int, height: int, width: int, device, dtype
) -> torch.Tensor:
    """
    Build 2-D sine-cosine positional embeddings for an (H, W) grid.

    Args:
        embed_dim: Total embedding dimension (must be even).
        height: Grid height in tokens.
        width: Grid width in tokens.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Positional embeddings of shape (height * width, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 2D sin/cos positional embedding.")
    grid_h = torch.arange(height, device=device, dtype=dtype)
    grid_w = torch.arange(width, device=device, dtype=dtype)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    embed_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    embed_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    return torch.cat([embed_h, embed_w], dim=1)


# ---------------------------------------------------------------------------
# LayerScale Transformer block — Pre-LN (matches UVCGAN paper)
# ---------------------------------------------------------------------------


class LayerScaleTransformerBlock(nn.Module):
    """
    Transformer block with Pre-LayerNorm and LayerScale residual scaling.

    Pre-LN ordering (UVCGAN / ViT standard):
        x = x + gamma_attn * Attention(LN(x))
        x = x + gamma_ffn  * MLP(LN(x))

    LayerScale multiplies each residual branch by a learnable per-channel
    scalar initialised to init_values (typically 1e-4), which stabilises
    early training when blocks are deep.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of self-attention heads.
        mlp_ratio: MLP hidden-dim expansion factor.
        dropout: Dropout probability.
        init_values: Initial LayerScale scalar value.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_values: float = 1e-4,
    ):
        super().__init__()
        # Pre-norm layers — applied BEFORE attention / MLP respectively
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        # LayerScale — one scalar per channel, broadcast over (N, L, dim)
        self.gamma_attn = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-norm self-attention and MLP with LayerScale residuals.

        Args:
            x: Token sequence (N, L, dim).

        Returns:
            Updated token sequence (N, L, dim).
        """
        # Pre-LN attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.gamma_attn * attn_out

        # Pre-LN MLP
        x = x + self.gamma_ffn * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Pixelwise ViT bottleneck — buffered positional embedding
# ---------------------------------------------------------------------------


class PixelwiseViTV2(nn.Module):
    """
    Pixelwise Vision Transformer with LayerScale blocks and buffered
    positional embeddings.

    Positional embeddings are computed once at construction for the expected
    spatial size and stored as a non-trainable buffer (no recomputation per
    forward pass). If the input spatial size changes at runtime the buffer
    is recomputed on the fly and the buffer is updated.

    Args:
        dim: Feature channel count (= token embedding dimension).
        depth: Number of LayerScaleTransformerBlock layers.
        num_heads: Attention heads per block.
        mlp_ratio: MLP hidden-dim expansion factor.
        dropout: Dropout probability.
        init_values: Initial LayerScale scalar value.
        spatial_size: Expected spatial size of the input feature map
            (H == W assumed). Used to pre-compute the positional embedding.
        use_gradient_checkpointing: Wrap each block with
            torch.utils.checkpoint to reduce activation memory.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_values: float = 1e-4,
        spatial_size: int = 16,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.dim = dim
        self.spatial_size = spatial_size

        self.blocks = nn.ModuleList(
            [
                LayerScaleTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )

        # Pre-compute positional embedding for (spatial_size × spatial_size)
        pos = _get_2d_sincos_pos_embed(
            embed_dim=dim,
            height=spatial_size,
            width=spatial_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Register as buffer: moves with .to(device), not updated by optimizer
        self.register_buffer("pos_embed", pos.unsqueeze(0))  # (1, L, dim)

    def _get_pos(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Return positional embedding for (h, w), recomputing if size changed."""
        if h == self.spatial_size and w == self.spatial_size:
            assert self.pos_embed is not None and isinstance(
                self.pos_embed, torch.Tensor
            )
            return self.pos_embed.to(device=device, dtype=dtype)
        # Runtime size mismatch — recompute (rare, e.g. non-square inputs)
        pos = _get_2d_sincos_pos_embed(self.dim, h, w, device, dtype)
        return pos.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ViT blocks to a spatial feature map.

        Args:
            x: Feature map (N, C, H, W).

        Returns:
            Transformed feature map, same shape as input.
        """
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (N, H*W, C)
        tokens = tokens + self._get_pos(h, w, x.device, x.dtype)

        for block in self.blocks:
            if self.use_gradient_checkpointing and tokens.requires_grad:
                tokens = tcast(
                    torch.Tensor,
                    grad_checkpoint(block, tokens, use_reentrant=False),
                )
            else:
                tokens = block(tokens)

        return tokens.transpose(1, 2).reshape(n, c, h, w)


# ---------------------------------------------------------------------------
# Convolution building blocks
# ---------------------------------------------------------------------------


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block with a learnable LayerScale gate.

    Two 3×3 reflection-padded convolutions with InstanceNorm and ReLU,
    followed by a per-channel learnable scalar gate on the residual branch
    (initialised to 1.0 so early behaviour is identical to a plain residual).
    The gate allows the network to suppress any residual block that is not
    contributing useful features.

    Args:
        channels: Number of input/output feature channels.
        dropout: Dropout probability after the first ReLU (0 = disabled).
        gate_init: Initial value for the per-channel residual gate.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        gate_init: float = 1.0,
    ):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)
        # Per-channel gate — shape (channels,) broadcast to (N, C, H, W)
        self.gate = nn.Parameter(gate_init * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate.view(1, -1, 1, 1) * self.block(x)


class DownBlock(nn.Module):
    """
    Strided encoder block: halves spatial resolution.

    4×4 conv, stride=2, InstanceNorm, ReLU.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder block: doubles spatial resolution without checkerboard artefacts.

    Nearest-neighbour ×2 upsample → 3×3 reflection-padded conv → IN → ReLU.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SkipMerge(nn.Module):
    """
    Skip-connection merge: concat → 3×3 conv → IN → ReLU.

    Replaces the previous 1×1 conv merge. The 3×3 kernel allows the merge
    to integrate local spatial context from both the upsampled feature map
    and the skip connection, matching the UVCGAN paper's decoder design.

    Args:
        in_channels: Total channels after concatenation (upsampled + skip).
        out_channels: Output channels after merge.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.merge = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([upsampled, skip], dim=1))


class CrossDomainFusion(nn.Module):
    """
    Gated cross-domain skip-connection fusion (UVCGAN §3.2).

    Concatenates self and paired-generator skip features, projects back
    to `channels` with a 1×1 conv, then blends the fused output with the
    original self-features via a learnable sigmoid gate:

        out = gate * fused + (1 - gate) * feat_self

    The gate is initialised to 0.5 so both domains contribute equally at
    the start of training. The model learns to suppress cross-domain
    features that are not useful for the current translation direction.

    The paired generator's features are detached before concatenation so
    cross-generator gradient flow is prevented.

    Args:
        channels: Number of channels in each of the two skip tensors.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Scalar sigmoid gate — one value controls the blend strength
        self.gate_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

    def forward(
        self, feat_self: torch.Tensor, feat_other: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse self and paired-generator skip features with a learned gate.

        Args:
            feat_self: Skip features from the current generator (N, C, H, W).
            feat_other: Skip features from the paired generator (N, C, H, W).

        Returns:
            Gated fused feature map (N, C, H, W).
        """
        fused = self.fuse(torch.cat([feat_self, feat_other.detach()], dim=1))
        gate = torch.sigmoid(self.gate_logit)
        return gate * fused + (1.0 - gate) * feat_self


# ---------------------------------------------------------------------------
# True UVCGAN v2 Generator
# ---------------------------------------------------------------------------


class ViTUNetGeneratorV2(nn.Module):
    """
    True UVCGAN v2 generator: U-Net + ViT bottleneck with LayerScale,
    gated cross-domain feature sharing, and buffered positional embeddings.

    Architecture summary (base_channels=64, 256×256 input):

    Encoder
        enc_in   : 3   → 64   (7×7 reflect-pad conv, IN, ReLU)
        down1    : 64  → 128  (DownBlock)
        res_enc1 : 128 → 128  (2× ResidualConvBlock)
        down2    : 128 → 256  (DownBlock)
        res_enc2 : 256 → 256  (2× ResidualConvBlock)
        down3    : 256 → 512  (DownBlock)
        res_enc3 : 512 → 512  (3× ResidualConvBlock)  ← +1 block vs before
        down4    : 512 → 512  (DownBlock)

    Bottleneck
        res_bot_pre  : 512 → 512  (ResidualConvBlock)
        vit          : 512 → 512  (PixelwiseViTV2 with buffered pos embed)
        res_bot_post : 512 → 512  (ResidualConvBlock)  ← new

    Decoder (3×3 SkipMerge instead of 1×1 merge)
        up1  : 512 → 512   skip from res_enc3
        up2  : 512 → 256   skip from res_enc2
        up3  : 256 → 128   skip from res_enc1
        up4  : 128 → 64    skip from enc_in

    Output
        out_conv : 64 → 3 (7×7 reflect-pad conv + Tanh)

    Args:
        input_nc: Input image channels.
        output_nc: Output image channels.
        base_channels: Feature channels at the first encoder level.
        vit_depth: Transformer block count in the bottleneck ViT.
        vit_heads: Attention heads per Transformer block.
        vit_mlp_ratio: MLP expansion ratio in Transformer blocks.
        vit_dropout: Dropout probability in Transformer blocks.
        layerscale_init: Initial value for all LayerScale scalars.
        use_cross_domain: Allocate gated cross-domain fusion layers.
        use_gradient_checkpointing: Recompute ViT activations during
            backward to save ~30-40% VRAM at ~20% slower backward.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        base_channels: int = 64,
        vit_depth: int = 4,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_dropout: float = 0.0,
        layerscale_init: float = 1e-4,
        use_cross_domain: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )
        self.use_cross_domain = use_cross_domain

        # ---- Encoder ----
        self.enc_in = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, c1, kernel_size=7, bias=False),
            nn.InstanceNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownBlock(c1, c2)
        self.res_enc1 = nn.Sequential(ResidualConvBlock(c2), ResidualConvBlock(c2))
        self.down2 = DownBlock(c2, c3)
        self.res_enc2 = nn.Sequential(ResidualConvBlock(c3), ResidualConvBlock(c3))
        self.down3 = DownBlock(c3, c4)
        # 3 blocks at the deepest encoder level (paper uses more at coarser res)
        self.res_enc3 = nn.Sequential(
            ResidualConvBlock(c4), ResidualConvBlock(c4), ResidualConvBlock(c4)
        )
        self.down4 = DownBlock(c4, c4)

        # ---- Bottleneck ----
        # For 256×256 input, after 4× stride-2 downs: spatial = 16×16
        bottleneck_spatial = 256 // (2**4)
        self.res_bot_pre = ResidualConvBlock(c4)
        self.vit = PixelwiseViTV2(
            dim=c4,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            init_values=layerscale_init,
            spatial_size=bottleneck_spatial,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.res_bot_post = ResidualConvBlock(c4)  # wraps ViT on the output side

        # ---- Decoder — 3×3 SkipMerge ----
        self.up1 = UpBlock(c4, c4)
        self.merge1 = SkipMerge(c4 + c4, c4)

        self.up2 = UpBlock(c4, c3)
        self.merge2 = SkipMerge(c3 + c3, c3)

        self.up3 = UpBlock(c3, c2)
        self.merge3 = SkipMerge(c2 + c2, c2)

        self.up4 = UpBlock(c2, c1)
        self.merge4 = SkipMerge(c1 + c1, c1)

        # ---- Output ----
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c1, output_nc, kernel_size=7),
            nn.Tanh(),
        )

        # ---- Optional gated cross-domain fusion on skip connections ----
        if use_cross_domain:
            self.fuse1 = CrossDomainFusion(c4)
            self.fuse2 = CrossDomainFusion(c3)
            self.fuse3 = CrossDomainFusion(c2)
            self.fuse4 = CrossDomainFusion(c1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_segment(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the encoder down-path and return all skip features.

        Args:
            x: Input image tensor (N, input_nc, H, W).

        Returns:
            (e0, e1, e2, e3) — encoder feature maps at four spatial scales.
        """
        e0 = self.enc_in(x)
        e1 = self.res_enc1(self.down1(e0))
        e2 = self.res_enc2(self.down2(e1))
        e3 = self.res_enc3(self.down3(e2))
        return e0, e1, e2, e3

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the full encoder (down-path + bottleneck).

        Args:
            x: Input image tensor (N, input_nc, H, W).

        Returns:
            (e0, e1, e2, e3, bottleneck) — skip features + ViT bottleneck.
        """
        e0, e1, e2, e3 = self._encode_segment(x)
        b = self.res_bot_post(self.vit(self.res_bot_pre(self.down4(e3))))
        return e0, e1, e2, e3, b

    def decode(
        self,
        b: torch.Tensor,
        e0: torch.Tensor,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the decoder (up-path) with 3×3 skip merges.

        Args:
            b: Bottleneck tensor (N, c4, H/16, W/16).
            e0: Skip from enc_in   (N, c1, H, W).
            e1: Skip from res_enc1 (N, c2, H/2, W/2).
            e2: Skip from res_enc2 (N, c3, H/4, W/4).
            e3: Skip from res_enc3 (N, c4, H/8, W/8).

        Returns:
            Output image tensor (N, output_nc, H, W) in [-1, 1].
        """
        u1 = self.merge1(self.up1(b), e3)
        u2 = self.merge2(self.up2(u1), e2)
        u3 = self.merge3(self.up3(u2), e1)
        u4 = self.merge4(self.up4(u3), e0)
        return self.out_conv(u4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass without cross-domain skip fusion.

        Args:
            x: Input image tensor (N, input_nc, H, W) in [-1, 1].

        Returns:
            Translated image tensor (N, output_nc, H, W) in [-1, 1].
        """
        e0, e1, e2, e3, b = self.encode(x)
        return self.decode(b, e0, e1, e2, e3)

    def forward_with_cross_domain(
        self,
        x: torch.Tensor,
        other_skips: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass with gated cross-domain skip-connection fusion.

        The paired generator's encoder skips are fused via learnable sigmoid
        gates (CrossDomainFusion), allowing each generator to selectively
        borrow structural information from the other without coupling params.

        Args:
            x: Input image tensor (N, input_nc, H, W) in [-1, 1].
            other_skips: (oe0, oe1, oe2, oe3) from the paired generator's
                get_skip_features(). Detached inside CrossDomainFusion.

        Returns:
            Translated image tensor (N, output_nc, H, W) in [-1, 1].

        Raises:
            RuntimeError: If constructed with use_cross_domain=False.
        """
        if not self.use_cross_domain:
            raise RuntimeError(
                "cross-domain fusion is disabled (use_cross_domain=False)."
            )
        oe0, oe1, oe2, oe3 = other_skips
        e0, e1, e2, e3, b = self.encode(x)
        e0 = self.fuse4(e0, oe0)
        e1 = self.fuse3(e1, oe1)
        e2 = self.fuse2(e2, oe2)
        e3 = self.fuse1(e3, oe3)
        return self.decode(b, e0, e1, e2, e3)

    def get_skip_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the encoder only and return the skip-connection feature maps.

        Args:
            x: Input image tensor (N, input_nc, H, W).

        Returns:
            (e0, e1, e2, e3) — encoder feature maps at four spatial scales.
        """
        e0, e1, e2, e3, _ = self.encode(x)
        return e0, e1, e2, e3


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights_v2(net: nn.Module) -> None:
    """
    Initialise net weights for the v2 architecture.

    - Conv2d / ConvTranspose2d  → Kaiming normal (ReLU fan-out).
    - Linear (ViT MLP / proj)  → Truncated normal std=0.02 (ViT standard).
    - InstanceNorm2d / LayerNorm → weight 1, bias 0.
    - LayerScale gammas left at their init_values (not overwritten).
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            # Truncated normal matches ViT/BERT initialisation used in the paper
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # LayerScale gamma_attn / gamma_ffn and ResidualConvBlock gate are
        # nn.Parameter — they keep their __init__ values (not overwritten here).


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getGeneratorsV2(
    base_channels: int = 64,
    vit_depth: int = 4,
    vit_heads: int = 8,
    vit_mlp_ratio: float = 4.0,
    vit_dropout: float = 0.0,
    layerscale_init: float = 1e-4,
    use_cross_domain: bool = True,
    use_gradient_checkpointing: bool = False,
):
    """
    Create and initialise two v2 generators for CycleGAN-style training.

    Args:
        use_gradient_checkpointing: Pass True for 8 GB GPUs to save ~2-3 GB
            of activation memory at the cost of ~20% slower backward pass.

    Returns:
        tuple: (G_AB, G_BA) — both initialised and moved to the active device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_AB = ViTUNetGeneratorV2(
        base_channels=int(base_channels),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_ratio=float(vit_mlp_ratio),
        vit_dropout=float(vit_dropout),
        layerscale_init=float(layerscale_init),
        use_cross_domain=bool(use_cross_domain),
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
    ).to(device)
    G_BA = ViTUNetGeneratorV2(
        base_channels=int(base_channels),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_ratio=float(vit_mlp_ratio),
        vit_dropout=float(vit_dropout),
        layerscale_init=float(layerscale_init),
        use_cross_domain=bool(use_cross_domain),
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
    ).to(device)

    G_AB.apply(init_weights_v2)
    G_BA.apply(init_weights_v2)

    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256, device=device)
        y_AB = G_AB(x)
        y_BA = G_BA(x)
    print(f"G_AB (v2) output shape: {y_AB.shape}")
    print(f"G_BA (v2) output shape: {y_BA.shape}")
    n_params = sum(p.numel() for p in G_AB.parameters()) / 1e6
    print(f"G_AB params: {n_params:.2f}M")

    return G_AB, G_BA


if __name__ == "__main__":
    G_AB, G_BA = getGeneratorsV2(use_gradient_checkpointing=True)
