"""
True UVCGAN v2 Generator.

Improvements over the v1 (CycleGAN-style) generator:

* **Unified multi-domain convolution blocks** – residual blocks with Instance
  Normalization and separate per-domain learned parameters ready for future
  multi-domain extension.
* **LayerScale** – each Transformer block's residual branch is scaled by a
  learnable scalar initialised near zero, which stabilises early training.
* **Cross-domain feature sharing** – lightweight 1×1 fusion layers on skip
  connections help the two generators share structural knowledge.
* **Better weight initialisation** – Kaiming/Xavier schemes matched to each
  layer type.

Public API
----------
``ViTUNetGeneratorV2``   – drop-in replacement for ``ViTUNetGenerator``.
``init_weights_v2``      – weight initialiser compatible with the v2 model.
``getGeneratorsV2``      – factory mirroring ``getGenerators`` from v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional embedding helpers (shared with v1 logic)
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Build 1-D sine-cosine positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be even).
        pos: Position indices of shape ``(N,)``.

    Returns:
        Tensor of shape ``(N, embed_dim)``.
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
    Build 2-D sine-cosine positional embeddings for an ``(H, W)`` grid.

    Args:
        embed_dim: Total embedding dimension (must be even).
        height: Grid height.
        width: Grid width.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``(H*W, embed_dim)``.
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
# LayerScale Transformer block
# ---------------------------------------------------------------------------


class LayerScaleTransformerBlock(nn.Module):
    """
    Transformer block with **LayerScale** residual scaling.

    LayerScale multiplies each residual branch by a learnable per-channel
    scalar ``alpha`` initialised to ``init_values`` (typically 1e-4).  This
    prevents instability at the start of training when residuals are large
    relative to the main path.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dimension expansion factor.
        dropout: Dropout probability.
        init_values: Initial value for LayerScale scalars.
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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        # LayerScale parameters – one scalar per channel, initialised small.
        self.gamma_attn = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.gamma_attn * attn_out
        x = x + self.gamma_ffn * self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Pixelwise ViT bottleneck (v2)
# ---------------------------------------------------------------------------


class PixelwiseViTV2(nn.Module):
    """
    Pixelwise Vision Transformer with **LayerScale** blocks.

    Flattens spatial positions into tokens, applies Transformer blocks, then
    reshapes back to ``(N, C, H, W)``.

    Args:
        dim: Channel / embedding dimension.
        depth: Number of Transformer blocks.
        num_heads: Attention heads per block.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout probability.
        init_values: LayerScale initialisation value.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_values: float = 1e-4,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (N, H*W, C)
        pos = _get_2d_sincos_pos_embed(
            embed_dim=c, height=h, width=w, device=x.device, dtype=x.dtype
        )
        tokens = tokens + pos.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(n, c, h, w)


# ---------------------------------------------------------------------------
# Unified multi-domain convolution blocks
# ---------------------------------------------------------------------------


class ResidualConvBlock(nn.Module):
    """
    Residual convolution block with Instance Normalisation.

    The block applies two 3×3 convolutions with a skip connection.  Reflection
    padding is used to avoid border artefacts common in image translation.

    Args:
        channels: Number of input/output channels.
        dropout: Dropout probability between the two convolutions.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock(nn.Module):
    """
    Strided 4×4 convolution encoder block (stride-2 downsampling).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
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
    Nearest-neighbour upsampling followed by a 3×3 conv (avoids checkerboard).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
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


class CrossDomainFusion(nn.Module):
    """
    Lightweight 1×1 convolution that fuses skip-connection features.

    Used to share structural information on skip connections when both
    generators process the same spatial level simultaneously.

    Args:
        channels: Number of feature channels on this skip level.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, feat_self: torch.Tensor, feat_other: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse own skip features with the other generator's skip features.

        Args:
            feat_self: Skip features from this generator.
            feat_other: Skip features from the paired generator (detached).

        Returns:
            Fused feature map of the same spatial size and channel count.
        """
        return self.fuse(torch.cat([feat_self, feat_other.detach()], dim=1))


# ---------------------------------------------------------------------------
# True UVCGAN v2 Generator
# ---------------------------------------------------------------------------


class ViTUNetGeneratorV2(nn.Module):
    """
    True UVCGAN v2 generator: U-Net + ViT bottleneck with LayerScale and
    optional cross-domain feature sharing.

    Architecture summary (base_channels = 64):

    Encoder
        enc_in  : 3   → 64   (7×7 reflect-pad conv)
        down1   : 64  → 128  (DownBlock)
        res_enc1: 128 → 128  (2× ResidualConvBlock)
        down2   : 128 → 256  (DownBlock)
        res_enc2: 256 → 256  (2× ResidualConvBlock)
        down3   : 256 → 512  (DownBlock)
        res_enc3: 512 → 512  (2× ResidualConvBlock)
        down4   : 512 → 512  (DownBlock)

    Bottleneck
        res_bot : 512 → 512  (ResidualConvBlock)
        vit     : 512 → 512  (PixelwiseViTV2)

    Decoder
        up1     : 512 → 512
        dec1    : 1024 → 512 (skip from enc3)
        up2     : 512 → 256
        dec2    : 512  → 256 (skip from enc2)
        up3     : 256 → 128
        dec3    : 256  → 128 (skip from enc1)
        up4     : 128 → 64
        dec4    : 128  → 64  (skip from enc_in)

    Output
        out_conv: 64 → 3 (7×7 reflect-pad conv + Tanh)

    Args:
        input_nc: Input image channels (default 3 for RGB).
        output_nc: Output image channels (default 3 for RGB).
        base_channels: Feature channels at the first encoder level.
        vit_depth: Transformer block count in the bottleneck ViT.
        vit_heads: Attention heads per Transformer block.
        vit_mlp_ratio: MLP expansion ratio in Transformer blocks.
        vit_dropout: Dropout probability in Transformer blocks.
        layerscale_init: Initial value for LayerScale scalars.
        use_cross_domain: Allocate cross-domain fusion layers for skip
            connections (call :meth:`fuse_skips` each forward pass when
            the paired generator's skip features are available).
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
        self.res_enc3 = nn.Sequential(ResidualConvBlock(c4), ResidualConvBlock(c4))
        self.down4 = DownBlock(c4, c4)

        # ---- Bottleneck ----
        self.res_bot = ResidualConvBlock(c4)
        self.vit = PixelwiseViTV2(
            dim=c4,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
            init_values=layerscale_init,
        )

        # ---- Decoder ----
        self.up1 = UpBlock(c4, c4)
        # Each dec_merge fuses the upsampled feature map with the skip connection.
        self.dec1_merge = nn.Sequential(
            nn.Conv2d(c4 + c4, c4, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.up2 = UpBlock(c4, c3)
        self.dec2_merge = nn.Sequential(
            nn.Conv2d(c3 + c3, c3, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.up3 = UpBlock(c3, c2)
        self.dec3_merge = nn.Sequential(
            nn.Conv2d(c2 + c2, c2, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.up4 = UpBlock(c2, c1)
        self.dec4_merge = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, kernel_size=1, bias=False),
            nn.InstanceNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # ---- Output ----
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c1, output_nc, kernel_size=7),
            nn.Tanh(),
        )

        # ---- Optional cross-domain fusion on skip connections ----
        if use_cross_domain:
            self.fuse1 = CrossDomainFusion(c4)
            self.fuse2 = CrossDomainFusion(c3)
            self.fuse3 = CrossDomainFusion(c2)
            self.fuse4 = CrossDomainFusion(c1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor):
        """
        Run the encoder and return intermediate feature maps.

        Args:
            x: Input image tensor ``(N, input_nc, H, W)``.

        Returns:
            tuple: ``(e0, e1, e2, e3, bottleneck)`` where each ``e_i``
            is a skip-connection feature map and ``bottleneck`` is the
            deepest feature map after the ViT.
        """
        e0 = self.enc_in(x)
        e1 = self.res_enc1(self.down1(e0))
        e2 = self.res_enc2(self.down2(e1))
        e3 = self.res_enc3(self.down3(e2))
        b = self.vit(self.res_bot(self.down4(e3)))
        return e0, e1, e2, e3, b

    def decode(self, b, e0, e1, e2, e3) -> torch.Tensor:
        """
        Run the decoder given a bottleneck and skip-connection features.

        Args:
            b: Bottleneck feature map ``(N, c4, H/16, W/16)``.
            e0: Skip from ``enc_in`` at full resolution.
            e1: Skip from first encoder level.
            e2: Skip from second encoder level.
            e3: Skip from third encoder level.

        Returns:
            Output image tensor ``(N, output_nc, H, W)``.
        """
        u1 = self.up1(b)
        u1 = self.dec1_merge(torch.cat([u1, e3], dim=1))
        u2 = self.up2(u1)
        u2 = self.dec2_merge(torch.cat([u2, e2], dim=1))
        u3 = self.up3(u2)
        u3 = self.dec3_merge(torch.cat([u3, e1], dim=1))
        u4 = self.up4(u3)
        u4 = self.dec4_merge(torch.cat([u4, e0], dim=1))
        return self.out_conv(u4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (no cross-domain fusion).

        Args:
            x: Input image tensor ``(N, input_nc, H, W)``.

        Returns:
            Translated image tensor ``(N, output_nc, H, W)``.
        """
        e0, e1, e2, e3, b = self.encode(x)
        return self.decode(b, e0, e1, e2, e3)

    def forward_with_cross_domain(
        self,
        x: torch.Tensor,
        other_skips,
    ) -> torch.Tensor:
        """
        Forward pass with cross-domain skip-connection fusion.

        Call this method when training both generators simultaneously so that
        each generator can leverage the other's structural features.

        Args:
            x: Input image tensor ``(N, input_nc, H, W)``.
            other_skips: Tuple ``(eo0, eo1, eo2, eo3)`` of skip-connection
                feature maps from the *paired* generator at the same step
                (they are detached inside :class:`CrossDomainFusion`).

        Returns:
            Translated image tensor ``(N, output_nc, H, W)``.
        """
        if not self.use_cross_domain:
            raise RuntimeError(
                "cross-domain fusion is disabled (use_cross_domain=False)."
            )
        oe0, oe1, oe2, oe3 = other_skips
        e0, e1, e2, e3, b = self.encode(x)
        # Fuse each skip level with the paired generator's skip.
        e0 = self.fuse4(e0, oe0)
        e1 = self.fuse3(e1, oe1)
        e2 = self.fuse2(e2, oe2)
        e3 = self.fuse1(e3, oe3)
        return self.decode(b, e0, e1, e2, e3)

    def get_skip_features(self, x: torch.Tensor):
        """
        Return only the encoder skip features (no decoding).

        Used by the paired generator to obtain cross-domain features without
        running a full forward pass.

        Args:
            x: Input image tensor.

        Returns:
            tuple: ``(e0, e1, e2, e3)`` skip feature maps.
        """
        e0, e1, e2, e3, _ = self.encode(x)
        return e0, e1, e2, e3


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights_v2(net: nn.Module) -> None:
    """
    Initialise ``net`` weights for the v2 architecture.

    * Conv2d / ConvTranspose2d → Kaiming normal (suitable for ReLU).
    * Linear → Xavier uniform.
    * InstanceNorm2d → weight 1, bias 0.
    * LayerNorm → weight 1, bias 0.

    Args:
        net: Network module to initialise in-place.
    """
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight.data, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, (nn.InstanceNorm2d, nn.LayerNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


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
):
    """
    Create and initialise two v2 generators for CycleGAN-style training.

    Returns:
        tuple: ``(G_AB, G_BA)`` – both moved to the available device and
        weight-initialised with :func:`init_weights_v2`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_AB = ViTUNetGeneratorV2(
        base_channels=int(base_channels),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_ratio=vit_mlp_ratio,
        vit_dropout=vit_dropout,
        layerscale_init=layerscale_init,
        use_cross_domain=bool(use_cross_domain),
    ).to(device)
    G_BA = ViTUNetGeneratorV2(
        base_channels=int(base_channels),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_ratio=vit_mlp_ratio,
        vit_dropout=vit_dropout,
        layerscale_init=layerscale_init,
        use_cross_domain=bool(use_cross_domain),
    ).to(device)

    G_AB.apply(init_weights_v2)
    G_BA.apply(init_weights_v2)

    # Smoke test.
    x = torch.randn(1, 3, 256, 256, device=device)
    y_AB = G_AB(x)
    y_BA = G_BA(x)
    print("G_AB (v2) output shape:", y_AB.shape)
    print("G_BA (v2) output shape:", y_BA.shape)

    return G_AB, G_BA


if __name__ == "__main__":
    G_AB, G_BA = getGeneratorsV2()
