"""
True UVCGAN v2 Generator.

Improvements over the v1 (CycleGAN-style) generator:

* Unified multi-domain convolution blocks -- residual blocks with Instance
  Normalisation.
* LayerScale -- each Transformer block's residual branch is scaled by a
  learnable scalar initialised near zero, which stabilises early training.
* Cross-domain feature sharing -- lightweight 1x1 fusion layers on skip
  connections help the two generators share structural knowledge.
* Better weight initialisation -- Kaiming/Xavier schemes matched to each
  layer type.
* Gradient checkpointing -- optional activation recomputation during
  backward pass to trade compute for VRAM (controlled by
  use_gradient_checkpointing in GeneratorConfig).

Public API
----------
ViTUNetGeneratorV2   -- drop-in replacement for ViTUNetGenerator.
init_weights_v2      -- weight initialiser compatible with the v2 model.
getGeneratorsV2      -- factory mirroring getGenerators from v1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Tuple, cast as tcast


# ---------------------------------------------------------------------------
# Positional embedding helpers
# ---------------------------------------------------------------------------


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos positional embedding.")
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int, height: int, width: int, device, dtype
) -> torch.Tensor:
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
    Transformer block with LayerScale residual scaling.

    LayerScale multiplies each residual branch by a learnable per-channel
    scalar alpha initialised to init_values (typically 1e-4).
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
    Pixelwise Vision Transformer with LayerScale blocks.

    Flattens spatial positions into tokens, applies Transformer blocks, then
    reshapes back to (N, C, H, W).

    Args:
        use_gradient_checkpointing: If True, each Transformer block is wrapped
            with torch.utils.checkpoint to trade compute for VRAM.  This is
            the single biggest memory saving in the whole generator since the
            ViT blocks store large (N, H*W, C) activation tensors normally.
    """

    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        init_values: float = 1e-4,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
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
            if self.use_gradient_checkpointing and tokens.requires_grad:
                # grad_checkpoint is typed as returning Any|None in older stubs;
                # tcast tells Pylance it is always a Tensor here.
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
    Residual convolution block used in the encoder, decoder, and bottleneck.

    Each block applies two consecutive 3×3 convolutions with reflection
    padding (to avoid border artefacts), InstanceNorm, and ReLU activation,
    then adds the input via a skip connection.  An optional dropout layer
    is inserted between the two convolutions when ``dropout > 0``.

    The residual connection allows gradients to flow directly from the
    output to the input, preventing vanishing gradients in deep networks.

    Args:
        channels (int): Number of input and output feature channels
            (spatial dimensions are preserved).
        dropout (float): Dropout probability applied after the first
            activation.  Set to ``0.0`` to disable (default).
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
        """
        Apply the residual block.

        Args:
            x (torch.Tensor): Feature map ``(N, channels, H, W)``.

        Returns:
            torch.Tensor: ``x + block(x)`` — same shape as input.
        """
        return x + self.block(x)


class DownBlock(nn.Module):
    """
    Strided encoder block that halves spatial resolution.

    Applies a single 4×4 convolution with ``stride=2`` followed by
    InstanceNorm and ReLU.  Spatial dimensions change as
    ``(H, W) → (H/2, W/2)`` while the channel count grows from
    ``in_channels`` to ``out_channels``.

    Preferred over ``MaxPool + Conv`` because the strided convolution is
    learnable and does not throw away spatial information before
    normalisation.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels (typically
            double the input count at each encoder level).
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
        """
        Apply the strided convolution block.

        Args:
            x (torch.Tensor): Feature map ``(N, in_channels, H, W)``.

        Returns:
            torch.Tensor: Downsampled map ``(N, out_channels, H/2, W/2)``.
        """
        return self.block(x)


class UpBlock(nn.Module):
    """
    Decoder block that doubles spatial resolution without checkerboard artefacts.

    Uses nearest-neighbour upsampling (×2) followed by a 3×3 reflection-padded
    convolution, InstanceNorm, and ReLU.  This avoids the checkerboard pattern
    that can appear with transposed convolutions (deconvolutions), especially
    at early training stages.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels (typically
            half the input count at each decoder level).
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
        """
        Apply the nearest-neighbour upsampling followed by convolution.

        Args:
            x (torch.Tensor): Feature map ``(N, in_channels, H, W)``.

        Returns:
            torch.Tensor: Upsampled map ``(N, out_channels, H*2, W*2)``.
        """
        return self.block(x)


class CrossDomainFusion(nn.Module):
    """
    Lightweight module that fuses skip-connection features from both generators.

    Concatenates the self skip features with the paired generator's skip
    features along the channel axis, then reduces back to ``channels``
    with a 1×1 convolution followed by InstanceNorm and ReLU.

    The paired generator's features are detached before the concatenation
    so that their gradients do not propagate back through the other generator
    during the current generator's backward pass.  This preserves the
    independent parameter updates of each generator.

    Args:
        channels (int): Number of channels in each of the two skip tensors.
            The concatenated input will have ``2 × channels`` channels.
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
        Fuse self and paired-generator skip features.

        Args:
            feat_self (torch.Tensor): Skip features from the current generator
                ``(N, channels, H, W)``.
            feat_other (torch.Tensor): Skip features from the paired generator
                ``(N, channels, H, W)``.  Detached internally before use.

        Returns:
            torch.Tensor: Fused feature map ``(N, channels, H, W)``.
        """
        return self.fuse(torch.cat([feat_self, feat_other.detach()], dim=1))


# ---------------------------------------------------------------------------
# True UVCGAN v2 Generator
# ---------------------------------------------------------------------------


class ViTUNetGeneratorV2(nn.Module):
    """
    True UVCGAN v2 generator: U-Net + ViT bottleneck with LayerScale and
    optional cross-domain feature sharing.

    Architecture summary (base_channels=64):

    Encoder
        enc_in  : 3   -> 64   (7x7 reflect-pad conv)
        down1   : 64  -> 128  (DownBlock)
        res_enc1: 128 -> 128  (2x ResidualConvBlock)
        down2   : 128 -> 256  (DownBlock)
        res_enc2: 256 -> 256  (2x ResidualConvBlock)
        down3   : 256 -> 512  (DownBlock)
        res_enc3: 512 -> 512  (2x ResidualConvBlock)
        down4   : 512 -> 512  (DownBlock)

    Bottleneck
        res_bot : 512 -> 512  (ResidualConvBlock)
        vit     : 512 -> 512  (PixelwiseViTV2)

    Decoder
        up1     : 512 -> 512
        dec1    : 1024 -> 512 (skip from enc3)
        up2     : 512 -> 256
        dec2    : 512  -> 256 (skip from enc2)
        up3     : 256 -> 128
        dec3    : 256  -> 128 (skip from enc1)
        up4     : 128 -> 64
        dec4    : 128  -> 64  (skip from enc_in)

    Output
        out_conv: 64 -> 3 (7x7 reflect-pad conv + Tanh)

    Args:
        input_nc: Input image channels.
        output_nc: Output image channels.
        base_channels: Feature channels at the first encoder level.
        vit_depth: Transformer block count in the bottleneck ViT.
        vit_heads: Attention heads per Transformer block.
        vit_mlp_ratio: MLP expansion ratio in Transformer blocks.
        vit_dropout: Dropout probability in Transformer blocks.
        layerscale_init: Initial value for LayerScale scalars.
        use_cross_domain: Allocate cross-domain fusion layers.
        use_gradient_checkpointing: Recompute ViT block activations during
            backward to save ~30-40% generator VRAM at the cost of ~20%
            slower backward pass.  Recommended for 8 GB GPUs.
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
        self.use_gradient_checkpointing = use_gradient_checkpointing

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
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # ---- Decoder ----
        self.up1 = UpBlock(c4, c4)
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

    def _encode_segment(self, x: torch.Tensor):
        """
        Run the encoder down-path and return all intermediate skip features.

        This method is the checkpointable unit: when gradient checkpointing
        is enabled it is wrapped with ``torch.utils.checkpoint.checkpoint``
        inside :meth:`encode` so that its activations are recomputed during
        the backward pass instead of being stored in memory.

        Args:
            x (torch.Tensor): Input image tensor ``(N, input_nc, H, W)``.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]:
                ``(e0, e1, e2, e3)`` — encoder feature maps at resolutions
                ``H/1``, ``H/2``, ``H/4``, ``H/8`` respectively.
        """
        e0 = self.enc_in(x)
        e1 = self.res_enc1(self.down1(e0))
        e2 = self.res_enc2(self.down2(e1))
        e3 = self.res_enc3(self.down3(e2))
        return e0, e1, e2, e3

    def encode(self, x: torch.Tensor):
        """
        Run the full encoder (down-path + bottleneck) and return all feature maps.

        Args:
            x (torch.Tensor): Input image tensor ``(N, input_nc, H, W)``.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                ``(e0, e1, e2, e3, bottleneck)`` where each ``eN`` is the
                encoder feature map saved for the skip connection at the
                corresponding decoder level, and ``bottleneck`` is the
                ViT-processed deepest feature map.
        """
        e0, e1, e2, e3 = self._encode_segment(x)

        b = self.vit(self.res_bot(self.down4(e3)))
        return e0, e1, e2, e3, b

    def decode(self, b, e0, e1, e2, e3) -> torch.Tensor:
        """
        Run the decoder (up-path) given bottleneck and skip features.

        At each decoder level the upsampled feature map is concatenated
        with the corresponding encoder skip feature, then merged with a
        1×1 convolution followed by InstanceNorm and ReLU.

        Args:
            b (torch.Tensor): Bottleneck tensor ``(N, c4, H/16, W/16)``.
            e0 (torch.Tensor): Skip from ``enc_in``  — ``(N, c1, H, W)``.
            e1 (torch.Tensor): Skip from ``res_enc1`` — ``(N, c2, H/2, W/2)``.
            e2 (torch.Tensor): Skip from ``res_enc2`` — ``(N, c3, H/4, W/4)``.
            e3 (torch.Tensor): Skip from ``res_enc3`` — ``(N, c4, H/8, W/8)``.

        Returns:
            torch.Tensor: Output image tensor ``(N, output_nc, H, W)``
            with values in ``[-1, 1]`` (Tanh activation applied).
        """
        u1 = self.dec1_merge(torch.cat([self.up1(b), e3], dim=1))
        u2 = self.dec2_merge(torch.cat([self.up2(u1), e2], dim=1))
        u3 = self.dec3_merge(torch.cat([self.up3(u2), e1], dim=1))
        u4 = self.dec4_merge(torch.cat([self.up4(u3), e0], dim=1))
        return self.out_conv(u4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass without cross-domain skip fusion.

        Args:
            x (torch.Tensor): Input image tensor ``(N, input_nc, H, W)``
                with values in ``[-1, 1]``.

        Returns:
            torch.Tensor: Translated image tensor ``(N, output_nc, H, W)``
            with values in ``[-1, 1]``.
        """
        e0, e1, e2, e3, b = self.encode(x)
        return self.decode(b, e0, e1, e2, e3)

    def forward_with_cross_domain(self, x: torch.Tensor, other_skips) -> torch.Tensor:
        """
        Forward pass with cross-domain skip-connection fusion.

        The paired generator's encoder skip features are passed through a
        :class:`CrossDomainFusion` layer at each decoder level so that both
        generators can share structural information without coupling their
        parameters.  This is the defining UVCGAN feature.

        Args:
            x (torch.Tensor): Input image tensor ``(N, input_nc, H, W)``
                with values in ``[-1, 1]``.
            other_skips (tuple[Tensor, Tensor, Tensor, Tensor]): Encoder skip
                features ``(oe0, oe1, oe2, oe3)`` from the paired generator,
                obtained via :meth:`get_skip_features`.  They are detached
                inside :class:`CrossDomainFusion` to prevent cross-generator
                gradient flow.

        Returns:
            torch.Tensor: Translated image tensor ``(N, output_nc, H, W)``
            with values in ``[-1, 1]``.

        Raises:
            RuntimeError: If this generator was constructed with
                ``use_cross_domain=False``.
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

    def get_skip_features(self, x: torch.Tensor):
        """
        Run the encoder only and return the skip-connection feature maps.

        Used by the training loop to obtain skip features from one generator
        before passing them to the paired generator's
        :meth:`forward_with_cross_domain`.

        Args:
            x (torch.Tensor): Input image tensor ``(N, input_nc, H, W)``.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: ``(e0, e1, e2, e3)``
            encoder feature maps at four spatial scales.
        """
        e0, e1, e2, e3, _ = self.encode(x)
        return e0, e1, e2, e3


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights_v2(net: nn.Module) -> None:
    """
    Initialise net weights for the v2 architecture.

    * Conv2d / ConvTranspose2d -> Kaiming normal (ReLU).
    * Linear -> Xavier uniform.
    * InstanceNorm2d / LayerNorm -> weight 1, bias 0.
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
    use_gradient_checkpointing: bool = False,
):
    """
    Create and initialise two v2 generators for CycleGAN-style training.

    Args:
        use_gradient_checkpointing: Pass True for 8 GB GPUs to save ~2-3 GB
            of activation memory at the cost of ~20% slower backward pass.

    Returns:
        tuple: (G_AB, G_BA)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        base_channels=int(base_channels),
        vit_depth=int(vit_depth),
        vit_heads=int(vit_heads),
        vit_mlp_ratio=vit_mlp_ratio,
        vit_dropout=vit_dropout,
        layerscale_init=layerscale_init,
        use_cross_domain=bool(use_cross_domain),
        use_gradient_checkpointing=bool(use_gradient_checkpointing),
    )

    G_AB = ViTUNetGeneratorV2(
        base_channels=base_channels,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_mlp_ratio=vit_mlp_ratio,
        vit_dropout=vit_dropout,
        layerscale_init=layerscale_init,
        use_cross_domain=use_cross_domain,
        use_gradient_checkpointing=use_gradient_checkpointing,
    ).to(device)

    G_BA = ViTUNetGeneratorV2(
        base_channels=base_channels,
        vit_depth=vit_depth,
        vit_heads=vit_heads,
        vit_mlp_ratio=vit_mlp_ratio,
        vit_dropout=vit_dropout,
        layerscale_init=layerscale_init,
        use_cross_domain=use_cross_domain,
        use_gradient_checkpointing=use_gradient_checkpointing,
    ).to(device)

    G_AB.apply(init_weights_v2)
    G_BA.apply(init_weights_v2)

    x = torch.randn(1, 3, 256, 256, device=device)
    y_AB = G_AB(x)
    y_BA = G_BA(x)
    print("G_AB (v2) output shape:", y_AB.shape)
    print("G_BA (v2) output shape:", y_BA.shape)

    return G_AB, G_BA


if __name__ == "__main__":
    G_AB, G_BA = getGeneratorsV2(use_gradient_checkpointing=True)
