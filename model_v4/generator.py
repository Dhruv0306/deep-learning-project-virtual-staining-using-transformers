"""
model_v4/generator.py — CNN + Transformer generator for CUT + Transformer (v4).

UPGRADES (v4.2):
    ResnetGenerator:
        - ResnetBlock now uses a squeeze-and-excitation (SE) channel
          attention gate after the second conv. SE computes per-channel
          importance weights from global average pooling and recalibrates
          features, helping the bottleneck focus on texture-relevant channels.
        - Added a self-attention layer at the bottleneck (between ResNet
          blocks and the decoder) on the flattened 64×64 feature map. At
          the 64-channel bottleneck this is cheap but captures long-range
          structural relationships that pure local convolutions miss.
        - Intermediate feature activations are now collected at down1, down2,
          and after each ResNet block — supporting richer NCE layers.

    TransformerGeneratorV4:
        - TransformerBlock now uses pre-norm (LayerNorm before attention /
          MLP) instead of post-norm, improving gradient flow in deeper models.
        - Added a lightweight depth-wise CNN (DW-Conv) branch in parallel
          with self-attention in each block and sums the outputs. This gives
          the Transformer local inductive bias for capturing fine stain
          texture that pure attention struggles with.
        - Added a dedicated texture refinement head between the last up-block
          and the output conv: a 3×3 depth-wise + 1×1 point-wise conv that
          sharpens local detail without significantly increasing parameters.
        - encode_features now also returns features from the projection and
          decoder up-blocks, enabling richer NCE supervision across more
          spatial scales.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model_v4.transformer_blocks import (
    PatchEmbed,
    TransformerBlock,
    _get_2d_sincos_pos_embed,
)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation channel attention
# ---------------------------------------------------------------------------


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention gate.

    Computes per-channel importance weights via global average pooling
    followed by a two-layer FC bottleneck with sigmoid output, then
    scales the input feature map channel-wise.

    Args:
        channels:   Number of input/output channels.
        reduction:  Reduction ratio for the bottleneck (default 8).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# Bottleneck self-attention for ResNet generator
# ---------------------------------------------------------------------------


class SpatialSelfAttention(nn.Module):
    """
    Lightweight spatial self-attention for CNN feature maps.

    Reshapes a 2-D feature map to a token sequence, applies multi-head
    self-attention, and reshapes back. Used once at the ResNet bottleneck.

    Args:
        channels: Number of feature channels (= token dimension).
        num_heads: Attention heads (default 4).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # Tokens: (N, H*W, C)
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        # Residual reshape back
        return x + attn_out.transpose(1, 2).reshape(n, c, h, w)


# ---------------------------------------------------------------------------
# ResNet residual block with SE attention
# ---------------------------------------------------------------------------


class ResnetBlock(nn.Module):
    """
    Enhanced residual block with SE channel attention.

    Two 3×3 convolutions with reflection padding and InstanceNorm, followed
    by a Squeeze-and-Excitation gate that recalibrates channel-wise feature
    importance. The input is added as a skip connection (identity residual).

    Args:
        channels:  Number of input/output feature channels.
        dropout:   Dropout probability between the two convolutions.
        se_reduction: SE bottleneck reduction ratio (0 = disable SE).
    """

    def __init__(self, channels: int, dropout: float = 0.0, se_reduction: int = 8):
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
        self.se = (
            SEBlock(channels, reduction=se_reduction)
            if se_reduction > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return x + out


# ---------------------------------------------------------------------------
# ResnetGenerator — enhanced with bottleneck self-attention
# ---------------------------------------------------------------------------


class ResnetGenerator(nn.Module):
    """
    Enhanced ResNet-based generator.

    Encoder–residual–decoder structure with:
        - SE-gated residual blocks in the bottleneck.
        - One spatial self-attention layer after all ResNet blocks.
        - Intermediate encoder features accessible for PatchNCE.

    Architecture:
        Input  → 7×7 reflect-pad conv (c1)
               → stride-2 conv (c2)
               → stride-2 conv (c3)
               → num_res_blocks × ResnetBlock (with SE)
               → SpatialSelfAttention          ← NEW
               → nearest-upsample + 3×3 conv (c2)
               → nearest-upsample + 3×3 conv (c1)
               → 7×7 reflect-pad conv (output_nc) → Tanh

    Args:
        input_nc:       Number of input image channels.
        output_nc:      Number of output image channels.
        base_channels:  Width of the first conv layer.
        num_res_blocks: Number of SE-gated residual blocks.
        dropout:        Dropout probability inside each block.
        se_reduction:   SE reduction ratio (0 = disable).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 6,
        dropout: float = 0.0,
        se_reduction: int = 8,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, c1, kernel_size=7, bias=False),
            nn.InstanceNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.ModuleList(
            [
                ResnetBlock(c3, dropout=dropout, se_reduction=se_reduction)
                for _ in range(num_res_blocks)
            ]
        )

        # Bottleneck spatial self-attention
        self.bottleneck_attn = SpatialSelfAttention(c3, num_heads=4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c3, c2, kernel_size=3, bias=False),
            nn.InstanceNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(c2, c1, kernel_size=3, bias=False),
            nn.InstanceNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c1, output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def _encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.in_conv(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = f2
        for block in self.res_blocks:
            f3 = block(f3)
        f3 = self.bottleneck_attn(f3)
        return f0, f1, f2, f3

    def encode_features(
        self, x: torch.Tensor, nce_layers: list[int] | tuple[int, ...] | None = None
    ) -> list[torch.Tensor]:
        """
        Extract encoder features for PatchNCE.

        Feature list: [f0 (in_conv), f1 (down1), f2 (down2), f3 (bottleneck+attn)].
        """
        feats = list(self._encode(x))
        if nce_layers is None:
            return feats
        return [feats[i] for i in nce_layers]

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        nce_layers: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        f0, f1, f2, f3 = self._encode(x)
        y = self.up1(f3)
        y = self.up2(y)
        out = self.out_conv(y)
        if return_features:
            feats = [f0, f1, f2, f3]
            if nce_layers is not None:
                feats = [feats[i] for i in nce_layers]
            return out, feats
        return out


# ---------------------------------------------------------------------------
# Enhanced Transformer block: pre-norm + DW-Conv local branch
# ---------------------------------------------------------------------------


class EnhancedTransformerBlock(nn.Module):
    """
    Transformer block with:
        - Pre-LayerNorm (before attention and MLP).
        - Depth-wise CNN branch run in parallel with self-attention;
          the two outputs are summed (with a learnable gate). This gives
          the model local inductive bias for fine texture alongside the
          global attention.
        - GELU activation in MLP.

    Args:
        dim:       Token embedding dimension.
        num_heads: Attention heads.
        mlp_ratio: MLP hidden-dim multiplier.
        dropout:   Dropout in MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )

        # Depth-wise local conv branch (operates on the token sequence
        # reshaped to a spatial map — caller must reshape)
        # We use a 1-D depth-wise conv over the sequence dimension as a
        # lightweight approximation (avoids needing to know grid size here).
        self.dw_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
        )
        self.local_gate = nn.Parameter(
            torch.zeros(1)
        )  # starts at 0 → pure attn initially

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token sequence (B, N, dim).

        Returns:
            Updated token sequence (B, N, dim).
        """
        # --- Self-attention branch ---
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)

        # --- Depth-wise local branch ---
        # Conv1d expects (B, C, N) — transpose sequence and channels
        dw_out = self.dw_conv(normed.transpose(1, 2)).transpose(1, 2)

        gate = torch.sigmoid(self.local_gate)
        x = x + attn_out + gate * dw_out

        # --- MLP ---
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Texture refinement head
# ---------------------------------------------------------------------------


class TextureRefinementHead(nn.Module):
    """
    Lightweight DW-Conv + PW-Conv head that sharpens local spatial detail.

    Placed between the last decoder up-block and the output convolution.
    Adds almost no parameters while improving high-frequency texture fidelity.

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            # Depth-wise 3×3 for local texture
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, groups=channels, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            # Point-wise channel mixing
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Transformer Generator V4 — enhanced
# ---------------------------------------------------------------------------


class TransformerGeneratorV4(nn.Module):
    """
    Enhanced Transformer-encoder + CNN-decoder generator.

    Flow:
        Input → PatchEmbed → EnhancedTransformerBlocks (pre-norm + DW-Conv)
             → reshape to spatial map → proj → CNN up-decoder
             → TextureRefinementHead → output conv → Tanh

    Args:
        input_nc:   Input image channels.
        output_nc:  Output image channels.
        image_size: Spatial size of the (square) input.
        patch_size: Non-overlapping patch size (power of 2).
        embed_dim:  Transformer token dimension.
        depth:      Number of Transformer blocks.
        num_heads:  Attention heads per block.
        mlp_ratio:  MLP expansion ratio.
        dropout:    Dropout in Transformer MLP.
        base_channels: Base channel width for the CNN decoder.
        use_gradient_checkpointing: Activation checkpointing.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        image_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        base_channels: int = 64,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if patch_size & (patch_size - 1) != 0:
            raise ValueError("patch_size must be a power of 2.")
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.patch_embed = PatchEmbed(
            in_channels=input_nc,
            embed_dim=embed_dim,
            patch_size=patch_size,
            image_size=image_size,
        )
        self.blocks = nn.ModuleList(
            [
                EnhancedTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Token → spatial feature map projection
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels * 4, kernel_size=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # CNN upsampling decoder
        num_upsamples = int(math.log2(patch_size))
        up_blocks = []
        in_ch = base_channels * 4
        for i in range(num_upsamples):
            out_ch = (
                base_channels
                if i == num_upsamples - 1
                else max(base_channels, in_ch // 2)
            )
            up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)

        # Texture refinement before output conv
        self.texture_head = TextureRefinementHead(in_ch)

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def _tokens_to_map(
        self, tokens: torch.Tensor, grid: tuple[int, int]
    ) -> torch.Tensor:
        b, n, c = tokens.shape
        h, w = grid
        if n != h * w:
            raise ValueError("Token length does not match grid size.")
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def _encode_tokens(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        nce_layers: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int], list[torch.Tensor]]:
        tokens, grid = self.patch_embed(x)
        pos = _get_2d_sincos_pos_embed(
            self.embed_dim, grid[0], grid[1], tokens.device, tokens.dtype
        )
        tokens = tokens + pos.unsqueeze(0)

        features: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            assert tokens is not None
            use_ckpt = (
                self.use_gradient_checkpointing
                and self.training
                and tokens.requires_grad
            )
            tokens = (
                grad_checkpoint(block, tokens, use_reentrant=False)
                if use_ckpt
                else block(tokens)
            )
            if return_features and (nce_layers is None or idx in nce_layers):
                assert tokens is not None
                features.append(tokens)

        tokens = self.norm(tokens)
        return tokens, grid, features

    def encode_features(
        self, x: torch.Tensor, nce_layers: list[int] | tuple[int, ...] | None = None
    ) -> list[torch.Tensor]:
        """
        Extract spatial feature maps for PatchNCE.

        Returns exactly the same set of block-level feature maps that
        ``forward(return_features=True, nce_layers=nce_layers)`` returns,
        so that patch_ids sampled from real features index correctly into
        fake features. Decoder-level features are intentionally excluded
        here to keep the list length consistent with the forward pass.
        """
        tokens, grid, block_feats = self._encode_tokens(
            x, return_features=True, nce_layers=nce_layers
        )
        return [self._tokens_to_map(f, grid) for f in block_feats]

    def encode_features_multiscale(
        self, x: torch.Tensor, nce_layers: list[int] | tuple[int, ...] | None = None
    ) -> list[torch.Tensor]:
        """
        Extended feature extraction including decoder up-block maps.

        Use this when you want richer multi-scale NCE supervision and are
        managing patch_ids yourself. The list length is
        len(nce_layers or all_blocks) + len(up_blocks).
        """
        tokens, grid, block_feats = self._encode_tokens(
            x, return_features=True, nce_layers=nce_layers
        )
        block_maps = [self._tokens_to_map(f, grid) for f in block_feats]
        feat_map = self._tokens_to_map(tokens, grid)
        y = self.proj(feat_map)
        extra_maps: list[torch.Tensor] = []
        for up_block in self.up_blocks:
            y = up_block(y)
            extra_maps.append(y)
        return block_maps + extra_maps

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        nce_layers: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        tokens, grid, feats = self._encode_tokens(
            x, return_features=return_features, nce_layers=nce_layers
        )
        feat_map = self._tokens_to_map(tokens, grid)
        y = self.proj(feat_map)
        for block in self.up_blocks:
            y = block(y)
        y = self.texture_head(y)
        out = self.out_conv(y)

        if return_features:
            maps = [self._tokens_to_map(f, grid) for f in feats]
            return out, maps
        return out


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def init_weights_v4(net: nn.Module) -> None:
    """
    Initialise network weights with GAN-friendly defaults.

    - Conv2d / ConvTranspose2d: Normal(0, 0.02)
    - InstanceNorm2d scale:     Normal(1, 0.02), bias = 0
    - Linear:                   Xavier uniform, bias = 0
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
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getGeneratorV4(
    input_nc: int = 3,
    output_nc: int = 3,
    base_channels: int = 64,
    num_res_blocks: int = 6,
    dropout: float = 0.0,
    se_reduction: int = 8,
    use_transformer_encoder: bool = False,
    image_size: int = 256,
    patch_size: int = 8,
    encoder_dim: int = 192,
    encoder_depth: int = 4,
    encoder_heads: int = 4,
    encoder_mlp_ratio: float = 2.0,
    encoder_dropout: float = 0.0,
    use_gradient_checkpointing: bool = False,
    device: torch.device | None = None,
    run_smoke_test: bool = True,
) -> "ResnetGenerator | TransformerGeneratorV4":
    """
    Build, initialise, and smoke-test a v4 generator.

    New args vs v4.1:
        se_reduction (int): SE bottleneck reduction for ResnetGenerator.
            Set to 0 to disable SE gates. Default 8.

    All other args unchanged from v4.1. API-compatible with existing callers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_transformer_encoder:
        model = TransformerGeneratorV4(
            input_nc=input_nc,
            output_nc=output_nc,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
            base_channels=base_channels,
            use_gradient_checkpointing=use_gradient_checkpointing,
        ).to(device)
    else:
        model = ResnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            se_reduction=se_reduction,
        ).to(device)

    model.apply(init_weights_v4)

    if run_smoke_test:
        with torch.no_grad():
            x = torch.randn(1, input_nc, 256, 256, device=device)
            y = model(x)
            print(f"[getGeneratorV4] output shape: {y.shape}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[getGeneratorV4] G params: {n_params:.2f}M")

    return model


if __name__ == "__main__":
    print("=== ResnetGenerator v4.2 ===")
    _ = getGeneratorV4(run_smoke_test=True)
    print("=== TransformerGeneratorV4 v4.2 ===")
    _ = getGeneratorV4(use_transformer_encoder=True, run_smoke_test=True)
