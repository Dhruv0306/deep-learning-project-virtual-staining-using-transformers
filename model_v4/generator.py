"""
model_v4/generator.py — Phase 1 CNN generator for CUT + Transformer (v4).

Provides a lightweight ResNet-style generator (ResnetGenerator) used as the
baseline GAN backbone before the Transformer encoder is introduced in Phase 3.

Architecture summary:
    Input → 7×7 reflect-pad conv → 2× stride-2 downsampling
          → N ResNet blocks → 2× nearest-neighbour upsampling
          → 7×7 reflect-pad conv → Tanh
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model_v4.transformer_blocks import (
    PatchEmbed,
    TransformerBlock,
    _get_2d_sincos_pos_embed,
)


class ResnetBlock(nn.Module):
    """
    Standard residual block with reflection padding and InstanceNorm.

    Each block applies two 3×3 convolutions with reflection padding so that
    spatial dimensions are preserved.  The input is added back as a skip
    connection (identity residual).

    Args:
        channels: Number of input/output feature channels.
        dropout: Dropout probability inserted between the two convolutions.
                 Set to 0.0 to disable.
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


class ResnetGenerator(nn.Module):
    """
    ResNet-based generator for the Phase 1 GAN baseline.

    Encoder–residual–decoder structure:
        Input  → 7×7 reflect-pad conv (c1 channels)
               → stride-2 conv (c2 channels)
               → stride-2 conv (c3 channels)
               → num_res_blocks × ResnetBlock
               → nearest-upsample + 3×3 conv (c2 channels)
               → nearest-upsample + 3×3 conv (c1 channels)
               → 7×7 reflect-pad conv (output_nc channels) → Tanh

    Channel widths: c1 = base_channels, c2 = 2×base_channels,
                    c3 = 4×base_channels.

    Args:
        input_nc:       Number of input image channels (default 3 for RGB).
        output_nc:      Number of output image channels.
        base_channels:  Width of the first conv layer; subsequent layers
                        double up to 4× this value.
        num_res_blocks: Number of residual blocks in the bottleneck.
        dropout:        Dropout probability inside each ResnetBlock.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 6,
        dropout: float = 0.0,
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

        blocks = [ResnetBlock(c3, dropout=dropout) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*blocks)

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

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        f0 = self.in_conv(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f3 = self.res_blocks(f2)
        return f0, f1, f2, f3

    def encode_features(
        self, x: torch.Tensor, nce_layers: list[int] | tuple[int, ...] | None = None
    ) -> list[torch.Tensor]:
        """
        Extract encoder features for PatchNCE.

        Args:
            x: Input image tensor.
            nce_layers: Indices into the feature list [f0, f1, f2, f3].
                If None, returns all features.
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
        x = self.up1(f3)
        x = self.up2(x)
        out = self.out_conv(x)
        if return_features:
            feats = [f0, f1, f2, f3]
            if nce_layers is not None:
                feats = [feats[i] for i in nce_layers]
            return out, feats
        return out


class TransformerGeneratorV4(nn.Module):
    """
    Phase 3 transformer-encoder + CNN-decoder generator.

    Flow:
        Input → PatchEmbed → Transformer blocks → tokens
             → reshape to spatial map → CNN upsampling decoder → output
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
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Token → feature map projection before decoder.
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, base_channels * 4, kernel_size=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )

        # Upsampling decoder: number of 2× upsamples equals log2(patch_size).
        num_upsamples = int(math.log2(patch_size))
        up_blocks = []
        in_ch = base_channels * 4
        for i in range(num_upsamples):
            out_ch = base_channels if i == num_upsamples - 1 else max(
                base_channels, in_ch // 2
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

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def _tokens_to_map(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
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
            if (
                self.use_gradient_checkpointing
                and self.training
                and tokens.requires_grad
            ):
                tokens = grad_checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            if return_features and (nce_layers is None or idx in nce_layers):
                features.append(tokens)

        tokens = self.norm(tokens)
        return tokens, grid, features

    def encode_features(
        self, x: torch.Tensor, nce_layers: list[int] | tuple[int, ...] | None = None
    ) -> list[torch.Tensor]:
        _, grid, feats = self._encode_tokens(
            x, return_features=True, nce_layers=nce_layers
        )
        return [self._tokens_to_map(f, grid) for f in feats]

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
        out = self.out_conv(y)

        if return_features:
            maps = [self._tokens_to_map(f, grid) for f in feats]
            return out, maps
        return out


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


def getGeneratorV4(
    input_nc: int = 3,
    output_nc: int = 3,
    base_channels: int = 64,
    num_res_blocks: int = 6,
    dropout: float = 0.0,
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
) -> nn.Module:
    """
    Build, initialise, and optionally smoke-test a Phase 1 ResnetGenerator.

    Args:
        input_nc:       Input image channels.
        output_nc:      Output image channels.
        base_channels:  Base feature-map width (default 64).
        num_res_blocks: Residual blocks in the bottleneck (default 6).
        dropout:        Dropout inside residual blocks (default 0 = off).
        use_transformer_encoder: If True, builds the Phase 3 transformer encoder.
        image_size:     Input image size (square, used for patch embedding).
        patch_size:     Transformer patch size (power of 2 recommended).
        encoder_dim:    Transformer token dimension.
        encoder_depth:  Number of transformer blocks.
        encoder_heads:  Attention heads per block.
        encoder_mlp_ratio: MLP expansion ratio in transformer blocks.
        encoder_dropout: Dropout inside transformer blocks.
        use_gradient_checkpointing: Enable activation checkpointing in the transformer.
        device:         Target device; defaults to CUDA if available.
        run_smoke_test: If True, runs a single forward pass on a random
                        256×256 tensor and prints the output shape.

    Returns:
        Initialised generator on the requested device.
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
        ).to(device)
    model.apply(init_weights_v4)

    if run_smoke_test:
        with torch.no_grad():
            x = torch.randn(1, input_nc, 256, 256, device=device)
            y = model(x)
            print(f"[getGeneratorV4] output shape: {y.shape}")

    return model


if __name__ == "__main__":
    _ = getGeneratorV4()
