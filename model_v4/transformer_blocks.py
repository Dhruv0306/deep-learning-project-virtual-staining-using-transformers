"""
Transformer blocks for the v4 Phase 3 generator.

Provides PatchEmbed and basic TransformerBlock utilities for a lightweight
ViT-style encoder used in unpaired translation.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


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


class PatchEmbed(nn.Module):
    """
    Patchify image and project to token embeddings.

    Returns tokens of shape (B, N, C) and grid size (H', W').
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 192,
        patch_size: int = 8,
        image_size: int = 256,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, (h, w)


class TransformerBlock(nn.Module):
    """
    Lightweight Transformer block with pre-norm attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
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
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


__all__ = [
    "PatchEmbed",
    "TransformerBlock",
    "_get_2d_sincos_pos_embed",
]
