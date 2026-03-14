"""
Swin-style transformer blocks used in the generator bottleneck.

This file implements window partition/reverse utilities and a simplified
Swin block with optional window shifts.
"""

import torch
import torch.nn as nn


def window_partition(x, window_size):
    """
    Partition a feature map into non-overlapping windows.

    Args:
        x (Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): Window size along H and W.

    Returns:
        Tensor: Windowed tensor of shape
            (B * num_windows, window_size, window_size, C).

    Note:
        H and W must be divisible by window_size.
    """

    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    windows = windows.view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.

    Args:
        windows (Tensor): Windowed tensor of shape
            (B * num_windows, window_size, window_size, C).
        window_size (int): Window size used during partition.
        H (int): Original height.
        W (int): Original width.

    Returns:
        Tensor: Reconstructed feature map of shape (B, H, W, C).
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))

    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )

    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    x = x.view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.

    Used inside the Swin Transformer block on flattened windows.

    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, dim, num_heads):

        super(WindowAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, C),
                where N is the number of tokens in a window.
        """

        B, N, C = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)

        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)

        return out


class SwinTransformerBlock(nn.Module):
    """
    Simplified Swin Transformer block used inside the generator bottleneck.

    Structure:
        WindowAttention (with optional window shift)
        -> MLP with residual connection

    Note:
        This implementation applies the MLP residual only.
    """

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):

        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Convert to channels-last for window partitioning.
        x = x.permute(0, 2, 3, 1)  # B, H, W, C

        # Shift feature map for alternating blocks (no attention mask applied).
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # Partition into windows and apply attention.
        windows = window_partition(x, self.window_size)

        windows = windows.view(-1, self.window_size * self.window_size, C)

        windows = self.norm1(windows)

        attn_windows = self.attn(windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse windows back to feature map.
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Undo shift for shifted blocks.
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Back to channels-first.
        x = x.permute(0, 3, 1, 2)

        # MLP over flattened spatial tokens with residual.
        x = x + self.mlp(self.norm2(x.flatten(2).transpose(1, 2))).transpose(1, 2).view(
            B, C, H, W
        )

        return x


class SwinTransformerBottleneck(nn.Module):
    """
    Transformer bottleneck used in the generator.

    Stacks multiple SwinTransformerBlocks with alternating window shifts.
    """

    def __init__(self, dim=256, n_blocks=6, window_size=8):

        super(SwinTransformerBottleneck, self).__init__()

        self.blocks = nn.ModuleList()

        for i in range(n_blocks):

            shift = 0 if i % 2 == 0 else window_size // 2

            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, num_heads=4, window_size=window_size, shift_size=shift
                )
            )

    def forward(self, x):

        # Sequentially apply all blocks.
        for block in self.blocks:
            x = block(x)

        return x
