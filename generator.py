"""
Generator definition for UVCGAN.

Includes:
- UVCGAN-style U-Net + ViT generator (default)
- Weight initialization helpers
- Helper to build the two generators required for bidirectional translation
"""

# Imports
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


# ---------------------------
# UVCGAN Generator (U-Net + ViT)
# ---------------------------


def _get_1d_sincos_pos_embed(embed_dim, pos):
    """
    Build 1D sine-cosine positional embeddings.

    Args:
        embed_dim (int): Embedding dimension.
        pos (torch.Tensor): Positions (N,).
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for sin/cos positional embedding.")
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(embed_dim, height, width, device, dtype):
    """
    Build 2D sine-cosine positional embeddings for (H, W) grid.
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even for 2D sin/cos positional embedding.")

    grid_h = torch.arange(height, device=device, dtype=dtype)
    grid_w = torch.arange(width, device=device, dtype=dtype)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid_h_flat = grid[0].reshape(-1)
    grid_w_flat = grid[1].reshape(-1)

    embed_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid_h_flat)
    embed_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid_w_flat)
    return torch.cat([embed_h, embed_w], dim=1)


class ReZeroTransformerBlock(nn.Module):
    """
    Transformer block with ReZero residual scaling.
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
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
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.alpha_ffn = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.alpha_attn * attn_out
        x = x + self.alpha_ffn * self.mlp(self.norm2(x))
        return x


class PixelwiseViT(nn.Module):
    """
    Pixelwise Vision Transformer operating on flattened spatial tokens.
    """

    def __init__(self, dim, depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ReZeroTransformerBlock(
                    dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        # x: (N, C, H, W) -> tokens: (N, H*W, C)
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        pos = _get_2d_sincos_pos_embed(
            embed_dim=c, height=h, width=w, device=x.device, dtype=x.dtype
        )
        tokens = tokens + pos.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(n, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class ViTUNetGenerator(nn.Module):
    """
    UVCGAN-style generator: U-Net backbone with a ViT bottleneck.
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        base_channels=64,
        vit_depth=4,
        vit_heads=8,
        vit_mlp_ratio=4.0,
        vit_dropout=0.0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.enc1 = ConvBlock(input_nc, c1)
        self.down1 = DownsampleBlock(c1, c2)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = DownsampleBlock(c2, c3)
        self.enc3 = ConvBlock(c3, c3)
        self.down3 = DownsampleBlock(c3, c4)
        self.enc4 = ConvBlock(c4, c4)
        self.down4 = DownsampleBlock(c4, c4)

        self.bottleneck = ConvBlock(c4, c4)
        self.vit = PixelwiseViT(
            dim=c4,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )

        self.up1 = UpsampleBlock(c4, c4)
        self.dec1 = ConvBlock(c4 + c4, c4)
        self.up2 = UpsampleBlock(c4, c3)
        self.dec2 = ConvBlock(c3 + c3, c3)
        self.up3 = UpsampleBlock(c3, c2)
        self.dec3 = ConvBlock(c2 + c2, c2)
        self.up4 = UpsampleBlock(c2, c1)
        self.dec4 = ConvBlock(c1 + c1, c1)

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c1, output_nc, kernel_size=7),
            nn.Tanh(),
        )

    def _cp(self, fn, *args):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _dec1(self, x, skip):
        return self.dec1(torch.cat((x, skip), dim=1))

    def _dec2(self, x, skip):
        return self.dec2(torch.cat((x, skip), dim=1))

    def _dec3(self, x, skip):
        return self.dec3(torch.cat((x, skip), dim=1))

    def _dec4(self, x, skip):
        return self.dec4(torch.cat((x, skip), dim=1))

    def forward(self, x):
        e1 = self._cp(self.enc1, x)
        d1 = self._cp(self.down1, e1)
        e2 = self._cp(self.enc2, d1)
        d2 = self._cp(self.down2, e2)
        e3 = self._cp(self.enc3, d2)
        d3 = self._cp(self.down3, e3)
        e4 = self._cp(self.enc4, d3)
        d4 = self._cp(self.down4, e4)

        b = self._cp(self.bottleneck, d4)
        b = self._cp(self.vit, b)

        u1 = self._cp(self.up1, b)
        u1 = self._cp(self._dec1, u1, e4)
        u2 = self._cp(self.up2, u1)
        u2 = self._cp(self._dec2, u2, e3)
        u3 = self._cp(self.up3, u2)
        u3 = self._cp(self._dec3, u3, e2)
        u4 = self._cp(self.up4, u3)
        u4 = self._cp(self._dec4, u4, e1)
        return self._cp(self.out_conv, u4)


# Weight Initialization
def init_weights(net):
    """
    Initialize network weights using a normal distribution.

    This initialization scheme is commonly used for GANs and helps with
    training stability. Convolutional layers are initialized with small
    random weights, while normalization layers get specific initialization.

    Args:
        net (nn.Module): Network to initialize.
    """
    for m in net.modules():
        # Initialize convolutional and transpose convolutional layers.
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Normal initialization with mean=0, std=0.02 for conv layers.
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # Initialize instance normalization layers.
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                # Normal initialization with mean=1, std=0.02 for scale parameter.
                nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                # Zero initialization for bias parameter.
                nn.init.constant_(m.bias.data, 0)


## Initialize Two Generators
# CycleGAN needs two generators for bidirectional translation:
# - G_AB (Unstained -> Stained)
# - G_BA (Stained -> Unstained)
def getGenerators(use_checkpoint=False):
    """
    Create and initialize two generators for CycleGAN.

    CycleGAN requires two generators for bidirectional translation between
    two domains. Both generators share the same architecture but learn
    different transformations.

    Returns:
        tuple: (G_AB, G_BA) - Two initialized UVCGAN generators.
    """
    # Determine device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two generators with identical architecture.
    G_AB = ViTUNetGenerator(use_checkpoint=use_checkpoint).to(device)
    G_BA = ViTUNetGenerator(use_checkpoint=use_checkpoint).to(device)

    # Apply weight initialization to both generators.
    G_AB.apply(init_weights)
    G_BA.apply(init_weights)

    # Quick smoke test to verify output shape.
    x = torch.randn(1, 3, 256, 256).to(device)  # Random 256x256 RGB image
    y_AB = G_AB(x)  # Forward pass through G_AB generator
    y_BA = G_BA(x)  # Forward pass through G_BA generator

    # Print output shapes for verification (both should be [1, 3, 256, 256]).
    print("G_AB output shape:", y_AB.shape)
    print("G_BA output shape:", y_BA.shape)
    return G_AB, G_BA


if __name__ == "__main__":
    G_AB, G_BA = getGenerators()
