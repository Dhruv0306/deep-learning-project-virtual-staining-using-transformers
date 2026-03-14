"""
UNet encoder and decoder blocks used in the hybrid generator.

These blocks are used to build the encoder-decoder structure around
the transformer bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    """
    Downsampling block for encoder stage.

    Reduces spatial resolution while increasing feature channels.

    Structure:
        Conv2d (stride 2)
        -> InstanceNorm
        -> ReLU

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """

    def __init__(self, in_features, out_features):
        super(DownsampleBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    Upsampling block for decoder stage.

    Restores spatial resolution and merges skip connections from the encoder.

    Structure:
        Bilinear upsample
        -> Concatenate skip connection
        -> Conv block

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.

    Note:
        `in_features` should match the channel count after concatenation:
        (decoder_channels + skip_channels).
    """

    def __init__(self, in_features, out_features):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=3,
                bias=False,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        )

    def forward(self, x, skip):
        """
        Forward pass with skip connection.

        Args:
            x (Tensor): Decoder input of shape (B, C, H, W).
            skip (Tensor): Encoder feature map of shape (B, C_skip, 2H, 2W).
        """

        # Upsample spatial resolution (bilinear for smoothness).
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Concatenate encoder feature map along channels.
        x = torch.cat([x, skip], dim=1)

        return self.conv(x)
