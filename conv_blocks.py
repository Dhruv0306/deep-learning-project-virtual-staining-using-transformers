"""
Common convolutional building blocks used across the generator architecture.

These blocks follow the same design conventions as the main CycleGAN generator:
- Reflection padding for image boundary preservation
- Instance normalization for style transfer tasks
- ReLU activation with inplace operations
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Basic convolutional block used for feature extraction.

    This block is commonly used in encoder/decoder stages of image
    translation networks.

    Structure:
        ReflectionPad2d
        -> Conv2d
        -> InstanceNorm2d
        -> ReLU

    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Reflection padding size applied before convolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
            ),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(True),
        )

    def forward(self, x):
        """
        Forward pass through the convolutional block.
        """
        return self.block(x)
