"""
Generator definition for CycleGAN.

Includes a hybrid encoder/decoder generator with a Swin-Transformer
bottleneck, weight initialization, and a helper to build the two
generators required for bidirectional translation.
"""

# Imports
import torch
import torch.nn as nn
from swin_blocks import SwinTransformerBottleneck


# CycleGAN Generator Structure (as implemented here)
"""
CycleGAN Generator Architecture:
    Input (H x W x C)
    ->
    7x7 Conv - Initial feature extraction with reflection padding
    ->
    Downsample (Conv stride 2) x3 - Reduce spatial dimensions while increasing channels
    ->
    Swin Transformer bottleneck xN - Local self-attention with windowed shifts
    ->
    Upsample x3 (ConvTranspose) - Restore spatial dimensions while reducing channels
    ->
    7x7 Conv - Final feature mapping to output channels
    ->
    Tanh - Output activation to range [-1, 1]
"""


# Generator Model
class Generator(nn.Module):
    """
    Hybrid encoder/decoder generator for CycleGAN with a Swin bottleneck.

    This generator uses a CNN encoder-decoder with a transformer bottleneck
    for image-to-image translation tasks where spatial structure should be
    preserved while allowing flexible feature transformation.

    Architecture:
    - Encoder: 7x7 conv + 3 downsampling layers.
    - Bottleneck: Swin Transformer blocks (n_blocks).
    - Decoder: 3 upsampling layers + 7x7 conv + tanh.

    Args:
        input_nc (int): Number of input channels (default: 3 for RGB).
        output_nc (int): Number of output channels (default: 3 for RGB).
        n_blocks (int): Number of transformer blocks in the bottleneck.
    """

    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super(Generator, self).__init__()

        model = []

        # Initial 7x7 convolution for low-level feature extraction.
        # Reflection padding reduces edge artifacts.
        model += [
            nn.ReflectionPad2d(3),  # Pad by 3 pixels on each side for 7x7 kernel
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),  # 7x7 conv to 64 channels
            nn.InstanceNorm2d(64),  # Instance normalization
            nn.ReLU(True),  # ReLU activation
        ]

        # Downsampling: reduce spatial resolution and expand channels.
        # Three layers: 64 -> 128 -> 256 -> 512 channels, H/W reduced by 8x.
        in_features = 64
        out_features = in_features * 2

        for _ in range(3):  # Three downsampling layers
            model += [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,  # Stride 2 for downsampling
                    padding=1,  # Maintain spatial relationships
                    bias=False,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features *= 2  # Double channels for next layer

        # Swin Transformer bottleneck for localized self-attention.
        model += [SwinTransformerBottleneck(dim=in_features, n_blocks=n_blocks, window_size=8)]

        # Upsampling: restore spatial resolution and reduce channels.
        # Three layers: 512 -> 256 -> 128 -> 64 channels, H/W increased by 8x.
        out_features = in_features // 2

        for _ in range(3):  # Three upsampling layers
            model += [
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=2,  # Stride 2 for upsampling
                    padding=1,
                    output_padding=1,  # Ensure correct output size
                    bias=False,
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features = in_features // 2  # Halve channels for next layer

        # Final layer: map features to output image with tanh activation.
        model += [
            nn.ReflectionPad2d(3),  # Reflection padding for 7x7 kernel
            nn.Conv2d(64, output_nc, kernel_size=7),  # Final 7x7 conv to output channels
            nn.Tanh(),  # Tanh activation to output range [-1, 1]
        ]

        # Combine all layers into a sequential model.
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input image tensor of shape
                (batch_size, input_nc, height, width).

        Returns:
            torch.Tensor: Generated image tensor of shape
                (batch_size, output_nc, height, width).
        """
        return self.model(x)


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
def getGenerators():
    """
    Create and initialize two generators for CycleGAN.

    CycleGAN requires two generators for bidirectional translation between
    two domains. Both generators share the same architecture but learn
    different transformations.

    Returns:
        tuple: (G_AB, G_BA) - Two initialized generator instances.
    """
    # Determine device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two generators with identical architecture.
    G_AB = Generator().to(device)  # Generator Unstained -> Stained
    G_BA = Generator().to(device)  # Generator Stained -> Unstained

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
