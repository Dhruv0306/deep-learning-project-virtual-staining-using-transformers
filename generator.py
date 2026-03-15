"""
Generator definition for CycleGAN.

Includes a ResNet-based generator, weight initialization, and a helper
to build the two generators required for bidirectional translation.
"""

# Imports
import torch
import torch.nn as nn


# CycleGAN Generator Structure
"""
CycleGAN Generator Architecture:
    Input (256x256x3)
    ->
    7x7 Conv - Initial feature extraction with reflection padding
    ->
    Downsample (Conv stride 2) x2 - Reduce spatial dimensions while increasing channels
    ->
    ResNet Blocks x6 (or x9 for 256x256) - Feature transformation while preserving structure
    ->
    Upsample x2 (ConvTranspose) - Restore spatial dimensions while reducing channels
    ->
    7x7 Conv - Final feature mapping to output channels
    ->
    Tanh - Output activation to range [-1, 1]
"""


# Residual Block (Core Component)
# This preserves structure -- extremely important in tissue morphology.
class ResnetBlock(nn.Module):
    """
    Residual block with skip connections for the CycleGAN generator.

    Uses reflection padding to avoid border artifacts and instance normalization
    for better style transfer performance. The skip connection helps preserve
    important structural information during transformation.

    Args:
        dim (int): Number of input and output channels.
    """

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        # Sequential block with two conv layers, normalization, and activation.
        # Reflection padding helps avoid border artifacts in image generation.
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),  # Pad with reflection to maintain image boundaries
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),  # 3x3 convolution
            nn.InstanceNorm2d(dim),  # Instance normalization for style transfer
            nn.ReLU(True),  # ReLU activation with inplace operation
            nn.ReflectionPad2d(1),  # Second reflection padding
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),  # Second 3x3 convolution
            nn.InstanceNorm2d(dim),  # Second instance normalization (no activation)
        )

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim, height, width).

        Returns:
            torch.Tensor: Output tensor with same shape as input.
        """
        # Add input to the output of the block (residual connection).
        return x + self.block(x)


# Generator Model
class ResnetGenerator(nn.Module):
    """
    ResNet-based generator for CycleGAN.

    This generator uses an encoder-decoder architecture with residual blocks
    in the middle. It is designed for image-to-image translation tasks where
    preserving structural information is crucial.

    Architecture:
    - Encoder: 7x7 conv + 2 downsampling layers.
    - Transformer: n_blocks residual blocks.
    - Decoder: 2 upsampling layers + 7x7 conv + tanh.

    Args:
        input_nc (int): Number of input channels (default: 3 for RGB).
        output_nc (int): Number of output channels (default: 3 for RGB).
        n_blocks (int): Number of residual blocks (default: 9 for 256x256 images).
    """

    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super(ResnetGenerator, self).__init__()

        model = []

        # Initial 7x7 convolution - feature extraction from input image.
        # Uses reflection padding to avoid border artifacts.
        model += [
            nn.ReflectionPad2d(3),  # Pad by 3 pixels on each side for 7x7 kernel
            nn.Conv2d(input_nc, 64, kernel_size=7, bias=False),  # 7x7 conv to 64 channels
            nn.InstanceNorm2d(64),  # Instance normalization
            nn.ReLU(True),  # ReLU activation
        ]

        # Downsampling - reduce spatial dimensions while increasing feature channels.
        # Two downsampling layers: 64->128->256 channels, size/4.
        in_features = 64
        out_features = in_features * 2

        for _ in range(2):  # Two downsampling layers
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

        # Residual blocks - feature transformation while preserving structure.
        # These blocks allow the network to learn complex transformations
        # while maintaining important structural information through skip connections.
        for _ in range(n_blocks):
            model += [ResnetBlock(in_features)]

        # Upsampling - restore spatial dimensions while reducing feature channels.
        # Two upsampling layers: 256->128->64 channels, size*4.
        out_features = in_features // 2

        for _ in range(2):  # Two upsampling layers
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

        # Final layer - map features to output image.
        # 7x7 convolution to output channels with tanh activation.
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
        tuple: (G_AB, G_BA) - Two initialized ResNet generators.
    """
    # Determine device (GPU if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two generators with identical architecture.
    G_AB = ResnetGenerator().to(device)  # Generator Unstained -> Stained
    G_BA = ResnetGenerator().to(device)  # Generator Stained -> Unstained

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
