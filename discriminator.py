"""
Discriminator definition for CycleGAN.

Implements a PatchGAN discriminator and a helper to build the two
required discriminators for the A and B domains.
"""

# Imports
import torch
import torch.nn as nn
from generator import init_weights

"""
PatchGAN Discriminator:
    Input (256x256x3)
    ->
    Conv (stride 2)
    ->
    Conv (stride 2)
    ->
    Conv (stride 2)
    ->
    Conv (stride 1)
    ->
    Conv -> Output feature map
"""


# Discriminator definition
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies overlapping patches of an image as real or fake.

    This discriminator uses a series of convolutional layers to downsample the input image
    and produce a patch-based output where each element corresponds to a patch of the input.

    Args:
        input_nc (int): Number of input channels (default: 3 for RGB images).
    """

    def __init__(self, input_nc=3):
        super(PatchDiscriminator, self).__init__()

        model = []

        # First layer (no normalization).
        # Initial convolution without normalization to avoid early artifacts.
        model += [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]

        # Downsampling layers.
        # Progressive downsampling with increasing number of features.
        in_features = 64
        out_features = in_features * 2

        # Add two downsampling blocks with instance normalization.
        for _ in range(1, 3):
            model += [
                nn.Conv2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,  # No bias when using normalization
                ),
                nn.InstanceNorm2d(out_features),  # Instance normalization for stability
                nn.LeakyReLU(0.2, True),  # Leaky ReLU with negative slope of 0.2
            ]
            in_features = out_features
            out_features *= 2

        # One more layer with stride=1.
        # Increases receptive field without further downsampling.
        model += [
            nn.Conv2d(
                in_features,
                in_features * 2,
                kernel_size=4,
                stride=1,  # Stride=1 to maintain spatial dimensions
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(in_features * 2),
            nn.LeakyReLU(0.2, True),
        ]

        # Final output layer.
        # Output single channel for real/fake classification per patch.
        model += [nn.Conv2d(in_features * 2, 1, kernel_size=4, stride=1, padding=1)]

        # Combine all layers into a sequential model.
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass through the discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_nc, height, width).

        Returns:
            torch.Tensor: Output tensor with patch-based predictions.
        """
        return self.model(x)


def getDiscriminators():
    """
    Create and initialize two PatchDiscriminator instances for CycleGAN.

    This function creates two discriminators (D_A and D_B) for the two domains
    in CycleGAN, initializes their weights, and tests them with random input.

    Returns:
        tuple: (D_A, D_B) - The two discriminator networks.
    """
    # Set device to GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two discriminator instances for domains A (unstained) and B (stained).
    D_A = PatchDiscriminator().to(device)
    D_B = PatchDiscriminator().to(device)

    # Apply weight initialization to both discriminators.
    D_A.apply(init_weights)
    D_B.apply(init_weights)

    # Test the discriminators with random input.
    x = torch.randn(1, 3, 256, 256).to(device)  # Random 256x256 RGB image
    y_A = D_A(x)  # Forward pass through discriminator A
    y_B = D_B(x)  # Forward pass through discriminator B

    # Print output shapes for verification.
    print(f"D_A output shape: {y_A.shape}")
    print(f"D_B output shape: {y_B.shape}")

    return D_A, D_B


# Main execution block
if __name__ == "__main__":
    # Create and test the discriminators when script is run directly.
    D_A, D_B = getDiscriminators()
