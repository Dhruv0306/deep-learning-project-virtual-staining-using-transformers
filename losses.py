"""
Loss definitions for the CycleGAN training loop.

Includes a VGG19-based perceptual loss and a composite CycleGAN loss
with GAN, cycle-consistency, identity, and perceptual terms.
"""

# Imports
import torch
import torch.nn as nn
import torchvision.models as models
from replay_buffer import ReplayBuffer

# Full Loss Structure
# L_G = L_GAN + lambda_cycle * L_cycle + lambda_identity * L_identity
#       + lambda_cycle_perceptual * L_cycle_perceptual
#       + lambda_identity_perceptual * L_identity_perceptual


# VGG19 Perceptual Loss Network
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 feature maps.

    Computes L1 distance between VGG19 activations of two images. Uses
    pretrained ImageNet weights and keeps all parameters frozen.
    """

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.eval()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """
        Compute perceptual loss between two image batches.

        Args:
            x (torch.Tensor): Generated images (N, C, H, W).
            y (torch.Tensor): Target images (N, C, H, W).

        Returns:
            torch.Tensor: Scalar perceptual loss.
        """
        # Expand grayscale to 3 channels if needed for VGG19.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        x = self.normalize(x)
        y = self.normalize(y)
        x_feat1, x_feat2, x_feat3 = self.extract_features(x)
        y_feat1, y_feat2, y_feat3 = self.extract_features(y)

        loss = (
            nn.functional.l1_loss(x_feat1, y_feat1)
            + nn.functional.l1_loss(x_feat2, y_feat2)
            + nn.functional.l1_loss(x_feat3, y_feat3)
        )
        return loss

    def extract_features(self, x):
        """
        Run inputs through selected VGG19 layers.

        Returns:
            tuple: Feature maps from relu1_2, relu2_2, relu3_4.
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h1, h2, h3

    def normalize(self, x):
        """
        Normalize inputs with ImageNet mean and std.
        """
        return (x - self.mean) / self.std


# CycleGAN Loss Class
class CycleGANLoss:
    """
    Composite loss for CycleGAN training.

    Combines GAN losses (LSGAN), cycle-consistency, identity, and
    optional perceptual components.
    """

    def __init__(
        self,
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_cycle_perceptual=0.1,
        lambda_identity_perceptual=0.05,
        device=None,
    ):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_cycle_perceptual = lambda_cycle_perceptual
        self.lambda_identity_perceptual = lambda_identity_perceptual

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Loss functions
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss().to(device=self.device)

        # Keep independent replay buffers across training steps for both domains.
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def get_identity_lambda(self, epoch, total_epochs):
        """
        Decay the identity loss after 55% of training.

        This helps stabilize early training but reduces the identity constraint
        later so generators can better focus on translation.
        """
        if epoch <= 0.55 * total_epochs:
            return self.lambda_identity
        return self.lambda_identity * (0.997 ** (epoch - 0.55 * total_epochs))

    def generator_loss(self, real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, total_epochs):
        """
        Compute generator loss and produce fake samples.

        Returns:
            tuple: (loss_G, fake_A, fake_B)
        """

        # ------------------
        # Identity Loss
        # ------------------
        current_lambda_identity = self.get_identity_lambda(epoch, total_epochs)
        idt_A = G_BA(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_A) * current_lambda_identity

        idt_B = G_AB(real_B)
        loss_idt_B = self.criterion_identity(idt_B, real_B) * current_lambda_identity

        # ------------------
        # GAN Loss
        # ------------------
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))

        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

        # ------------------
        # Cycle Loss
        # ------------------
        rec_A = G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle

        rec_B = G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        # ------------------
        # Perceptual Cycle Loss
        # ------------------
        loss_cycle_perceptual_A = (
            self.criterion_perceptual(rec_A, real_A.detach())
            * self.lambda_cycle_perceptual
        )
        loss_cycle_perceptual_B = (
            self.criterion_perceptual(rec_B, real_B.detach())
            * self.lambda_cycle_perceptual
        )

        # ------------------
        # Perceptual Identity Loss
        # ------------------
        loss_identity_perceptual_A = (
            self.criterion_perceptual(idt_A, real_A.detach())
            * self.lambda_identity_perceptual
        )
        loss_identity_perceptual_B = (
            self.criterion_perceptual(idt_B, real_B.detach())
            * self.lambda_identity_perceptual
        )

        # ------------------
        # Total Generator Loss
        # ------------------
        loss_G = (
            loss_GAN_AB
            + loss_GAN_BA
            + loss_cycle_A
            + loss_cycle_B
            + loss_idt_A
            + loss_idt_B
            + loss_cycle_perceptual_A
            + loss_cycle_perceptual_B
            + loss_identity_perceptual_A
            + loss_identity_perceptual_B
        )

        return loss_G, fake_A, fake_B

    def discriminator_loss(self, D, real, fake, replay_buffer=None):
        """
        Compute discriminator loss with optional replay buffer.

        Args:
            D (nn.Module): Discriminator for the target domain.
            real (torch.Tensor): Real samples from the domain.
            fake (torch.Tensor): Newly generated samples.
            replay_buffer (ReplayBuffer | None): Optional buffer of past fakes.

        Returns:
            torch.Tensor: Scalar discriminator loss.
        """
        # Real loss (LSGAN uses targets close to 1).
        pred_real = D(real)
        loss_real = self.criterion_GAN(pred_real, 0.8 * torch.ones_like(pred_real))

        # Fake loss (use buffered fakes to reduce model oscillation).
        fake_buffer = replay_buffer.push_and_pop(fake) if replay_buffer else fake
        pred_fake = D(fake_buffer.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        # Total
        loss_D = (loss_real + loss_fake) * 0.5

        return loss_D


if __name__ == "__main__":
    # Example usage
    loss_fn = CycleGANLoss(
        lambda_cycle=10.0,
        lambda_identity=5.0,
        lambda_cycle_perceptual=0.1,
        lambda_identity_perceptual=0.05,
    )
    print("CycleGAN Loss initialized successfully.")
    from generator import getGenerators
    from discriminator import getDiscriminators

    G_AB, G_BA = getGenerators()
    D_A, D_B = getDiscriminators()

    # Keep example tensors on the same device as the models.
    device = next(G_AB.parameters()).device
    real_A = torch.randn(1, 3, 256, 256, device=device)  # Example input
    real_B = torch.randn(1, 3, 256, 256, device=device)  # Example input

    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B, 1, 1
    )
    print("Generator loss:", loss_G)

    loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A)
    loss_D_B = loss_fn.discriminator_loss(D_B, real_B, fake_B)
    print("Discriminator A loss:", loss_D_A)
    print("Discriminator B loss:", loss_D_B)
