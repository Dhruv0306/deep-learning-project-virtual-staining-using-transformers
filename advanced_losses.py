"""
Advanced loss functions for UVCGAN v2 training.

Provides:

* :class:`VGGPerceptualLossV2`  – multi-backbone VGG perceptual loss.
* :class:`SpectralLoss`         – frequency-domain loss for colour/texture.
* :class:`ContrastiveLoss`      – NT-Xent contrastive loss for domain alignment.
* :class:`WGANGPLoss`           – Wasserstein GAN with gradient penalty.
* :class:`AdvancedCycleGANLoss` – composite loss combining all of the above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# VGG perceptual loss (v2 – multi-level)
# ---------------------------------------------------------------------------


class VGGPerceptualLossV2(nn.Module):
    """
    Perceptual loss using four VGG19 feature levels.

    Computes the L1 distance between VGG19 activations of two images at
    ``relu1_2``, ``relu2_2``, ``relu3_4``, and ``relu4_4``.  All parameters
    are frozen.

    Args:
        resize_to: Spatial resolution to which images are interpolated before
            being passed to VGG.  ``None`` skips resizing.
        weights: Per-level loss weights ``(w1, w2, w3, w4)``.
    """

    def __init__(
        self,
        resize_to: int = 128,
        weights: tuple = (1.0, 1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.resize_to = resize_to
        self.weights = weights

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])  # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_4
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27])  # relu4_4

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = self.mean.to(x.device).to(x.dtype)  # type: ignore
        std: torch.Tensor = self.std.to(x.device).to(x.dtype)  # type: ignore
        return (x - mean) / std

    def _extract(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-level perceptual loss.

        Args:
            x: Generated image batch ``(N, C, H, W)``.
            y: Target image batch ``(N, C, H, W)``.

        Returns:
            Scalar perceptual loss.
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        if self.resize_to is not None:
            h, w = x.shape[-2], x.shape[-1]
            if h != self.resize_to or w != self.resize_to:
                x = F.interpolate(
                    x,
                    size=(self.resize_to, self.resize_to),
                    mode="bilinear",
                    align_corners=False,
                )
                y = F.interpolate(
                    y,
                    size=(self.resize_to, self.resize_to),
                    mode="bilinear",
                    align_corners=False,
                )

        x = self._normalize(x)
        y = self._normalize(y)
        xf = self._extract(x)
        yf = self._extract(y)

        loss = sum(
            (w * F.l1_loss(xfi, yfi) for w, xfi, yfi in zip(self.weights, xf, yf)),
            torch.tensor(0.0, device=x.device),
        )
        return loss


# ---------------------------------------------------------------------------
# Spectral (frequency-domain) loss
# ---------------------------------------------------------------------------


class SpectralLoss(nn.Module):
    """
    Frequency-domain loss that encourages generated images to match the
    power spectrum of real images.

    The loss is computed as the L1 distance between the log-magnitude of the
    2-D FFT of the generated and target images, averaged over channels.  This
    penalises systematic colour/texture deviations that are invisible in the
    spatial domain.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss.

        Args:
            x: Generated image batch ``(N, C, H, W)``.
            y: Target image batch ``(N, C, H, W)``.

        Returns:
            Scalar spectral loss.
        """
        # Use float32 for numerical stability in FFT.
        x_f = torch.fft.rfft2(x.float(), norm="ortho")
        y_f = torch.fft.rfft2(y.float(), norm="ortho")
        x_mag = torch.log1p(x_f.abs())
        y_mag = torch.log1p(y_f.abs())
        return F.l1_loss(x_mag, y_mag)


# ---------------------------------------------------------------------------
# Contrastive (NT-Xent) loss for domain alignment
# ---------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss for domain alignment.

    Encourages a translated image ``fake_B = G_AB(real_A)`` to be similar
    to real samples from domain B and dissimilar to real samples from domain A
    in a projected embedding space.

    A lightweight MLP projection head maps pooled feature vectors to a
    normalised embedding space where cosine similarities are used.

    Args:
        in_features: Dimension of the input feature vector (e.g. flattened
            bottleneck channels after global average pooling).
        proj_dim: Projection head output dimension.
        temperature: Temperature for the softmax.
    """

    def __init__(
        self,
        in_features: int = 512,
        proj_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, proj_dim),
        )

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pool ``(N, C, H, W)`` then project."""
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # (N, C)
        z = self.projection(x)
        return F.normalize(z, dim=1)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            anchor: Feature map of the generated image (e.g. bottleneck of
                ``G_AB(real_A)``).
            positive: Feature map of a real sample from the target domain B.
            negative: Feature map of a real sample from the source domain A.

        Returns:
            Scalar contrastive loss.
        """
        z_a = self._project(anchor)  # (N, proj_dim)
        z_p = self._project(positive)  # (N, proj_dim)
        z_n = self._project(negative)  # (N, proj_dim)

        sim_pos = (z_a * z_p).sum(dim=1) / self.temperature  # (N,)
        sim_neg = (z_a * z_n).sum(dim=1) / self.temperature  # (N,)

        # NT-Xent: cross-entropy with two classes (positive vs negative).
        logits = torch.stack([sim_pos, sim_neg], dim=1)  # (N, 2)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# WGAN-GP loss
# ---------------------------------------------------------------------------


class WGANGPLoss:
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).

    WGAN-GP replaces the standard GAN / LSGAN objective with the Wasserstein
    distance, which provides more stable gradients and avoids mode collapse.
    The gradient penalty enforces the 1-Lipschitz constraint on the
    discriminator without weight clipping.

    Reference: Gulrajani et al., "Improved Training of Wasserstein GANs", 2017.
    """

    @staticmethod
    def generator_loss(disc_fake_outputs) -> torch.Tensor:
        """
        Generator loss: maximise the discriminator score on fakes.

        Accepts outputs from either a single discriminator (tensor) or a
        multi-scale discriminator (list of tensors).

        Args:
            disc_fake_outputs: Discriminator logits for generated images.

        Returns:
            Scalar generator loss.
        """
        if isinstance(disc_fake_outputs, (list, tuple)):
            return -sum((o.mean() for o in disc_fake_outputs), torch.tensor(0.0)) / len(
                disc_fake_outputs
            )
        return -disc_fake_outputs.mean()

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_fake_outputs) -> torch.Tensor:
        """
        Discriminator loss: maximise real − fake Wasserstein estimate.

        Args:
            disc_real_outputs: Discriminator logits for real images.
            disc_fake_outputs: Discriminator logits for generated images.

        Returns:
            Scalar discriminator loss (positive → discriminator improving).
        """
        if isinstance(disc_real_outputs, (list, tuple)):
            loss_real = sum(
                (o.mean() for o in disc_real_outputs), torch.tensor(0.0)
            ) / len(disc_real_outputs)
            loss_fake = sum(
                (o.mean() for o in disc_fake_outputs), torch.tensor(0.0)
            ) / len(disc_fake_outputs)
        else:
            loss_real = disc_real_outputs.mean()
            loss_fake = disc_fake_outputs.mean()
        return loss_fake - loss_real

    @staticmethod
    def gradient_penalty(
        D: nn.Module, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the gradient penalty for a discriminator.

        For a multi-scale discriminator the penalty is averaged across scales.

        Args:
            D: Discriminator module.
            real: Real image batch.
            fake: Generated image batch (detached from the generator graph).

        Returns:
            Scalar gradient penalty.
        """
        batch_size = real.size(0)
        eps = torch.rand(batch_size, 1, 1, 1, device=real.device)
        with torch.autocast(device_type=real.device.type, enabled=False):
            real_f = real.detach().float()
            fake_f = fake.detach().float()
            interp = (eps * real_f + (1.0 - eps) * fake_f).requires_grad_(True)
            pred = D(interp)
            if isinstance(pred, (list, tuple)):
                # Sum across scales so grad flows through all.
                pred_sum = sum((p.sum() for p in pred), torch.tensor(0.0))
                grad = torch.autograd.grad(
                    outputs=pred_sum,
                    inputs=interp,
                    grad_outputs=None,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
            else:
                grad = torch.autograd.grad(
                    outputs=pred,
                    inputs=interp,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
            grad = grad.view(batch_size, -1)
            return ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Composite advanced loss
# ---------------------------------------------------------------------------


class AdvancedCycleGANLoss:
    """
    Composite loss for the UVCGAN v2 training loop.

    Combines:

    * WGAN-GP adversarial loss (or LSGAN if ``use_wgan_gp=False``).
    * Cycle-consistency (L1).
    * Identity (L1).
    * Multi-level VGG perceptual loss on cycle and identity outputs.
    * Contrastive domain-alignment loss.
    * Spectral (frequency-domain) loss.

    Args:
        lambda_cycle: Cycle-consistency loss weight.
        lambda_identity: Identity loss weight.
        lambda_cycle_perceptual: Perceptual cycle loss weight.
        lambda_identity_perceptual: Perceptual identity loss weight.
        lambda_gp: Gradient-penalty weight.
        lambda_contrastive: Contrastive loss weight.
        lambda_spectral: Spectral loss weight.
        perceptual_resize: Image size for VGG perceptual loss.
        use_wgan_gp: Use WGAN-GP; if ``False`` falls back to LSGAN.
        contrastive_temperature: Temperature for the contrastive loss.
        lsgan_real_label: Label smoothing target for real samples in LSGAN
            (values < 1 implement one-sided label smoothing).
        identity_decay_start: Fraction of training after which identity loss
            weight begins to decay (default 0.5 = 50 %).
        identity_decay_rate: Per-epoch multiplicative decay applied to the
            identity weight after ``identity_decay_start`` (default 0.997).
        device: Training device (auto-detected if ``None``).
    """

    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_cycle_perceptual: float = 0.1,
        lambda_identity_perceptual: float = 0.05,
        lambda_gp: float = 10.0,
        lambda_contrastive: float = 0.1,
        lambda_spectral: float = 0.05,
        perceptual_resize: int = 128,
        use_wgan_gp: bool = True,
        contrastive_temperature: float = 0.07,
        lsgan_real_label: float = 0.97,
        identity_decay_start: float = 0.5,
        identity_decay_rate: float = 0.997,
        device=None,
    ):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_cycle_perceptual = lambda_cycle_perceptual
        self.lambda_identity_perceptual = lambda_identity_perceptual
        self.lambda_gp = lambda_gp
        self.lambda_contrastive = lambda_contrastive
        self.lambda_spectral = lambda_spectral
        self.use_wgan_gp = use_wgan_gp
        self.lsgan_real_label = lsgan_real_label
        self.identity_decay_start = identity_decay_start
        self.identity_decay_rate = identity_decay_rate

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Loss modules
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLossV2(resize_to=perceptual_resize).to(
            self.device
        )
        self.criterion_spectral = SpectralLoss()
        self.wgan = WGANGPLoss()

        # Contrastive loss (bottleneck channels = 512 by default).
        if lambda_contrastive > 0.0:
            self.criterion_contrastive = ContrastiveLoss(
                in_features=512,
                temperature=contrastive_temperature,
            ).to(self.device)
        else:
            self.criterion_contrastive = None

        # LSGAN fallback
        self.criterion_GAN = nn.MSELoss()

        # Replay buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    # ------------------------------------------------------------------
    # Identity weight schedule
    # ------------------------------------------------------------------

    def get_identity_lambda(self, epoch: int, total_epochs: int) -> float:
        """
        Decay the identity weight after ``identity_decay_start`` of training.

        Args:
            epoch: Current epoch (0-indexed).
            total_epochs: Total number of training epochs.

        Returns:
            Effective identity loss weight.
        """
        if epoch <= self.identity_decay_start * total_epochs:
            return self.lambda_identity
        return self.lambda_identity * (
            self.identity_decay_rate
            ** (epoch - self.identity_decay_start * total_epochs)
        )

    # ------------------------------------------------------------------
    # Discriminator losses (helper)
    # ------------------------------------------------------------------

    def _disc_gan_loss(self, real_outputs, fake_outputs) -> torch.Tensor:
        """
        Compute discriminator GAN loss (WGAN or LSGAN).

        Handles both single-scale (tensor) and multi-scale (list) outputs.

        Args:
            real_outputs: Discriminator outputs for real images.
            fake_outputs: Discriminator outputs for fake images.

        Returns:
            Scalar discriminator GAN loss.
        """
        if self.use_wgan_gp:
            return self.wgan.discriminator_loss(real_outputs, fake_outputs)

        # LSGAN fallback.
        def _lsgan(real_out, fake_out):
            if isinstance(real_out, (list, tuple)):
                r = sum(
                    (
                        self.criterion_GAN(
                            r, self.lsgan_real_label * torch.ones_like(r)
                        )
                        for r in real_out
                    ),
                    torch.tensor(0.0),
                ) / len(real_out)
                f = sum(
                    (self.criterion_GAN(f, torch.zeros_like(f)) for f in fake_out),
                    torch.tensor(0.0),
                ) / len(fake_out)
            else:
                r = self.criterion_GAN(
                    real_out, self.lsgan_real_label * torch.ones_like(real_out)
                )
                f = self.criterion_GAN(fake_out, torch.zeros_like(fake_out))
            return (r + f) * 0.5

        return _lsgan(real_outputs, fake_outputs)

    def _gen_gan_loss(self, disc_fake_outputs) -> torch.Tensor:
        """
        Generator GAN loss (WGAN or LSGAN).

        Args:
            disc_fake_outputs: Discriminator outputs for generated images.

        Returns:
            Scalar generator GAN loss.
        """
        if self.use_wgan_gp:
            return self.wgan.generator_loss(disc_fake_outputs)

        # LSGAN fallback.
        if isinstance(disc_fake_outputs, (list, tuple)):
            return sum(
                (self.criterion_GAN(o, torch.ones_like(o)) for o in disc_fake_outputs),
                torch.tensor(0.0),
            ) / len(disc_fake_outputs)
        return self.criterion_GAN(disc_fake_outputs, torch.ones_like(disc_fake_outputs))

    # ------------------------------------------------------------------
    # Public loss API
    # ------------------------------------------------------------------

    def generator_loss(self, real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, total_epochs):
        """
        Compute the full generator loss.

        Args:
            real_A: Real images from domain A ``(N, C, H, W)``.
            real_B: Real images from domain B ``(N, C, H, W)``.
            G_AB: Generator A→B.
            G_BA: Generator B→A.
            D_A: Discriminator for domain A.
            D_B: Discriminator for domain B.
            epoch: Current training epoch (0-indexed).
            total_epochs: Total training epochs.

        Returns:
            tuple: ``(loss_G, fake_A, fake_B)``
        """
        current_lambda_identity = self.get_identity_lambda(epoch, total_epochs)

        # ---- Identity ----
        idt_A = G_BA(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_A) * current_lambda_identity
        idt_B = G_AB(real_B)
        loss_idt_B = self.criterion_identity(idt_B, real_B) * current_lambda_identity

        # ---- GAN ----
        fake_B = G_AB(real_A)
        loss_GAN_AB = self._gen_gan_loss(D_B(fake_B))
        fake_A = G_BA(real_B)
        loss_GAN_BA = self._gen_gan_loss(D_A(fake_A))

        # ---- Cycle ----
        rec_A = G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
        rec_B = G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle

        # ---- Perceptual cycle ----
        loss_cyc_perc_A = (
            self.criterion_perceptual(rec_A, real_A.detach())
            * self.lambda_cycle_perceptual
        )
        loss_cyc_perc_B = (
            self.criterion_perceptual(rec_B, real_B.detach())
            * self.lambda_cycle_perceptual
        )

        # ---- Perceptual identity ----
        loss_idt_perc_A = (
            self.criterion_perceptual(idt_A, real_A.detach())
            * self.lambda_identity_perceptual
        )
        loss_idt_perc_B = (
            self.criterion_perceptual(idt_B, real_B.detach())
            * self.lambda_identity_perceptual
        )

        # ---- Spectral ----
        loss_spectral = torch.tensor(0.0, device=real_A.device)
        if self.lambda_spectral > 0.0:
            loss_spectral = (
                self.criterion_spectral(fake_B, real_B)
                + self.criterion_spectral(fake_A, real_A)
            ) * self.lambda_spectral

        # ---- Contrastive ----
        loss_contrastive = torch.tensor(0.0, device=real_A.device)
        if self.criterion_contrastive is not None and self.lambda_contrastive > 0.0:
            # Use global-average-pooled bottleneck features as embeddings.
            if hasattr(G_AB, "encode"):
                _, _, _, _, bot_AB = G_AB.encode(real_A)
                _, _, _, _, bot_B = G_AB.encode(real_B)
                _, _, _, _, bot_BA = G_BA.encode(real_B)
                _, _, _, _, bot_A = G_BA.encode(real_A)
                loss_contrastive = (
                    self.criterion_contrastive(bot_AB, bot_B, bot_A)
                    + self.criterion_contrastive(bot_BA, bot_A, bot_B)
                ) * self.lambda_contrastive

        # ---- Total ----
        loss_G = (
            loss_GAN_AB
            + loss_GAN_BA
            + loss_cycle_A
            + loss_cycle_B
            + loss_idt_A
            + loss_idt_B
            + loss_cyc_perc_A
            + loss_cyc_perc_B
            + loss_idt_perc_A
            + loss_idt_perc_B
            + loss_spectral
            + loss_contrastive
        )

        return loss_G, fake_A, fake_B

    def discriminator_loss(
        self, D, real, fake, replay_buffer: Optional[ReplayBuffer] = None
    ) -> torch.Tensor:
        """
        Compute discriminator loss with optional replay buffer.

        Args:
            D: Discriminator module.
            real: Real image batch.
            fake: Newly generated image batch.
            replay_buffer: Optional buffer of past generated images.

        Returns:
            Scalar discriminator loss.
        """
        # Optionally draw from the replay buffer.
        fake_buf = replay_buffer.push_and_pop(fake) if replay_buffer else fake

        real_out = D(real)
        fake_out = D(fake_buf.detach())
        loss_D = self._disc_gan_loss(real_out, fake_out)

        if self.lambda_gp > 0.0:
            gp = self.wgan.gradient_penalty(D, real, fake)
            loss_D = loss_D + self.lambda_gp * gp

        return loss_D


if __name__ == "__main__":
    loss_fn = AdvancedCycleGANLoss()
    print("AdvancedCycleGANLoss initialised successfully.")

    from uvcgan_v2_generator import getGeneratorsV2
    from spectral_norm_discriminator import getDiscriminatorsV2

    G_AB, G_BA = getGeneratorsV2()
    D_A, D_B = getDiscriminatorsV2()

    device = next(G_AB.parameters()).device
    real_A = torch.randn(1, 3, 256, 256, device=device)
    real_B = torch.randn(1, 3, 256, 256, device=device)

    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B, 0, 10
    )
    print("Generator loss:", loss_G.item())

    loss_DA = loss_fn.discriminator_loss(D_A, real_A, fake_A)
    loss_DB = loss_fn.discriminator_loss(D_B, real_B, fake_B)
    print("D_A loss:", loss_DA.item())
    print("D_B loss:", loss_DB.item())
