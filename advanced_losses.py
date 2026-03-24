"""
Advanced loss functions for UVCGAN v2 training.

Provides:

* :class:`VGGPerceptualLossV2`      – multi-level VGG19 perceptual loss.
* :class:`SpectralLoss`             – frequency-domain loss for colour/texture.
* :class:`ContrastiveLoss`          – NT-Xent contrastive loss for domain alignment.
* :class:`LSGANGradientPenalty`     – paper-correct one-sided gradient penalty (gamma=100).
* :class:`UVCGANLoss`               – composite loss combining all of the above.

Paper reference:
    Prokopenko et al., "UVCGAN v2: An Improved Cycle-Consistent GAN for
    Unpaired Image-to-Image Translation", 2023.

Key design decisions (paper-aligned):
  - GAN objective   : LSGAN (MSE), NOT Wasserstein.
                      Paper Table 2 shows LSGAN + GP is the best configuration.
  - Gradient penalty: ONE-SIDED, target gamma=100.
                      GP = E[max(0, ||grad D(x_hat)||_2 - gamma)^2] / gamma^2
                      Only penalises gradients that EXCEED gamma, so it never
                      prevents D from having small-norm gradients near real data.
  - n_critic        : 1 (standard for LSGAN; no multi-step D updates needed).
  - Adam betas      : (0.5, 0.999) -- standard for LSGAN/CycleGAN.
  - lambda_gp       : 0.1 (paper value, much smaller than WGAN's 10).
  - Cross-domain    : forward_with_cross_domain() activated when available.
  - AMP safety      : GP always computed in float32 outside autocast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional

from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# VGG perceptual loss (v2 - multi-level)
# ---------------------------------------------------------------------------


class VGGPerceptualLossV2(nn.Module):
    """
    Perceptual loss using four VGG19 feature levels.

    Computes the L1 distance between VGG19 activations of two images at
    relu1_2, relu2_2, relu3_4, and relu4_4.  All parameters are frozen.

    Args:
        resize_to: Spatial resolution to which images are interpolated before
            being passed to VGG.  None skips resizing.
        weights: Per-level loss weights (w1, w2, w3, w4).
    """

    def __init__(
        self,
        resize_to: int = 128,
        weights: tuple = (1.0, 1.0, 1.0, 1.0),
    ):
        """
        Initialize VGGPerceptualLossV2.

        Args:
            resize_to: Spatial resolution for VGG input.  ``None`` skips
                resizing.
            weights: Per-level loss weights ``(w1, w2, w3, w4)`` applied to
                relu1_2, relu2_2, relu3_4, and relu4_4 respectively.
        """
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
        """
        Apply ImageNet mean/std normalisation.

        The registered buffers ``mean`` and ``std`` are cast to the dtype
        and device of ``x`` so the method is safe to call under AMP
        (float16) and on any device.

        Args:
            x (torch.Tensor): Image batch in ``[0, 1]`` range after
                de-normalising from ``[-1, 1]``.

        Returns:
            torch.Tensor: ImageNet-normalised tensor, same shape as ``x``.
        """
        mean: torch.Tensor = self.mean.to(x.device).to(x.dtype)  # type: ignore
        std: torch.Tensor = self.std.to(x.device).to(x.dtype)  # type: ignore
        return (x - mean) / std

    def _extract(self, x: torch.Tensor):
        """
        Extract multi-level VGG19 feature activations.

        Passes ``x`` sequentially through the four feature-extraction slices,
        each slice picking up where the previous left off so that activations
        at increasing depths are returned.

        Args:
            x (torch.Tensor): ImageNet-normalised image tensor
                ``(N, 3, H, W)``.

        Returns:
            tuple[Tensor, Tensor, Tensor, Tensor]: ``(h1, h2, h3, h4)``
            feature maps at relu1_2, relu2_2, relu3_4, and relu4_4.
        """
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the multi-level VGG19 perceptual loss between two image batches.

        Grayscale inputs (1-channel) are broadcast to 3 channels before
        processing.  Both images are optionally resized to ``resize_to``
        before feature extraction.

        Args:
            x (torch.Tensor): First image batch ``(N, C, H, W)`` in the
                ``[-1, 1]`` range (model output or cycle-reconstructed image).
            y (torch.Tensor): Second image batch ``(N, C, H, W)`` in the
                ``[-1, 1]`` range (target/real image, detach before passing).

        Returns:
            torch.Tensor: Scalar perceptual loss — weighted sum of L1
            distances at four VGG feature levels.
        """
        x = x.float()
        y = y.float()

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

        x = (x + 1.0) / 2.0  # De-normalise from [-1, 1] to [0, 1].
        y = (y + 1.0) / 2.0  # De-normalise from [-1, 1] to [0, 1].

        x = x.clamp(0.0, 1.0)  # Clamp to [0, 1] after de-normalisation
        y = y.clamp(0.0, 1.0)  # Clamp to [0, 1] after de-normalisation

        x = self._normalize(x)
        y = self._normalize(y)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = self._extract(x.float())
            yf = self._extract(y.float())

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
    Frequency-domain loss via log-magnitude FFT L1 distance.

    Measures the difference between two images in the frequency domain by
    comparing the log-magnitude of their 2-D real FFT (``rfft2``).  Using
    the logarithm compresses the wide dynamic range of Fourier coefficients
    and makes the loss sensitive to both low-frequency colour/brightness
    differences and high-frequency texture/edge differences.

    This loss encourages generated images to match the spectral envelope of
    the target distribution, which helps with texture fidelity.  It is
    disabled by default (``lambda_spectral=0``); enable by setting
    ``lambda_spectral > 0`` in :class:`UVCGANLoss`.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-magnitude spectral L1 loss between two image batches.

        Both tensors are cast to float32 before the FFT so the loss is safe
        to call inside an AMP context (float16 inputs are accepted).

        Args:
            x (torch.Tensor): First image batch ``(N, C, H, W)``.
            y (torch.Tensor): Second image batch ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Scalar L1 distance between
            ``log(1 + |FFT(x)|)`` and ``log(1 + |FFT(y)|)``.
        """
        x_f = torch.fft.rfft2(x.float(), norm="ortho")
        y_f = torch.fft.rfft2(y.float(), norm="ortho")
        return F.l1_loss(torch.log1p(x_f.abs()), torch.log1p(y_f.abs()))


# ---------------------------------------------------------------------------
# Contrastive (NT-Xent) loss for domain alignment
# ---------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    """
    NT-Xent contrastive loss for domain alignment.

    Args:
        in_features: Dimension of the GAP bottleneck (512 for base_channels=64).
        proj_dim: Projection head output dimension.
        temperature: Softmax temperature.
    """

    def __init__(
        self, in_features: int = 512, proj_dim: int = 128, temperature: float = 0.07
    ):
        """
        Initialize ContrastiveLoss.

        Args:
            in_features: Dimension of the GAP bottleneck feature vector
                (512 for ``base_channels=64``).
            proj_dim: Output dimension of the projection head.
            temperature: Softmax temperature for the NT-Xent loss.
        """
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, proj_dim),
        )

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a feature tensor to the contrastive embedding space.

        4-D feature maps ``(N, C, H, W)`` are first reduced to ``(N, C)``
        via global average pooling before being passed through the
        projection MLP.  The output is L2-normalised so that cosine
        similarities are equivalent to dot products.

        Args:
            x (torch.Tensor): Feature tensor ``(N, C)`` or ``(N, C, H, W)``.

        Returns:
            torch.Tensor: L2-normalised projection ``(N, proj_dim)``.
        """
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])  # global average pool
        return F.normalize(self.projection(x), dim=1)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the NT-Xent (normalised temperature-scaled cross-entropy) loss.

        Each sample in the batch is treated as an ``(anchor, positive,
        negative)`` triplet.  The loss encourages the anchor to be close to
        the positive in embedding space and far from the negative.

        Args:
            anchor (torch.Tensor): Bottleneck features from the translated
                image ``(N, C)`` or ``(N, C, H, W)``.
            positive (torch.Tensor): Bottleneck features from the real target
                image in the same domain ``(N, C)`` or ``(N, C, H, W)``.
            negative (torch.Tensor): Bottleneck features from the real source
                image in the other domain ``(N, C)`` or ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        z_a = self._project(anchor)
        z_p = self._project(positive)
        z_n = self._project(negative)
        sim_pos = (z_a * z_p).sum(dim=1) / self.temperature
        sim_neg = (z_a * z_n).sum(dim=1) / self.temperature
        logits = torch.stack([sim_pos, sim_neg], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# UVCGAN paper gradient penalty  (one-sided, LSGAN-compatible, gamma=100)
# ---------------------------------------------------------------------------


class LSGANGradientPenalty:
    """
    One-sided gradient penalty from the UVCGAN v2 paper.

    Standard WGAN-GP (Gulrajani 2017) uses a TWO-SIDED penalty:
        GP = E[(||grad D(x_hat)||_2 - 1)^2]

    The UVCGAN paper instead uses a ONE-SIDED penalty with target gamma=100:
        GP = E[max(0, ||grad D(x_hat)||_2 - gamma)^2] / gamma^2

    This only penalises gradients that EXCEED gamma, leaving D free to have
    small-norm gradients near real data. This is appropriate because the GAN
    objective is LSGAN (not Wasserstein) -- D does not need to be
    1-Lipschitz everywhere, only bounded from above.

    The gamma^2 normalisation keeps the penalty magnitude scale-invariant,
    so lambda_gp = 0.1 works across different gamma choices.

    Because LSGAN + GP both produce losses >= 0, the EarlyStopping divergence
    check works correctly without any sign correction (unlike WGAN).

    Reference: Prokopenko et al., "UVCGAN v2", 2023, Eq. (4).

    IMPORTANT: Must be called with autocast DISABLED (float32 only).
    """

    GAMMA: float = 100.0  # Paper value.

    @staticmethod
    def gradient_penalty(
        D: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the one-sided gradient penalty on interpolated samples.

        Always runs in float32 regardless of the caller's AMP state.

        Args:
            D   : Discriminator in train mode, all params require_grad=True.
            real: Real image batch -- detached and cast to float32 internally.
            fake: Generated image batch -- detached and cast to float32 internally.

        Returns:
            Scalar penalty >= 0.
        """
        gamma = LSGANGradientPenalty.GAMMA
        batch_size = real.size(0)
        device = real.device

        real_f = real.detach().float()
        fake_f = fake.detach().float()

        eps = torch.rand(batch_size, 1, 1, 1, device=device, dtype=torch.float32)
        interp = (eps * real_f + (1.0 - eps) * fake_f).requires_grad_(True)

        pred = D(interp)

        # Reduce discriminator output to a single scalar so that grad_outputs
        # can be None (autograd default for scalar outputs).
        # torch.stack avoids Python's built-in sum() returning int|Any.
        if isinstance(pred, (list, tuple)):
            pred_scalar: torch.Tensor = torch.stack([p.sum() for p in pred]).sum()
        else:
            pred_scalar = pred.sum()

        # grad_outputs=None is valid and correct when outputs is a 0-dim scalar.
        grads = torch.autograd.grad(
            outputs=pred_scalar,
            inputs=interp,
            grad_outputs=None,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # One-sided hinge: penalise only gradients that exceed gamma.
        grad_norms = grads.view(batch_size, -1).norm(2, dim=1)  # (N,)
        penalty = (F.relu(grad_norms - gamma) ** 2).mean() / (gamma**2)
        return penalty


# ---------------------------------------------------------------------------
# Composite UVCGAN loss
# ---------------------------------------------------------------------------


class UVCGANLoss:
    """
    Composite loss for the UVCGAN v2 training loop.

    GAN objective : LSGAN (MSE) -- paper Table 2 best configuration.
    Gradient penalty: one-sided, gamma=100, lambda=0.1 -- paper Eq. (4).

    All component losses (cycle, identity, perceptual, spectral, contrastive)
    are >= 0, so the EarlyStopping divergence check works without any
    sign correction. This is a key stability advantage over WGAN.

    Cross-domain skip fusion (forward_with_cross_domain) is activated
    automatically when both generators expose the required API.

    Args:
        lambda_cycle: Cycle-consistency loss weight.
        lambda_identity: Identity loss weight.
        lambda_cycle_perceptual: Perceptual cycle loss weight.
        lambda_identity_perceptual: Perceptual identity loss weight.
        lambda_gp: Gradient-penalty weight. Paper value is 0.1.
        lambda_contrastive: Contrastive loss weight (0 = disabled).
        lambda_spectral: Spectral loss weight (0 = disabled).
        perceptual_resize: Image size for VGG perceptual loss.
        contrastive_temperature: Temperature for NT-Xent loss.
        lsgan_real_label: Label-smoothing target for real samples (< 1).
        identity_decay_start: Fraction of training at which identity weight
            begins to decay.
        identity_decay_rate: Per-epoch multiplicative decay for identity weight.
        device: Training device (auto-detected when None).
    """

    def __init__(
        self,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        lambda_cycle_perceptual: float = 0.1,
        lambda_identity_perceptual: float = 0.05,
        lambda_gp: float = 0.1,
        lambda_contrastive: float = 0.0,
        lambda_spectral: float = 0.0,
        perceptual_resize: int = 128,
        contrastive_temperature: float = 0.07,
        lsgan_real_label: float = 0.9,
        identity_decay_start: float = 0.5,
        identity_decay_rate: float = 0.997,
        device=None,
    ):
        """
        Initialize UVCGANLoss.

        Args:
            lambda_cycle: Weight for the cycle-consistency L1 loss.
            lambda_identity: Initial weight for the identity L1 loss.
            lambda_cycle_perceptual: Weight for the VGG perceptual cycle loss.
            lambda_identity_perceptual: Weight for the VGG perceptual identity
                loss.
            lambda_gp: Gradient-penalty weight (paper value: 0.1).
            lambda_contrastive: NT-Xent contrastive loss weight.  ``0``
                disables the contrastive term entirely.
            lambda_spectral: Spectral (frequency-domain) loss weight.  ``0``
                disables the spectral term.
            perceptual_resize: Spatial size used by
                :class:`VGGPerceptualLossV2`.
            contrastive_temperature: Temperature for the NT-Xent loss.
            lsgan_real_label: Label-smoothing target for real discriminator
                outputs (< 1).
            identity_decay_start: Fraction of total training epochs at which
                the identity weight starts decaying.
            identity_decay_rate: Per-epoch multiplicative decay applied to the
                identity weight after ``identity_decay_start``.
            device: Target device.  Auto-detected when ``None``.
        """
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_cycle_perceptual = lambda_cycle_perceptual
        self.lambda_identity_perceptual = lambda_identity_perceptual
        self.lambda_gp = lambda_gp
        self.lambda_contrastive = lambda_contrastive
        self.lambda_spectral = lambda_spectral
        self.lsgan_real_label = lsgan_real_label
        self.identity_decay_start = identity_decay_start
        self.identity_decay_rate = identity_decay_rate

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Core loss modules.
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_GAN = nn.MSELoss()  # LSGAN
        self.criterion_perceptual = VGGPerceptualLossV2(resize_to=perceptual_resize).to(
            self.device
        )
        self.criterion_spectral = SpectralLoss()
        self.gp = LSGANGradientPenalty()

        # Contrastive loss -- only instantiated when enabled.
        self.criterion_contrastive: Optional[ContrastiveLoss]
        if lambda_contrastive > 0.0:
            self.criterion_contrastive = ContrastiveLoss(
                in_features=512,
                temperature=contrastive_temperature,
            ).to(self.device)
        else:
            self.criterion_contrastive = None

        # Replay buffers for discriminator stabilisation.
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    # ------------------------------------------------------------------
    # Identity weight schedule
    # ------------------------------------------------------------------

    def get_identity_lambda(self, epoch: int, total_epochs: int) -> float:
        """Decay the identity weight after identity_decay_start of training."""
        if epoch <= self.identity_decay_start * total_epochs:
            return self.lambda_identity
        return self.lambda_identity * (
            self.identity_decay_rate
            ** (epoch - self.identity_decay_start * total_epochs)
        )

    # ------------------------------------------------------------------
    # LSGAN helpers
    # ------------------------------------------------------------------

    def _lsgan_disc_loss(self, real_outputs, fake_outputs) -> torch.Tensor:
        """
        LSGAN discriminator loss with one-sided label smoothing on real targets.
        Handles both single-scale (Tensor) and multi-scale (list) outputs.
        """

        def _single(r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
            """
            Compute LSGAN loss for one discriminator scale.

            Args:
                r: Real discriminator output at this scale.
                f: Fake discriminator output at this scale.

            Returns:
                torch.Tensor: Scalar discriminator loss for this scale.
            """
            loss_real = self.criterion_GAN(
                r, self.lsgan_real_label * torch.ones_like(r)
            )
            loss_fake = self.criterion_GAN(f, torch.zeros_like(f))
            return (loss_real + loss_fake) * 0.5

        if isinstance(real_outputs, (list, tuple)):
            return torch.stack(
                [_single(r, f) for r, f in zip(real_outputs, fake_outputs)]
            ).mean()
        return _single(real_outputs, fake_outputs)

    def _lsgan_gen_loss(self, disc_fake_outputs) -> torch.Tensor:
        """
        LSGAN generator loss: push fake outputs toward 1.
        Handles both single-scale (Tensor) and multi-scale (list) outputs.
        """
        if isinstance(disc_fake_outputs, (list, tuple)):
            return torch.stack(
                [self.criterion_GAN(o, torch.ones_like(o)) for o in disc_fake_outputs]
            ).mean()
        return self.criterion_GAN(disc_fake_outputs, torch.ones_like(disc_fake_outputs))

    # ------------------------------------------------------------------
    # Public loss API
    # ------------------------------------------------------------------

    def generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        G_AB: nn.Module,
        G_BA: nn.Module,
        D_A: nn.Module,
        D_B: nn.Module,
        epoch: int,
        total_epochs: int,
    ):
        """
        Compute the full UVCGAN generator loss.

        Activates cross-domain skip fusion (forward_with_cross_domain)
        when both generators support it -- the defining UVCGAN feature.

        Returns:
            tuple: (loss_G, fake_A, fake_B)
        """
        current_lambda_identity = self.get_identity_lambda(epoch, total_epochs)

        # ---- Cross-domain fusion check ----
        use_cross = (
            getattr(G_AB, "use_cross_domain", False)
            and getattr(G_BA, "use_cross_domain", False)
            and hasattr(G_AB, "forward_with_cross_domain")
            and hasattr(G_BA, "forward_with_cross_domain")
            and hasattr(G_AB, "get_skip_features")
            and hasattr(G_BA, "get_skip_features")
        )

        # ---- Identity loss (standard forward) ----
        idt_A = G_BA(real_A)
        loss_idt_A = self.criterion_identity(idt_A, real_A) * current_lambda_identity
        idt_B = G_AB(real_B)
        loss_idt_B = self.criterion_identity(idt_B, real_B) * current_lambda_identity

        # ---- Translation with optional cross-domain skip fusion ----
        # type: ignore[operator] silences Pylance's "Tensor not callable" false-positive:
        # G_AB/G_BA are nn.Module subclasses at runtime and these methods are
        # already guarded by the hasattr() checks in the use_cross block above.
        if use_cross:
            skips_A = G_AB.get_skip_features(real_A)  # type: ignore[operator]
            skips_B = G_BA.get_skip_features(real_B)  # type: ignore[operator]
            fake_B = G_AB.forward_with_cross_domain(real_A, skips_B)  # type: ignore[operator]
            fake_A = G_BA.forward_with_cross_domain(real_B, skips_A)  # type: ignore[operator]
        else:
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

        # ---- LSGAN adversarial loss ----
        loss_GAN_AB = self._lsgan_gen_loss(D_B(fake_B))
        loss_GAN_BA = self._lsgan_gen_loss(D_A(fake_A))

        # ---- Cycle-consistency ----
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

        # ---- Spectral loss (disabled by default) ----
        loss_spectral = torch.tensor(0.0, device=real_A.device)
        if self.lambda_spectral > 0.0:
            loss_spectral = (
                self.criterion_spectral(fake_B, real_B)
                + self.criterion_spectral(fake_A, real_A)
            ) * self.lambda_spectral

        # ---- Contrastive loss (disabled by default) ----
        loss_contrastive = torch.tensor(0.0, device=real_A.device)
        if self.criterion_contrastive is not None and self.lambda_contrastive > 0.0:
            if hasattr(G_AB, "encode") and hasattr(G_BA, "encode"):
                _, _, _, _, bot_AB = G_AB.encode(real_A)  # type: ignore[operator]
                _, _, _, _, bot_B = G_AB.encode(real_B)  # type: ignore[operator]
                _, _, _, _, bot_BA = G_BA.encode(real_B)  # type: ignore[operator]
                _, _, _, _, bot_A = G_BA.encode(real_A)  # type: ignore[operator]
                loss_contrastive = (
                    self.criterion_contrastive(bot_AB, bot_B, bot_A)
                    + self.criterion_contrastive(bot_BA, bot_A, bot_B)
                ) * self.lambda_contrastive

        # ---- Total generator loss (all terms >= 0 with LSGAN) ----
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
        self,
        D: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> torch.Tensor:
        """
        Compute the LSGAN discriminator loss + one-sided gradient penalty.

        The GP is always computed in float32 with autocast disabled, regardless
        of the caller's AMP state, to avoid NaN gradients under mixed precision.

        Returns:
            Scalar loss >= 0.  LSGAN and the one-sided GP are both non-negative,
            so the EarlyStopping divergence check works without sign correction.
        """
        fake_buf = (
            replay_buffer.push_and_pop(fake.detach())
            if replay_buffer
            else fake.detach()
        )

        real_out = D(real)
        fake_out = D(fake_buf)
        loss_D = self._lsgan_disc_loss(real_out, fake_out)

        if self.lambda_gp > 0.0:
            # Disable autocast unconditionally -- GP must be float32.
            with torch.autocast(device_type=real.device.type, enabled=False):
                gp = self.gp.gradient_penalty(D, real, fake)
            loss_D = loss_D + self.lambda_gp * gp.to(loss_D.dtype)

        return loss_D


if __name__ == "__main__":
    loss_fn = UVCGANLoss()
    print("UVCGANLoss initialised successfully.")

    from uvcgan_v2_generator import getGeneratorsV2
    from spectral_norm_discriminator import getDiscriminatorsV2

    G_AB, G_BA = getGeneratorsV2()
    D_A, D_B = getDiscriminatorsV2()

    device = next(G_AB.parameters()).device
    real_A = torch.randn(2, 3, 256, 256, device=device)
    real_B = torch.randn(2, 3, 256, 256, device=device)

    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B, 0, 10
    )
    print(f"Generator loss : {loss_G.item():.4f}  (should be > 0)")

    loss_DA = loss_fn.discriminator_loss(D_A, real_A, fake_A)
    loss_DB = loss_fn.discriminator_loss(D_B, real_B, fake_B)
    print(f"D_A loss       : {loss_DA.item():.4f}  (should be > 0)")
    print(f"D_B loss       : {loss_DB.item():.4f}  (should be > 0)")
