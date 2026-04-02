"""
Loss helpers for the v3 diffusion training loop.

Component structure:
    1) LSGAN losses (generator and discriminator)
    2) R1 penalty loss (regularization)
    3) Cycle consistency loss
    4) Identity loss and weight schedule
    5) Diffusion loss (denoising + perceptual)

Primary shapes:
    z_t, noise, eps_pred: (N, 4, 32, 32)
    real_B:              (N, 3, 256, 256)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model_v3.noise_scheduler import DDPMScheduler
from model_v3.vae_wrapper import VAEWrapper


def _lsgan_gen_loss(fake_outputs) -> torch.Tensor:
    """
    LSGAN generator loss: E[(D(fake) - 1)²].

    Accepts either a single logit tensor or a list of tensors (multi-branch
    discriminator).  List outputs are averaged across branches.

    Args:
        fake_outputs: Tensor or list of tensors from D(fake).

    Returns:
        Scalar loss tensor.
    """
    if isinstance(fake_outputs, (list, tuple)):
        return torch.stack(
            [F.mse_loss(o, torch.ones_like(o)) for o in fake_outputs]
        ).mean()
    return F.mse_loss(fake_outputs, torch.ones_like(fake_outputs))


def _lsgan_disc_loss(real_outputs, fake_outputs) -> torch.Tensor:
    """
    LSGAN discriminator loss: E[(D(real) - 1)²] + E[D(fake)²].

    Accepts single tensors or lists (multi-branch discriminator); list
    outputs are averaged across branches.

    Args:
        real_outputs: Tensor or list of tensors from D(real).
        fake_outputs: Tensor or list of tensors from D(fake).

    Returns:
        Scalar loss tensor.
    """

    def _single(r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        loss_real = F.mse_loss(r, torch.ones_like(r))
        loss_fake = F.mse_loss(f, torch.zeros_like(f))
        return 0.5 * (loss_real + loss_fake)

    if isinstance(real_outputs, (list, tuple)):
        return torch.stack(
            [_single(r, f) for r, f in zip(real_outputs, fake_outputs)]
        ).mean()
    return _single(real_outputs, fake_outputs)


def _r1_penalty_loss(
    discriminator: torch.nn.Module,
    real_images: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    R1 gradient penalty: penalise discriminator gradients on real images.

    Computes 0.5 * gamma * E[||∇D(x_real)||²], which encourages D to have
    small gradients near real data and improves training stability.

    Always run in float32 (outside autocast) to avoid NaN gradients.

    Args:
        discriminator: Discriminator module (ProjectionDiscriminator).
        real_images:   Real image batch (N, 3, H, W) with requires_grad=True.
        gamma:         R1 penalty coefficient (cfg.diffusion.r1_gamma).

    Returns:
        Scalar R1 penalty tensor.
    """
    real_scores = discriminator(real_images)
    if isinstance(real_scores, (list, tuple)):
        real_score_mean = torch.stack([s.mean() for s in real_scores]).sum()
    else:
        real_score_mean = real_scores.mean()

    grad_real = torch.autograd.grad(
        outputs=real_score_mean,
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.size(0), -1).sum(dim=1).mean()
    return 0.5 * gamma * grad_penalty


def _ddim_shortcut_from_xt(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    z_t_start: torch.Tensor,
    t_start: torch.Tensor,
    condition: torch.Tensor,
    target_domain: int,
    prediction_type: str,
    num_steps: int,
    eta: float,
) -> torch.Tensor:
    """
    Short deterministic DDIM denoising from a provided x_t starting point.

    Used in the cycle-consistency loss to reconstruct x0 from a noisy
    latent built with shared (epsilon, t) from the primary forward pass.
    Starting from x_t rather than pure noise keeps the cycle semantically
    consistent with the denoising objective.

    Args:
        model:          CycleDiTGenerator.
        scheduler:      DDPMScheduler with precomputed alpha buffers.
        z_t_start:      Noisy latent to start denoising from (N, 4, 32, 32).
        t_start:        Per-sample starting timestep (N,).
        condition:      Conditioning image or tokens for the reverse direction.
        target_domain:  Domain id for the reverse translation (0 or 1).
        prediction_type: ``"v"`` or ``"eps"``.
        num_steps:      Number of DDIM steps (typically cfg.cycle_ddim_steps).
        eta:            DDIM stochasticity (0 = deterministic).

    Returns:
        Denoised latent estimate z0 (N, 4, 32, 32).
    """
    z_t = z_t_start
    steps = max(1, int(num_steps))

    # Build per-sample time trajectory from each sample's t_start down to 0.
    t_start_f = t_start.float()
    for k in range(steps):
        frac_cur = float(steps - k) / float(steps)
        frac_prev = float(max(steps - k - 1, 0)) / float(steps)

        t_cur = (
            torch.round(t_start_f * frac_cur)
            .long()
            .clamp(min=0, max=scheduler.num_timesteps - 1)
        )
        t_prev = (
            torch.round(t_start_f * frac_prev)
            .long()
            .clamp(min=0, max=scheduler.num_timesteps - 1)
        )

        out = model(
            z_t,
            t_cur,
            condition,
            target_domain=target_domain,
            scheduler=scheduler,
            prediction_type=prediction_type,
        )
        v_or_eps = out["v_pred"] if isinstance(out, dict) else out

        if prediction_type == "v":
            eps_pred = scheduler.predict_eps_from_v(
                z_t.float(), v_or_eps.float(), t_cur
            )
        elif prediction_type == "eps":
            eps_pred = v_or_eps.float()
        else:
            raise ValueError(
                f"prediction_type must be 'v' or 'eps', got {prediction_type!r}"
            )

        z0_pred = scheduler.predict_x0(z_t.float(), eps_pred, t_cur)

        alpha_bar_t = scheduler._extract(
            scheduler.alphas_cumprod, t_cur, z_t.shape
        ).float()
        alpha_bar_prev = scheduler._extract(
            scheduler.alphas_cumprod, t_prev, z_t.shape
        ).float()

        sigma_arg = (
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t).clamp(min=1e-8)
            * (1.0 - alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
        ).clamp(min=0.0)
        sigma = eta * torch.sqrt(sigma_arg)

        dir_arg = (1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0)
        dir_xt = torch.sqrt(dir_arg) * eps_pred

        z_t = torch.sqrt(alpha_bar_prev) * z0_pred + dir_xt
        if eta > 0.0:
            z_t = z_t + sigma * torch.randn_like(z_t)

    return z_t


def _compute_cycle_loss(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    z0_A: torch.Tensor,
    z0_B: torch.Tensor,
    z0_fake_B: torch.Tensor,
    z0_fake_A: torch.Tensor,
    noise_A: torch.Tensor,
    noise_B: torch.Tensor,
    t_A: torch.Tensor,
    t_B: torch.Tensor,
    fake_B_img: torch.Tensor,
    fake_A_img: torch.Tensor,
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    prediction_type: str,
    cycle_ddim_steps: int,
    cycle_ddim_eta: float,
) -> torch.Tensor:
    """
    Compute latent-space cycle-consistency loss using shared (epsilon, t).

    Builds noisy versions of the fake latents using the same noise and
    timesteps as the primary forward pass, then denoises back via short
    DDIM to obtain reconstructions:

        rec_A = G(z_t(fake_B), condition=fake_B_img, domain=A)
        rec_B = G(z_t(fake_A), condition=fake_A_img, domain=B)

    Loss = L1(rec_A, z0_A) + L1(rec_B, z0_B).

    Args:
        model:            CycleDiTGenerator.
        scheduler:        DDPMScheduler.
        z0_A, z0_B:       Clean VAE latents for real_A and real_B.
        z0_fake_B, z0_fake_A: x0 predictions from the primary forward pass.
        noise_A, noise_B: Noise tensors used in the primary forward pass.
        t_A, t_B:         Timesteps used in the primary forward pass.
        fake_B_img, fake_A_img: Decoded fake images used as conditions.
        real_A, real_B:   Unused (reserved for future extensions).
        prediction_type:  ``"v"`` or ``"eps"``.
        cycle_ddim_steps: DDIM steps for the short reverse path.
        cycle_ddim_eta:   DDIM eta (0 = deterministic).

    Returns:
        Scalar cycle-consistency loss tensor.
    """
    z_t_rec_A = scheduler.add_noise(z0_fake_B.detach(), noise_A, t_A)
    z_t_rec_B = scheduler.add_noise(z0_fake_A.detach(), noise_B, t_B)

    z0_rec_A = _ddim_shortcut_from_xt(
        model=model,
        scheduler=scheduler,
        z_t_start=z_t_rec_A,
        t_start=t_A,
        condition=fake_B_img.detach(),
        target_domain=0,
        prediction_type=prediction_type,
        num_steps=cycle_ddim_steps,
        eta=cycle_ddim_eta,
    )
    z0_rec_B = _ddim_shortcut_from_xt(
        model=model,
        scheduler=scheduler,
        z_t_start=z_t_rec_B,
        t_start=t_B,
        condition=fake_A_img.detach(),
        target_domain=1,
        prediction_type=prediction_type,
        num_steps=cycle_ddim_steps,
        eta=cycle_ddim_eta,
    )

    loss_cyc = F.l1_loss(z0_rec_A, z0_A) + F.l1_loss(z0_rec_B, z0_B)
    return loss_cyc


def _compute_identity_loss(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    z0_A: torch.Tensor,
    z0_B: torch.Tensor,
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    device: torch.device,
    prediction_type: str,
) -> torch.Tensor:
    """
    Compute latent-space identity loss at t=0.

    Passes clean latents through the model with t=0 and same-domain
    conditioning.  At t=0 the model should predict minimal change, so
    the x0_pred should be close to the input z0:

        loss = L1(x0_pred_A, z0_A) + L1(x0_pred_B, z0_B)

    Args:
        model:          CycleDiTGenerator.
        scheduler:      DDPMScheduler (used to derive x0_pred).
        z0_A, z0_B:     Clean VAE latents for real_A and real_B.
        real_A, real_B: Conditioning images for same-domain passes.
        device:         Device for the zero timestep tensor.
        prediction_type: ``"v"`` or ``"eps"``.

    Returns:
        Scalar identity loss tensor.
    """
    b = z0_A.size(0)
    t_idt = torch.zeros(b, device=device, dtype=torch.long)

    out_idt_A = model(
        z0_A,
        t_idt,
        real_A,
        target_domain=0,
        scheduler=scheduler,
        prediction_type=prediction_type,
    )
    out_idt_B = model(
        z0_B,
        t_idt,
        real_B,
        target_domain=1,
        scheduler=scheduler,
        prediction_type=prediction_type,
    )

    z0_idt_A = out_idt_A["x0_pred"] if isinstance(out_idt_A, dict) else out_idt_A
    z0_idt_B = out_idt_B["x0_pred"] if isinstance(out_idt_B, dict) else out_idt_B

    loss_idt = F.l1_loss(z0_idt_A, z0_A) + F.l1_loss(z0_idt_B, z0_B)
    return loss_idt


def _compute_identity_weight(
    epoch: int, num_epochs: int, l_start: float, l_end: float, decay_ratio: float
) -> float:
    """
    Linearly decay the identity loss weight over the first portion of training.

    Schedule:
        [0, decay_end_epoch)  — linear ramp from l_start to l_end
        [decay_end_epoch, ∞)  — constant l_end

    where decay_end_epoch = int(num_epochs * decay_ratio).

    Args:
        epoch:       Current epoch (0-indexed).
        num_epochs:  Total training epochs.
        l_start:     Initial identity weight.
        l_end:       Final identity weight (typically 0.0).
        decay_ratio: Fraction of training over which decay occurs.

    Returns:
        Float identity weight for the current epoch.
    """
    decay_end_epoch = int(num_epochs * decay_ratio)
    if epoch < decay_end_epoch:
        progress = epoch / max(1, decay_end_epoch)
        return l_start + (l_end - l_start) * progress
    return l_end


def compute_diffusion_loss(
    z0: Tensor,
    z_t: Tensor,
    t: Tensor,
    noise: Tensor,
    model_pred: Tensor,
    real_B: Tensor,
    scheduler: DDPMScheduler,
    vae: VAEWrapper,
    perceptual_loss: Optional[nn.Module],
    lambda_perc: float,
    prediction_type: str = "v",
    min_snr_gamma: float = 0.0,
    global_step: int = 0,
    perceptual_every_n_steps: int = 1,
    perceptual_batch_fraction: float = 1.0,
) -> Tuple[Tensor, Tensor, float]:
    """
    Compute the diffusion denoising loss with an optional perceptual term.

    Pipeline:
        1. MSE prediction loss (optionally Min-SNR weighted):
               v-mode:   MSE(v_pred, v_target)
               eps-mode: MSE(eps_pred, noise)
        2. Optional perceptual branch (every perceptual_every_n_steps steps):
               predict x0 -> VAE decode -> perceptual(fake_B_pred, real_B)
        3. total = lambda_denoising * mse + lambda_perc * perceptual

    Args:
        z0:            Clean latent (N, 4, 32, 32).
        z_t:           Noisy latent at timestep t (N, 4, 32, 32).
        t:             Integer timesteps (N,).
        noise:         Noise added to z0 to produce z_t (N, 4, 32, 32).
        model_pred:    Model output — v_pred or eps_pred (N, 4, 32, 32).
        real_B:        Target domain image for perceptual supervision
                       (N, 3, 256, 256).
        scheduler:     DDPMScheduler for v-target and x0 reconstruction.
        vae:           VAEWrapper for decoding x0 to image space.
        perceptual_loss: VGG perceptual loss module, or None to disable.
        lambda_perc:   Weight for the perceptual term (0.0 = disabled).
        prediction_type: ``"v"`` or ``"eps"``.
        min_snr_gamma: Min-SNR loss weighting gamma (0.0 = disabled).
        global_step:   Global training step counter for perceptual scheduling.
        perceptual_every_n_steps: Run perceptual loss every N steps.
        perceptual_batch_fraction: Fraction of the batch used for perceptual.

    Returns:
        tuple:
            loss       — total scalar tensor (MSE + perceptual).
            loss_simple — MSE-only scalar tensor (logged separately).
            loss_perc_val — perceptual scalar as float (0.0 when disabled).
    """

    noise = noise.to(dtype=model_pred.dtype)

    def _mse_per_sample(pred: Tensor, target: Tensor) -> Tensor:
        dims = tuple(range(1, pred.dim()))
        return ((pred - target) ** 2).mean(dim=dims)

    def _min_snr_weights() -> Tensor:
        alpha_bar = scheduler.get_alpha_bar(t).to(model_pred.dtype)
        snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
        snr_cap = torch.full_like(snr, float(min_snr_gamma))
        if prediction_type == "v":
            return torch.minimum(snr, snr_cap) / (snr + 1.0)
        return torch.minimum(snr, snr_cap) / snr.clamp(min=1e-8)

    if prediction_type == "v":
        target = scheduler.get_v_target(z0.float(), noise.float(), t).to(
            model_pred.dtype
        )
        loss_per = _mse_per_sample(model_pred, target)
        x0_pred = scheduler.predict_x0_from_v(z_t.float(), model_pred.float(), t)
    elif prediction_type == "eps":
        loss_per = _mse_per_sample(model_pred, noise)
        x0_pred = scheduler.predict_x0(z_t.float(), model_pred.float(), t)
    else:
        raise ValueError(
            f"prediction_type must be 'v' or 'eps', got {prediction_type!r}"
        )

    if min_snr_gamma > 0.0:
        loss_per = loss_per * _min_snr_weights()

    loss_simple = loss_per.mean().float()

    loss = loss_simple
    loss_perc_val = 0.0

    use_perc_this_step = (global_step % max(1, perceptual_every_n_steps)) == 0
    if perceptual_loss is not None and lambda_perc > 0.0 and use_perc_this_step:
        # Perceptual term runs in FP32 outside autocast.
        bs = real_B.size(0)
        frac = min(1.0, max(0.1, float(perceptual_batch_fraction)))
        n_perc = max(1, int(bs * frac))
        x0_sub = x0_pred[:n_perc]
        real_sub = real_B[:n_perc].detach()

        fake_B_pred = vae.decode(x0_sub)
        loss_perc = perceptual_loss(fake_B_pred, real_sub)
        loss = loss + lambda_perc * loss_perc
        loss_perc_val = float(loss_perc.item())

    return loss, loss_simple, loss_perc_val
