"""
Noise scheduler and DDIM sampler for the v3 latent diffusion pipeline.

Component structure:
    1) beta schedule builders
    2) DDPMScheduler (forward process helpers)
    3) DDIMSampler (reverse sampling)

Primary latent shape:
    z: (N, 4, 32, 32)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn, Tensor


def _linear_beta_schedule(num_timesteps: int) -> Tensor:
    """Linear beta schedule from 1e-4 to 2e-2 over num_timesteps steps."""
    return torch.linspace(1e-4, 2e-2, num_timesteps, dtype=torch.float32)


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Tensor:
    """
    Cosine beta schedule (Nichol & Dhariwal, 2021).

    Computes betas from the cosine-squared alpha_bar curve, clamped to
    a maximum of 0.999 to prevent singular diffusion steps.

    Args:
        num_timesteps: Total diffusion steps T.
        s:             Small offset to prevent beta from being too small
                       near t=0 (default 0.008 per the paper).

    Returns:
        Float32 tensor of shape (num_timesteps,).
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    f = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    f = f / f[0]
    betas = 1.0 - (f[1:] / f[:-1])
    return betas.clamp(max=0.999)


class DDPMScheduler(nn.Module):
    """
    DDPM forward-process scheduler with precomputed alpha buffers.

    Registers all schedule tensors as non-trainable buffers so they move
    with the module when calling ``.to(device)``.

    Key buffers (all shape (T,)):
        betas, alphas, alphas_cumprod,
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

    Args:
        num_timesteps: Total diffusion steps T (default 1000).
        beta_schedule: ``"cosine"`` (recommended) or ``"linear"``.
    """


    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor

    def __init__(self, num_timesteps: int = 1000, beta_schedule: str = "cosine"):
        super().__init__()
        if beta_schedule not in ("linear", "cosine"):
            raise ValueError(
                f"beta_schedule must be 'linear' or 'cosine', got {beta_schedule!r}"
            )

        betas = (
            _linear_beta_schedule(num_timesteps)
            if beta_schedule == "linear"
            else _cosine_beta_schedule(num_timesteps)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.num_timesteps = int(num_timesteps)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def _extract(self, arr: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
        """
        Gather per-sample schedule values and broadcast to x_shape.

        Args:
            arr:     1-D schedule buffer of length T.
            t:       Integer timestep tensor (N,) or scalar.
            x_shape: Shape of the target tensor for broadcasting.

        Returns:
            Tensor of shape (N, 1, 1, ...) broadcastable to x_shape.
        """
        if t.dim() == 0:
            t = t.view(1)
        out = arr.gather(0, t.to(arr.device))
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def add_noise(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise

        Shapes:
            x0, noise: (N, C, H, W)
            t:         (N,)
            x_t:       (N, C, H, W)
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    def predict_x0(self, x_t: Tensor, eps_pred: Tensor, t: Tensor) -> Tensor:
        """
        Reconstruct x0 estimate from noisy latent and predicted noise.

        Shapes:
            x_t, eps_pred: (N, C, H, W)
            t:             (N,)
            x0_pred:       (N, C, H, W)
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar

    def get_v_target(self, x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        """
        Compute v-parameterization target.

        v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x0
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_alpha_bar * noise - sqrt_one_minus * x0

    def predict_eps_from_v(self, x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
        """
        Recover epsilon prediction from v-parameterisation output.

        eps = sqrt(alpha_bar) * v + sqrt(1 - alpha_bar) * x_t
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_bar * v_pred + sqrt_one_minus * x_t

    def predict_x0_from_v(self, x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
        """
        Recover x0 prediction from v-parameterisation output.

        x0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_bar * x_t - sqrt_one_minus * v_pred

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        """
        Return alpha_bar(t) for scalar or batched timesteps.

        Args:
            t: Integer timestep tensor (N,) or scalar.

        Returns:
            alpha_bar values with the same shape as t.
        """
        return self._extract(self.alphas_cumprod, t, t.shape)


class DDIMSampler:
    """
    Deterministic DDIM reverse sampler (Song et al., 2020).

    Iterates a linearly-spaced subsequence of timesteps from T-1 down to 0,
    applying the DDIM update rule at each step.  Setting ``eta=0`` gives
    fully deterministic sampling; ``eta=1`` recovers DDPM stochasticity.

    Uses ``alphas_cumprod[0]`` as the terminal alpha_bar_prev when the
    subsequence reaches t=0 to keep the final update well-defined.

    Args:
        scheduler: DDPMScheduler instance whose alpha buffers are used.
    """


    def __init__(self, scheduler: DDPMScheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition: Tensor,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        num_steps: int = 50,
        eta: float = 0.0,
        prediction_type: str = "v",
        cfg_scale: float = 1.0,
        uncond_condition: Tensor | None = None,
        target_domain: int = 1,
    ) -> Tensor:
        """
        Sample a denoised latent z0 from pure Gaussian noise using DDIM.

        Args:
            model:             CycleDiTGenerator denoiser.
            condition:         Conditioning image or tokens (N, 3, H, W) or
                               (N, L, hidden_dim).
            shape:             Output latent shape (N, 4, 32, 32).
            device:            Target device.
            num_steps:         DDIM denoising steps (default 50).
            eta:               Stochasticity coefficient.  0 = deterministic
                               DDIM; 1 ≈ DDPM.
            prediction_type:   ``"v"`` or ``"eps"``.
            cfg_scale:         Classifier-free guidance scale.  1.0 = no
                               guidance; >1.0 amplifies the conditional signal.
            uncond_condition:  Unconditional condition for CFG.  Defaults to
                               zeros when cfg_scale > 1.0 and not provided.
            target_domain:     Target domain id (0 = unstained, 1 = stained).

        Returns:
            Denoised latent tensor of shape ``shape``.
        """
        b = shape[0]
        z_t = torch.randn(shape, device=device)

        timesteps = torch.linspace(
            0, self.scheduler.num_timesteps - 1, num_steps, device=device
        ).long()
        timesteps = timesteps.flip(0)  # [T-1, ..., t_1, t_0]

        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            cond_out = model(z_t, t_batch, condition, target_domain=target_domain)
            cond_out_tensor = cond_out["v_pred"] if isinstance(cond_out, dict) else cond_out
            model_out = cond_out_tensor
            if cfg_scale > 1.0:
                if uncond_condition is None:
                    uncond_condition = torch.zeros_like(condition)
                uncond_out = model(
                    z_t,
                    t_batch,
                    uncond_condition,
                    target_domain=target_domain,
                )
                uncond_out = (
                    uncond_out["v_pred"] if isinstance(uncond_out, dict) else uncond_out
                )
                model_out = uncond_out + cfg_scale * (cond_out_tensor - uncond_out)

            if prediction_type == "v":
                eps_pred = self.scheduler.predict_eps_from_v(z_t, model_out, t_batch)
            elif prediction_type == "eps":
                eps_pred = model_out
            else:
                raise ValueError(
                    f"prediction_type must be 'v' or 'eps', got {prediction_type!r}"
                )

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = self.scheduler.alphas_cumprod[
                    torch.zeros(1, device=device, dtype=torch.long)
                ].squeeze()

            z0_pred = self.scheduler.predict_x0(z_t, eps_pred, t_batch)

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
