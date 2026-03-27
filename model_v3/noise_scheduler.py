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
    return torch.linspace(1e-4, 2e-2, num_timesteps, dtype=torch.float32)


def _cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> Tensor:
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    f = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    f = f / f[0]
    betas = 1.0 - (f[1:] / f[:-1])
    return betas.clamp(max=0.999)


class DDPMScheduler(nn.Module):
    """
    DDPM forward-process scheduler with precomputed buffers.

Key buffer shapes:
    betas, alphas, alphas_cumprod: (T,)
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
        Gather values from a buffer for batched timesteps and reshape
        for broadcasting.
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

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        """
        Return alpha_bar(t) for scalar or batched timesteps.
        """
        return self._extract(self.alphas_cumprod, t, t.shape)


class DDIMSampler:
    """
    Deterministic DDIM sampler.

Sample output shape equals requested ``shape`` argument.
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
    ) -> Tensor:
        """
        Sample a denoised latent z0 from pure noise.
        """
        b = shape[0]
        z_t = torch.randn(shape, device=device)

        timesteps = torch.linspace(
            0, self.scheduler.num_timesteps - 1, num_steps, device=device
        ).long()
        timesteps = timesteps.flip(0)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), int(t.item()), device=device, dtype=torch.long)
            eps_pred = model(z_t, t_batch, condition)

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.scheduler.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(
                    1.0, device=device, dtype=alpha_bar_t.dtype
                )

            z0_pred = self.scheduler.predict_x0(z_t, eps_pred, t_batch)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_xt = torch.sqrt((1 - alpha_bar_prev) - sigma**2) * eps_pred

            z_t = torch.sqrt(alpha_bar_prev) * z0_pred + dir_xt
            if eta > 0.0:
                z_t = z_t + sigma * torch.randn_like(z_t)

        return z_t
