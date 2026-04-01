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

    Key buffers:
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
        Recover eps prediction from v prediction.
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_bar * v_pred + sqrt_one_minus * x_t

    def predict_x0_from_v(self, x_t: Tensor, v_pred: Tensor, t: Tensor) -> Tensor:
        """
        Recover x0 prediction from v prediction.
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_alpha_bar * x_t - sqrt_one_minus * v_pred

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        """
        Return alpha_bar(t) for scalar or batched timesteps.
        """
        return self._extract(self.alphas_cumprod, t, t.shape)


class DDIMSampler:
    """
    Deterministic DDIM sampler.

    Uses alphas_cumprod[0] as the terminal alpha_bar_prev when the
    subsequence reaches t=0 to keep the final update well-defined.
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
        Sample a denoised latent z0 from pure noise using DDIM.

        Args:
            model:     DiTGenerator -- accepts (z_t, t_batch, condition).
            condition: Pre-computed condition vector (N, hidden_dim).
            shape:     Output shape (N, 4, 32, 32).
            device:    Target device.
            num_steps: Number of DDIM denoising steps (default 50).
            eta:       Stochasticity coefficient.  0 = deterministic DDIM;
                       1 = equivalent to DDPM.

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
