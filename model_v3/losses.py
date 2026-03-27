"""
Loss helpers for the v3 diffusion training loop.

Component structure:
    1) compute_diffusion_loss

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
) -> Tuple[Tensor, Tensor, float]:
    """
     Compute diffusion training loss with optional perceptual term.

     Dataflow:
          1) MSE prediction loss:
              - eps mode: mse(eps_pred, noise)
              - v mode:   mse(v_pred, v_target)
          2) optional perceptual branch:
              predict x0 -> decode to fake_B_pred -> perceptual(fake_B_pred, real_B)
          3) total = mse + lambda_perc * perceptual

    Returns:
        loss: total scalar tensor
        loss_simple: scalar tensor (MSE term)
        loss_perc_val: perceptual scalar as float (0.0 when disabled)
    """
    noise = noise.to(dtype=model_pred.dtype)
    if prediction_type == "v":
        target = scheduler.get_v_target(z0.float(), noise.float(), t).to(model_pred.dtype)
        loss_simple = F.mse_loss(model_pred, target).float()
        x0_pred = scheduler.predict_x0_from_v(z_t.float(), model_pred.float(), t)
    elif prediction_type == "eps":
        loss_simple = F.mse_loss(model_pred, noise).float()
        x0_pred = scheduler.predict_x0(z_t.float(), model_pred.float(), t)
    else:
        raise ValueError(
            f"prediction_type must be 'v' or 'eps', got {prediction_type!r}"
        )

    loss = loss_simple
    loss_perc_val = 0.0

    if perceptual_loss is not None and lambda_perc > 0.0:
        # Perceptual term runs in FP32 outside autocast.
        fake_B_pred = vae.decode(x0_pred)
        loss_perc = perceptual_loss(fake_B_pred, real_B)
        loss = loss + lambda_perc * loss_perc
        loss_perc_val = float(loss_perc.item())

    return loss, loss_simple, loss_perc_val

