"""
VAE wrapper for the v3 latent diffusion pipeline.

Provides a thin interface around diffusers.AutoencoderKL with the
Stable Diffusion latent scaling factor applied.
"""

from __future__ import annotations

from typing import Any, Optional, cast
import warnings

import torch
from torch import nn, Tensor

try:
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "diffusers is required for v3. Install with `pip install diffusers>=0.27.0`."
    ) from exc


class VAEWrapper(nn.Module):
    """
    Frozen VAE wrapper for latent diffusion.

    The VAE is always frozen (no trainable params), but gradients may
    still flow through it unless the caller wraps encode/decode in
    torch.no_grad().
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.model_id = model_id
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `local_dir_use_symlinks` argument is deprecated",
                category=UserWarning,
            )
            self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.latent_scale = 0.18215

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode an image batch into latents.

        Args:
            x: (N, 3, H, W) in [-1, 1]

        Returns:
            (N, 4, H/8, W/8) latent scaled by 0.18215.
        """
        x = x.clamp(-1.0, 1.0)
        enc_out = self.vae.encode(x)
        latents = cast(Any, enc_out).latent_dist.sample()
        return latents * self.latent_scale

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode a latent batch into images.

        Args:
            z: (N, 4, H, W) latent (scaled)

        Returns:
            (N, 3, H*8, W*8) in [-1, 1]
        """
        z = (z / self.latent_scale).float()
        dec_out = self.vae.decode(cast(torch.FloatTensor, z))
        img = cast(Any, dec_out).sample
        return img.clamp(-1.0, 1.0)
