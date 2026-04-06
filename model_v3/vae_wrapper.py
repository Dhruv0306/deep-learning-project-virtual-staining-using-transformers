"""
VAE wrapper for the v3 latent diffusion pipeline.

Component structure:
    1) pretrained AutoencoderKL loader
    2) encode helper (image -> latent)
    3) decode helper (latent -> image)

Scaling convention:
    Stable Diffusion latent scaling factor 0.18215 is applied on encode
    and undone on decode.
"""

from __future__ import annotations

import os
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
    Frozen Stable Diffusion VAE wrapper for latent diffusion.

    Downloads and caches the AutoencoderKL checkpoint from HuggingFace on
    first use (~335 MB).  All VAE parameters are frozen; the wrapper is
    always in eval mode.

    Scaling convention: SD latents are multiplied by 0.18215 on encode and
    divided by 0.18215 on decode, matching the original SD training setup.

    Args:
        model_id: HuggingFace model ID for the VAE
                  (default ``"stabilityai/sd-vae-ft-mse"``).
    """

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        cache_dir: Optional[str] = None,
        offline_first: bool = True,
    ):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.offline_first = offline_first
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The `local_dir_use_symlinks` argument is deprecated",
                category=UserWarning,
            )
            self.vae = self._load_vae()
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.latent_scale = 0.18215

    def _load_vae(self) -> AutoencoderKL:
        """
        Load VAE weights with offline-first behavior.

        Strategy:
            1) If ``model_id`` is a local path, load from that path only.
            2) If ``offline_first`` is enabled, try HuggingFace cache only
               (``local_files_only=True``).
            3) If not found in cache, fall back to online download.
        """
        common_kwargs: dict[str, Any] = {}
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir

        if os.path.isdir(self.model_id):
            return AutoencoderKL.from_pretrained(
                self.model_id,
                local_files_only=True,
                **common_kwargs,
            )

        cached_exc: Optional[Exception] = None
        if self.offline_first:
            try:
                return AutoencoderKL.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                    **common_kwargs,
                )
            except Exception as exc:
                cached_exc = exc

        try:
            return AutoencoderKL.from_pretrained(self.model_id, **common_kwargs)
        except Exception as exc:
            if cached_exc is not None:
                raise RuntimeError(
                    "Could not load VAE from local HuggingFace cache, and online "
                    "download also failed. If you are offline, connect once to "
                    "download the model or provide a local model directory path "
                    f"for model_id. model_id={self.model_id!r}"
                ) from exc
            raise

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode an image batch into scaled latents.

        Pipeline: clamp → VAE encode → sample from latent dist → scale.

        Args:
            x: Image tensor (N, 3, H, W) in [-1, 1].

        Returns:
            Scaled latent tensor (N, 4, H/8, W/8).
        """
        x = x.clamp(-1.0, 1.0)
        enc_out = self.vae.encode(x)
        latents = cast(Any, enc_out).latent_dist.sample()
        return latents * self.latent_scale

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode a scaled latent batch into images.

        Pipeline: unscale → VAE decode → clamp to [-1, 1].

        Args:
            z: Scaled latent tensor (N, 4, H, W).

        Returns:
            Image tensor (N, 3, H*8, W*8) in [-1, 1].
        """
        z = (z / self.latent_scale).float()
        dec_out = self.vae.decode(cast(torch.FloatTensor, z))
        img = cast(Any, dec_out).sample
        return img.clamp(-1.0, 1.0)
