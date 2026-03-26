# `model_v3/vae_wrapper.py` — VAE Wrapper

Source of truth: `../../model_v3/vae_wrapper.py`

**Role:** Loads a pretrained Stable Diffusion VAE and exposes `encode` / `decode` for latent diffusion.

---

## Overview

The v3 pipeline operates in the latent space of a pretrained VAE. `VAEWrapper`:

- Loads `diffusers.AutoencoderKL` from a model id or local path.
- Freezes all VAE parameters (`requires_grad=False`).
- Applies the Stable Diffusion latent scaling factor `0.18215`.

---

## Interface

```python
vae = VAEWrapper(model_id="stabilityai/sd-vae-ft-mse")

z = vae.encode(x)   # x: (N, 3, 256, 256) -> z: (N, 4, 32, 32)
img = vae.decode(z) # z: (N, 4, 32, 32) -> img: (N, 3, 256, 256)
```

### `encode(x)`
- Input: `(N, 3, H, W)` in `[-1, 1]`
- Output: `(N, 4, H/8, W/8)` scaled by `0.18215`

### `decode(z)`
- Input: `(N, 4, H, W)` scaled
- Output: `(N, 3, H*8, W*8)` in `[-1, 1]`

---

## Notes

- The VAE is frozen, but gradients can still flow through it unless you wrap calls in `torch.no_grad()`.
- If `diffusers` is missing, `VAEWrapper` raises a clear `ImportError` with install instructions.

