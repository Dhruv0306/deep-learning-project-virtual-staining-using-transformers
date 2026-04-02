# model_v3/vae_wrapper.py - VAE Wrapper

Source of truth: ../../model_v3/vae_wrapper.py

This module wraps Stable Diffusion's `AutoencoderKL` for v3 latent diffusion
training and inference.

## Public Component

- `VAEWrapper`

## `VAEWrapper`

Behavior:

- loads the HuggingFace VAE identified by `model_id`
- freezes all parameters
- keeps the module in eval mode
- uses the Stable Diffusion latent scaling convention of `0.18215`

The wrapper caches the downloaded VAE checkpoint locally through the
`diffusers`/`huggingface_hub` stack.

## `encode`

Input:

- image tensor `(N, 3, H, W)` in `[-1, 1]`

Flow:

1. clamp the image to `[-1, 1]`
2. run VAE encoding
3. sample from the latent distribution
4. scale by `0.18215`

Output:

- scaled latent tensor `(N, 4, H/8, W/8)`

## `decode`

Input:

- scaled latent tensor `(N, 4, h, w)`

Flow:

1. divide by `0.18215`
2. cast to float32
3. run VAE decoding
4. clamp output to `[-1, 1]`

Output:

- image tensor `(N, 3, h*8, w*8)`

## Notes

- The wrapper stays frozen and in eval mode.
- Gradients can still flow through encode/decode if the caller does not use
	`torch.no_grad()`.
- An informative `ImportError` is raised when `diffusers` is unavailable.
