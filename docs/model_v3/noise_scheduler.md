# model_v3/noise_scheduler.py - DDPM Scheduler and DDIM Sampler

Source of truth: ../../model_v3/noise_scheduler.py

This module provides the diffusion forward-process scheduler and the DDIM
reverse sampler used by v3.

## Public Components

1. `_linear_beta_schedule`
2. `_cosine_beta_schedule`
3. `DDPMScheduler`
4. `DDIMSampler`

## `_linear_beta_schedule`

Returns a simple linear beta ramp from `1e-4` to `2e-2` over `T` steps.

## `_cosine_beta_schedule`

Implements the cosine schedule from Nichol & Dhariwal with a small offset
`s=0.008` and a maximum beta clamp of `0.999`.

## `DDPMScheduler`

Stores the diffusion schedule as module buffers so the tensors move with
the module across devices.

Key buffers:

- `betas`
- `alphas`
- `alphas_cumprod`
- `sqrt_alphas_cumprod`
- `sqrt_one_minus_alphas_cumprod`

Important methods:

- `_extract(arr, t, x_shape)`: gather per-timestep schedule values and
  broadcast them to the target tensor shape
- `add_noise(x0, noise, t)`: construct `x_t` from clean latent plus noise
- `predict_x0(x_t, eps_pred, t)`: reconstruct the clean latent from epsilon
- `get_v_target(x0, noise, t)`: build the v-parameterization target
- `predict_eps_from_v(x_t, v_pred, t)`: convert v-prediction to epsilon
- `predict_x0_from_v(x_t, v_pred, t)`: convert v-prediction directly to `x0`
- `get_alpha_bar(t)`: return `alpha_bar` for scalar or batched timesteps

## `DDIMSampler`

Performs reverse sampling from Gaussian noise back to a latent sample.

Behavior:

1. initialize `z_t` with random noise
2. iterate a timestep subsequence from high to low
3. call the model with the current latent, timestep, and condition
4. optionally apply classifier-free guidance via `cfg_scale`
5. convert the model output to epsilon or `x0` as needed
6. apply the DDIM update rule

Sampling arguments now support:

- `prediction_type`: `"v"` or `"eps"`
- `cfg_scale`: classifier-free guidance scale
- `uncond_condition`: unconditional conditioning input for CFG
- `target_domain`: translation direction id

The sampler is used both for full image generation and for the short
cycle-consistency denoising path.
