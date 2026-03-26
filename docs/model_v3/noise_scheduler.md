# `model_v3/noise_scheduler.py` — DDPM Scheduler + DDIM Sampler

Source of truth: `../../model_v3/noise_scheduler.py`

**Role:** Implements the forward diffusion process and deterministic DDIM sampling.

---

## `DDPMScheduler`

Initializes a beta schedule and precomputes:

- `betas`
- `alphas`
- `alphas_cumprod` (`alpha_bar`)
- `sqrt_alphas_cumprod`
- `sqrt_one_minus_alphas_cumprod`

### Methods

```python
sch = DDPMScheduler(num_timesteps=1000, beta_schedule="cosine")
```

- `add_noise(x0, noise, t)`  
  Returns `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise`

- `predict_x0(x_t, eps_pred, t)`  
  Reconstructs `x0` from noisy latent and predicted noise

- `get_alpha_bar(t)`  
  Returns `alpha_bar_t`

---

## `DDIMSampler`

Deterministic reverse diffusion. Supports `eta=0.0` for fully deterministic sampling.

```python
sampler = DDIMSampler(sch)
z0 = sampler.sample(model, condition, shape=(N, 4, 32, 32), device=device, num_steps=50)
```

### Notes

- `num_steps=50` is the default validation setting.
- Lower step counts (e.g., 10) are acceptable for quick previews.


