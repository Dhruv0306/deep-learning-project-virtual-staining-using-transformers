# `training_loop_v3.py` — v3 Diffusion Training Loop

Source of truth: `../training_loop_v3.py`

**Role:** Trains the DiT diffusion model with optional perceptual loss, EMA, AMP, and validation.

---

## Key Steps

1. Load data via `getDataLoader` (expects aligned `(real_A, real_B)` pairs).
2. Encode `real_B` to latents using `VAEWrapper` (frozen).
3. Sample timestep `t` and noise `eps`.
4. Create noisy latent `z_t` with `DDPMScheduler.add_noise`.
5. Encode condition `c = ConditionEncoder(real_A)`.
6. Predict noise with `DiTGenerator(z_t, t, c)`.
7. Loss = MSE(noise prediction) + optional VGG perceptual on decoded `z0_pred`.
8. Optimizer: AdamW (lr=1e-4), AMP, gradient clipping.
9. Maintain EMA of DiT weights.
10. Periodic validation (DDIM + MetricsCalculator). `_run_validation_v3` uses `is_test=False`.
11. Checkpoint save (every 20 epochs and at end).
12. Final test-set inference using the same v3 validation path (saved to `test_images`) with `is_test=True`.

---

## Checkpoint Schema

```python
{
  "dit_state_dict": ...,
  "cond_encoder_state_dict": ...,
  "optimizer_state_dict": ...,
  "ema_state_dict": ...,
  "config": ...
}
```

---

## Logged Scalars

- `Loss/DiT`
- `Loss/Perceptual`
- `LR/DiT`
- `Diagnostics/GradNorm`
- `Validation/ssim_B`, `Validation/psnr_B`, optional `Validation/fid`

## Perceptual Resize

If `lambda_perceptual_v3 > 0`, the VGG perceptual loss uses
`cfg.loss.perceptual_resize` (fallback to 128 if unset).
