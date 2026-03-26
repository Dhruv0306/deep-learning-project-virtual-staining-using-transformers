# `v3_losses.py` — v3 Diffusion Loss Helpers

Source of truth: `../v3_losses.py`

**Role:** Encapsulates the v3 diffusion loss computation so the training loop stays clean.

---

## `compute_diffusion_loss`

```python
loss, loss_simple, loss_perc_val = compute_diffusion_loss(
    z_t,
    t,
    noise,
    eps_pred,
    real_B,
    scheduler,
    vae,
    perceptual_loss,
    lambda_perc,
)
```

### Inputs
- `z_t`: Noisy latent `(N, 4, 32, 32)`
- `t`: Timestep tensor `(N,)`
- `noise`: Ground-truth noise `(N, 4, 32, 32)`
- `eps_pred`: Predicted noise `(N, 4, 32, 32)`
- `real_B`: Target image `(N, 3, 256, 256)` in `[-1, 1]`
- `scheduler`: `DDPMScheduler` used for `predict_x0`
- `vae`: `VAEWrapper` for decode
- `perceptual_loss`: Optional `VGGPerceptualLossV2`
- `lambda_perc`: Weight for perceptual term

### Outputs
- `loss`: Total loss (MSE + optional perceptual)
- `loss_simple`: MSE noise prediction loss
- `loss_perc_val`: Perceptual loss value as float

### Notes
- Perceptual loss is computed in FP32 outside autocast.
- The VAE is frozen but gradients can flow through unless the caller wraps in `torch.no_grad()`.
