# model_v3/losses.py - v3 Loss Helpers

Source of truth: ../../model_v3/losses.py

Role: Computes v3 diffusion objective with optional perceptual term.

---

## Component Structure

1. compute_diffusion_loss

---

## 1) compute_diffusion_loss

Inputs:
- z_t: (N,4,32,32)
- t: (N,)
- noise: (N,4,32,32)
- eps_pred: (N,4,32,32)
- real_B: (N,3,256,256)
- scheduler: DDPMScheduler
- vae: VAEWrapper
- perceptual_loss: module or None
- lambda_perc: float

### Dataflow

Step A: base denoising objective
1. cast noise to eps_pred dtype
2. MSE term:
   - loss_simple = mse(eps_pred, noise)
   - scalar tensor
3. initialize total loss:
   - loss = loss_simple

Step B: optional perceptual branch (only if enabled)
4. x0 prediction in latent space:
   - z0_pred = scheduler.predict_x0(z_t, eps_pred, t)
   - shape: (N,4,32,32)
5. decode predicted latent:
   - fake_B_pred = vae.decode(z0_pred)
   - shape: (N,3,256,256)
6. perceptual scalar:
   - loss_perc = perceptual_loss(fake_B_pred, real_B)
7. weighted add:
   - loss = loss + lambda_perc * loss_perc

Outputs:
- loss: scalar tensor total
- loss_simple: scalar tensor MSE-only term
- loss_perc_val: python float (0.0 when branch disabled)

---

## Shape Summary

- latent tensors remain (N,4,32,32)
- decoded image tensor is (N,3,256,256)
- all returned losses are scalars (plus float mirror for perceptual logging)
