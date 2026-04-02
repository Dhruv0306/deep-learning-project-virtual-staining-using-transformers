# Model V3 Dataflow Pipeline

Model v3 is the diffusion-based CycleDiT pipeline for virtual staining. It combines a frozen Stable Diffusion VAE, a DiT generator, a DDPM/DDIM scheduler pair, and dual projection discriminators to learn both stain synthesis and cycle consistency in latent space.

## What Lives Here

- [generator.md](generator.md) - CycleDiT generator, conditioning tokenizer, and domain embeddings
- [discriminator.md](discriminator.md) - local, global, and FFT discriminator branches
- [noise_scheduler.md](noise_scheduler.md) - DDPM forward scheduler and DDIM reverse sampler
- [vae_wrapper.md](vae_wrapper.md) - frozen Stable Diffusion VAE wrapper
- [losses.md](losses.md) - diffusion, adversarial, cycle, identity, and perceptual losses
- [training_loop.md](training_loop.md) - full adversarial diffusion training pipeline
- [history_utils.md](history_utils.md) - CSV and plotting helpers for training history

## End-to-End Dataflow

```
real_A / real_B images
    -> shared data loader
    -> normalize to [-1, 1]
    -> encode target image with frozen VAE
    -> sample noise and timestep
    -> add noise with DDPM scheduler
    -> CycleDiT generator predicts v or eps
    -> scheduler converts prediction back to x0 / denoised latent
    -> VAE decodes latent to image space
    -> discriminators evaluate real and fake images
    -> losses combine diffusion + adversarial + cycle + identity + perceptual terms
    -> optimizer step + EMA update
```

## Training Pipeline

### 1. Input preparation

A batch contains paired domain samples, but they are treated as unpaired training signals:

- `real_A` is the source-domain image used as conditioning input
- `real_B` is the target-domain image used for latent supervision, cycle reconstruction, and validation

Both domains are resized and normalized to the generator's expected range before entering the model.

### 2. Latent diffusion path

The v3 training loop encodes the clean target latent with the VAE, samples a timestep `t`, adds Gaussian noise, and asks the generator to predict the diffusion target in latent space. The generator can operate in either `v`-prediction or `eps`-prediction mode.

### 3. Conditioning path

The raw conditioning image is tokenized with patch embeddings, positional encodings, and domain conditioning. The resulting sequence is fused with the timestep embedding inside the DiT blocks, so the generator sees both where the structure came from and which translation direction it should produce.

### 4. Loss path

The final objective is a weighted sum of:

- diffusion denoising loss
- adversarial LSGAN loss from the projection discriminators
- cycle consistency loss from short DDIM reconstruction
- identity loss for same-domain stability
- optional perceptual loss on decoded predictions
- optional R1 regularization on discriminators

### 5. Sampling and evaluation path

Validation and testing use the EMA generator plus DDIM sampling to produce full-resolution outputs. The evaluation path reports SSIM, PSNR, and optional FID for both translation directions, then writes comparison grids to disk.

## Model Map

| File | Role in the pipeline |
|---|---|
| `generator.py` | Builds the latent-space DiT translator and domain conditioning path |
| `noise_scheduler.py` | Creates noisy latents during training and denoises them during sampling |
| `vae_wrapper.py` | Moves between pixel space and latent space |
| `discriminator.py` | Scores local texture, global structure, and frequency-domain artifacts |
| `losses.py` | Combines diffusion, GAN, cycle, identity, and perceptual objectives |
| `training_loop.py` | Orchestrates optimization, EMA, validation, checkpointing, and logging |
| `history_utils.py` | Tracks and visualizes losses across epochs |

## Recommended Reading Order

1. Start with [training_loop.md](training_loop.md) to understand how the full system is assembled.
2. Read [generator.md](generator.md) and [noise_scheduler.md](noise_scheduler.md) to see the latent diffusion path.
3. Read [losses.md](losses.md) and [discriminator.md](discriminator.md) to see how training signals are formed.
4. Use [vae_wrapper.md](vae_wrapper.md) for the latent/image conversion details.