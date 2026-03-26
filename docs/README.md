# Documentation Index

Reference documentation for all key source files in the UVCGAN v2 histology stain/unstain translation project.

---

## Model V1 — Hybrid UVCGAN + CycleGAN

| File | Documentation | Description |
|---|---|---|
| `model_v1/generator.py` | [generator.md](model_v1/generator.md) | V1 generator: U-Net + ViT with ReZero Transformer blocks. Architecture diagram, all building blocks, weight init |
| `model_v1/discriminator.py` | [discriminator.md](model_v1/discriminator.md) | V1 discriminator: single-scale PatchGAN. Architecture, receptive field, data flow |
| `model_v1/losses.py` | [losses.md](model_v1/losses.md) | V1 loss functions: LSGAN + cycle + identity + VGG19 perceptual + two-sided GP. All parameters and data flow |
| `model_v1/training_loop.py` | [training_loop.md](model_v1/training_loop.md) | V1 training loop: full step-by-step flow, hardcoded hyperparameters, differences from v2 |

## Model V2 — True UVCGAN

| File | Documentation | Description |
|---|---|---|
| `model_v2/generator.py` | [generator.md](model_v2/generator.md) | V2 generator: U-Net + ViT with LayerScale, cross-domain fusion, gradient checkpointing. Full architecture diagram and every class/function |
| `model_v2/discriminator.py` | [discriminator.md](model_v2/discriminator.md) | V2 discriminator: multi-scale spectral-norm PatchGAN. Architecture at each scale, spectral norm explanation |
| `model_v2/losses.py` | [losses.md](model_v2/losses.md) | V2 loss functions: LSGAN + one-sided GP (?=100) + contrastive + spectral. Paper-aligned formulas and full data flow |
| `model_v2/training_loop.py` | [training_loop.md](model_v2/training_loop.md) | V2 training loop: gradient accumulation, AMP safety, per-batch step detail, validation/early stopping separation |

## Model V3 — DiT Diffusion

| File | Documentation | Description |
|---|---|---|
| `model_v3/generator.py` | [generator.md](model_v3/generator.md) | DiT backbone, ConditionEncoder, adaLN-Zero blocks |
| `model_v3/noise_scheduler.py` | [noise_scheduler.md](model_v3/noise_scheduler.md) | DDPM scheduler and DDIM sampler |
| `model_v3/vae_wrapper.py` | [vae_wrapper.md](model_v3/vae_wrapper.md) | SD VAE wrapper for latent diffusion |
| `model_v3/training_loop.py` | [training_loop.md](model_v3/training_loop.md) | Diffusion training loop with EMA and AMP |
| `model_v3/losses.py` | [losses.md](model_v3/losses.md) | v3 diffusion loss helper functions |
| `model_v3/data_loader.py` | [data_loader.md](model_v3/data_loader.md) | Paired A/B dataloader for v3 diffusion |
| `model_v3/history_utils.py` | [history_utils.md](model_v3/history_utils.md) | v3 training history CSV/plots (no discriminator terms) |

## Shared

| File | Documentation | Description |
|---|---|---|
| `config.py` | [config.md](config.md) | All hyperparameters across all dataclasses. Every field explained. `get_8gb_config()` VRAM savings table |
| `shared/data_loader.py` | [data_loader.md](shared/data_loader.md) | Unpaired dataset class, transform pipeline, `getDataLoader` factory, and `denormalize` helper |
| `shared/replay_buffer.py` | [replay_buffer.md](shared/replay_buffer.md) | Fixed-size pool of past fake images for discriminator stabilisation |
| `shared/metrics.py` | [metrics.md](shared/metrics.md) | SSIM, PSNR, and FID metrics via `MetricsCalculator`. InceptionV3 feature extraction |
| `shared/validation.py` | [validation.md](shared/validation.md) | Per-epoch validation image saving and quantitative metric computation |
| `shared/testing.py` | [testing.md](shared/testing.md) | Final test-set inference and image export |
| `shared/EarlyStopping.py` | [early_stopping.md](shared/early_stopping.md) | SSIM-plateau and loss-divergence early stopping |
| `shared/history_utils.py` | [history_utils.md](shared/history_utils.md) | Training history visualisation, CSV save/append/load |
| `preprocess_data.py` | [preprocess_data.md](preprocess_data.md) | Whole-slide patch extraction, tissue filtering, train/test split |
| `trainModel.py` | [train_model.md](train_model.md) | Interactive training entry point. Prompts for parameters and dispatches to v1/v2/v3 loops |
| `app.py` | [app.md](app.md) | Patch-based whole-slide inference with cosine blending. Supports all model versions |
