# Documentation Index

Reference documentation for all key source files in the UVCGAN v2 histology stain/unstain translation project.

---

## Model V1 — Hybrid UVCGAN + CycleGAN

| File | Documentation | Description |
|---|---|---|
| `generator.py` | [generator.md](generator.md) | V1 generator: U-Net + ViT with ReZero Transformer blocks. Architecture diagram, all building blocks, weight init |
| `discriminator.py` | [discriminator.md](discriminator.md) | V1 discriminator: single-scale PatchGAN. Architecture, receptive field, data flow |
| `losses.py` | [losses.md](losses.md) | V1 loss functions: LSGAN + cycle + identity + VGG19 perceptual + two-sided GP. All parameters and data flow |
| `training_loop.py` | [training_loop.md](training_loop.md) | V1 training loop: full step-by-step flow, hardcoded hyperparameters, differences from v2 |

## Model V2 — True UVCGAN

| File | Documentation | Description |
|---|---|---|
| `uvcgan_v2_generator.py` | [uvcgan_v2_generator.md](uvcgan_v2_generator.md) | V2 generator: U-Net + ViT with LayerScale, cross-domain fusion, gradient checkpointing. Full architecture diagram and every class/function |
| `spectral_norm_discriminator.py` | [spectral_norm_discriminator.md](spectral_norm_discriminator.md) | V2 discriminator: multi-scale spectral-norm PatchGAN. Architecture at each scale, spectral norm explanation |
| `advanced_losses.py` | [advanced_losses.md](advanced_losses.md) | V2 loss functions: LSGAN + one-sided GP (γ=100) + contrastive + spectral. Paper-aligned formulas and full data flow |
| `training_loop_v2.py` | [training_loop_v2.md](training_loop_v2.md) | V2 training loop: gradient accumulation, AMP safety, per-batch step detail, validation/early stopping separation |

## Shared

| File | Documentation | Description |
|---|---|---|
| `config.py` | [config.md](config.md) | All hyperparameters across all dataclasses. Every field explained. `get_8gb_config()` VRAM savings table |
| `data_loader.py` | [data_loader.md](data_loader.md) | Unpaired dataset class, transform pipeline, `getDataLoader` factory, and `denormalize` helper |
| `replay_buffer.py` | [replay_buffer.md](replay_buffer.md) | Fixed-size pool of past fake images for discriminator stabilisation |
| `metrics.py` | [metrics.md](metrics.md) | SSIM, PSNR, and FID metrics via `MetricsCalculator`. InceptionV3 feature extraction |
| `validation.py` | [validation.md](validation.md) | Per-epoch validation image saving and quantitative metric computation |
| `testing.py` | [testing.md](testing.md) | Final test-set inference and image export |
| `EarlyStopping.py` | [early_stopping.md](early_stopping.md) | SSIM-plateau and loss-divergence early stopping |
| `history_utils.py` | [history_utils.md](history_utils.md) | Training history visualisation, CSV save/append/load |
| `preprocess_data.py` | [preprocess_data.md](preprocess_data.md) | Whole-slide patch extraction, tissue filtering, train/test split |
| `trainModel.py` | [train_model.md](train_model.md) | Interactive training entry point. Prompts for parameters and dispatches to v1 or v2 loop |
| `app.py` | [app.md](app.md) | Patch-based whole-slide inference with cosine blending. Supports both model versions |
