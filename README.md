# True UVCGAN v2 for Histology Stain/Unstain Translation

This project implements a **true UVCGAN v2** pipeline (Prokopenko et al., 2023) for unpaired image-to-image translation between unstained and H&E-stained histology tissue images. It includes:

- Dataset preprocessing to create 256×256 patches with tissue/background filtering
- Three model variants: a v1 hybrid (UVCGAN + CycleGAN), a v2 true UVCGAN, and a v3 DiT diffusion model
- Paper-aligned training with LSGAN + one-sided gradient penalty, cross-domain feature sharing, and ViT bottleneck with LayerScale
- VRAM-optimised configuration for 8 GB GPUs
- Validation every N epochs with SSIM/PSNR/FID metrics, TensorBoard logging, and early stopping
- Inference to stain or unstain whole-slide images via patch-based reconstruction

## Dataset

This project uses the **E-stainind DermaRepo H&E staining dataset** with **unstained** (`Un_Stained`) and **stained** (`C_Stained`) image domains. All rights for the dataset are held by its original owners and licensors.

---

## Project Layout

### Entry Points

| File | Purpose |
|---|---|
| `trainModel.py` | Training entry point � prompts for epoch size, epochs, test size, and model version |
| `app.py` | Inference script — translates whole-slide images via patch-based staining/unstaining |
| `preprocess_data.py` | Patch extraction, tissue/background filtering, and train/test split |
| `unzip.py` | Helper to extract the dataset ZIP archive into the expected directory layout |

### Configuration

| File | Purpose |
|---|---|
| `config.py` | Centralised configuration via `UVCGANConfig` dataclasses. Provides `get_default_config(model_version=1\|2)` and `get_8gb_config()` for v1/v2 plus `get_dit_config()` and `get_dit_8gb_config()` for v3 |

### Training Loops

| File | Purpose |
|---|---|
| `model_v1/training_loop.py` | v1 training loop — CycleGAN-style with LSGAN, single PatchGAN discriminator, gradient clipping, AMP |
| `model_v2/training_loop.py` | v2 training loop — paper-aligned LSGAN + one-sided GP, multi-scale spectral-norm discriminators, warm-up + linear LR decay, gradient accumulation, and per-interval validation |
| `model_v3/training_loop.py` | v3 training loop � DiT diffusion with EMA, DDPM/DDIM sampling, and validation/testing |

### Models

| File | Purpose |
|---|---|
| `model_v1/generator.py` | v1 generator — U-Net + ViT bottleneck with ReZero Transformer blocks |
| `model_v2/generator.py` | v2 generator — U-Net + ViT with LayerScale, cross-domain skip fusion, Kaiming/Xavier weight init, and optional gradient checkpointing |
| `model_v3/generator.py` | v3 generator � DiT backbone with conditional encoder for latent diffusion |
| `model_v1/discriminator.py` | v1 discriminator — standard PatchGAN |
| `model_v2/discriminator.py` | v2 discriminator — spectral-norm multi-scale PatchGAN |
| `model_v3/noise_scheduler.py` | v3 scheduler � DDPM scheduler + DDIM sampler |
| `model_v3/vae_wrapper.py` | v3 VAE wrapper � SD VAE encode/decode for latents |

### Losses

| File | Purpose |
|---|---|
| `model_v1/losses.py` | v1 composite loss — LSGAN, cycle-consistency, identity, VGG19 perceptual, gradient penalty |
| `model_v2/losses.py` | v2 composite loss (`UVCGANLoss`) — LSGAN + one-sided GP (γ=100, λ=0.1), cycle, identity, multi-level VGG19 perceptual, optional NT-Xent contrastive, optional spectral frequency loss |
| `model_v3/losses.py` | v3 diffusion loss helpers (noise prediction + perceptual terms) |

### Data & Utilities

| File | Purpose |
|---|---|
| `shared/data_loader.py` | Unpaired dataset loader and augmentation transforms |
| `shared/EarlyStopping.py` | Early stopping on SSIM improvement and loss-divergence detection |
| `shared/replay_buffer.py` | Fixed-size replay buffer mixing old and new fake samples to stabilise discriminator training |
| `shared/metrics.py` | SSIM, PSNR, and FID metrics via InceptionV3 |
| `shared/validation.py` | Per-interval validation — runs generators, computes metrics, saves comparison images |
| `shared/testing.py` | End-of-training test inference and comparison image export |
| `shared/history_utils.py` | Training history visualisation and CSV persistence |
| `model_v3/data_loader.py` | v3 paired data loader for aligned A/B training pairs |
| `model_v3/history_utils.py` | v3 training history CSV/plots (no discriminator terms) |

---

## Data Layout

Place your dataset under `data\E_Staining_DermaRepo\H_E-Staining_dataset`:

```
data/
  E_Staining_DermaRepo/
    H_E-Staining_dataset/
      Un_Stained/          ← original whole-slide unstained images
      C_Stained/           ← original whole-slide stained images
      trainA/              ← created by preprocess_data.py
      trainB/
      testA/
      testB/
      models_v2_YYYY_MM_DD_HH_MM_SS/   ← created at training time
```

---

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

GPU is optional but strongly recommended. For CUDA acceleration install a CUDA-compatible PyTorch build — the project was developed on PyTorch 2.x with CUDA 12.x.

---

## Preprocess the Dataset

Extracts 256×256 patches, applies tissue/background filtering, and creates the CycleGAN-style `trainA/trainB/testA/testB` folders:

```bash
python preprocess_data.py
```

Filtering defaults (configurable inside `preprocess_data.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `tissue_threshold` | 0.1 | Minimum tissue fraction to keep a patch |
| `background_keep_ratio` | 0.2 | Fraction of background patches to keep |
| `white_thresh` | 220 | RGB threshold for near-white background |
| `sat_thresh` | 0.05 | Saturation threshold for low-colour background |

---

## Train the Model
```bash
python trainModel.py
```


You will be prompted for:

| Prompt | Description |
|---|---|
| `Epoch Size` | Number of samples drawn per epoch |
| `Number of Epochs` | Total training epochs |
| `Test Size` | Number of test samples to export at the end |
| `Model Version` | `1` for v1 Hybrid (UVCGAN + CycleGAN), `2` for true UVCGAN v2, `3` for DiT diffusion v3 |

Model v2 automatically uses `get_8gb_config()` and v3 uses `get_dit_8gb_config()` for 8 GB VRAM (see **Configuration** below).

### Training artifacts

All outputs are written to a timestamped directory:

```
data\E_Staining_DermaRepo\H_E-Staining_dataset\
  models_v2_YYYY_MM_DD_HH_MM_SS\
    checkpoint_epoch_N.pth         ← saved every 20 epochs
    final_checkpoint_epoch_N.pth   ← saved at end of training
    training_history.csv
    training_history.png
    validation_images\             ← comparison images every N epochs
    test_images\
    tensorboard_logs\
```

---



| Version | Generator | Training | Notes |
|---|---|---|---|
| v1 | U-Net + ViT (ReZero) | CycleGAN + LSGAN | `model_v1/training_loop.py` |
| v2 | U-Net + ViT (LayerScale, cross-domain) | UVCGAN v2 + one-sided GP | `model_v2/training_loop.py` |
| v3 | DiT (full Transformer, adaLN-Zero) | Latent diffusion (DDPM/DDIM) | `model_v3/training_loop.py` |

### v1 — Hybrid UVCGAN + CycleGAN (`model_v1/training_loop.py`)

A hybrid model that uses the UVCGAN generator architecture inside a standard CycleGAN training framework.

**Generator (`model_v1/generator.py` → `ViTUNetGenerator`)**

U-Net backbone with 4 encoder levels (64 → 128 → 256 → 512 channels) and a PixelwiseViT bottleneck:

```
Encoder:  3 → 64 (ConvBlock) → 128 → 256 → 512 → 512
Bottleneck: ConvBlock → PixelwiseViT (ReZero Transformer blocks)
Decoder:  512 → 512 → 256 → 128 → 64 → 3 (skip connections at each level)
```

The PixelwiseViT flattens spatial positions into tokens `(N, H×W, C)`, adds 2D sine-cosine positional embeddings, passes them through `vit_depth` **ReZero Transformer blocks**, then reshapes back to `(N, C, H, W)`. ReZero scales each residual branch by a learnable scalar initialised to 0, allowing the network to start as identity and learn residuals gradually.

Weight initialisation: Normal `N(0, 0.02)` for convolutions, Xavier uniform for linear layers, `N(1, 0.02)` for InstanceNorm scale parameters.

**Discriminator (`model_v1/discriminator.py` → `PatchDiscriminator`)**

Standard single-scale PatchGAN with 5 convolutional layers:

```
Conv(stride=2) → Conv(stride=2, IN) → Conv(stride=2, IN) → Conv(stride=1, IN) → Conv → output map
```

Each output element covers a 70×70 receptive field of the 256×256 input. One discriminator per domain (D_A for unstained, D_B for stained).

**Loss (`model_v1/losses.py` → `CycleGANLoss`)**

| Term | Formula | Weight |
|---|---|---|
| GAN (LSGAN) | MSE against real=1, fake=0 targets | 1.0 |
| Cycle-consistency | L1(G_BA(G_AB(A)), A) + L1(G_AB(G_BA(B)), B) | λ=10.0 |
| Identity | L1(G_BA(A), A) + L1(G_AB(B), B) | λ=5.0 (decays after epoch 50%) |
| Perceptual cycle | L1 on VGG19 relu1\_2/relu2\_2/relu3\_4 features | λ=0.2 |
| Perceptual identity | Same VGG19 features on identity outputs | λ=0.1 |
| Gradient penalty | Two-sided: E[(‖∇D(x̂)‖₂ - 1)²] | λ=10.0 |

Label smoothing: real targets use 0.97 instead of 1.0. Discriminator fakes are drawn from a replay buffer (size 50) to reduce oscillation.

**Training (`model_v1/training_loop.py`)**
- Adam `lr=2e-4`, betas `(0.5, 0.999)` for all optimisers
- Linear LR decay from epoch 100 to 200 (constant before epoch 100)
- Mixed precision (AMP) when CUDA is available
- Early stopping: checks every 10 epochs after a warmup of 80 epochs, patience of 40 epochs on SSIM, divergence detection if all losses exceed 5× their best value simultaneously
- Validation images and SSIM/PSNR/FID metrics every 10 epochs after warmup

---

### v2 — True UVCGAN (`model_v2/training_loop.py`)
| `model_v3/training_loop.py` | v3 training loop � DiT diffusion with EMA, DDPM/DDIM sampling, and validation/testing |

Paper-aligned implementation of Prokopenko et al., *UVCGAN v2*, 2023.

**Generator (`model_v2/generator.py` → `ViTUNetGeneratorV2`)**
| `model_v3/generator.py` | v3 generator � DiT backbone with conditional encoder for latent diffusion |

Revised U-Net + ViT architecture with several structural improvements over v1:

```
Encoder:  3 → 64 (7×7 reflect-pad conv) → 128 (DownBlock + 2×ResidualConvBlock)
            → 256 (DownBlock + 2×ResidualConvBlock) → 512 (DownBlock + 2×ResidualConvBlock) → 512 (DownBlock)
Bottleneck: ResidualConvBlock → PixelwiseViTV2 (LayerScale Transformer blocks)
Decoder:  512 → 512 → 256 → 128 → 64 → 3 (1×1 conv skip merges at each level)
```

Key differences from v1:

| Feature | v1 | v2 |
|---|---|---|
| Residual blocks | Simple ConvBlock | ResidualConvBlock (skip connection) |
| Transformer residual scaling | ReZero (scalar per block, init=0) | LayerScale (vector per channel, init=1e-4) |
| Skip connection merge | Concatenate → ConvBlock | Concatenate → 1×1 conv + IN + ReLU |
| Cross-domain fusion | None | CrossDomainFusion on all 4 skip levels |
| Weight init | Normal N(0, 0.02) for convs | Kaiming normal for convs, Xavier uniform for linear |
| Gradient checkpointing | Not supported | Supported on ViT blocks |

**Cross-domain skip fusion** is the defining UVCGAN feature. Both generators run simultaneously, and each fuses its own skip features with the paired generator's skip features at matching spatial levels via a lightweight 1×1 convolution. This allows G_AB and G_BA to share structural information without coupling their parameters:

```
fake_B = G_AB.forward_with_cross_domain(real_A, skips_from_G_BA(real_B))
fake_A = G_BA.forward_with_cross_domain(real_B, skips_from_G_AB(real_A))
```

**Discriminator (`model_v2/discriminator.py` → `MultiScaleDiscriminator`)**
| `model_v3/noise_scheduler.py` | v3 scheduler � DDPM scheduler + DDIM sampler |
| `model_v3/vae_wrapper.py` | v3 VAE wrapper � SD VAE encode/decode for latents |

Wraps N independent `SpectralNormDiscriminator` instances, each operating on a progressively downsampled version of the input (2× average-pool between scales):

- **Scale 0** — original 256×256 input → fine texture discrimination
- **Scale 1** — 128×128 downsampled input → mid-level structure
- **Scale 2** — 64×64 downsampled input → global colour/layout

**Spectral normalisation** divides each conv weight matrix by its largest singular value, bounding the discriminator's Lipschitz constant and preventing it from becoming too strong relative to the generator.

**Loss (`model_v2/losses.py` → `UVCGANLoss`)**
| `model_v3/losses.py` | v3 diffusion loss helpers (noise prediction + perceptual terms) |

| Term | Formula | Weight |
|---|---|---|
| GAN (LSGAN) | MSE against real=0.9 (label smoothing), fake=0 | 1.0 |
| One-sided GP | E[max(0, ‖∇D(x̂)‖₂ - γ)²] / γ²,  γ=100 | λ=0.1 |
| Cycle-consistency | L1(G_BA(G_AB(A)), A) + L1(G_AB(G_BA(B)), B) | λ=10.0 |
| Identity | L1(G_BA(A), A) + L1(G_AB(B), B) | λ=5.0 (decays after epoch 50%) |
| Perceptual cycle | L1 on VGG19 relu1\_2/relu2\_2/relu3\_4/relu4\_4 | λ=0.1 |
| Perceptual identity | Same 4-level VGG19 features on identity outputs | λ=0.05 |
| Contrastive (NT-Xent) | Domain alignment on bottleneck features | λ=0.0 (disabled by default) |
| Spectral frequency | L1 on log-magnitude FFT of fake vs real | λ=0.0 (disabled by default) |

The one-sided GP is softer than WGAN-GP — it only penalises gradients that exceed γ=100, leaving D free to have small-norm gradients near real data. This is appropriate because the GAN objective is LSGAN (not Wasserstein). The GP is always computed in float32 regardless of AMP state to avoid NaN gradients.

**Training (`model_v2/training_loop.py`)**
| `model_v3/training_loop.py` | v3 training loop � DiT diffusion with EMA, DDPM/DDIM sampling, and validation/testing |
- Adam `lr=2e-4`, betas `(0.5, 0.999)`, `n_critic=1`
- Warm-up LR ramp for `warmup_epochs` epochs, then constant, then linear decay from `decay_start_epoch`
- Gradient clipping (max norm 1.0)
- Gradient accumulation over `accumulate_grads` batches (effective batch = `batch_size × accumulate_grads`)
- Replay buffer (size 50) for discriminator stabilisation
- Mixed precision (AMP) for generator step; GP always in float32
- Validation after `validation_warmup_epochs` epochs
- Early stopping activates after `early_stopping_warmup` epochs, monitors SSIM improvement and loss divergence

---

### v3 — DiT Diffusion (`model_v3/training_loop.py`)

v3 training assumes paired A/B patches with matching filenames in `trainA/trainB` and `testA/testB` (see `model_v3/data_loader.py`).

Conditional latent diffusion with a full Transformer backbone:

- VAE latent space (4 channels, 32x32 for 256x256 inputs)
- Conditioning via external `ConditionEncoder` and adaLN-Zero
- Noise prediction objective (MSE), optional VGG perceptual term
- DDIM sampling for inference (default 50 steps)

Note: the VAE checkpoint downloads ~335 MB on first run.

---

## Configuration

All hyperparameters live in `config.py` as typed dataclasses. There are two main presets:

### `get_default_config(model_version=2)` — 12+ GB VRAM

Full paper-aligned settings: `batch_size=4`, `vit_depth=4`, `num_scales=3`, no gradient checkpointing.



| Change from default | VRAM saving | Quality impact |
|---|---|---|
| `use_gradient_checkpointing=False` | 0 GB (faster training) | None |
| `batch_size=2`, `accumulate_grads=2` | ~1.5 GB | None (effective batch stays 4) |
| `num_scales=3` | 0 GB (full multi-scale) | Best stability/quality |
| `vit_depth=4` | 0 GB (full depth) | Best generator capacity |
| `perceptual_resize=180` | Slight increase vs 128 | Better perceptual supervision |

This repository's 8 GB profile prioritizes a balanced speed/quality setup and only
applies the most impactful memory reduction (batch size + accumulation). If needed,
you can still enable gradient checkpointing manually in `config.py`.

To customise further, edit the config before passing it to `train_v2`:

```python
from config import get_8gb_config

cfg = get_8gb_config()
cfg.training.num_epochs = 500
cfg.loss.lambda_cycle = 15.0
cfg.training.validation_warmup_epochs = 10  # validate after 10 epochs

history, G_AB, G_BA, D_A, D_B = train_v2(
    epoch_size=500,
    num_epochs=500,
    model_dir="my_run",
    cfg=cfg,
)
```

### Key hyperparameters

| Config class | Parameter | Default (v2) | Description |
|---|---|---|---|
| `GeneratorConfig` | `vit_depth` | 4 | ViT Transformer blocks in bottleneck |
| `GeneratorConfig` | `use_cross_domain` | True | Cross-domain skip fusion |
| `GeneratorConfig` | `use_gradient_checkpointing` | False | Recompute ViT activations during backward |
| `DiscriminatorConfig` | `num_scales` | 3 | Multi-scale discriminator levels |
| `LossConfig` | `lambda_cycle` | 10.0 | Cycle-consistency weight |
| `LossConfig` | `lambda_identity` | 5.0 | Identity loss weight |
| `LossConfig` | `lambda_gp` | 0.1 | One-sided gradient penalty weight |
| `LossConfig` | `lambda_contrastive` | 0.0 | NT-Xent contrastive weight (0 = off) |
| `LossConfig` | `lambda_spectral` | 0.0 | Spectral frequency loss weight (0 = off) |
| `TrainingConfig` | `accumulate_grads` | 1 | Gradient accumulation steps |
| `TrainingConfig` | `validation_warmup_epochs` | 10 | Validate every N epochs |
| `TrainingConfig` | `early_stopping_warmup` | 80 | Epoch before early stopping can trigger |

---

## Monitor with TensorBoard

```bash
tensorboard --logdir data\E_Staining_DermaRepo\H_E-Staining_dataset\models_v2_YYYY_MM_DD_HH_MM_SS\tensorboard_logs
```

Logged scalars include: `Loss/Generator`, `Loss/Discriminator_A`, `Loss/Discriminator_B`, `LR/Generator`, `Diagnostics/GradNorm_G`, `Validation/ssim_A`, `Validation/ssim_B`, `Validation/psnr_A`, `Validation/psnr_B`, `EarlyStopping/ssim`, `EarlyStopping/counter`.

---

## Inference (Stain / Unstain)

`app.py` loads a checkpoint and translates whole-slide images by splitting them into 256×256 patches, running inference on each, and reconstructing the output with blended overlapping windows:

```bash
python app.py
```

You will be prompted for paths to an unstained and a stained image. Update the checkpoint path inside `app.py` to point at your trained model:

```
data\E_Staining_DermaRepo\H_E-Staining_dataset\models_v2_YYYY_MM_DD_HH_MM_SS\final_checkpoint_epoch_XXX.pth
```

Outputs:
- `data\reconstructed_stained_output.png`
- `data\reconstructed_unstained_output.png`

> **Note:** Checkpoints from v1 are not compatible with the v2 generator architecture and vice versa.

---

## Metrics

Validation runs every `validation_warmup_epochs` epochs and reports:

| Metric | Description |
|---|---|
| SSIM | Structural similarity (higher = better) |
| PSNR | Peak signal-to-noise ratio in dB (higher = better) |
| FID | Fréchet Inception Distance on a small subset (lower = better) |

All metrics are logged to TensorBoard and printed to the console. Early stopping monitors SSIM improvement and triggers if all losses diverge simultaneously.

---

## Notes


- Patch size is fixed at 256×256. If you change it, update `preprocess_data.py` and `app.py` together.
- To enable the optional contrastive or spectral losses once training is stable, set `cfg.loss.lambda_contrastive = 0.1` and/or `cfg.loss.lambda_spectral = 0.05`.
- The gradient penalty (`lambda_gp`) and its target (`LSGANGradientPenalty.GAMMA = 100`) are paper-aligned. Do not change `GAMMA` without re-tuning `lambda_gp`.

---

## References

- Prokopenko et al., *UVCGAN v2: An Improved Cycle-Consistent GAN for Unpaired Image-to-Image Translation*, 2023
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
- Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017




