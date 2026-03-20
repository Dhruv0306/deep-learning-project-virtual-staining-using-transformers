# True UVCGAN v2 for Histology Stain/Unstain Translation

This project implements a **true UVCGAN v2** pipeline (Prokopenko et al., 2023) for unpaired image-to-image translation between unstained and H&E-stained histology tissue images. It includes:

- Dataset preprocessing to create 256×256 patches with tissue/background filtering
- Two model variants: a v1 hybrid (UVCGAN + CycleGAN) and a v2 true UVCGAN
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
| `trainModel.py` | Training entry point — prompts for epoch size, epochs, test size, and model version |
| `app.py` | Inference script — translates whole-slide images via patch-based staining/unstaining |
| `preprocess_data.py` | Patch extraction, tissue/background filtering, and train/test split |
| `unzip.py` | Helper to extract the dataset ZIP archive into the expected directory layout |

### Configuration

| File | Purpose |
|---|---|
| `config.py` | Centralised configuration via `UVCGANConfig` dataclasses. Provides `get_default_config(model_version=1\|2)` for standard hardware and `get_8gb_config()` for 8 GB GPUs |

### Training Loops

| File | Purpose |
|---|---|
| `training_loop.py` | v1 training loop — CycleGAN-style with LSGAN, single PatchGAN discriminator, gradient clipping, AMP |
| `training_loop_v2.py` | v2 training loop — paper-aligned LSGAN + one-sided GP, multi-scale spectral-norm discriminators, warm-up + linear LR decay, gradient accumulation, and per-interval validation |

### Models

| File | Purpose |
|---|---|
| `generator.py` | v1 generator — U-Net + ViT bottleneck with ReZero Transformer blocks |
| `uvcgan_v2_generator.py` | v2 generator — U-Net + ViT with LayerScale, cross-domain skip fusion, Kaiming/Xavier weight init, and optional gradient checkpointing |
| `discriminator.py` | v1 discriminator — standard PatchGAN |
| `spectral_norm_discriminator.py` | v2 discriminator — spectral-norm multi-scale PatchGAN |

### Losses

| File | Purpose |
|---|---|
| `losses.py` | v1 composite loss — LSGAN, cycle-consistency, identity, VGG19 perceptual, gradient penalty |
| `advanced_losses.py` | v2 composite loss (`UVCGANLoss`) — LSGAN + one-sided GP (γ=100, λ=0.1), cycle, identity, multi-level VGG19 perceptual, optional NT-Xent contrastive, optional spectral frequency loss |

### Data & Utilities

| File | Purpose |
|---|---|
| `data_loader.py` | Unpaired dataset loader and augmentation transforms |
| `EarlyStopping.py` | Early stopping on SSIM improvement and loss-divergence detection |
| `replay_buffer.py` | Fixed-size replay buffer mixing old and new fake samples to stabilise discriminator training |
| `metrics.py` | SSIM, PSNR, and FID metrics via InceptionV3 |
| `validation.py` | Per-interval validation — runs generators, computes metrics, saves comparison images |
| `testing.py` | End-of-training test inference and comparison image export |
| `history_utils.py` | Training history visualisation and CSV persistence |

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
| `background_keep_ratio` | 0.4 | Fraction of background patches to keep |
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
| `Model Version` | `1` for v1 Hybrid (UVCGAN + CycleGAN), `2` for true UVCGAN v2 |

Model v2 automatically uses `get_8gb_config()` which is optimised for 8 GB VRAM (see **Configuration** below).

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

## Model Versions

### v1 — Hybrid UVCGAN + CycleGAN (`training_loop.py`)

- Generator: U-Net + ViT bottleneck with ReZero Transformer blocks
- Discriminator: single-scale PatchGAN
- Loss: LSGAN + cycle-consistency + identity + VGG19 perceptual + gradient penalty
- Standard CycleGAN Adam betas `(0.5, 0.999)`, `lr=2e-4`

### v2 — True UVCGAN (`training_loop_v2.py`)

Paper-aligned implementation of Prokopenko et al., *UVCGAN v2*, 2023:

**Generator (`uvcgan_v2_generator.py`)**
- U-Net backbone with 4 encoder levels (64 → 128 → 256 → 512 channels)
- ViT bottleneck with **LayerScale** Transformer blocks (stabilises early training)
- **Cross-domain skip fusion** — each generator receives skip features from the other generator at matching spatial levels, enabling structural knowledge sharing
- Optional **gradient checkpointing** on ViT blocks (saves ~2 GB VRAM)
- Kaiming normal init for convolutions, Xavier uniform for linear layers

**Discriminator (`spectral_norm_discriminator.py`)**
- Multi-scale PatchGAN: independent discriminators at 3 spatial scales
- **Spectral normalisation** on every conv layer for Lipschitz stability

**Loss (`advanced_losses.py` → `UVCGANLoss`)**
- **LSGAN** adversarial loss (paper Table 2 best configuration)
- **One-sided gradient penalty** (GP): `E[max(0, ‖∇D(x̂)‖₂ - γ)²] / γ²` with γ=100, λ=0.1. Only penalises gradients exceeding γ — softer than standard WGAN-GP and correct for LSGAN
- Cycle-consistency (L1)
- Identity (L1, with decay after 50% of training)
- Multi-level **VGG19 perceptual** loss on cycle and identity outputs
- Optional **NT-Xent contrastive** domain-alignment loss (disabled by default)
- Optional **spectral frequency** loss (disabled by default)

**Training (`training_loop_v2.py`)**
- Warm-up + linear decay LR schedule
- Gradient clipping
- Gradient accumulation (effective batch size independent of physical batch size)
- Replay buffer for discriminator stabilisation
- Mixed precision (AMP) — GP always computed in float32 regardless
- Per-interval validation with SSIM/PSNR/FID; early stopping on SSIM + divergence

---

## Configuration

All hyperparameters live in `config.py` as typed dataclasses. There are two main presets:

### `get_default_config(model_version=2)` — 12+ GB VRAM

Full paper-aligned settings: `batch_size=4`, `vit_depth=4`, `num_scales=3`, no gradient checkpointing.

### `get_8gb_config()` — 8 GB VRAM *(used by default in `trainModel.py`)*

| Change from default | VRAM saving | Quality impact |
|---|---|---|
| `use_gradient_checkpointing=True` (ViT blocks only) | ~2.5 GB | None (~20% slower backward) |
| `batch_size=2`, `accumulate_grads=2` | ~1.5 GB | None (effective batch stays 4) |
| `num_scales=2` | ~0.4 GB | Minor (loses coarsest D scale) |
| `vit_depth=2` | ~0.3 GB | Small (fewer ViT blocks) |
| `perceptual_resize=64` | ~0.2 GB | Very minor (perceptual terms only) |

To customise further, edit the config before passing it to `train_v2`:

```python
from config import get_8gb_config

cfg = get_8gb_config()
cfg.training.num_epochs = 500
cfg.loss.lambda_cycle = 15.0
cfg.training.validation_warmup_epochs = 5  # validate every 5 epochs

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

- Paths are Windows-style (`\`). On Linux/macOS replace `\` with `/` throughout `data_loader.py`, `trainModel.py`, and `app.py`.
- Patch size is fixed at 256×256. If you change it, update `preprocess_data.py` and `app.py` together.
- To enable the optional contrastive or spectral losses once training is stable, set `cfg.loss.lambda_contrastive = 0.1` and/or `cfg.loss.lambda_spectral = 0.05`.
- The gradient penalty (`lambda_gp`) and its target (`LSGANGradientPenalty.GAMMA = 100`) are paper-aligned. Do not change `GAMMA` without re-tuning `lambda_gp`.

---

## References

- Prokopenko et al., *UVCGAN v2: An Improved Cycle-Consistent GAN for Unpaired Image-to-Image Translation*, 2023
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
- Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017