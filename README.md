# Histology Stain/Unstain Translation with Transformers

This project implements four model variants for unpaired image-to-image translation between unstained and H&E-stained histology tissue images:

- **v1** — Hybrid UVCGAN + CycleGAN baseline
- **v2** — True UVCGAN v2 (Prokopenko et al., 2023) with cross-domain skip fusion and LayerScale ViT bottleneck
- **v3** — CycleDiT latent diffusion with local+global attention, multi-scale conditioning, and a three-branch discriminator
- **v4** — CUT baseline with PatchNCE contrastive loss and a Transformer encoder + SE-gated ResNet generator

All variants share a common preprocessing pipeline, data loader, metrics, and early stopping infrastructure.

## Dataset

This project uses the **E-Staining DermaRepo H&E staining dataset** with **unstained** (`Un_Stained`) and **stained** (`C_Stained`) image domains. All rights for the dataset are held by its original owners and licensors.

---

## Project Layout

### Entry Points

| File | Purpose |
|---|---|
| `trainModel.py` | Training entry point — prompts for epoch size, epochs, test size, and model version (1–4) |
| `app.py` | Inference script — translates whole-slide images via patch-based staining/unstaining |
| `preprocess_data.py` | Patch extraction, tissue/background filtering, and train/test split |
| `unzip.py` | Helper to extract the dataset ZIP archive into the expected directory layout |

### Configuration

| File | Purpose |
|---|---|
| `config.py` | Centralised configuration via typed dataclasses. `UVCGANConfig` for v1/v2/v3 (`get_default_config`, `get_8gb_config`, `get_dit_config`, `get_dit_8gb_config`). `V4Config` for v4 (`get_v4_config`, `get_v4_8gb_config`) |

### Training Loops

| File | Purpose |
|---|---|
| `model_v1/training_loop.py` | v1 — CycleGAN-style LSGAN, single PatchGAN discriminator, AMP |
| `model_v2/training_loop.py` | v2 — LSGAN + one-sided GP, multi-scale spectral-norm discriminators, warm-up + linear LR decay, gradient accumulation |
| `model_v3/training_loop.py` | v3 — DiT diffusion + adversarial training, cycle + identity losses, dual ProjectionDiscriminators, EMA, R1 penalty, ReplayBuffer |
| `model_v4/training_loop.py` | v4 — LSGAN + PatchNCE contrastive loss + identity loss, EMA generators, linear warmup + decay LR, replay buffer |

### Models

| File | Purpose |
|---|---|
| `model_v1/generator.py` | v1 — U-Net + ViT bottleneck with ReZero Transformer blocks |
| `model_v2/generator.py` | v2 — U-Net + ViT with LayerScale, cross-domain skip fusion, Kaiming/Xavier init, gradient checkpointing |
| `model_v3/generator.py` | v3.2 — CycleDiTGenerator: overlapping PatchEmbed stem, multi-scale ConditionTokenizer, DiTBlocks with local+global SA and 12-chunk adaLN-Zero, alternating cross-attention |
| `model_v4/generator.py` | v4.2 — ResnetGenerator (SE-gated blocks + bottleneck self-attention) or TransformerGeneratorV4 (pre-norm + DW-Conv blocks + TextureRefinementHead) |
| `model_v1/discriminator.py` | v1 — standard PatchGAN |
| `model_v2/discriminator.py` | v2 — spectral-norm multi-scale PatchGAN |
| `model_v3/discriminator.py` | v3.2 — ProjectionDiscriminator: LocalPatchBranch+MinibatchStdDev, GlobalBranch+self-attention, color-aware FFT branch, learnable branch weights |
| `model_v4/discriminator.py` | v4.2 — PatchGANDiscriminator with spectral norm, auxiliary multi-scale head, and MinibatchStdDev |
| `model_v3/noise_scheduler.py` | v3 — DDPMScheduler (cosine/linear beta) + DDIMSampler (deterministic, CFG support) |
| `model_v3/vae_wrapper.py` | v3 — frozen SD VAE (`stabilityai/sd-vae-ft-mse`) encode/decode for latents |

### Losses

| File | Purpose |
|---|---|
| `model_v1/losses.py` | v1 — LSGAN, cycle, identity, VGG19 perceptual, two-sided gradient penalty |
| `model_v2/losses.py` | v2 — LSGAN + one-sided GP (γ=100, λ=0.1), cycle, identity, multi-level VGG19 perceptual, optional NT-Xent contrastive, optional spectral frequency |
| `model_v3/losses.py` | v3 — denoising MSE (v-mode + eps-mode), Min-SNR weighting, LSGAN adversarial, R1 penalty, latent cycle, latent identity |
| `model_v4/nce_loss.py` | v4 — PatchNCELoss: InfoNCE with per-layer lazy MLP projection heads |

### Data & Utilities

| File | Purpose |
|---|---|
| `shared/data_loader.py` | Unpaired dataset loader and augmentation transforms |
| `shared/EarlyStopping.py` | Early stopping on SSIM improvement and loss-divergence detection |
| `shared/replay_buffer.py` | Fixed-size replay buffer mixing old and new fake samples |
| `shared/metrics.py` | SSIM, PSNR, and FID metrics via InceptionV3 |
| `shared/validation.py` | Per-interval validation — runs generators, computes metrics, saves comparison images |
| `shared/testing.py` | End-of-training test inference and comparison image export |
| `shared/history_utils.py` | Training history visualisation and CSV persistence |
| `model_v3/history_utils.py` | v3 training history CSV/plots — denoising, adversarial, cycle, identity, discriminator, gradient-norm terms |
| `model_v4/patch_sampler.py` | Spatial patch sampling at shared locations for PatchNCE |
| `model_v4/transformer_blocks.py` | Shared ViT utilities: `PatchEmbed`, `TransformerBlock`, 2-D sincos positional embeddings |

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
      models_v<i>_YYYY_MM_DD_HH_MM_SS/   ← created at training time (i = 1,2,3,4 based on the model)
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

GPU is optional but strongly recommended. The project was developed on PyTorch 2.x with CUDA 12.x.

---

## Preprocess the Dataset

Extracts 256×256 patches, applies tissue/background filtering, and creates the `trainA/trainB/testA/testB` folders:

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
| `Model Version` | `1` v1 Hybrid, `2` true UVCGAN v2, `3` DiT diffusion v3, `4` CUT + Transformer v4 |

Config presets used automatically:

| Version | Config preset |
|---|---|
| v1 | `get_default_config(model_version=1)` |
| v2 | `get_8gb_config()` |
| v3 | `get_dit_8gb_config()` |
| v4 | `get_v4_8gb_config()` |

### Training artifacts

All versions write to a timestamped directory under the dataset root:

```
data\E_Staining_DermaRepo\H_E-Staining_dataset\
  models_YYYY_MM_DD_HH_MM_SS\          ← v1
  models_v2_YYYY_MM_DD_HH_MM_SS\       ← v2
  models_v3_YYYY_MM_DD_HH_MM_SS\       ← v3
  models_v4_YYYY_MM_DD_HH_MM_SS\       ← v4
```

Contents of each run directory:

```
  checkpoint_epoch_N.pth         ← saved every 20 epochs
  final_checkpoint.pth           ← saved at end of training
  training_history.csv           ← All Models
  training_history.png           ← All Models
  validation_images\
  test_images\
  tensorboard_logs\
```

---

## Model Variants

| Version | Generator | Discriminator | Training objective |
|---|---|---|---|
| v1 | U-Net + ViT (ReZero) | Single PatchGAN | CycleGAN + LSGAN |
| v2 | U-Net + ViT (LayerScale, cross-domain) | Multi-scale spectral-norm PatchGAN | UVCGAN v2 + one-sided GP |
| v3 | CycleDiT (local+global SA, multi-scale cond) | ProjectionDiscriminator (local+global+FFT) | Latent diffusion (DDPM/DDIM) + adversarial |
| v4 | ResNet (SE blocks) or Transformer + CNN decoder | PatchGAN (SN + multi-scale + MBStdDev) | LSGAN + PatchNCE + identity |

### v1 — Hybrid UVCGAN + CycleGAN

**Generator (`model_v1/generator.py` → `ViTUNetGenerator`)**

U-Net with 4 encoder levels (64→128→256→512) and a PixelwiseViT bottleneck using ReZero Transformer blocks. ReZero scales each residual by a learnable scalar initialised to 0.

**Discriminator (`model_v1/discriminator.py` → `PatchDiscriminator`)**

Standard 70×70 PatchGAN. One discriminator per domain.

**Loss (`model_v1/losses.py` → `CycleGANLoss`)**

| Term | Weight |
|---|---|
| LSGAN | 1.0 |
| Cycle-consistency | λ=10.0 |
| Identity | λ=5.0 (decays after 50% of training) |
| Perceptual cycle (VGG19 relu1_2/2_2/3_4) | λ=0.2 |
| Perceptual identity | λ=0.1 |
| Two-sided gradient penalty | λ=10.0 |

**Training** — AdamW for generator (`weight_decay=0.01`), Adam for discriminators; `lr=2e-4`, betas `(0.5, 0.999)`, linear LR decay from epoch 100, AMP, early stopping after warmup of 80 epochs.

---

### v2 — True UVCGAN

Paper-aligned implementation of Prokopenko et al., *UVCGAN v2*, 2023.

**Generator (`model_v2/generator.py` → `ViTUNetGeneratorV2`)**

```
Encoder:  3 → 64 (7×7 reflect-pad) → 128 → 256 → 512 → 512
Bottleneck: ResidualConvBlock → PixelwiseViTV2 (LayerScale Transformer blocks)
Decoder:  512 → 512 → 256 → 128 → 64 → 3 (1×1 conv skip merges)
```

Cross-domain skip fusion runs both generators simultaneously and fuses skip features at each level via a 1×1 conv.

**Discriminator (`model_v2/discriminator.py` → `MultiScaleDiscriminator`)**

3 independent `SpectralNormDiscriminator` instances on 256×256, 128×128, and 64×64 inputs.

**Loss (`model_v2/losses.py` → `UVCGANLoss`)**

| Term | Weight |
|---|---|
| LSGAN (real=0.9 label smoothing) | 1.0 |
| One-sided GP (γ=100) | λ=0.1 |
| Cycle-consistency | λ=10.0 |
| Identity | λ=5.0 (decays after 50%) |
| Perceptual cycle (VGG19 4-level) | λ=0.1 |
| Perceptual identity | λ=0.05 |
| NT-Xent contrastive | λ=0.0 (off by default) |
| Spectral frequency | λ=0.0 (off by default) |

**Training** — AdamW for generator (`weight_decay=0.01`), Adam for discriminators; `lr=2e-4`, warm-up + constant + linear decay LR, gradient clipping, gradient accumulation, AMP (GP always in float32).

---

### v3 — CycleDiT Diffusion (v3.2)

Conditional latent diffusion with a full Transformer backbone and CycleGAN-style consistency losses.

**Generator (`model_v3/generator.py` → `CycleDiTGenerator`)**

- `PatchEmbed` — overlapping two-conv stem (DeiT-III style) for smoother patch boundary gradients
- `ConditionTokenizer` — multi-scale fusion of full-resolution and 2× downsampled conditioning tokens
- `DiTBlock` — global SA + local window SA (gated) + optional cross-attention + GELU MLP with intermediate LayerNorm; 12-chunk adaLN-Zero
- `DiTGenerator` — alternating cross-attention: odd blocks receive full condition tokens, even blocks receive pooled summary
- `DomainEmbedding` — learned 2-entry embedding (0=unstained, 1=stained)

```
z_t:(N,4,32,32) → PatchEmbed → tokens:(N,L,Hd)
                → depth × DiTBlock(cond_tokens, timestep+domain)
                → head → unpatchify → v_pred:(N,4,32,32)
```

**Discriminator (`model_v3/discriminator.py` → `ProjectionDiscriminator`)**

| Branch | Enhancement | Purpose |
|---|---|---|
| `LocalPatchBranchWithMBStd` | MinibatchStdDev before final conv | Texture + diversity penalty |
| `GlobalDiscriminatorBranch` | Self-attention on 4×4 feature map | Color balance, tissue layout |
| `FFTDiscriminatorBranch` | Grayscale + R/G/B channel FFT (4-channel) | Periodic artifacts, channel imbalance |

Learnable `branch_logweights` (softmax-normalised) allow training to up-weight the most informative branch.

**Noise Scheduler (`model_v3/noise_scheduler.py`)**

- `DDPMScheduler` — cosine or linear beta schedule; provides `add_noise`, `predict_x0`, `get_v_target`, `predict_x0_from_v`, `predict_eps_from_v`
- `DDIMSampler` — deterministic reverse sampler; `eta=0` fully deterministic, `eta=1` recovers DDPM; supports CFG via `cfg_scale`

**VAE (`model_v3/vae_wrapper.py`)**

Frozen `stabilityai/sd-vae-ft-mse` (~335 MB, downloaded on first run). Encodes 256×256 → `(N,4,32,32)` latents scaled by 0.18215.

**Loss (`model_v3/losses.py`)**

| Stage | Term | Weight |
|---|---|---|
| 1 — Denoising | MSE v-pred + eps-pred, Min-SNR weighting | `lambda_denoising` |
| 1 — Denoising | VGG19 perceptual on decoded x0 | `lambda_perceptual_v3` |
| 2 — Adversarial | LSGAN generator | `lambda_adv_v3` (warm-up ramp) |
| 2 — Cycle | Latent L1 via short DDIM | `lambda_cycle_v3` |
| 2 — Identity | Latent L1 at t=0 | `lambda_identity_v3` (linear decay) |
| D | LSGAN + R1 penalty | 1.0 / `r1_gamma` |

**Training** — AdamW `lr=1e-4` for generator, Adam `lr=2e-4` for discriminators; cosine LR decay; EMA decay=0.9999; adaptive D update; AMP (R1 always float32).

---

### v4 — CUT + Transformer (v4.2)

GAN + PatchNCE contrastive learning with an optional Transformer encoder.

**Generator (`model_v4/generator.py`)**

Two variants selectable via `use_transformer_encoder`:

- `ResnetGenerator` — SE-gated residual blocks + bottleneck `SpatialSelfAttention` + nearest-upsample decoder
- `TransformerGeneratorV4` — `EnhancedTransformerBlock` (pre-norm + DW-Conv1d local branch) + `TextureRefinementHead` + CNN up-decoder

Both expose `encode_features(x, nce_layers)` for PatchNCE feature extraction.

**Discriminator (`model_v4/discriminator.py` → `PatchGANDiscriminator`)**

Spectral-norm PatchGAN with:
- Auxiliary score head tapped at `n_layers//2` for multi-scale feedback
- `MinibatchStdDev` before the final conv
- `forward()` returns averaged (main + aux) score map; `forward_multiscale()` returns both separately

**Loss**

| Term | Weight |
|---|---|
| LSGAN adversarial | `lambda_gan` (default 5.0) |
| PatchNCE (InfoNCE per layer) | `lambda_nce` (default 2.0) |
| Identity L1 | `lambda_identity` (default 5.0) |

**PatchNCE (`model_v4/nce_loss.py` → `PatchNCELoss`)**

Per-layer lazy MLP projection heads (2-layer, ReLU). Queries and keys are L2-normalised; InfoNCE cross-entropy with diagonal positives. Projectors are keyed by true layer index for checkpoint stability.

**Training** — AdamW for generator when Transformer encoder is enabled (`weight_decay=0.01`), Adam otherwise; Adam for discriminators; `lr=2e-4`, linear warmup + linear decay LR; EMA generators (decay=0.999); replay buffers; gradient clipping; AMP; early stopping on SSIM.

---

## Configuration

All hyperparameters live in `config.py` as typed dataclasses.

### v2 — key hyperparameters

| Config class | Parameter | Default | Description |
|---|---|---|---|
| `GeneratorConfig` | `vit_depth` | 4 | ViT Transformer blocks |
| `GeneratorConfig` | `use_cross_domain` | True | Cross-domain skip fusion |
| `GeneratorConfig` | `use_gradient_checkpointing` | False | ~30–40% VRAM saving |
| `DiscriminatorConfig` | `num_scales` | 3 | Multi-scale discriminator levels |
| `LossConfig` | `lambda_cycle` | 10.0 | Cycle-consistency weight |
| `LossConfig` | `lambda_gp` | 0.1 | One-sided GP weight |
| `TrainingConfig` | `accumulate_grads` | 1 | Gradient accumulation steps |
| `TrainingConfig` | `validation_warmup_epochs` | 10 | Validate every N epochs |

### v3 — key hyperparameters

| Config class | Parameter | Default | Description |
|---|---|---|---|
| `DiffusionConfig` | `dit_hidden_dim` | 512 | Token embedding dimension |
| `DiffusionConfig` | `dit_depth` | 8 | DiT Transformer blocks |
| `DiffusionConfig` | `use_cross_attention` | True | Cross-attention in DiTBlocks |
| `DiffusionConfig` | `lambda_cycle_v3` | 10.0 | Latent cycle weight |
| `DiffusionConfig` | `r1_gamma` | 10.0 | R1 penalty coefficient |
| `DiffusionConfig` | `disc_use_fft` | True | FFT discriminator branch |

### v4 — key hyperparameters

| Config class | Parameter | Default | Description |
|---|---|---|---|
| `V4ModelConfig` | `use_transformer_encoder` | True | Transformer vs ResNet generator |
| `V4ModelConfig` | `encoder_depth` | 6 | Transformer blocks |
| `V4ModelConfig` | `encoder_dim` | 384 | Token embedding dimension |
| `V4TrainingConfig` | `lambda_nce` | 2.0 | PatchNCE weight |
| `V4TrainingConfig` | `nce_layers` | (0,1,2,3,4,5) | Encoder layers for NCE |
| `V4TrainingConfig` | `use_ema` | True | EMA generator copies |

### Customising a config

```python
from config import get_8gb_config

cfg = get_8gb_config()
cfg.training.num_epochs = 500
cfg.loss.lambda_cycle = 15.0

history, G_AB, G_BA, D_A, D_B = train_v2(
    epoch_size=500,
    num_epochs=500,
    model_dir="my_run",
    cfg=cfg,
)
```

```python
from config import get_v4_8gb_config
from model_v4.training_loop import train_v4

cfg = get_v4_8gb_config()
cfg.training.lambda_nce = 3.0

history, G_AB, G_BA, D_A, D_B = train_v4(cfg=cfg)
```

---

## Monitor with TensorBoard

```bash
tensorboard --logdir data\E_Staining_DermaRepo\H_E-Staining_dataset\models_v2_YYYY_MM_DD_HH_MM_SS\tensorboard_logs
```

v1/v2 logged scalars: `Loss/Generator`, `Loss/Discriminator_A`, `Loss/Discriminator_B`, `LR/Generator`, `Diagnostics/GradNorm_G`, `Validation/ssim_A`, `Validation/ssim_B`, `Validation/psnr_A`, `Validation/psnr_B`, `EarlyStopping/ssim`, `EarlyStopping/counter`.

v3 logged scalars: `Loss/DiT`, `Loss/Perceptual`, `Loss/G_adv`, `Loss/D_A`, `Loss/D_B`, `Loss/Cycle`, `Loss/Identity`, `LR/DiT`, `LR/D_A`, `LR/D_B`, `Diagnostics/GradNorm`, `Diagnostics/TimestepMean`, `Diagnostics/TimestepStd`, `Weights/lambda_adv_current`, `Weights/lambda_identity_current`, `Validation/ssim_A`, `Validation/ssim_B`, `Validation/psnr_A`, `Validation/psnr_B`, `EarlyStopping/ssim`, `EarlyStopping/counter`, `EarlyStopping/divergence_counter`.

v4 logged scalars: `Loss/Generator`, `Loss/GAN`, `Loss/NCE`, `Loss/Identity`, `Loss/Generator_AB`, `Loss/Generator_BA`, `Loss/Discriminator_A`, `Loss/Discriminator_B`, `LR/Generator`, `LR/Discriminator_A`, `LR/Discriminator_B`, `Diagnostics/GradNorm_G`, `Validation/ssim_A`, `Validation/ssim_B`, `Validation/psnr_A`, `Validation/psnr_B`, `EarlyStopping/ssim`, `EarlyStopping/counter`, `EarlyStopping/divergence_counter`.

---

## Inference (Stain / Unstain)

`app.py` loads a checkpoint and translates whole-slide images by splitting them into 256×256 patches, running inference on each, and reconstructing the output with blended overlapping windows:

```bash
python app.py
```

You will be prompted for:

| Prompt | v1/v2 | v3 | v4 |
|---|---|---|---|
| Model path | ✓ | ✓ | ✓ |
| Model version | ✓ | ✓ | ✓ |
| Unstained image path | ✓ | ✓ | ✓ |
| Stained image path | ✓ | — | ✓ |

Model-version behaviour:
- v1/v2/v4 — bidirectional translation using `G_AB` (unstained→stained) and `G_BA` (stained→unstained).
- v3 — A→B only (unstained→stained) via DDIM sampling with domain conditioning; architecture is auto-detected from the checkpoint.
- v4 — architecture hyperparameters are loaded from the `"config"` key stored in the checkpoint; falls back to shape-inference for legacy checkpoints.

Patches are extracted with 50% overlap (`stride = patch_size // 2`) and blended with a 2-D Hann window for seamless reconstruction.

Outputs:
- `data\reconstructed_stained_output.png`
- `data\reconstructed_unstained_output.png`

> **Note:** Checkpoints are not cross-compatible between model versions.

---

## Metrics

Validation runs every N epochs and reports:

| Metric | Description |
|---|---|
| SSIM | Structural similarity (higher = better) |
| PSNR | Peak signal-to-noise ratio in dB (higher = better) |
| FID | Fréchet Inception Distance on a small subset (lower = better) |

All metrics are logged to TensorBoard and printed to the console. Early stopping monitors SSIM improvement and triggers if losses diverge simultaneously.

---

## Notes

- Patch size is fixed at 256×256. If you change it, update `preprocess_data.py` and `app.py` together.
- v3: the VAE checkpoint downloads ~335 MB on first run.
- v2: to enable optional losses once training is stable, set `cfg.loss.lambda_contrastive = 0.1` and/or `cfg.loss.lambda_spectral = 0.05`.
- v3: the gradient penalty (`lambda_gp`) and its target (`GAMMA=100`) are paper-aligned — do not change `GAMMA` without re-tuning `lambda_gp`.
- v4: `nce_layers` indices are automatically clamped to the number of encoder blocks; invalid indices are silently dropped.

---

## References

- Prokopenko et al., *UVCGAN v2: An Improved Cycle-Consistent GAN for Unpaired Image-to-Image Translation*, 2023
- Park et al., *Contrastive Unpaired Translation*, ECCV 2020
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, ICCV 2017
- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023
- Gulrajani et al., *Improved Training of Wasserstein GANs*, NeurIPS 2017
