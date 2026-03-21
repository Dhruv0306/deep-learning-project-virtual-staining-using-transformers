# `config.py` — Configuration Manager

**Shared by:** Both v1 and v2  
**Role:** Centralised hyperparameter management. V2 training reads hyperparameters from `UVCGANConfig`; v1 still keeps several values hardcoded inside `training_loop.py` for legacy compatibility.

---

## Design

The config is structured as nested dataclasses — one per concern — all wrapped in a top-level `UVCGANConfig`. This makes it easy to override individual values without touching anything else:

```python
cfg = get_8gb_config()
cfg.training.num_epochs = 500        # just change what you need
cfg.loss.lambda_cycle = 15.0
```

---

## `GeneratorConfig`

Controls the architecture of `ViTUNetGeneratorV2` (v2 only — v1 does not use this config).

| Field | Type | Default | Description |
|---|---|---|---|
| `input_nc` | int | 3 | Input image channels |
| `output_nc` | int | 3 | Output image channels |
| `base_channels` | int | 64 | Channels at the first encoder level. Subsequent levels are ×2, ×4, ×8. Directly determines model size — doubling this roughly quadruples peak activation memory |
| `vit_depth` | int | 4 | Number of LayerScale Transformer blocks in the ViT bottleneck. Each block adds `(N, H×W, C)` activation storage during training. 8GB config reduces to 2 |
| `vit_heads` | int | 8 | Attention heads per Transformer block. Must evenly divide the bottleneck channel count (`base_channels × 8 = 512`) |
| `vit_mlp_ratio` | float | 4.0 | MLP hidden dimension = `bottleneck_channels × vit_mlp_ratio` |
| `vit_dropout` | float | 0.0 | Dropout inside each Transformer block. 0 is standard for GAN training |
| `use_layerscale` | bool | True | Enable LayerScale residual scaling in ViT blocks |
| `layerscale_init` | float | 1e-4 | Initial value for LayerScale per-channel vectors. Near-zero init means the block starts as identity |
| `use_cross_domain` | bool | True | Allocate CrossDomainFusion layers on skip connections. This is the defining UVCGAN feature |
| `use_gradient_checkpointing` | bool | False | Recompute ViT block activations during backward instead of storing them. Saves ~2.5 GB VRAM at the cost of ~20% slower backward |

---

## `DiscriminatorConfig`

Controls the architecture of `MultiScaleDiscriminator` (v2) or `PatchDiscriminator` (v1).

| Field | Type | Default | Description |
|---|---|---|---|
| `input_nc` | int | 3 | Input image channels |
| `base_channels` | int | 64 | Channels after the first conv layer |
| `n_layers` | int | 3 | Number of strided downsampling layers (excluding the stride-1 layer and the output layer) |
| `num_scales` | int | 3 | Number of spatial scales in the multi-scale discriminator. 8GB config reduces to 2 (drops the coarsest 64×64 scale) |
| `use_spectral_norm` | bool | True | Apply spectral normalisation to all conv layers |

---

## `LossConfig`

Controls all loss weights and the gradient penalty configuration.

| Field | Type | Default | Description |
|---|---|---|---|
| `lambda_cycle` | float | 10.0 | Cycle-consistency L1 weight. Standard CycleGAN value |
| `lambda_identity` | float | 5.0 | Identity L1 weight at the start of training. Decays after 50% of training |
| `lambda_cycle_perceptual` | float | 0.1 | VGG perceptual loss weight on cycle outputs |
| `lambda_identity_perceptual` | float | 0.05 | VGG perceptual loss weight on identity outputs |
| `lambda_gp` | float | 0.1 | One-sided gradient penalty weight. Paper value. Much smaller than WGAN-GP's typical 10.0 because the γ²-normalised GP has a different scale |
| `lambda_contrastive` | float | 0.0 | NT-Xent contrastive loss weight. Disabled by default — enable only after GAN training has stabilised |
| `lambda_spectral` | float | 0.0 | Spectral frequency loss weight. Disabled by default for the same reason |
| `perceptual_resize` | int | 128 | Bilinear resize applied to images before VGG19. 8GB config uses 64 |
| `use_wgan_gp` | bool | False | Legacy field kept for v1 compatibility. Always False for v2 — LSGAN is the paper's best configuration |
| `contrastive_temperature` | float | 0.07 | NT-Xent softmax temperature |

---

## `TrainingConfig`

Controls the training loop behaviour — optimiser, scheduling, and stopping.

| Field | Type | Default | Description |
|---|---|---|---|
| `num_epochs` | int | 200 | Total training epochs |
| `epoch_size` | int | 3000 | Maximum samples drawn per epoch. The dataloader wraps around if the dataset is smaller |
| `test_size` | int | 200 | Number of test samples to export in the final evaluation |
| `lr` | float | 2e-4 | Base learning rate for all Adam optimisers |
| `beta1` | float | 0.5 | Adam β₁. Standard CycleGAN/LSGAN value. Lower than the default 0.9 reduces momentum and stabilises GAN training |
| `beta2` | float | 0.999 | Adam β₂. Standard value |
| `warmup_epochs` | int | 5 | Number of epochs to linearly ramp the LR from 0 to `lr`. Prevents large gradient steps at the start of training |
| `decay_start_epoch` | int | 100 | Epoch at which the LR begins its linear decay from `lr` to 0. Constant between `warmup_epochs` and `decay_start_epoch` |
| `grad_clip_norm` | float | 1.0 | Maximum L2 norm for gradient clipping. 0 disables clipping |
| `early_stopping_patience` | int | 40 | Number of validation checks without SSIM improvement before stopping. Measured in check intervals, not epochs |
| `early_stopping_warmup` | int | 80 | Epoch before early stopping can trigger, even if validation runs earlier |
| `early_stopping_interval` | int | 10 | How often to compute validation metrics and run early stopping checks (in epochs) |
| `divergence_threshold` | float | 5.0 | If all losses simultaneously exceed 5× their best-ever value, the divergence counter increments |
| `divergence_patience` | int | 2 | Consecutive divergence checks before training stops |
| `use_amp` | bool | True | Use PyTorch Automatic Mixed Precision (float16 activations, float32 master weights). The gradient penalty always runs in float32 regardless |
| `replay_buffer_size` | int | 50 | Number of past generated images stored in each replay buffer |
| `n_critic` | int | 1 | Discriminator update steps per generator update. 1 is correct for LSGAN |
| `accumulate_grads` | int | 1 | Gradient accumulation steps. Effective batch = `batch_size × accumulate_grads`. 8GB config uses 2 with `batch_size=2` to match the effective batch of the default `batch_size=4` |
| `validation_warmup_epochs` | int | 10 | In v2, controls when validation image export starts. Validation images run every epoch after this threshold |

---

## `DataConfig`

Controls data loading.

| Field | Type | Default | Description |
|---|---|---|---|
| `data_root` | str | `data/E_Staining_DermaRepo/...` | Root directory containing `trainA`, `trainB`, `testA`, `testB` |
| `image_size` | int | 256 | Images are resized to this resolution. Changing this requires updating `preprocess_data.py` and `app.py` together |
| `batch_size` | int | 4 | Mini-batch size. 8GB config uses 2 |
| `num_workers` | int | 4 | DataLoader worker processes for parallel data loading |
| `augment` | bool | True | Apply random horizontal flips and colour jitter during training |

---

## `UVCGANConfig`

Top-level container. Holds one instance of each sub-config above plus output path overrides.

| Field | Type | Default | Description |
|---|---|---|---|
| `model_version` | int | 2 | `1` for v1 hybrid, `2` for true UVCGAN v2 |
| `generator` | `GeneratorConfig` | default | Generator architecture settings |
| `discriminator` | `DiscriminatorConfig` | default | Discriminator architecture settings |
| `loss` | `LossConfig` | default | Loss weights and settings |
| `training` | `TrainingConfig` | default | Training loop settings |
| `data` | `DataConfig` | default | Data loading settings |
| `model_dir` | `str` or `None` | None | Override checkpoint output directory |
| `val_dir` | `str` or `None` | None | Override validation images directory |

### `__post_init__()`

Validates cross-field constraints at construction time:
- `model_version` must be 1 or 2
- `decay_start_epoch` must be at least 2 epochs before `num_epochs` (otherwise the LR decay period is too short to be useful)

---

## LR Schedule

The learning rate follows a three-phase schedule defined by `warmup_epochs`, `decay_start_epoch`, and `num_epochs`:

```
Epoch                LR multiplier
─────────────────────────────────────────
0 → warmup_epochs    Linear ramp: 0 → 1
warmup → decay_start Constant:       1
decay_start → total  Linear decay: 1 → 0
```

With defaults (`warmup=5`, `decay_start=100`, `total=200`):
```
Epochs 0–5:    ramp from 0 to 2e-4
Epochs 5–100:  constant at 2e-4
Epochs 100–200: linear decay from 2e-4 to 0
```

---

## Factory Functions

### `get_default_config(model_version=2)`

Returns a standard config suitable for 12+ GB VRAM. For `model_version=1`, overrides v2-specific fields to their v1 equivalents (no cross-domain, no spectral norm, single discriminator scale, `batch_size=2`).

### `get_8gb_config()`

Returns a VRAM-optimised config for 8 GB GPUs. Changes from default:

| Change | VRAM saving | Quality impact |
|---|---|---|
| `use_gradient_checkpointing=True` | ~2.5 GB | None (~20% slower backward) |
| `batch_size=2`, `accumulate_grads=2` | ~1.5 GB | None (effective batch stays 4) |
| `num_scales=2` | ~0.4 GB | Minor (loses coarsest D scale) |
| `vit_depth=2` | ~0.3 GB | Small (fewer ViT blocks) |
| `perceptual_resize=64` | ~0.2 GB | Very minor |
