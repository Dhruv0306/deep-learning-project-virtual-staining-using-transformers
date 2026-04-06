# config.py — Configuration Reference

Source of truth: `../config.py`

Centralised configuration for all four model families (v1/v2/v3/v4).

## Configuration Containers

### UVCGANConfig (v1 / v2 / v3)

Top-level container passed to `train_v1`, `train_v2`, and `train_v3`.

Fields:

| Field | Type | Description |
|---|---|---|
| `model_version` | int | `1`, `2`, or `3` |
| `generator` | `GeneratorConfig` | v2 generator architecture |
| `discriminator` | `DiscriminatorConfig` | multi-scale PatchGAN settings |
| `loss` | `LossConfig` | loss weights and options |
| `training` | `TrainingConfig` | optimiser, scheduler, loop |
| `data` | `DataConfig` | dataset path, batch, workers |
| `diffusion` | `DiffusionConfig` | DiT settings (v3 only) |
| `model_dir` | `str \| None` | checkpoint root (auto-timestamped when `None`) |
| `val_dir` | `str \| None` | validation image dir (defaults to `model_dir/validation_images`) |

`__post_init__` validation:

- `model_version` must be `1`, `2`, or `3`
- `training.decay_start_epoch` must be at least 2 epochs before `training.num_epochs`

### V4Config (v4 only)

Top-level container passed to `train_v4`.

Fields:

| Field | Type | Description |
|---|---|---|
| `model_version` | int | must be `4` |
| `model` | `V4ModelConfig` | generator + discriminator architecture |
| `training` | `V4TrainingConfig` | optimiser, NCE, loop, early stopping |
| `data` | `V4DataConfig` | image size, batch, workers |
| `model_dir` | `str \| None` | checkpoint root |
| `val_dir` | `str \| None` | validation image dir |

`__post_init__` enforces `model_version == 4`.

---

## Sub-Config Groups — v1 / v2 / v3

### GeneratorConfig

UVCGAN v2 generator architecture controls.

| Field | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input image channels |
| `output_nc` | 3 | Output image channels |
| `base_channels` | 64 | Feature channels at first encoder level |
| `vit_depth` | 4 | ViT Transformer blocks in bottleneck |
| `vit_heads` | 8 | Attention heads |
| `vit_mlp_ratio` | 4.0 | MLP hidden-dim multiplier |
| `vit_dropout` | 0.0 | Dropout in ViT blocks |
| `use_layerscale` | `True` | LayerScale residual scaling |
| `layerscale_init` | 1e-4 | LayerScale initial value |
| `use_cross_domain` | `True` | Cross-domain skip fusion |
| `use_gradient_checkpointing` | `False` | Recompute ViT activations during backward (~30–40% VRAM saving) |

### DiscriminatorConfig

Spectral-norm multi-scale PatchGAN settings.

| Field | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input channels |
| `base_channels` | 64 | Feature width |
| `n_layers` | 3 | Strided downsampling layers |
| `num_scales` | 3 | Independent PatchGAN scales |
| `use_spectral_norm` | `True` | Spectral norm on all convs |

### LossConfig

| Field | Default | Description |
|---|---|---|
| `lambda_cycle` | 10.0 | Cycle-consistency weight |
| `lambda_identity` | 5.0 | Identity loss weight |
| `lambda_cycle_perceptual` | 0.1 | VGG19 perceptual on cycle outputs |
| `lambda_identity_perceptual` | 0.05 | VGG19 perceptual on identity outputs |
| `lambda_gp` | 0.1 | One-sided gradient penalty weight |
| `lambda_contrastive` | 0.0 | NT-Xent contrastive weight (0 = off) |
| `lambda_spectral` | 0.0 | Spectral frequency loss weight (0 = off) |
| `perceptual_resize` | 128 | VGG19 input resolution (lower = less VRAM) |
| `use_wgan_gp` | `False` | Two-sided WGAN-GP instead of one-sided |
| `contrastive_temperature` | 0.07 | NT-Xent softmax temperature |

### TrainingConfig

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 200 | Total training epochs |
| `epoch_size` | 3000 | Samples drawn per epoch |
| `test_size` | 200 | Test samples exported at end |
| `lr` | 2e-4 | Adam learning rate |
| `beta1` / `beta2` | 0.5 / 0.999 | Adam betas |
| `warmup_epochs` | 5 | LR warm-up epochs |
| `decay_start_epoch` | 100 | Epoch at which linear LR decay begins |
| `grad_clip_norm` | 1.0 | Gradient clipping max norm |
| `early_stopping_patience` | 40 | Max epochs without SSIM improvement |
| `early_stopping_warmup` | 80 | Earliest epoch early stopping activates |
| `early_stopping_interval` | 10 | Check every N epochs |
| `divergence_threshold` | 5.0 | Loss explosion ratio vs best baseline |
| `divergence_patience` | 2 | Consecutive divergence checks before stop |
| `use_amp` | `True` | Mixed precision (halves activation VRAM) |
| `replay_buffer_size` | 50 | Discriminator fake-sample history size |
| `n_critic` | 1 | Discriminator steps per generator step |
| `accumulate_grads` | 1 | Gradient accumulation steps |
| `validation_warmup_epochs` | 10 | Validate every N epochs |
| `validation_size` | 100 | Validation samples |
| `validation_fid_samples` | 200 | FID sample count |
| `validation_fid_min_samples` | 50 | Minimum FID samples |

### DataConfig

| Field | Default | Description |
|---|---|---|
| `data_root` | `data/E_Staining_DermaRepo/H_E-Staining_dataset` | Dataset root |
| `image_size` | 256 | Patch resize target |
| `batch_size` | 4 | Mini-batch size |
| `num_workers` | 4 | DataLoader workers |
| `prefetch_factor` | 2 | Batches prefetched per worker |
| `augment` | `True` | Reserved for future augmentation |

### DiffusionConfig

v3 DiT diffusion hyperparameters.

| Field | Default | Description |
|---|---|---|
| `num_timesteps` | 1000 | DDPM training steps |
| `beta_schedule` | `"cosine"` | Noise schedule (`"cosine"` or `"linear"`) |
| `dit_hidden_dim` | 512 | Token embedding dimension |
| `dit_depth` | 8 | DiT Transformer blocks |
| `dit_heads` | 8 | Attention heads per block |
| `dit_patch_size` | 2 | Latent patch size |
| `dit_mlp_ratio` | 4.0 | MLP hidden-dim multiplier |
| `prediction_type` | `"v"` | `"v"` (v-prediction) or `"epsilon"` |
| `num_inference_steps` | 50 | DDIM sampling steps at inference |
| `cfg_scale` | 2.0 | Classifier-free guidance scale (1.0 = off) |
| `cond_dropout_prob` | 0.1 | Condition dropout for CFG training |
| `cond_patch_size` | 16 | Conditioning image patch size |
| `cond_token_pool_stride` | 1 | Avg-pool stride on condition tokens |
| `use_cross_attention` | `True` | Cross-attention in DiTBlocks |
| `use_gradient_checkpointing` | `False` | Recompute DiT activations during backward |
| `vae_model_id` | `"stabilityai/sd-vae-ft-mse"` | HuggingFace VAE model ID |
| `min_snr_gamma` | 5.0 | Min-SNR loss weighting gamma |
| `perceptual_every_n_steps` | 4 | Perceptual loss cadence |
| `perceptual_batch_fraction` | 0.5 | Fraction of batch used for perceptual |
| `lambda_denoising` | 1.0 | Denoising loss weight |
| `lambda_adv_v3` | 0.5 | Adversarial loss weight |
| `lambda_adv_warmup_steps` | 3000 | Steps to ramp adversarial weight |
| `lambda_cycle_v3` | 10.0 | Cycle-consistency weight |
| `lambda_identity_v3_start` | 5.0 | Identity loss weight at start |
| `lambda_identity_v3_end` | 0.0 | Identity loss weight at end |
| `identity_decay_end_ratio` | 0.3 | Fraction of training over which identity decays |
| `cycle_ddim_steps` | 4 | DDIM steps for cycle reconstruction |
| `cycle_ddim_eta` | 0.0 | DDIM eta (0 = deterministic) |
| `use_r1_penalty` | `True` | R1 gradient penalty on discriminator |
| `r1_gamma` | 10.0 | R1 penalty coefficient |
| `r1_interval` | 16 | Apply R1 every N discriminator steps |
| `adaptive_d_update` | `True` | Skip D step when `loss_D < threshold` |
| `adaptive_d_loss_threshold` | 0.1 | Threshold for adaptive D update |
| `grad_clip_norm_g` | 1.0 | Generator gradient clip max norm |
| `disc_use_local` | `True` | Enable local PatchGAN branch |
| `disc_use_global` | `True` | Enable global branch |
| `disc_use_fft` | `True` | Enable FFT branch |
| `disc_base_channels` | 64 | Local branch feature width |
| `disc_global_base_channels` | 64 | Global branch feature width |
| `disc_fft_base_channels` | 32 | FFT branch feature width |
| `disc_n_layers` | 3 | Strided layers in local branch |

---

## Sub-Config Groups — v4

### V4ModelConfig

v4.1 improved defaults.

| Field | Default | Description |
|---|---|---|
| `input_nc` / `output_nc` | 3 | Image channels |
| `base_channels` | 128 | CNN decoder base width |
| `num_res_blocks` | 15 | ResNet bottleneck depth |
| `use_transformer_encoder` | `True` | Use `TransformerGeneratorV4`; else `ResnetGenerator` |
| `image_size` | 256 | Square input size |
| `patch_size` | 8 | Transformer patch size |
| `encoder_dim` | 384 | Token embedding dimension |
| `encoder_depth` | 6 | Transformer blocks |
| `encoder_heads` | 8 | Attention heads |
| `encoder_mlp_ratio` | 4.0 | MLP expansion ratio |
| `encoder_dropout` | 0.0 | Dropout in Transformer blocks |
| `use_gradient_checkpointing` | `False` | Activation checkpointing |
| `disc_base_channels` | 128 | PatchGAN feature width |
| `disc_n_layers` | 4 | PatchGAN strided layers |

### V4TrainingConfig

| Field | Default | Description |
|---|---|---|
| `num_epochs` | 200 | Total epochs |
| `epoch_size` | 3000 | Samples per epoch |
| `lr` | 2e-4 | Adam learning rate |
| `beta1` / `beta2` | 0.5 / 0.999 | Adam betas |
| `grad_clip_norm` | 1.0 | Gradient clip max norm |
| `use_amp` | `True` | Mixed precision |
| `accumulate_grads` | 1 | Gradient accumulation steps |
| `save_every` | 20 | Checkpoint every N epochs |
| `validation_every` | 5 | Validate every N epochs |
| `lambda_gan` | 5.0 | LSGAN adversarial weight |
| `lambda_nce` | 2.0 | PatchNCE contrastive weight |
| `lambda_identity` | 5.0 | Identity loss weight |
| `nce_layers` | `(0,1,2,3,4,5)` | Encoder block indices for NCE |
| `nce_num_patches` | 256 | Spatial patches per feature map |
| `nce_temperature` | 0.07 | InfoNCE softmax temperature |
| `nce_proj_dim` | 256 | MLP projector output dimension |
| `use_replay_buffer` | `True` | Discriminator fake-sample history |
| `replay_buffer_size` | 50 | History buffer size |
| `use_ema` | `True` | EMA copy of generators |
| `ema_decay` | 0.999 | EMA decay factor |
| `use_lr_schedule` | `True` | Linear warmup + linear decay |
| `lr_warmup_epochs` | 5 | LR warm-up epochs |
| `lr_decay_start_epoch` | 100 | Epoch at which decay begins |
| `early_stopping_patience` | 40 | Max epochs without improvement |
| `early_stopping_warmup` | 80 | Earliest epoch early stopping activates |
| `early_stopping_interval` | 10 | Check every N epochs |
| `early_stopping_min_delta` | 1e-5 | Minimum SSIM gain to reset patience |
| `divergence_threshold` | 5.0 | Loss explosion ratio |
| `divergence_patience` | 2 | Consecutive divergence checks before stop |

### V4DataConfig

| Field | Default | Description |
|---|---|---|
| `image_size` | 256 | Patch resize target |
| `batch_size` | 4 | Mini-batch size |
| `num_workers` | 4 | DataLoader workers |
| `prefetch_factor` | 2 | Batches prefetched per worker |

---

## Factory Functions

| Function | Returns | Notes |
|---|---|---|
| `get_default_config(model_version=2)` | `UVCGANConfig` | Full-VRAM defaults; v1 overrides applied when `model_version=1` |
| `get_8gb_config()` | `UVCGANConfig` | v2 tuned for 8 GB: `batch_size=2`, `accumulate_grads=2`, gradient checkpointing on |
| `get_dit_config()` | `UVCGANConfig` | v3 baseline: `hidden_dim=256`, `depth=4`, `heads=4` |
| `get_dit_8gb_config()` | `UVCGANConfig` | v3 for 8 GB: `hidden_dim=512`, `depth=8`, `heads=8`, checkpointing on, FFT branch off |
| `get_v4_config()` | `V4Config` | v4 full-capacity defaults |
| `get_v4_8gb_config()` | `V4Config` | v4 for 8 GB: gradient checkpointing on |

### get_8gb_config() active settings

- `generator.use_gradient_checkpointing = True`
- `generator.vit_depth = 4` (reduce to 2 for more headroom)
- `discriminator.num_scales = 3` (reduce to 2 to save ~15%)
- `data.batch_size = 2`, `training.accumulate_grads = 2` (effective batch = 4)
- `loss.perceptual_resize = 180`
- `training.use_amp = True` (do not disable)

### get_dit_8gb_config() active settings

- `dit_hidden_dim=768`, `dit_depth=8`, `dit_heads=12`, `dit_patch_size=8`
- `use_gradient_checkpointing=True`, `use_cross_attention=False`
- `cond_patch_size=32`, `cond_token_pool_stride=4`
- `disc_use_fft=False` (memory saving), `disc_use_global=True`, `disc_use_local=True`
- `disc_base_channels=128`, `disc_global_base_channels=32`, `disc_n_layers=4`
- `data.batch_size=2`, `training.accumulate_grads=2`
