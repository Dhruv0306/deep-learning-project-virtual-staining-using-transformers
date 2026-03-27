"""
Centralized configuration manager for training and inference presets.

Provides dataclasses for generator, discriminator, loss, training, data, and
diffusion hyperparameters. The top-level ``model_version`` switch supports:
    - v1: Hybrid CycleGAN/UVCGAN baseline
    - v2: True UVCGAN v2
    - v3: DiT diffusion pipeline

v2 defaults are paper-aligned (Prokopenko et al., UVCGAN v2, 2023):
  - GAN objective  : LSGAN, NOT Wasserstein
  - Gradient penalty: one-sided, gamma=100, lambda_gp=0.1
  - n_critic        : 1 (LSGAN does not need multi-step D updates)
  - Adam betas      : (0.5, 0.999), lr=2e-4  (standard LSGAN/CycleGAN values)
  - lambda_contrastive / lambda_spectral: 0.0 initially; enable once stable

For 8 GB VRAM, prefer ``get_8gb_config()`` for v2 and
``get_dit_8gb_config()`` for v3.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneratorConfig:
    """
    Hyperparameters for the UVCGAN v2 generator.

    Memory-relevant attributes (8 GB tips in parentheses):
        base_channels : Feature channels at the first encoder level.
                        Quadruples peak activation memory. Keep at 64.
        vit_depth     : Number of Transformer blocks in the ViT bottleneck.
                        Each block stores activations for backprop.
                        Reduce from 4 → 2 to save ~15% generator VRAM.
        use_gradient_checkpointing: Trade compute for memory by recomputing
                        activations during backprop instead of storing them.
                        Saves ~30-40% generator activation memory with
                        ~20% slower backward pass. Best single option for 8 GB.
    """

    input_nc: int = 3
    output_nc: int = 3
    base_channels: int = 64
    vit_depth: int = 4
    vit_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    use_layerscale: bool = True
    layerscale_init: float = 1e-4
    use_cross_domain: bool = True
    # Memory optimisation: recompute activations during backward instead of
    # storing them.  Reduces generator activation VRAM by ~30-40%.
    # Optional for 8 GB GPUs when additional memory headroom is needed.
    use_gradient_checkpointing: bool = False


@dataclass
class DiscriminatorConfig:
    """
    Hyperparameters for the spectral-norm multi-scale discriminator.

    Memory-relevant attributes:
        num_scales: 3 scales = 3 independent PatchGAN forward passes per step.
                   Reduce from 3 → 2 to save ~15% discriminator VRAM with
                   minor quality impact (coarsest scale is least useful).
    """

    input_nc: int = 3
    base_channels: int = 64
    n_layers: int = 3
    num_scales: int = 3
    use_spectral_norm: bool = True


@dataclass
class LossConfig:
    """
    Loss function weights and settings.

    Memory-relevant attributes:
        perceptual_resize: VGG19 input resolution. Lower = less VRAM.
                          64 instead of 128 saves ~200 MB with minor quality
                          loss on the perceptual terms only.
    """

    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_cycle_perceptual: float = 0.1
    lambda_identity_perceptual: float = 0.05
    lambda_gp: float = 0.1
    lambda_contrastive: float = 0.0
    lambda_spectral: float = 0.0
    perceptual_resize: int = 128
    use_wgan_gp: bool = False
    contrastive_temperature: float = 0.07


@dataclass
class TrainingConfig:
    """
    Training loop hyperparameters.

    Memory-relevant attributes:
        use_amp          : Mixed precision (float16 activations). Already True
                          by default. Cuts activation memory roughly in half.
                          Do NOT disable this on 8 GB.
        accumulate_grads : Gradient accumulation steps. Simulates a larger
                          effective batch size without increasing peak VRAM.
                          accumulate_grads=2 with batch_size=2 is equivalent
                          to batch_size=4 in terms of gradient statistics,
                          but only 2 samples live in VRAM at once.
    """

    num_epochs: int = 200
    epoch_size: int = 3000
    test_size: int = 200
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    warmup_epochs: int = 5
    decay_start_epoch: int = 100
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 40
    early_stopping_warmup: int = 80
    early_stopping_interval: int = 10
    divergence_threshold: float = 5.0
    divergence_patience: int = 2
    use_amp: bool = True
    replay_buffer_size: int = 50
    n_critic: int = 1
    # Gradient accumulation: accumulate gradients over this many batches
    # before stepping the optimiser.  Effective batch = batch_size * this.
    # Set to 2 when batch_size=2 to keep effective batch = 4.
    accumulate_grads: int = 1
    validation_warmup_epochs: int = 10
    validation_size: int = 100
    validation_fid_samples: int = 200
    validation_fid_min_samples: int = 50


@dataclass
class DataConfig:
    """
    Data loading configuration.

    Fields:
        data_root (str): Root directory of the preprocessed CycleGAN dataset
            (must contain ``trainA/``, ``trainB/``, ``testA/``, ``testB/``).
        image_size (int): Spatial size to which every patch is resized
            during loading.  Must match the patch size used in
            ``preprocess_data.py`` (default 256).
        batch_size (int): Number of sample pairs per training mini-batch.
        num_workers (int): DataLoader worker processes.  Set to 0 to
            disable multiprocessing (useful for debugging on Windows).
        augment (bool): Reserved for future augmentation support.
            Currently unused.
    """

    data_root: str = os.path.join(
        "data", "E_Staining_DermaRepo", "H_E-Staining_dataset"
    )
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    augment: bool = True


@dataclass
class DiffusionConfig:
    """
    Hyperparameters for the v3 diffusion model.
    """

    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    dit_hidden_dim: int = 512
    dit_depth: int = 8
    dit_heads: int = 8
    dit_patch_size: int = 2
    dit_mlp_ratio: float = 4.0
    lambda_perceptual_v3: float = 0.0
    num_inference_steps: int = 50
    use_gradient_checkpointing: bool = False
    vae_model_id: str = "stabilityai/sd-vae-ft-mse"
    prediction_type: str = "v"
    cond_dropout_prob: float = 0.1
    cfg_scale: float = 2.0
    cond_patch_size: int = 16


@dataclass
class UVCGANConfig:
    """
    Top-level configuration container for a UVCGAN training run.

    All sub-configurations are grouped into typed dataclass fields so that
    every hyperparameter is discoverable and type-checked at construction
    time.  Pass an instance of this class to ``train_v2()`` to control the
    full training pipeline.

    Fields:
        model_version (int): ``1`` for the hybrid UVCGAN + CycleGAN model;
            ``2`` for the true UVCGAN v2 model; ``3`` for DiT diffusion.
        generator (GeneratorConfig): Generator architecture settings.
        discriminator (DiscriminatorConfig): Discriminator settings.
        loss (LossConfig): Loss function weights and options.
        training (TrainingConfig): Optimiser, scheduler, and loop settings.
        data (DataConfig): Dataset path, batch size, and worker settings.
        model_dir (str | None): Root directory for checkpoints, TensorBoard
            logs, and CSV history.  Auto-generated from a timestamp when
            ``None``.
        val_dir (str | None): Directory for per-epoch validation images.
            Defaults to ``model_dir/validation_images`` when ``None``.
    """

    model_version: int = 2
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model_dir: Optional[str] = None
    val_dir: Optional[str] = None

    def __post_init__(self):
        """
        Validate inter-field constraints after dataclass initialisation.

        Raises:
            ValueError: If ``model_version`` is not 1, 2, or 3, or if
                ``decay_start_epoch`` is too close to ``num_epochs`` to
                allow a meaningful linear decay phase (fewer than 2 epochs
                of decay would remain).
        """
        if self.model_version not in (1, 2, 3):
            raise ValueError(
                f"model_version must be 1, 2, or 3, got {self.model_version!r}."
            )
        if self.training.decay_start_epoch >= self.training.num_epochs - 1:
            raise ValueError(
                "decay_start_epoch must be at least 2 epochs before num_epochs."
            )


def get_default_config(model_version: int = 2) -> UVCGANConfig:
    """
    Return a default UVCGANConfig for the requested model version.
    Assumes sufficient VRAM (roughly 12+ GB).

    Args:
        model_version: ``1`` (hybrid baseline), ``2`` (true UVCGAN v2),
            or ``3`` (DiT diffusion).

    Notes:
        Explicit architecture overrides are applied only for ``model_version=1``
        to keep legacy behavior aligned with the v1 training loop.
    """
    cfg = UVCGANConfig(model_version=model_version)

    if model_version == 1:
        cfg.loss.use_wgan_gp = False
        cfg.loss.lambda_gp = 10.0
        cfg.loss.lambda_contrastive = 0.0
        cfg.loss.lambda_spectral = 0.0
        cfg.generator.use_layerscale = False
        cfg.generator.use_cross_domain = False
        cfg.generator.use_gradient_checkpointing = False
        cfg.discriminator.use_spectral_norm = False
        cfg.discriminator.num_scales = 1
        cfg.training.n_critic = 1
        cfg.training.lr = 2e-4
        cfg.training.beta1 = 0.5
        cfg.training.beta2 = 0.999
        cfg.data.batch_size = 2

    return cfg


def get_8gb_config() -> UVCGANConfig:
    """
    Return a user-tuned UVCGANConfig for 8 GB GPUs.

    NOTE: These values reflect current memory testing and are not the
    original VRAM-minimizing profile. Comments below describe the
    active settings.
    """
    cfg = UVCGANConfig(model_version=2)

    # --- Gradient checkpointing ---
    # Disabled based on measured memory headroom to keep training faster.
    cfg.generator.use_gradient_checkpointing = False

    # --- ViT depth ---
    # Kept at 4 to preserve capacity; not reduced in this profile.
    cfg.generator.vit_depth = 4

    # --- Discriminator scales ---
    # Kept at 3 (full multi-scale) for stability/quality.
    cfg.discriminator.num_scales = 3

    # --- Batch size: 4 -> 2, with gradient accumulation to compensate ---
    # batch_size=2 halves activation memory. accumulate_grads=2 means the
    # optimiser steps every 2 batches, so the effective gradient batch is
    # still 4. Loss scaling is handled in model_v2/training_loop.
    cfg.data.batch_size = 2
    cfg.training.accumulate_grads = 2

    # --- Perceptual loss resize ---
    # Set to 180 based on current memory/quality trade-offs.
    cfg.loss.perceptual_resize = 180

    # --- Everything else stays paper-aligned ---
    cfg.generator.use_cross_domain = True
    cfg.generator.use_layerscale = True
    cfg.loss.use_wgan_gp = False
    cfg.loss.lambda_gp = 0.1
    cfg.training.use_amp = True  # critical - do not disable
    cfg.training.n_critic = 1

    return cfg


def get_dit_config() -> UVCGANConfig:
    """
    Return a default config for v3 diffusion training.

    The returned profile keeps gradient checkpointing disabled and uses the
    standard v3 architecture settings defined in :class:`DiffusionConfig`.
    """
    cfg = UVCGANConfig(model_version=3)
    cfg.diffusion.dit_hidden_dim = 512
    cfg.diffusion.dit_depth = 8
    cfg.diffusion.dit_heads = 8
    cfg.diffusion.dit_patch_size = 2
    cfg.diffusion.dit_mlp_ratio = 4.0
    cfg.diffusion.use_gradient_checkpointing = True
    cfg.diffusion.prediction_type = "v"
    cfg.diffusion.cond_dropout_prob = 0.1
    cfg.diffusion.cfg_scale = 2.0
    cfg.diffusion.cond_patch_size = 16
    cfg.data.batch_size = 4
    cfg.training.accumulate_grads = 1
    cfg.training.validation_fid_samples = 200
    cfg.training.validation_fid_min_samples = 50
    return cfg


def get_dit_8gb_config() -> UVCGANConfig:
    """
    Return a VRAM-optimised config for v3 diffusion training.

    Relative to :func:`get_dit_config`, this profile enables gradient
    checkpointing for DiT blocks and uses a lighter validation setup.
    """
    cfg = UVCGANConfig(model_version=3)
    cfg.diffusion.dit_hidden_dim = 384
    cfg.diffusion.dit_depth = 6
    cfg.diffusion.dit_heads = 6
    cfg.diffusion.dit_patch_size = 2
    cfg.diffusion.dit_mlp_ratio = 4.0
    cfg.diffusion.use_gradient_checkpointing = True
    cfg.diffusion.prediction_type = "v"
    cfg.diffusion.cond_dropout_prob = 0.1
    cfg.diffusion.cfg_scale = 1.8
    cfg.diffusion.cond_patch_size = 16
    cfg.data.batch_size = 4
    cfg.training.accumulate_grads = 1
    cfg.loss.perceptual_resize = 256
    cfg.diffusion.lambda_perceptual_v3 = 0.0
    cfg.training.validation_size = 20
    cfg.training.validation_fid_samples = 200
    cfg.training.validation_fid_min_samples = 30

    return cfg

