"""
Centralized configuration manager for UVCGAN training.

Provides dataclasses for generator, discriminator, loss, training, and data
hyperparameters.  Supports switching between the original v1 (CycleGAN/
UVCGAN-style) and the new v2 (true UVCGAN) architecture via model_version.

v2 defaults are paper-aligned (Prokopenko et al., UVCGAN v2, 2023):
  - GAN objective  : LSGAN, NOT Wasserstein
  - Gradient penalty: one-sided, gamma=100, lambda_gp=0.1
  - n_critic        : 1 (LSGAN does not need multi-step D updates)
  - Adam betas      : (0.5, 0.999), lr=2e-4  (standard LSGAN/CycleGAN values)
  - lambda_contrastive / lambda_spectral: 0.0 initially; enable once stable

For 8 GB VRAM use get_8gb_config() instead of get_default_config().
"""

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
    # Set True for 8 GB GPUs.
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

    data_root: str = "data/E_Staining_DermaRepo/H_E-Staining_dataset"
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    augment: bool = True


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
            ``2`` for the true UVCGAN v2 model.
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
    model_dir: Optional[str] = None
    val_dir: Optional[str] = None

    def __post_init__(self):
        """
        Validate inter-field constraints after dataclass initialisation.

        Raises:
            ValueError: If ``model_version`` is not 1 or 2, or if
                ``decay_start_epoch`` is too close to ``num_epochs`` to
                allow a meaningful linear decay phase (fewer than 2 epochs
                of decay would remain).
        """
        if self.model_version not in (1, 2):
            raise ValueError(
                f"model_version must be 1 or 2, got {self.model_version!r}."
            )
        if self.training.decay_start_epoch >= self.training.num_epochs - 1:
            raise ValueError(
                "decay_start_epoch must be at least 2 epochs before num_epochs."
            )


def get_default_config(model_version: int = 2) -> UVCGANConfig:
    """
    Return a default UVCGANConfig for the requested model version.
    Assumes sufficient VRAM (12+ GB).

    Args:
        model_version: 1 (original CycleGAN/UVCGAN) or 2 (true UVCGAN v2).
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
    Return a VRAM-optimised UVCGANConfig for 8 GB GPUs.

    Changes from the default config and their estimated VRAM savings:

    ┌─────────────────────────────────────┬──────────┬───────────────────────────┐
    │ Change                              │ Saving   │ Quality impact            │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ use_gradient_checkpointing = True   │ ~2.5 GB  │ None (same gradients,     │
    │                                     │          │ ~20% slower backward)     │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ batch_size 4 → 2                    │ ~1.5 GB  │ Compensated by            │
    │ accumulate_grads 1 → 2              │          │ accumulate_grads=2        │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ num_scales 3 → 2                    │ ~0.4 GB  │ Minor: loses coarsest     │
    │                                     │          │ discriminator scale       │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ vit_depth 4 → 2                     │ ~0.3 GB  │ Small: fewer ViT blocks   │
    │                                     │          │ in bottleneck             │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ perceptual_resize 128 → 64          │ ~0.2 GB  │ Very minor: only affects  │
    │                                     │          │ perceptual loss terms     │
    ├─────────────────────────────────────┼──────────┼───────────────────────────┤
    │ use_amp = True  (already default)   │ ~2.0 GB  │ None                      │
    └─────────────────────────────────────┴──────────┴───────────────────────────┘

    Total estimated saving: ~5–7 GB, bringing peak VRAM from ~11 GB to ~5–6 GB.
    use_amp is already True so that saving is already baked into the 11 GB figure.

    NOTE: If you still OOM after applying this config, set vit_depth=1 as a last
    resort. The ViT bottleneck is still present and active; it just has 1 block
    instead of 2.
    """
    cfg = UVCGANConfig(model_version=2)

    # --- Biggest win: gradient checkpointing ---
    # Recomputes activations during backward instead of storing them.
    # Saves ~2-3 GB with ~20% slower training. Zero quality impact.
    cfg.generator.use_gradient_checkpointing = True

    # --- ViT depth: 4 → 2 ---
    # Each ViT block stores its full activation tensor (N, H*W/256, 512) for backprop.
    # Halving the depth halves that storage. Quality impact is small for histology
    # images since the bottleneck is still present and spatially informed.
    cfg.generator.vit_depth = 2

    # --- Discriminator scales: 3 → 2 ---
    # The third (coarsest) scale operates on 64×64 images and adds the least
    # discriminative signal for 256×256 patches. Dropping it saves one full
    # D forward/backward pass per step.
    cfg.discriminator.num_scales = 2

    # --- Batch size: 4 → 2, with gradient accumulation to compensate ---
    # batch_size=2 halves activation memory. accumulate_grads=2 means the
    # optimiser steps every 2 batches, so the effective gradient batch is
    # still 4. Loss scaling is handled in training_loop_v2.
    cfg.data.batch_size = 2
    cfg.training.accumulate_grads = 2

    # --- Perceptual loss resize: 128 → 64 ---
    # VGG19 processes images at this resolution. Halving it saves ~200 MB and
    # the quality difference on the perceptual terms is negligible.
    cfg.loss.perceptual_resize = 64

    # --- Everything else stays paper-aligned ---
    cfg.generator.use_cross_domain = True
    cfg.generator.use_layerscale = True
    cfg.loss.use_wgan_gp = False
    cfg.loss.lambda_gp = 0.1
    cfg.training.use_amp = True  # critical - do not disable
    cfg.training.n_critic = 1

    return cfg
