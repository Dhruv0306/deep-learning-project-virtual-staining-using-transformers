"""
Centralized configuration manager for UVCGAN training.

Provides dataclasses for generator, discriminator, loss, training, and data
hyperparameters.  Supports switching between the original v1 (CycleGAN/
UVCGAN-style) and the new v2 (true UVCGAN) architecture via ``model_version``.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneratorConfig:
    """
    Hyperparameters for the UVCGAN generator.

    Attributes:
        input_nc: Number of input channels.
        output_nc: Number of output channels.
        base_channels: Feature channels at the first encoder level.
        vit_depth: Number of Transformer blocks in the ViT bottleneck.
        vit_heads: Multi-head attention head count.
        vit_mlp_ratio: MLP expansion ratio inside each Transformer block.
        vit_dropout: Dropout probability applied inside Transformer blocks.
        use_layerscale: Enable LayerScale residual scaling (v2 only).
        layerscale_init: Initial value for LayerScale parameters.
        use_cross_domain: Enable cross-domain feature sharing (v2 only).
    """

    input_nc: int = 3
    output_nc: int = 3
    base_channels: int = 64
    vit_depth: int = 4
    vit_heads: int = 8
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.0
    # v2-only knobs
    use_layerscale: bool = True
    layerscale_init: float = 1e-4
    use_cross_domain: bool = True


@dataclass
class DiscriminatorConfig:
    """
    Hyperparameters for the spectral-norm multi-scale discriminator.

    Attributes:
        input_nc: Number of input channels.
        base_channels: Feature channels at the first layer.
        n_layers: Number of strided convolution layers.
        num_scales: Number of discriminator scales (multi-scale approach).
        use_spectral_norm: Apply spectral normalisation to every conv layer.
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

    Attributes:
        lambda_cycle: Weight for cycle-consistency loss.
        lambda_identity: Weight for identity loss.
        lambda_cycle_perceptual: Weight for perceptual cycle loss.
        lambda_identity_perceptual: Weight for perceptual identity loss.
        lambda_gp: Weight for WGAN-GP gradient penalty.
        lambda_contrastive: Weight for contrastive domain-alignment loss (v2).
        lambda_spectral: Weight for frequency-domain spectral loss (v2).
        perceptual_resize: Spatial size to which images are resized before VGG.
        use_wgan_gp: Use WGAN-GP instead of LSGAN.
        contrastive_temperature: Temperature for NT-Xent contrastive loss.
    """

    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_cycle_perceptual: float = 0.1
    lambda_identity_perceptual: float = 0.05
    lambda_gp: float = 10.0
    lambda_contrastive: float = 0.1
    lambda_spectral: float = 0.05
    perceptual_resize: int = 128
    use_wgan_gp: bool = True
    contrastive_temperature: float = 0.07


@dataclass
class TrainingConfig:
    """
    Training loop hyperparameters.

    Attributes:
        num_epochs: Total training epochs.
        epoch_size: Maximum samples drawn per epoch.
        test_size: Number of test samples for final evaluation.
        lr: Base learning rate for all optimisers.
        beta1: Adam β₁ coefficient.
        beta2: Adam β₂ coefficient.
        warmup_epochs: Linear LR warm-up period.
        decay_start_epoch: Epoch at which linear LR decay begins.
        grad_clip_norm: Max-norm for gradient clipping (0 to disable).
        early_stopping_patience: Patience in *check intervals* before stopping.
        early_stopping_warmup: Epoch before early-stopping activates.
        early_stopping_interval: How often (epochs) to evaluate for early stopping.
        divergence_threshold: Loss ratio that triggers divergence detection.
        divergence_patience: Check intervals of divergence before stopping.
        use_amp: Use automatic mixed precision (True only when CUDA is available).
        replay_buffer_size: Size of replay buffers for discriminator training.
        n_critic: Discriminator update steps per generator update (WGAN style).
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


@dataclass
class DataConfig:
    """
    Data loading configuration.

    Attributes:
        data_root: Root directory that contains paired A/B image folders.
        image_size: Spatial size (H = W) to which images are resized.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker count.
        augment: Apply random flips and colour jitter during training.
    """

    data_root: str = "data/E_Staining_DermaRepo/H_E-Staining_dataset"
    image_size: int = 256
    batch_size: int = 2
    num_workers: int = 4
    augment: bool = True


@dataclass
class UVCGANConfig:
    """
    Top-level configuration container.

    Attributes:
        model_version: ``1`` for the original CycleGAN/UVCGAN-style pipeline;
            ``2`` for the new true-UVCGAN pipeline.
        generator: Generator hyperparameters.
        discriminator: Discriminator hyperparameters.
        loss: Loss hyperparameters.
        training: Training-loop hyperparameters.
        data: Data-loading hyperparameters.
        model_dir: Directory for checkpoints, logs, and validation outputs.
        val_dir: Sub-directory for per-epoch validation images.
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
        """Validate cross-field constraints."""
        if self.model_version not in (1, 2):
            raise ValueError(
                f"model_version must be 1 or 2, got {self.model_version!r}."
            )
        if self.training.decay_start_epoch >= self.training.num_epochs - 1:
            raise ValueError(
                "decay_start_epoch must be at least 2 epochs before num_epochs "
                "to allow a meaningful decay period."
            )


def get_default_config(model_version: int = 2) -> UVCGANConfig:
    """
    Return a default :class:`UVCGANConfig` for the requested model version.

    Args:
        model_version: ``1`` (original) or ``2`` (true UVCGAN).

    Returns:
        UVCGANConfig: Fully populated configuration object.
    """
    cfg = UVCGANConfig(model_version=model_version)
    if model_version == 1:
        # v1 uses LSGAN, no contrastive/spectral losses.
        cfg.loss.use_wgan_gp = False
        cfg.loss.lambda_contrastive = 0.0
        cfg.loss.lambda_spectral = 0.0
        cfg.generator.use_layerscale = False
        cfg.generator.use_cross_domain = False
        cfg.discriminator.use_spectral_norm = False
        cfg.discriminator.num_scales = 1
        cfg.training.n_critic = 1
    return cfg
