"""
Centralized configuration manager for training and inference presets.

Provides dataclasses for generator, discriminator, loss, training, data, and
diffusion hyperparameters. The top-level ``model_version`` switch supports:
    - v1: Hybrid CycleGAN/UVCGAN baseline
    - v2: True UVCGAN v2
    - v3: DiT diffusion pipeline
    - v4: CUT baseline (GAN + PatchNCE + Transformer)

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
    prefetch_factor: int = 2
    augment: bool = True


@dataclass
class DiffusionConfig:
    """
    Hyperparameters for the v3 DiT diffusion model.

    Key fields:
        num_timesteps:          DDPM diffusion steps during training.
        beta_schedule:          Noise schedule type (``"cosine"`` or ``"linear"``).
        dit_hidden_dim:         Token embedding dimension for the DiT backbone.
        dit_depth:              Number of DiT Transformer blocks.
        dit_heads:              Attention heads per DiT block.
        dit_patch_size:         Latent patch size fed into the DiT.
        prediction_type:        Noise prediction target — ``"v"`` (v-prediction)
                                or ``"epsilon"``.
        num_inference_steps:    DDIM sampling steps at inference time.
        cfg_scale:              Classifier-free guidance scale (1.0 = no guidance).
        min_snr_gamma:          Min-SNR loss weighting gamma (5.0 per paper).
        use_gradient_checkpointing: Recompute DiT activations during backward
                                to reduce peak VRAM.
        vae_model_id:           HuggingFace model ID for the SD VAE encoder/decoder.
        lambda_denoising:       Weight for the diffusion noise-prediction loss.
        lambda_adv_v3:          Adversarial loss weight (ramped up over
                                ``lambda_adv_warmup_steps`` steps).
        lambda_cycle_v3:        Cycle-consistency loss weight.
        lambda_identity_v3_start/end: Identity loss weight, linearly decayed
                                from start to end over the first
                                ``identity_decay_end_ratio`` of training.
        use_r1_penalty:         Enable R1 gradient penalty on the discriminator.
        r1_gamma:               R1 penalty coefficient.
        r1_interval:            Apply R1 every N discriminator steps.
        disc_use_local/global/fft: Toggle the three ProjectionDiscriminator branches.
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
    cond_token_pool_stride: int = 1
    use_cross_attention: bool = True
    min_snr_gamma: float = 5.0
    perceptual_every_n_steps: int = 4
    perceptual_batch_fraction: float = 0.5
    # --- CycleDiT v3 (phase-0/phase-1/phase-2) controls ---
    lambda_denoising: float = 1.0
    lambda_adv_v3: float = 0.5
    lambda_adv_warmup_steps: int = 3000
    lambda_cycle_v3: float = 10.0
    lambda_identity_v3_start: float = 5.0
    lambda_identity_v3_end: float = 0.0
    identity_decay_end_ratio: float = 0.3
    cycle_ddim_steps: int = 4
    cycle_ddim_eta: float = 0.0
    use_r1_penalty: bool = True
    r1_gamma: float = 10.0
    r1_interval: int = 16
    adaptive_d_update: bool = True
    adaptive_d_loss_threshold: float = 0.1
    grad_clip_norm_g: float = 1.0
    # ProjectionDiscriminator toggles
    disc_use_local: bool = True
    disc_use_global: bool = True
    disc_use_fft: bool = True
    disc_base_channels: int = 64
    disc_global_base_channels: int = 64
    disc_fft_base_channels: int = 32
    disc_n_layers: int = 3


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
            Note: v4 uses the separate :class:`V4Config` container.
        generator (GeneratorConfig): Generator architecture settings.
        discriminator (DiscriminatorConfig): Discriminator settings.
        loss (LossConfig): Loss function weights and options.
        training (TrainingConfig): Optimiser, scheduler, and loop settings.
        data (DataConfig): Dataset path, batch size, and worker settings.
        diffusion (DiffusionConfig): DiT diffusion settings (v3 only).
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
            ValueError: If ``model_version`` is not 1, 2, or 3 (use
                :class:`V4Config` for v4), or if ``decay_start_epoch`` is
                too close to ``num_epochs`` to allow a meaningful linear
                decay phase (fewer than 2 epochs of decay would remain).
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
        cfg.data.batch_size = 1

    return cfg


def get_8gb_config() -> UVCGANConfig:
    """
    Return a user-tuned UVCGANConfig for 8 GB GPUs.

    NOTE: These values reflect current memory testing and are not the
    original VRAM-minimizing profile. Comments below describe the
    active settings.
    """
    cfg = UVCGANConfig(model_version=2)

    # Recompute ViT activations during backward to save ~30-40% activation VRAM.
    cfg.generator.use_gradient_checkpointing = True
    cfg.generator.vit_depth = 4  # full depth; reduce to 2 for more headroom
    cfg.discriminator.num_scales = 3  # full multi-scale; reduce to 2 to save ~15%
    # batch_size=2 + accumulate_grads=2 keeps effective batch=4 at half the VRAM cost.
    cfg.data.batch_size = 2
    cfg.training.accumulate_grads = 2
    cfg.loss.perceptual_resize = 180  # slightly larger than 128 for better supervision
    cfg.generator.use_cross_domain = True
    cfg.generator.use_layerscale = True
    cfg.loss.use_wgan_gp = False
    cfg.loss.lambda_gp = 0.1
    cfg.training.use_amp = True  # do not disable — halves activation memory
    cfg.training.n_critic = 1

    return cfg


def get_dit_config() -> UVCGANConfig:
    """
    Return a default config for v3 diffusion training.

    The returned profile enables gradient checkpointing and uses the
    standard v3 architecture settings defined in :class:`DiffusionConfig`.
    """
    cfg = UVCGANConfig(model_version=3)
    cfg.diffusion.dit_hidden_dim = 256
    cfg.diffusion.dit_depth = 4
    cfg.diffusion.dit_heads = 4
    cfg.diffusion.dit_patch_size = 8
    cfg.diffusion.dit_mlp_ratio = 2.0
    cfg.diffusion.use_gradient_checkpointing = True
    cfg.diffusion.prediction_type = "v"
    cfg.diffusion.cond_dropout_prob = 0.1
    cfg.diffusion.cfg_scale = 2.5
    cfg.diffusion.cond_patch_size = 8
    cfg.diffusion.num_inference_steps = 20
    cfg.diffusion.min_snr_gamma = 5.0
    cfg.diffusion.lambda_perceptual_v3 = 0.0
    cfg.diffusion.perceptual_every_n_steps = 4
    cfg.diffusion.perceptual_batch_fraction = 0.5
    # Batch/accumulation: keep effective batch = 4 while lowering step overhead.
    cfg.data.batch_size = 2
    cfg.training.accumulate_grads = 2
    cfg.training.validation_size = 100
    cfg.training.validation_fid_samples = 600
    cfg.training.validation_fid_min_samples = 50
    return cfg


def get_dit_8gb_config() -> UVCGANConfig:
    """
    Return a config for v3 diffusion training tuned for 8 GB GPUs.

    Relative to :func:`get_dit_config`, this profile uses a larger DiT
    (hidden_dim=512, depth=8, heads=8) but pairs it with gradient
    checkpointing, AMP, a reduced batch size, and a lightweight validation
    setup to stay within 8 GB VRAM.

    DiT size notes:
        hidden_dim=512, depth=8, and heads=8 increase model capacity. Combined
        with gradient checkpointing this is still trainable on 8 GB; reduce
        these values if OOM is observed.

    Discriminator notes:
        disc_use_global=True keeps the global branch active.
        disc_base_channels=64 matches the default channel width.
        disc_use_fft=False disables the FFT discriminator, which is the
        primary memory saving relative to a fully featured setup.
    """
    cfg = UVCGANConfig(model_version=3)
    cfg.diffusion.dit_hidden_dim = 768
    cfg.diffusion.dit_depth = 8
    cfg.diffusion.dit_heads = 12
    cfg.diffusion.dit_patch_size = 8
    cfg.diffusion.dit_mlp_ratio = 2.0
    cfg.diffusion.use_gradient_checkpointing = True
    cfg.diffusion.prediction_type = "v"
    cfg.diffusion.cond_dropout_prob = 0.1
    cfg.diffusion.cfg_scale = 1.0
    cfg.diffusion.cond_patch_size = 32
    cfg.diffusion.cond_token_pool_stride = 4
    cfg.diffusion.use_cross_attention = False
    cfg.diffusion.num_inference_steps = 20
    cfg.diffusion.min_snr_gamma = 5.0
    cfg.diffusion.perceptual_every_n_steps = 1
    cfg.diffusion.perceptual_batch_fraction = 0.5
    # Batch/accumulation: keep effective batch = 4 while lowering step overhead.
    cfg.data.batch_size = 2
    cfg.training.accumulate_grads = 2
    # Slightly higher worker count helps keep GPU fed on fast local SSDs.
    cfg.data.num_workers = 4
    cfg.data.prefetch_factor = 2
    cfg.loss.perceptual_resize = 256
    cfg.diffusion.lambda_perceptual_v3 = 0.0
    cfg.training.validation_size = 20
    cfg.training.validation_fid_samples = 600
    cfg.training.validation_fid_min_samples = 50
    cfg.diffusion.disc_use_fft = False  # FFT branch is memory-intensive; disabled
    cfg.diffusion.disc_use_global = True  # global branch kept for layout supervision
    cfg.diffusion.disc_use_local = True  # local PatchGAN kept for texture detail
    cfg.diffusion.disc_base_channels = 128
    cfg.diffusion.disc_global_base_channels = 32
    cfg.diffusion.disc_n_layers = 4

    return cfg


# ---------------------------------------------------------------------------
# v4 config (GAN + PatchNCE + Transformer)
# ---------------------------------------------------------------------------


@dataclass
class V4ModelConfig:
    """
    Architecture hyperparameters for the v4 generator and discriminator.

    IMPROVED (v4.1) settings:
    - Generator base_channels increased from 128→192 for richer feature representation
    - num_res_blocks increased from 9→15 for deeper bottleneck processing
    - Transformer encoder_dim increased from 224→384 for higher capacity encoding
    - encoder_depth increased from 4→6 for complex feature transformation
    - encoder_heads increased from 4→8 for multi-scale attention patterns
    - encoder_mlp_ratio increased from 3.0→4.0 for stronger non-linearities
    - Discriminator channels increased from 64→128 for finer feature discrimination
    - disc_n_layers increased from 3→4 for deeper receptive field analysis

    Generator fields:
        input_nc / output_nc:   Input and output image channels.
        base_channels:          Base feature-map width for the CNN decoder.
        num_res_blocks:         ResNet bottleneck depth (ResnetGenerator only).
        use_transformer_encoder: If True, use TransformerGeneratorV4;
                                 otherwise use ResnetGenerator.
        image_size:             Square input image size.
        patch_size:             Transformer patch size (power of 2).
        encoder_dim:            Transformer token embedding dimension.
        encoder_depth:          Number of Transformer blocks.
        encoder_heads:          Attention heads per Transformer block.
        encoder_mlp_ratio:      MLP hidden-dim multiplier in Transformer blocks.
        encoder_dropout:        Dropout inside Transformer blocks.
        use_gradient_checkpointing: Recompute Transformer activations during
                                backward to reduce peak VRAM.

    Discriminator fields:
        disc_base_channels:     Base feature-map width for PatchGAN.
        disc_n_layers:          Number of strided downsampling layers.
    """

    input_nc: int = 3
    output_nc: int = 3
    base_channels: int = 128
    num_res_blocks: int = 15
    disc_base_channels: int = 128
    disc_n_layers: int = 4
    use_transformer_encoder: bool = True
    image_size: int = 256
    patch_size: int = 8
    encoder_dim: int = 384
    encoder_depth: int = 6
    encoder_heads: int = 8
    encoder_mlp_ratio: float = 4.0
    encoder_dropout: float = 0.0
    use_gradient_checkpointing: bool = False


@dataclass
class V4TrainingConfig:
    """
    Training loop hyperparameters for the v4 GAN + PatchNCE baseline.

    Key fields:
        lambda_gan:         Weight for the LSGAN adversarial loss.
        lambda_nce:         Weight for the PatchNCE contrastive loss.
        lambda_identity:    Weight for the identity (self-reconstruction) loss.
        nce_layers:         Encoder block indices used for NCE feature extraction.
        nce_num_patches:    Spatial patches sampled per feature map per step.
        nce_temperature:    InfoNCE softmax temperature.
        nce_proj_dim:       Output dimension of the per-layer MLP projectors.
        use_replay_buffer:  Stabilise discriminator with a history of fake samples.
        use_ema:            Maintain an EMA copy of both generators.
        ema_decay:          EMA decay factor (higher = slower update).
        use_lr_schedule:    Enable linear warmup + linear decay LR schedule.
        lr_warmup_epochs:   Epochs over which LR ramps from 0 to ``lr``.
        lr_decay_start_epoch: Epoch at which linear LR decay begins.
        accumulate_grads:   Gradient accumulation steps before an optimiser step.
        validation_every:   Run validation after this many epochs.
        early_stopping_patience: Max epochs-without-improvement before stopping.
        early_stopping_warmup: Earliest epoch at which early stopping is active.
        early_stopping_interval: Evaluate early stopping every N epochs.
        early_stopping_min_delta: Minimum SSIM gain required to reset patience.
        divergence_threshold: Loss explosion ratio vs best loss baseline.
        divergence_patience: Consecutive divergence checks before hard stop.
        save_every:         Save a checkpoint every N epochs.
    """

    num_epochs: int = 200
    epoch_size: int = 3000
    test_size: int = 200
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    accumulate_grads: int = 1
    log_every: int = 50
    save_every: int = 20
    validation_every: int = 5
    validation_samples: int = 10
    validation_max_batches: int = 50
    validation_fid_samples: int = 200
    validation_fid_min_samples: int = 50
    early_stopping_patience: int = 40
    early_stopping_warmup: int = 80
    early_stopping_interval: int = 10
    early_stopping_min_delta: float = 1e-5
    divergence_threshold: float = 5.0
    divergence_patience: int = 2
    lambda_gan: float = 5.0
    lambda_nce: float = 2.0
    lambda_identity: float = 5.0
    nce_layers: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    nce_num_patches: int = 256
    nce_temperature: float = 0.07
    nce_proj_dim: int = 256
    use_replay_buffer: bool = True
    replay_buffer_size: int = 50
    use_ema: bool = True
    ema_decay: float = 0.999
    use_lr_schedule: bool = True
    lr_warmup_epochs: int = 5
    lr_decay_start_epoch: int = 100


@dataclass
class V4DataConfig:
    """
    Data loading configuration for v4 training.

    Fields:
        image_size:      Spatial size to which patches are resized (square).
        batch_size:      Samples per mini-batch.
        num_workers:     DataLoader worker processes (0 = main process only).
        prefetch_factor: Batches prefetched per worker.
    """

    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2


@dataclass
class V4Config:
    """
    Top-level configuration container for v4 training.

    Groups all sub-configurations into typed fields and validates that
    ``model_version`` is exactly 4.  Pass an instance to ``train_v4()``.

    Fields:
        model_version: Must be ``4``.
        model:         Generator and discriminator architecture settings.
        training:      Optimiser, scheduler, NCE, and loop settings.
        data:          Dataset path, batch size, and worker settings.
        model_dir:     Root output directory for checkpoints and logs.
                       Auto-generated when ``None``.
        val_dir:       Directory for per-epoch validation images.
                       Defaults to ``model_dir/validation_images`` when ``None``.
    """

    model_version: int = 4
    model: V4ModelConfig = field(default_factory=V4ModelConfig)
    training: V4TrainingConfig = field(default_factory=V4TrainingConfig)
    data: V4DataConfig = field(default_factory=V4DataConfig)
    model_dir: Optional[str] = None
    val_dir: Optional[str] = None

    def __post_init__(self):
        if self.model_version != 4:
            raise ValueError(
                f"model_version must be 4 for V4Config, got {self.model_version!r}."
            )


def get_v4_config() -> V4Config:
    """
    Return a default V4Config with full-capacity settings.

    Suitable for GPUs with 12+ GB VRAM.  Gradient checkpointing is disabled
    for maximum training speed.
    """
    return V4Config()


def get_v4_8gb_config() -> V4Config:
    """
    Return a VRAM-optimised V4Config for 8 GB GPUs.

    Gradient checkpointing is currently disabled (set to False) — enable it
    by setting ``cfg.model.use_gradient_checkpointing = True`` if OOM is
    observed.  All other settings are identical to :func:`get_v4_config`.
    """
    cfg = V4Config()
    cfg.model.use_gradient_checkpointing = True
    return cfg
