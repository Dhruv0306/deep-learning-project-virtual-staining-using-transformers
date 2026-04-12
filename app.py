"""
Inference script for whole-slide stain / unstain translation.

Loads a trained checkpoint and translates arbitrarily large histology
images by splitting them into 256×256 patches, running each patch through
the appropriate generator, and then blending overlapping patches back into
a seamless full-resolution output image.

Both model versions are supported:
  - v1 (``ViTUNetGenerator``)   — loaded when ``model_version=1``
  - v2 (``ViTUNetGeneratorV2``) — loaded when ``model_version=2``
  - v3 (``DiTGenerator``)       — loaded when ``model_version=3``
  - v4 (``TransformerGeneratorV4``) — loaded when ``model_version=4``

Loader helpers are provided for each version so the saved config can be
retrieved alongside the model components:

    - ``load_v1_components``
    - ``load_v2_components``
    - ``load_v3_components``
    - ``load_v4_components``

For v2 and v4 the architecture hyperparameters are also inferred from the
checkpoint state dict so the instantiated model matches the saved weights.

CLI usage::

    python app.py

You will be prompted for:
  - Path to the trained checkpoint (``.pth`` file)
    - Model version (1, 2, 3, or 4)
    - Path to an unstained image  (A -> B translation)
    - Path to a stained image     (B -> A translation, v1/v2/v4 only)

Model-version behavior:
    - v1/v2 run bidirectional translation using ``G_AB`` and ``G_BA``.
    - v3 runs A→B only (unstained→stained) via DDIM sampling with ``target_domain=1``.
    - v4 runs bidirectional translation using the Transformer + PatchNCE generators.

Outputs are written to:
  - ``data/reconstructed_stained_output.png``
  - ``data/reconstructed_unstained_output.png``
"""

import os
import math
import dataclasses
from typing import Optional

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from PIL import ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_checkpoint(checkpoint_path: str, map_location):
    """
    Load a local trusted checkpoint across PyTorch versions.

    PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
    checkpoints that include config dataclasses. We explicitly disable that
    default for trusted local checkpoints, and fall back for older versions
    that do not support the weights_only argument.
    """
    try:
        return torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def is_v3_checkpoint(ckpt: dict) -> bool:
    """
    Return True when a checkpoint dictionary appears to follow v3 schema.

    The current heuristic checks for ``dit_state_dict``.
    """
    return "dit_state_dict" in ckpt


def _load_uvcgan_config(checkpoint: dict, model_version: int):
    """
    Load a UVCGAN config object from a checkpoint or fall back to defaults.

    Older v1/v2 checkpoints may not persist a config object, so this helper
    always returns a valid config for the requested model version.
    """
    from config import UVCGANConfig, get_default_config

    ckpt_config = checkpoint.get("config")
    if (
        isinstance(ckpt_config, UVCGANConfig)
        and ckpt_config.model_version == model_version
    ):
        return ckpt_config

    if ckpt_config is not None:
        print(
            f"[warn] checkpoint config has unexpected type {type(ckpt_config).__name__!r} "
            f"for model_version={model_version}; falling back to get_default_config()."
        )

    return get_default_config(model_version)


def load_v3_components(checkpoint_path: str, device: str):
    """
    Load v3 DiT + VAE + sampler components.

    The function prefers a serialized diffusion config stored in the
    checkpoint under ``config`` and falls back to ``get_dit_config()`` when
    not present.

    Returns:
        tuple: ``(dit_model, cond_encoder, vae, sampler, diff_cfg)``.
        ``cond_encoder`` is kept as ``None`` for backward compatibility,
        since v3 now tokenizes conditions inside the generator.
    """
    from config import get_dit_config
    from model_v3.generator import getGeneratorV3
    from model_v3.noise_scheduler import DDPMScheduler, DDIMSampler
    from model_v3.vae_wrapper import VAEWrapper

    checkpoint = _load_checkpoint(checkpoint_path, map_location=device)
    diff_cfg = checkpoint.get("config")
    if diff_cfg is None:
        diff_cfg = get_dit_config().diffusion

    dit_model = getGeneratorV3(diff_cfg, device=torch.device(device))
    if "ema_state_dict" in checkpoint:
        dit_model.load_state_dict(checkpoint["ema_state_dict"])
    elif "dit_state_dict" in checkpoint:
        dit_model.load_state_dict(checkpoint["dit_state_dict"])
    else:
        raise KeyError("Checkpoint missing both 'ema_state_dict' and 'dit_state_dict'.")
    dit_model.eval()

    vae = VAEWrapper(diff_cfg.vae_model_id).to(device)
    vae.eval()

    scheduler = DDPMScheduler(diff_cfg.num_timesteps, diff_cfg.beta_schedule).to(device)
    sampler = DDIMSampler(scheduler)

    cond_encoder = None
    return dit_model, cond_encoder, vae, sampler, diff_cfg


def load_v1_components(checkpoint_path: str, device: str = "cpu"):
    """
    Load v1 generators together with the saved or default UVCGAN config.

    Older v1 checkpoints do not persist a config object, so the loader falls
    back to ``get_default_config(1)``.
    """
    from model_v1.generator import ViTUNetGenerator

    checkpoint = _load_checkpoint(checkpoint_path, map_location=device)
    cfg = _load_uvcgan_config(checkpoint, model_version=1)

    G_AB = ViTUNetGenerator().to(device)
    G_BA = ViTUNetGenerator().to(device)
    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    G_AB.eval()
    G_BA.eval()

    return G_AB, G_BA, cfg


def load_v2_components(checkpoint_path: str, device: str = "cpu"):
    """
    Load v2 generators together with the saved or default UVCGAN config.

    The generator architecture is reconstructed from the checkpoint state dict
    so the loaded model always matches the saved weights.
    """
    from model_v2.generator import ViTUNetGeneratorV2

    checkpoint = _load_checkpoint(checkpoint_path, map_location=device)
    cfg = _load_uvcgan_config(checkpoint, model_version=2)

    kwargs = _infer_v2_kwargs(checkpoint["G_AB"])
    G_AB = ViTUNetGeneratorV2(**kwargs).to(device)
    G_BA = ViTUNetGeneratorV2(**kwargs).to(device)

    def _load_v2_state_with_compat(model, state_dict: dict, tag: str):
        """
        Load v2 weights with a legacy fallback for older checkpoint schemas.

        Newer ``ViTUNetGeneratorV2`` checkpoints should load strictly. For older
        v2 checkpoints (pre-v2.2) we remap known renamed keys and then load only
        shape-compatible tensors with ``strict=False``.
        """
        try:
            model.load_state_dict(state_dict)
            return
        except RuntimeError as exc:
            print(
                f"[warn] {tag}: strict v2 load failed, trying compatibility fallback."
            )

            remapped = {}
            for key, value in state_dict.items():
                new_key = key
                # Older v2 checkpoints used `res_bot.*`; newer model splits this
                # into pre/post bottleneck residual blocks.
                if key.startswith("res_bot."):
                    new_key = "res_bot_pre." + key[len("res_bot.") :]
                remapped[new_key] = value

            model_state = model.state_dict()
            compatible = {
                key: value
                for key, value in remapped.items()
                if key in model_state and model_state[key].shape == value.shape
            }

            model.load_state_dict(compatible, strict=False)

            loaded_count = len(compatible)
            total_count = len(model_state)
            print(
                f"[warn] {tag}: loaded {loaded_count}/{total_count} tensors via compatibility mode. "
                "Missing newer layers are left at init values."
            )
            print(f"[warn] {tag}: original strict-load error: {exc}")

    _load_v2_state_with_compat(G_AB, checkpoint["G_AB"], tag="G_AB")
    _load_v2_state_with_compat(G_BA, checkpoint["G_BA"], tag="G_BA")
    G_AB.eval()
    G_BA.eval()

    return G_AB, G_BA, cfg


def _infer_v2_kwargs(state_dict: dict) -> dict:
    """
    Read architecture hyperparameters directly from a v2 checkpoint state dict
    so the model we build always matches what was saved, regardless of which
    config was used during training.

    Detects:
        vit_depth        — number of ViT blocks (count unique block indices)
        use_cross_domain — True if fuse* layers are present in the weights
    """
    block_indices = set()
    for key in state_dict:
        # keys look like "vit.blocks.0.gamma_attn", "vit.blocks.1.norm1.weight" ...
        if key.startswith("vit.blocks."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                block_indices.add(int(parts[2]))

    vit_depth = len(block_indices) if block_indices else 4
    use_cross_domain = any(k.startswith("fuse") for k in state_dict)

    print(
        f"[load_model] auto-detected vit_depth={vit_depth}, use_cross_domain={use_cross_domain}"
    )
    return {"vit_depth": vit_depth, "use_cross_domain": use_cross_domain}


def load_model(checkpoint_path=None, device="cpu", model_version=2):
    """
    Load a trained CycleGAN checkpoint and return the two generator models.

    For model version 2, the architecture hyperparameters are auto-detected
    from the checkpoint state dict via :func:`_infer_v2_kwargs`, so the model
    that is instantiated always matches the one that was saved.

    Args:
        checkpoint_path (str): Path to a ``.pth`` checkpoint file produced by
            ``model_v1.training_loop.train()`` or ``model_v2.training_loop.train_v2()``.
        device (str): Device to load the model onto, e.g. ``"cpu"`` or
            ``"cuda"``.
        model_version (int): ``1`` loads :class:`~model_v1.generator.ViTUNetGenerator`;
            ``2`` loads :class:`~model_v2.generator.ViTUNetGeneratorV2`.
            For v3 use :func:`load_v3_components`; for v4 use :func:`load_v4_components`.

    Returns:
        tuple[nn.Module, nn.Module]: ``(G_AB, G_BA)`` — both in ``eval()``
        mode on the specified device.

    Raises:
        ValueError: If ``checkpoint_path`` is ``None`` or ``model_version``
            is not 1 or 2.
    """
    if checkpoint_path is None:
        raise ValueError("Checkpoint_path is required")

    if model_version == 1:
        G_AB, G_BA, _ = load_v1_components(checkpoint_path, device=device)
    elif model_version == 2:
        G_AB, G_BA, _ = load_v2_components(checkpoint_path, device=device)
    else:
        raise ValueError(f"model_version must be 1 or 2, got {model_version!r}")

    return G_AB, G_BA


def _infer_v4_kwargs(state_dict: dict, fallback_cfg) -> dict:
    """
    Infer v4 generator architecture parameters from a checkpoint state dict.

    Detects whether the checkpoint contains a Transformer encoder or a ResNet
    generator by checking for ``patch_embed.proj`` keys, then reads
    ``patch_size``, ``encoder_dim``, ``encoder_depth``, ``mlp_ratio``, and
    ``base_channels`` directly from weight shapes.  Falls back to
    ``fallback_cfg`` values for any parameter that cannot be inferred.

    Args:
        state_dict:   Generator state dict from the checkpoint.
        fallback_cfg: V4ModelConfig instance used as fallback for missing params.

    Returns:
        dict of keyword arguments suitable for passing to ``getGeneratorV4``.
    """
    if any(k.startswith("patch_embed.proj") for k in state_dict):
        patch_w = state_dict["patch_embed.proj.weight"].shape[2]
        encoder_dim = state_dict["patch_embed.proj.weight"].shape[0]
        block_indices = set()
        for key in state_dict:
            if key.startswith("blocks."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.add(int(parts[1]))
        encoder_depth = (
            len(block_indices) if block_indices else fallback_cfg.encoder_depth
        )

        mlp_ratio = fallback_cfg.encoder_mlp_ratio
        for key, weight in state_dict.items():
            if key.startswith("blocks.") and key.endswith("mlp.0.weight"):
                mlp_ratio = weight.shape[0] / encoder_dim
                break

        base_channels = fallback_cfg.base_channels
        proj_weight = state_dict.get("proj.0.weight")
        if proj_weight is not None and proj_weight.shape[0] % 4 == 0:
            base_channels = proj_weight.shape[0] // 4

        return dict(
            use_transformer_encoder=True,
            patch_size=patch_w,
            encoder_dim=encoder_dim,
            encoder_depth=encoder_depth,
            encoder_mlp_ratio=mlp_ratio,
            base_channels=base_channels,
        )

    # ResNet fallback
    base_channels = fallback_cfg.base_channels
    in_conv_weight = state_dict.get("in_conv.1.weight")
    if in_conv_weight is not None:
        base_channels = in_conv_weight.shape[0]

    res_block_indices = set()
    for key in state_dict:
        if key.startswith("res_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                res_block_indices.add(int(parts[1]))
    num_res_blocks = (
        len(res_block_indices) if res_block_indices else fallback_cfg.num_res_blocks
    )

    return dict(
        use_transformer_encoder=False,
        base_channels=base_channels,
        num_res_blocks=num_res_blocks,
    )


def load_v4_components(
    checkpoint_path: str, device: str, image_size: Optional[int] = None
):
    """
    Load v4 generators together with the saved or default V4 model config.

    Prefers EMA weights (``ema_G_AB_state_dict`` / ``ema_G_BA_state_dict``) and
    falls back to the raw generator weights when EMA is not present.

    Architecture hyperparameters are loaded from the ``"config"`` key stored in
    the checkpoint (a ``V4ModelConfig`` saved by ``train_v4``). For older
    checkpoints that do not contain this key the function falls back to
    :func:`~config.get_v4_8gb_config` and additionally attempts to infer
    shape-visible fields via :func:`_infer_v4_kwargs`.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint saved by ``train_v4``.
        device:          Device string, e.g. ``"cpu"`` or ``"cuda"``.
        image_size:      Square input image size used to configure patch embedding.

    Returns:
        tuple[nn.Module, nn.Module, V4ModelConfig]: ``(G_AB, G_BA, mcfg)``.

    Raises:
        KeyError: If neither EMA nor raw generator state dicts are found.
    """
    from config import V4ModelConfig, get_v4_8gb_config
    from model_v4.generator import getGeneratorV4

    checkpoint = _load_checkpoint(checkpoint_path, map_location=device)

    # Prefer the config saved at training time; fall back to the 8 GB preset for
    # legacy checkpoints that pre-date config persistence.
    ckpt_config = checkpoint.get("config")
    if isinstance(ckpt_config, V4ModelConfig):
        mcfg = ckpt_config
    else:
        if ckpt_config is not None:
            print(
                f"[warn] load_v4_model: checkpoint 'config' has unexpected type "
                f"{type(ckpt_config).__name__!r}; falling back to get_v4_8gb_config()."
            )
        else:
            print(
                "[warn] load_v4_model: checkpoint has no 'config' key; "
                "falling back to get_v4_8gb_config() for legacy checkpoint."
            )
        mcfg = get_v4_8gb_config().model

    def _pick_state(ema_key: str, raw_key: str):
        state = checkpoint.get(ema_key)
        if state is None:
            state = checkpoint.get(raw_key)
        return state

    state_ab = _pick_state("ema_G_AB_state_dict", "G_AB_state_dict")
    state_ba = _pick_state("ema_G_BA_state_dict", "G_BA_state_dict")
    if state_ab is None or state_ba is None:
        raise KeyError("Checkpoint missing v4 generator state dicts.")

    # For legacy checkpoints without a stored config, infer all shape-visible
    # fields from the state dict so model construction matches the saved weights.
    if not isinstance(ckpt_config, V4ModelConfig):
        inferred = _infer_v4_kwargs(state_ab, mcfg)
        mcfg = dataclasses.replace(
            mcfg,
            base_channels=inferred.get("base_channels", mcfg.base_channels),
            num_res_blocks=inferred.get("num_res_blocks", mcfg.num_res_blocks),
            use_transformer_encoder=inferred.get(
                "use_transformer_encoder", mcfg.use_transformer_encoder
            ),
            patch_size=inferred.get("patch_size", mcfg.patch_size),
            encoder_dim=inferred.get("encoder_dim", mcfg.encoder_dim),
            encoder_depth=inferred.get("encoder_depth", mcfg.encoder_depth),
            encoder_mlp_ratio=inferred.get("encoder_mlp_ratio", mcfg.encoder_mlp_ratio),
        )

    resolved_image_size = image_size if image_size is not None else mcfg.image_size

    G_AB = getGeneratorV4(
        input_nc=mcfg.input_nc,
        output_nc=mcfg.output_nc,
        base_channels=mcfg.base_channels,
        num_res_blocks=mcfg.num_res_blocks,
        use_transformer_encoder=mcfg.use_transformer_encoder,
        image_size=resolved_image_size,
        patch_size=mcfg.patch_size,
        encoder_dim=mcfg.encoder_dim,
        encoder_depth=mcfg.encoder_depth,
        encoder_heads=mcfg.encoder_heads,
        encoder_mlp_ratio=mcfg.encoder_mlp_ratio,
        encoder_dropout=mcfg.encoder_dropout,
        use_gradient_checkpointing=False,
        device=torch.device(device),
        run_smoke_test=False,
    )
    G_BA = getGeneratorV4(
        input_nc=mcfg.output_nc,
        output_nc=mcfg.input_nc,
        base_channels=mcfg.base_channels,
        num_res_blocks=mcfg.num_res_blocks,
        use_transformer_encoder=mcfg.use_transformer_encoder,
        image_size=resolved_image_size,
        patch_size=mcfg.patch_size,
        encoder_dim=mcfg.encoder_dim,
        encoder_depth=mcfg.encoder_depth,
        encoder_heads=mcfg.encoder_heads,
        encoder_mlp_ratio=mcfg.encoder_mlp_ratio,
        encoder_dropout=mcfg.encoder_dropout,
        use_gradient_checkpointing=False,
        device=torch.device(device),
        run_smoke_test=False,
    )

    def _load_v4_state_with_compat(model, state_dict: dict, tag: str):
        """
        Load v4 weights with a compatibility fallback for older checkpoints.

        Newer checkpoints should load strictly. For older checkpoints that
        predate added layers (e.g. local attention gate/texture head), load
        only shape-compatible tensors with ``strict=False``.
        """
        try:
            model.load_state_dict(state_dict)
            return
        except RuntimeError as exc:
            print(
                f"[warn] {tag}: strict v4 load failed, trying compatibility fallback."
            )

            model_state = model.state_dict()
            compatible = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            model.load_state_dict(compatible, strict=False)

            loaded_count = len(compatible)
            total_count = len(model_state)
            print(
                f"[warn] {tag}: loaded {loaded_count}/{total_count} tensors via compatibility mode. "
                "Missing newer layers are left at init values."
            )
            print(f"[warn] {tag}: original strict-load error: {exc}")

    _load_v4_state_with_compat(G_AB, state_ab, tag="G_AB")
    _load_v4_state_with_compat(G_BA, state_ba, tag="G_BA")
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA, mcfg


def load_v4_model(checkpoint_path: str, device: str, image_size: int) -> tuple:
    """
    Backward-compatible wrapper that returns only the v4 generator pair.
    """
    G_AB, G_BA, _ = load_v4_components(
        checkpoint_path=checkpoint_path, device=device, image_size=image_size
    )
    return G_AB, G_BA


def stain_image(image, model, device="cpu"):
    """
    Apply a generator model to a pre-processed image tensor (A → B direction).

    A thin wrapper around ``model.forward`` that moves the tensor to the
    specified device, runs inference under ``torch.no_grad``, and returns
    the output on CPU.

    Args:
        image (torch.Tensor): Input tensor of shape ``(1, C, H, W)`` in the
            ``[-1, 1]`` normalised range.
        model (nn.Module): Generator (e.g. ``G_AB``) in ``eval()`` mode.
        device (str): Device to use for inference.

    Returns:
        torch.Tensor: Generated tensor on CPU with shape ``(1, C, H, W)``.
    """
    with torch.no_grad():
        image = image.to(device)
        generated_image = model(image)
        return generated_image.cpu()


def unstain_image(image, model, device="cpu"):
    """
    Apply a generator model to a pre-processed image tensor (B → A direction).

    Functionally identical to :func:`stain_image`; provided as a named
    counterpart for readability when calling with ``G_BA``.

    Args:
        image (torch.Tensor): Input tensor of shape ``(1, C, H, W)`` in the
            ``[-1, 1]`` normalised range.
        model (nn.Module): Generator (e.g. ``G_BA``) in ``eval()`` mode.
        device (str): Device to use for inference.

    Returns:
        torch.Tensor: Generated tensor on CPU with shape ``(1, C, H, W)``.
    """
    with torch.no_grad():
        image = image.to(device)
        generated_image = model(image)
        return generated_image.cpu()


def pad_to_patch_multiple(image, patch_size=256):
    """
    Pad a PIL image with white pixels so that both dimensions are exact
    multiples of ``patch_size``.

    Padding is added to the right and bottom edges only, so the top-left
    corner of the original image is preserved without any shift.  The
    padded region is filled with ``(255, 255, 255)`` to approximate the
    white slide background common in histology images.

    Args:
        image (PIL.Image.Image): Source RGB image of arbitrary size.
        patch_size (int): Required spatial multiple (default 256).

    Returns:
        tuple[PIL.Image.Image, tuple[int, int]]:
            ``(padded_image, original_size)`` where ``original_size`` is
            ``(width, height)`` of the unpadded input.  If the input is
            already an exact multiple, it is returned unchanged along with
            its own size.
    """
    width, height = image.size
    padded_width = math.ceil(width / patch_size) * patch_size
    padded_height = math.ceil(height / patch_size) * patch_size

    if (padded_width, padded_height) == (width, height):
        return image, (width, height)

    padded = Image.new("RGB", (padded_width, padded_height), color=(255, 255, 255))
    padded.paste(image, (0, 0))
    return padded, (width, height)


def extract_patches_with_coords(pil_image, patch_size=256, stride=256):
    """
    Extract all ``patch_size × patch_size`` tiles from a PIL image and
    return them together with their pixel coordinates.

    Tiles are generated in row-major order (top-to-bottom, left-to-right).
    When the image dimensions are not exact multiples of ``stride``, an
    extra row/column of patches is appended so that the far right and
    bottom edges are always fully covered.

    Args:
        pil_image (PIL.Image.Image): Source image.  Should already be padded
            to a multiple of ``patch_size`` via :func:`pad_to_patch_multiple`.
        patch_size (int): Tile side length in pixels (default 256).
        stride (int): Step between tile origins.  ``stride == patch_size``
            gives non-overlapping tiles; ``stride < patch_size`` gives
            overlapping tiles that can be blended during reconstruction.

    Returns:
        tuple[list[PIL.Image.Image], list[tuple[int, int]]]:
            ``(patches, positions)`` where ``positions[i] = (top, left)``
            is the pixel offset of ``patches[i]`` within ``pil_image``.
    """
    width, height = pil_image.size
    top_positions = list(range(0, height - patch_size + 1, stride))
    left_positions = list(range(0, width - patch_size + 1, stride))

    if top_positions[-1] != height - patch_size:
        top_positions.append(height - patch_size)
    if left_positions[-1] != width - patch_size:
        left_positions.append(width - patch_size)

    patches = []
    positions = []
    for top in top_positions:
        for left in left_positions:
            patch = pil_image.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)
            positions.append((top, left))

    return patches, positions


def _blend_window(patch_size, device, dtype, eps=0.05):
    """
    Create a 2-D Hann (raised-cosine) blending window for seamless patch
    reconstruction.

    The 1-D window is ``sin²(π × k / patch_size)`` for ``k ∈ [0, patch_size)``,
    which is 0 at the edges and 1 at the centre.  A small floor ``eps`` is
    added so that edge pixels always receive a non-zero weight and no
    division-by-zero can occur in the weight map.

    The 2-D window is the outer product of two 1-D windows, giving a smooth
    tent-like surface that sums to a near-constant value when tiles overlap
    by exactly 50%.

    Args:
        patch_size (int): Tile side length (must be > 1).
        device (torch.device): Device on which to allocate the tensor.
        dtype (torch.dtype): Floating-point dtype to use.
        eps (float): Minimum window value at the edges (default 0.05).

    Returns:
        torch.Tensor: Shape ``(patch_size, patch_size)`` blending weights,
        all values in ``[eps, 1]``.
    """
    if patch_size <= 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    window_1d = (
        torch.sin(torch.linspace(0, math.pi, patch_size, device=device, dtype=dtype))
        ** 2
    )
    window_1d = window_1d * (1.0 - eps) + eps
    window_2d = window_1d[:, None] * window_1d[None, :]
    return window_2d


def reconstruct_tensor_from_patches(
    patches, positions, image_size, patch_size=256, stride=256
):
    """
    Reassemble translated patches into a full-resolution image tensor.

    Each patch is placed at its recorded ``(top, left)`` position and
    multiplied by a per-pixel weight.  When ``stride < patch_size``
    (overlapping tiles), a 2-D Hann window (:func:`_blend_window`) is used
    so that contributions from adjacent patches blend smoothly and no
    hard seam is visible.  When ``stride == patch_size`` (non-overlapping),
    each pixel receives a uniform weight of 1.

    The final reconstructed tensor is divided by the accumulated weight map
    (clamped to ``1e-6``) to normalise the weighted sum.

    Args:
        patches (list[torch.Tensor]): Translated patch tensors, each with
            shape ``(3, patch_size, patch_size)`` and values in ``[-1, 1]``.
        positions (list[tuple[int, int]]): ``(top, left)`` pixel offsets
            matching ``patches``, as returned by
            :func:`extract_patches_with_coords`.
        image_size (tuple[int, int]): ``(width, height)`` of the padded
            source image (not the original, unpadded size).
        patch_size (int): Tile side length in pixels (default 256).
        stride (int): Step between tile origins (default 256).

    Returns:
        torch.Tensor: Reconstructed image tensor of shape
        ``(3, height, width)`` with values in ``[-1, 1]``.
    """
    width, height = image_size
    dtype = patches[0].dtype
    device = patches[0].device

    # Keep output in normalized [-1, 1] tensor space to avoid patch-wise quantization.
    reconstructed = torch.zeros(3, height, width, dtype=dtype, device=device)
    weight_map = torch.zeros(1, height, width, dtype=dtype, device=device)

    if stride < patch_size:
        window = _blend_window(patch_size, device=device, dtype=dtype).unsqueeze(0)
    else:
        window = torch.ones(1, patch_size, patch_size, dtype=dtype, device=device)

    for patch, (top, left) in zip(patches, positions):
        reconstructed[:, top : top + patch_size, left : left + patch_size] += (
            patch * window
        )
        weight_map[:, top : top + patch_size, left : left + patch_size] += window

    reconstructed = reconstructed / weight_map.clamp_min(1e-6)
    return reconstructed


def translate_image_from_patches(
    input_image_path,
    model,
    transform,
    output_path,
    patch_size=256,
    stride=256,
    device="cpu",
):
    """
    Translate a whole-slide image by applying a generator patch-by-patch.

    The full pipeline is:

    1. Load the image from ``input_image_path`` as RGB.
    2. Pad to the nearest multiple of ``patch_size`` with white pixels.
    3. Extract ``patch_size × patch_size`` tiles with the specified stride.
    4. Apply ``transform`` to each tile and run ``model`` inference.
    5. Reconstruct the full image with weighted blending (Hann window when
       ``stride < patch_size``, uniform window otherwise).
    6. Crop back to the original image dimensions.
    7. Save to ``output_path`` with ``torchvision.utils.save_image``.

    A progress message is printed every 100 patches.

    Args:
        input_image_path (str): Path to the source whole-slide image.
        model (nn.Module): Generator in ``eval()`` mode.
        transform (callable): Preprocessing transform applied to each PIL
            patch before inference (should produce a ``[-1, 1]`` tensor).
        output_path (str): Path where the translated image is saved (PNG).
        patch_size (int): Tile side length in pixels (default 256).
        stride (int): Step between tile origins.  Use ``patch_size // 2``
            for 50% overlap and smoother seams (default 256 = no overlap).
        device (str): Inference device (default ``"cpu"``).

    Returns:
        tuple[tuple[int,int], tuple[int,int], int, str]:
            ``(original_size, padded_size, num_patches, output_path)``
            where sizes are ``(width, height)`` tuples.
    """
    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    padded_image, _ = pad_to_patch_multiple(input_image, patch_size=patch_size)

    input_patches, positions = extract_patches_with_coords(
        padded_image, patch_size=patch_size, stride=stride
    )
    translated_patches = []

    with torch.inference_mode():
        for patch in input_patches:
            patch_tensor = transform(patch).unsqueeze(0).to(device)  # [1, 3, H, W]
            translated_patch = model(patch_tensor).cpu().squeeze(0)
            translated_patches.append(translated_patch)

            if (
                translated_patches.__len__() % 100 == 0
                or translated_patches.__len__() == input_patches.__len__()
            ):
                print(
                    f"Processed {translated_patches.__len__()} / {input_patches.__len__()} patches"
                )

    reconstructed_padded = reconstruct_tensor_from_patches(
        translated_patches,
        positions,
        padded_image.size,
        patch_size=patch_size,
        stride=stride,
    )
    reconstructed = reconstructed_padded[:, : original_size[1], : original_size[0]]

    save_image(
        reconstructed.unsqueeze(0),
        output_path,
        normalize=True,
        value_range=(-1, 1),
    )

    return original_size, padded_image.size, len(input_patches), output_path


def translate_image_from_patches_v3(
    input_image_path,
    dit_model,
    cond_encoder,
    vae,
    sampler,
    transform,
    output_path,
    patch_size=256,
    stride=256,
    device="cpu",
    batch_size=1,
    num_steps=50,
    prediction_type="v",
    cfg_scale=1.0,
):
    """
    Translate a whole-slide image using the v3 diffusion pipeline.

    This path supports A -> B translation only (unstained -> stained).
    Patches are processed in mini-batches, denoised with DDIM sampling, then
    decoded by the VAE and blended back into a full-resolution image.

    Args:
        input_image_path (str): Path to the source image.
        dit_model (nn.Module): DiT denoiser network.
        cond_encoder (nn.Module | None): Deprecated placeholder for backward
            compatibility. The generator handles condition tokenization.
        vae (nn.Module): VAE decoder wrapper used to decode latent samples.
        sampler: DDIM sampler instance.
        transform (callable): Preprocessing transform returning normalized tensors.
        output_path (str): Output image path.
        patch_size (int): Patch side length.
        stride (int): Patch stride; use ``patch_size // 2`` for overlap.
        device (str): Inference device.
        batch_size (int): Number of patches per diffusion batch.
        num_steps (int): DDIM inference steps.
        prediction_type (str): Noise prediction target -- ``"v"`` or ``"epsilon"``.
        cfg_scale (float): Classifier-free guidance scale (1.0 = no guidance).

    Returns:
        tuple[tuple[int, int], tuple[int, int], int, str]:
            ``(original_size, padded_size, num_patches, output_path)``.
    """
    from torch.amp.autocast_mode import autocast

    input_image = Image.open(input_image_path).convert("RGB")
    original_size = input_image.size
    padded_image, _ = pad_to_patch_multiple(input_image, patch_size=patch_size)

    input_patches, positions = extract_patches_with_coords(
        padded_image, patch_size=patch_size, stride=stride
    )

    translated_patches = []
    use_amp = device == "cuda"

    with torch.inference_mode():
        for start in range(0, len(input_patches), batch_size):
            batch = input_patches[start : start + batch_size]
            batch_tensor = torch.stack([transform(p) for p in batch]).to(device)

            with autocast("cuda", enabled=use_amp):
                z0 = sampler.sample(
                    dit_model,
                    batch_tensor,
                    shape=(batch_tensor.size(0), 4, 32, 32),
                    device=torch.device(device),
                    num_steps=num_steps,
                    eta=0.0,
                    prediction_type=prediction_type,
                    cfg_scale=cfg_scale,
                    target_domain=1,
                )
                fake_B = vae.decode(z0)

            for j in range(fake_B.size(0)):
                translated_patches.append(fake_B[j].cpu())

            if len(translated_patches) % 100 == 0 or len(translated_patches) == len(
                input_patches
            ):
                print(
                    f"Processed {len(translated_patches)} / {len(input_patches)} patches"
                )

    reconstructed_padded = reconstruct_tensor_from_patches(
        translated_patches,
        positions,
        padded_image.size,
        patch_size=patch_size,
        stride=stride,
    )
    reconstructed = reconstructed_padded[:, : original_size[1], : original_size[0]]

    save_image(
        reconstructed.unsqueeze(0),
        output_path,
        normalize=True,
        value_range=(-1, 1),
    )

    return original_size, padded_image.size, len(input_patches), output_path


def _build_transform(patch_size: int):
    """Return the standard RGB normalisation transform for a patch size."""
    return transforms.Compose(
        [
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_path = input("Enter model path: ")
    model_version = int(
        input(
            "Enter 1 for Hybrid UVCGAN based, 2 for True-UVCGAN based, 3 for DiT diffusion, 4 for v4 Transformer: "
        )
    )

    dataset_root = os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset")

    if model_version == 3:
        if not model_path or not model_path.strip():
            raise ValueError("Checkpoint_path is required")
        dit_model, cond_encoder, vae, sampler, diff_cfg = load_v3_components(
            model_path, device=device
        )
        patch_size = 256
        stride = patch_size // 2
        transform = _build_transform(patch_size)

        unstained_image_path = input("Provide Path to Unstained Image: ")
        unstained_image_path = (
            os.path.join(
                dataset_root,
                "Un_Stained",
                "HC21-01338(A3-1).10X unstained.jpg",
            )
            if not unstained_image_path
            else os.path.normpath(unstained_image_path)
        )
        print(f"Unstained Image Path: {unstained_image_path}")
        # Output path is dataset_root / model_dir_name / V_Stained / original_filename
        model_dir_name = os.path.basename(os.path.dirname(model_path))
        output_path = os.path.join(dataset_root, model_dir_name, "V_Stained")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, os.path.basename(unstained_image_path))

        batch_size = 4 if device == "cuda" else 1
        original_size, padded_size, num_patches, stained_output_path = (
            translate_image_from_patches_v3(
                input_image_path=unstained_image_path,
                dit_model=dit_model,
                cond_encoder=cond_encoder,
                vae=vae,
                sampler=sampler,
                transform=transform,
                output_path=output_path,
                patch_size=patch_size,
                stride=stride,
                device=device,
                batch_size=batch_size,
                num_steps=diff_cfg.num_inference_steps,
                prediction_type=diff_cfg.prediction_type,
                cfg_scale=diff_cfg.cfg_scale,
            )
        )

        print(f"[Stain/v3] Original Image size: {original_size}")
        print(f"[Stain/v3] Padded Image size: {padded_size}")
        print(f"[Stain/v3] Num patches: {num_patches}")
        print(f"[Stain/v3] Saved reconstructed stained image at: {stained_output_path}")
        print(f"[Stain/v3] Patch stride: {stride}")
    elif model_version == 4:
        if not model_path or not model_path.strip():
            raise ValueError("Checkpoint_path is required")
        G_AB, G_BA, v4_cfg = load_v4_components(
            checkpoint_path=model_path, device=device
        )
        patch_size = v4_cfg.image_size
        stride = patch_size // 2
        transform = _build_transform(patch_size)

        unstained_image_path = input("Provide Path to Unstained Image: ")
        stained_image_path = input("Provide Path to Stained Image: ")
        unstained_image_path = (
            os.path.join(
                dataset_root,
                "Un_Stained",
                "HC21-01338(A3-1).10X unstained.jpg",
            )
            if not unstained_image_path
            else os.path.normpath(unstained_image_path)
        )
        stained_image_path = (
            os.path.join(
                dataset_root,
                "C_Stained",
                "HC21-01338(A3-2).10X unstained.jpg",
            )
            if not stained_image_path
            else os.path.normpath(stained_image_path)
        )

        # Output path is dataset_root / model_dir_name / V_Stained / original_filename
        model_dir_name = os.path.basename(os.path.dirname(model_path))
        output_path = os.path.join(dataset_root, model_dir_name, "V_Stained")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, os.path.basename(unstained_image_path))

        print(f"Unstained Image Path: {unstained_image_path}")
        print(f"Stained Image Path: {stained_image_path}")

        # A -> B (unstained -> stained)
        original_size, padded_size, num_patches, stained_output_path = (
            translate_image_from_patches(
                input_image_path=unstained_image_path,
                model=G_AB,
                transform=transform,
                output_path=output_path,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )

        print(f"[Stain/v4] Original Image size: {original_size}")
        print(f"[Stain/v4] Padded Image size: {padded_size}")
        print(f"[Stain/v4] Num patches: {num_patches}")
        print(f"[Stain/v4] Saved reconstructed stained image at: {stained_output_path}")
        print(f"[Stain/v4] Patch stride: {stride}")

        # B -> A (stained -> unstained)
        original_size, padded_size, num_patches, unstained_output_path = (
            translate_image_from_patches(
                input_image_path=stained_image_path,
                model=G_BA,
                transform=transform,
                output_path=os.path.join("data", "reconstructed_unstained_output.png"),
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )
        print(f"[Unstain/v4] Original Image size: {original_size}")
        print(f"[Unstain/v4] Padded Image size: {padded_size}")
        print(f"[Unstain/v4] Num patches: {num_patches}")
        print(
            f"[Unstain/v4] Saved reconstructed unstained image at: {unstained_output_path}"
        )
        print(f"[Unstain/v4] Patch stride: {stride}")
    elif model_version in (1, 2):
        if not model_path or not model_path.strip():
            raise ValueError("Checkpoint_path is required")
        if model_version == 1:
            G_AB, G_BA, uv_cfg = load_v1_components(model_path, device=device)
        else:
            G_AB, G_BA, uv_cfg = load_v2_components(model_path, device=device)

        patch_size = uv_cfg.data.image_size
        stride = patch_size // 2
        transform = _build_transform(patch_size)

        unstained_image_path = input("Provide Path to Unstained Image: ")
        stained_image_path = input("Provide Path to Stained Image: ")
        unstained_image_path = (
            os.path.join(
                dataset_root,
                "Un_Stained",
                "HC21-01338(A3-1).10X unstained.jpg",
            )
            if not unstained_image_path
            else os.path.normpath(unstained_image_path)
        )
        stained_image_path = (
            os.path.join(
                dataset_root,
                "C_Stained",
                "HC21-01338(A3-2).10X unstained.jpg",
            )
            if not stained_image_path
            else os.path.normpath(stained_image_path)
        )
        # Output path is dataset_root / model_dir_name / V_Stained / original_filename
        model_dir_name = os.path.basename(os.path.dirname(model_path))
        output_path = os.path.join(dataset_root, model_dir_name, "V_Stained")
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, os.path.basename(unstained_image_path))

        print(f"Unstained Image Path: {unstained_image_path}")
        print(f"Stained Image Path: {stained_image_path}")

        # A -> B (unstained -> stained)
        original_size, padded_size, num_patches, stained_output_path = (
            translate_image_from_patches(
                input_image_path=unstained_image_path,
                model=G_AB,
                transform=transform,
                output_path=output_path,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )

        print(f"[Stain] Original Image size: {original_size}")
        print(f"[Stain] Padded Image size: {padded_size}")
        print(f"[Stain] Num patches: {num_patches}")
        print(f"[Stain] Saved reconstructed stained image at: {stained_output_path}")
        print(f"[Stain] Patch stride: {stride}")

        # B -> A (stained -> unstained)
        original_size, padded_size, num_patches, unstained_output_path = (
            translate_image_from_patches(
                input_image_path=stained_image_path,
                model=G_BA,
                transform=transform,
                output_path=os.path.join("data", "reconstructed_unstained_output.png"),
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
        )
        print(f"[Unstain] Original Image size: {original_size}")
        print(f"[Unstain] Padded Image size: {padded_size}")
        print(f"[Unstain] Num patches: {num_patches}")
        print(
            f"[Unstain] Saved reconstructed unstained image at: {unstained_output_path}"
        )
        print(f"[Unstain] Patch stride: {stride}")
    else:
        raise ValueError(
            f"Invalid model_version: {model_version}. Must be 1, 2, 3, or 4."
        )
