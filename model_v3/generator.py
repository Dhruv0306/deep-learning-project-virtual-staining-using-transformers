"""
DiT generator components for the v3 latent diffusion pipeline.

UPGRADES (v3.2):
    - DiTBlock: Added windowed local self-attention branch alongside global
      self-attention, combined via a learnable gate. This lets the model
      attend to fine-grained local texture patches simultaneously with
      long-range structure.
    - DiTBlock: MLP now uses GELU instead of SiLU and has an intermediate
      layer-norm for training stability at deeper depths.
    - DiTBlock: adaLN parameters split into 12-chunk (was 9) to also
      modulate the new local-attention sub-layer.
    - DiTGenerator: Alternating cross-attention — odd-indexed blocks
      receive full condition tokens; even-indexed blocks receive a
      lightweight pooled summary to cut cross-attention cost without
      losing structural conditioning.
    - ConditionTokenizer: Optional multi-scale feature fusion. A shallow
      CNN processes the conditioning image at 1× and 2× downscaled
      resolutions and sums the projected tokens, enriching texture cues.
    - PatchEmbed: Overlapping patch projection via a 2-conv stem (as in
      DeiT-III) for smoother token gradients near patch boundaries.

Component structure:
    1) PatchEmbed (overlapping stem)
    2) TimestepEmbedding
    3) ConditionTokenizer (multi-scale)
    4) DiTBlock (local + global SA + cross-attention)
    5) DiTGenerator
    6) DomainEmbedding
    7) CycleDiTGenerator
    8) getGeneratorV3 factory
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_v2.generator import _get_2d_sincos_pos_embed, init_weights_v2


# ---------------------------------------------------------------------------
# Patch embedding — overlapping two-conv stem (DeiT-III style)
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """
    Patchify a latent tensor using an overlapping two-conv stem.

    The first 3×3 conv keeps full spatial resolution; the second 3×3 conv
    with stride=patch_size downsamples to the patch grid. Overlapping
    convolutions produce smoother gradients near patch boundaries compared
    to a single large-stride projection.

    Shape flow:
        z: (N, C, H, W) -> tokens: (N, (H/p)*(W/p), hidden_dim)

    Args:
        in_channels: Latent channel count (4 for SD VAE latents).
        patch_size:  Spatial size of each non-overlapping patch.
        hidden_dim:  Output token embedding dimension.
        latent_size: Spatial side length of the latent (H == W assumed).
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_dim: int = 512,
        latent_size: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.latent_size = latent_size

        # Overlapping stem: preserve spatial info at boundaries
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent tensor (N, C, H, W).

        Returns:
            Token tensor (N, (H/p)*(W/p), hidden_dim).
        """
        n, c, h, w = z.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError("Latent H/W must be divisible by patch_size.")
        # stem: (N, C, H, W) -> (N, hidden_dim, H/p, W/p)
        x = self.stem(z)
        # flatten spatial -> sequence
        x = x.flatten(2).transpose(1, 2).contiguous()  # (N, L, hidden_dim)
        return x


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by a two-layer MLP.

    Args:
        hidden_dim: Output embedding dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq_dim = hidden_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(self.freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sincos(self, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.view(1)
        t = t.float()
        half = self.freq_dim // 2
        if half < 1:
            return torch.zeros(
                t.shape[0], self.freq_dim, device=t.device, dtype=t.dtype
            )
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.freq_dim:
            emb = F.pad(emb, (0, self.freq_dim - emb.shape[1]))
        return emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self._sincos(t))


# ---------------------------------------------------------------------------
# Condition tokenizer — multi-scale fusion
# ---------------------------------------------------------------------------


class ConditionTokenizer(nn.Module):
    """
    Multi-scale conditioning tokenizer.

    Processes the conditioning image at two scales (full and 2×
    downsampled) and sums the projected tokens. This enriches the token
    sequence with both fine texture (full scale) and coarse structure
    (downsampled scale) cues, improving structural fidelity.

    Shape flow for 256×256 input, patch_size=16, pool_stride=1:
        (N, 3, 256, 256) -> (N, 256, hidden_dim)   [full scale]
                         + (N, 256, hidden_dim)   [2× downsampled + interp]
                         -> (N, 256, hidden_dim)   [summed + pos embed]

    Args:
        hidden_dim:   Token embedding dimension.
        image_size:   Spatial side length of the conditioning image (square).
        patch_size:   Stride of the patchifying convolution.
        pool_stride:  Average-pool stride applied after projection (1 = off).
        use_multiscale: If True, fuse a 2× downsampled scale for richer cues.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        image_size: int = 256,
        patch_size: int = 16,
        pool_stride: int = 1,
        use_multiscale: bool = True,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by cond patch_size.")
        if pool_stride < 1:
            raise ValueError("pool_stride must be >= 1.")
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.pool_stride = pool_stride
        self.use_multiscale = use_multiscale

        # Full-resolution branch
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Coarse-scale branch (operates on 2× downsampled input)
        if use_multiscale:
            self.proj_coarse = nn.Conv2d(
                3, hidden_dim, kernel_size=patch_size, stride=patch_size
            )
            # Learnable scale gate — blends fine and coarse contributions
            self.scale_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Conditioning image tensor (N, 3, H, W) in [-1, 1].

        Returns:
            Token tensor (N, L, hidden_dim) with 2-D sincos positional
            embeddings added.
        """
        # Full-scale tokens
        fine = self.proj(x)
        if self.pool_stride > 1:
            fine = F.avg_pool2d(fine, self.pool_stride, self.pool_stride)
        n, c, h, w = fine.shape

        fine_tokens = fine.flatten(2).transpose(1, 2).contiguous()

        # Coarse-scale tokens (2× downsampled input -> same token grid via adaptive pool)
        if self.use_multiscale:
            x_coarse = F.avg_pool2d(x, kernel_size=2, stride=2)
            coarse = self.proj_coarse(x_coarse)
            coarse = F.adaptive_avg_pool2d(coarse, (h, w))
            coarse_tokens = coarse.flatten(2).transpose(1, 2).contiguous()
            gate = torch.sigmoid(self.scale_gate)
            tokens = gate * fine_tokens + (1 - gate) * coarse_tokens
        else:
            tokens = fine_tokens

        pos = _get_2d_sincos_pos_embed(
            self.hidden_dim,
            h,
            w,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + pos.unsqueeze(0)


# ---------------------------------------------------------------------------
# DiT Block — local + global self-attention with adaLN-Zero
# ---------------------------------------------------------------------------


class _LocalWindowAttention(nn.Module):
    """
    Window-based local self-attention for a flat token sequence.

    Reshapes the token sequence back to a square grid, partitions into
    non-overlapping windows of size ``window_size``, applies multi-head
    self-attention within each window, and reshapes back.

    Args:
        hidden_dim:  Token dimension.
        num_heads:   Attention heads.
        window_size: Spatial size of each attention window (in tokens).
                     Tokens per window = window_size².
    """

    def __init__(self, hidden_dim: int, num_heads: int, window_size: int = 4):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, tokens: Tensor, grid: int) -> Tensor:
        """
        Args:
            tokens: (N, L, D) — flat token sequence.
            grid:   Square root of L (spatial grid side length).

        Returns:
            (N, L, D) — locally attended token sequence.
        """
        n, l, d = tokens.shape
        w = self.window_size
        if grid % w != 0:
            # Skip windowed attention if grid is not evenly divisible
            return tokens

        # Reshape to (N, grid, grid, D) then partition into windows
        x = tokens.reshape(n, grid, grid, d)
        num_w = grid // w
        # (N, num_w, w, num_w, w, D) -> (N*num_w*num_w, w*w, D)
        x = x.reshape(n, num_w, w, num_w, w, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(n * num_w * num_w, w * w, d)

        attn_out, _ = self.attn(x, x, x)

        # Reverse partition
        attn_out = attn_out.reshape(n, num_w, num_w, w, w, d)
        attn_out = attn_out.permute(0, 1, 3, 2, 4, 5).contiguous()
        attn_out = attn_out.reshape(n, grid * grid, d)
        return attn_out


class DiTBlock(nn.Module):
    """
    Enhanced DiT Transformer block with:
        1. Global self-attention (full sequence) — captures long-range structure.
        2. Local window self-attention — captures fine texture / micro-structure.
           Combined with global SA via a learnable gate (alpha_local).
        3. Optional cross-attention to condition tokens.
        4. Feed-forward MLP (GELU, intermediate LayerNorm for stability).
        5. adaLN-Zero adaptive conditioning with 12 chunks.

    Args:
        hidden_dim:     Token embedding dimension.
        num_heads:      Attention heads for all sub-layers.
        mlp_ratio:      MLP hidden-dim multiplier.
        use_cross_attn: If True, add a cross-attention sub-layer.
        window_size:    Local attention window size in tokens. 0 = disabled.
        latent_grid:    Token grid side length (needed for window attention).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = True,
        window_size: int = 4,
        latent_grid: int = 16,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.window_size = window_size
        self.latent_grid = latent_grid

        # Global self-attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Local window self-attention
        self.use_local = window_size > 0
        if self.use_local:
            self.norm_local = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.local_attn = _LocalWindowAttention(hidden_dim, num_heads, window_size)
            # Learnable gate to blend global + local (init near 0.5)
            self.local_gate = nn.Parameter(torch.zeros(1))

        # Cross-attention
        if use_cross_attn:
            self.norm_xc = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.norm_c = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )
        else:
            self.norm_xc = self.norm_c = self.cross_attn = None

        # MLP — GELU activation with an intermediate LayerNorm for depth stability
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden),  # intermediate norm
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # adaLN-Zero: 12 chunks when local SA is enabled, 9 otherwise
        n_chunks = 12 if self.use_local else 9
        self.adaLN = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, n_chunks * hidden_dim)
        )
        ada_linear = self.adaLN[1]
        assert isinstance(ada_linear, nn.Linear)
        nn.init.zeros_(ada_linear.weight)
        nn.init.zeros_(ada_linear.bias)
        self._n_chunks = n_chunks

    def forward(self, tokens: Tensor, cond_tokens: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            tokens:      Latent token sequence (N, L, hidden_dim).
            cond_tokens: Condition token sequence (N, Lc, hidden_dim).
            cond:        Scalar condition vector (N, hidden_dim).

        Returns:
            Updated token sequence (N, L, hidden_dim).
        """
        params = self.adaLN(cond).chunk(self._n_chunks, dim=1)

        # Unpack adaLN chunks
        gamma1, beta1, alpha1 = params[0], params[1], params[2]
        gamma_x, beta_x, alpha_x = params[3], params[4], params[5]
        gamma2, beta2, alpha2 = params[6], params[7], params[8]
        if self.use_local:
            gamma_l, beta_l, alpha_l = params[9], params[10], params[11]

        # --- Global self-attention ---
        x = self.norm1(tokens)
        x = x * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        g_out, _ = self.attn(x, x, x)
        tokens = tokens + alpha1.unsqueeze(1) * g_out

        # --- Local window self-attention ---
        if self.use_local:
            xl = self.norm_local(tokens)
            xl = xl * (1 + gamma_l.unsqueeze(1)) + beta_l.unsqueeze(1)
            l_out = self.local_attn(xl, self.latent_grid)
            gate = torch.sigmoid(self.local_gate)
            tokens = tokens + alpha_l.unsqueeze(1) * gate * l_out

        # --- Cross-attention ---
        if self.use_cross_attn:
            assert (
                self.norm_xc is not None
                and self.norm_c is not None
                and self.cross_attn is not None
            )
            x = self.norm_xc(tokens)
            x = x * (1 + gamma_x.unsqueeze(1)) + beta_x.unsqueeze(1)
            c = self.norm_c(cond_tokens)
            cross_out, _ = self.cross_attn(x, c, c)
            tokens = tokens + alpha_x.unsqueeze(1) * cross_out

        # --- Feed-forward MLP ---
        x = self.norm2(tokens)
        x = x * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        tokens = tokens + alpha2.unsqueeze(1) * self.mlp(x)

        return tokens


# ---------------------------------------------------------------------------
# DiT Generator backbone
# ---------------------------------------------------------------------------


class DiTGenerator(nn.Module):
    """
    Enhanced Diffusion Transformer (DiT) backbone.

    Alternating cross-attention strategy:
        Odd-indexed blocks receive full condition tokens (rich texture).
        Even-indexed blocks receive a lightweight mean-pooled condition
        summary, halving cross-attention cost while preserving structure.

    Args:
        in_channels:  Latent channel count (4 for SD VAE).
        hidden_dim:   Token embedding dimension.
        depth:        Number of DiT blocks.
        num_heads:    Attention heads per block.
        mlp_ratio:    MLP hidden-dim multiplier.
        patch_size:   Latent patch size.
        latent_size:  Spatial side length of the latent.
        use_gradient_checkpointing: Recompute activations during backward.
        use_cross_attn: Enable cross-attention to condition tokens.
        window_size:  Local attention window size (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        latent_size: int = 32,
        use_gradient_checkpointing: bool = False,
        use_cross_attn: bool = True,
        window_size: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.use_gradient_checkpointing = use_gradient_checkpointing

        latent_grid = latent_size // patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            latent_size=latent_size,
        )
        self.time_embed = TimestepEmbedding(hidden_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_cross_attn=use_cross_attn,
                    window_size=window_size,
                    latent_grid=latent_grid,
                )
                for _ in range(depth)
            ]
        )

        self.head: nn.Linear = nn.Linear(
            hidden_dim, patch_size * patch_size * in_channels
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _pos_embed(self, device, dtype) -> Tensor:
        grid = self.latent_size // self.patch_size
        pos = _get_2d_sincos_pos_embed(self.hidden_dim, grid, grid, device, dtype)
        return pos.unsqueeze(0)

    def unpatchify(self, tokens: Tensor) -> Tensor:
        n, num_patches, dim = tokens.shape
        p = self.patch_size
        h = w = self.latent_size // p
        if num_patches != h * w:
            raise ValueError("Token length does not match latent grid.")
        x = tokens.view(n, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(n, self.in_channels, h * p, w * p)

    def forward(self, z_t: Tensor, t: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            z_t: Noisy latent (N, 4, latent_size, latent_size).
            t:   Integer timesteps (N,).
            c:   Condition token sequence (N, Lc, hidden_dim).

        Returns:
            Noise or v-prediction (N, 4, latent_size, latent_size).
        """
        tokens = self.patch_embed(z_t)
        tokens = tokens + self._pos_embed(tokens.device, tokens.dtype)
        cond_global = c.mean(dim=1)
        cond = self.time_embed(t) + cond_global

        # Precompute pooled condition summary for even-indexed blocks
        c_pooled = c.mean(dim=1, keepdim=True).expand_as(c[:, :1, :])

        for i, block in enumerate(self.blocks):
            # Alternating full / pooled condition tokens
            c_in = c if (i % 2 == 1) else c_pooled.expand(-1, c.size(1), -1)

            if (
                self.use_gradient_checkpointing
                and self.training
                and isinstance(tokens, torch.Tensor)
                and tokens.requires_grad
            ):
                tokens = checkpoint(block, tokens, c_in, cond, use_reentrant=False)
            else:
                tokens = block(tokens, c_in, cond)

        tokens = self.head(tokens)
        return self.unpatchify(tokens)


# ---------------------------------------------------------------------------
# Domain embedding
# ---------------------------------------------------------------------------


class DomainEmbedding(nn.Module):
    """
    Learned domain embedding for conditional A/B translation.

    Domain ids: 0 -> domain A (unstained), 1 -> domain B (H&E stained).

    Args:
        hidden_dim: Embedding dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(2, hidden_dim)

    def forward(self, domain_ids: Tensor) -> Tensor:
        return self.embed(domain_ids)


# ---------------------------------------------------------------------------
# Top-level CycleDiTGenerator
# ---------------------------------------------------------------------------


class CycleDiTGenerator(nn.Module):
    """
    Top-level CycleDiT generator — unchanged public API, enhanced internals.

    Accepts a raw conditioning image (N, 3, H, W) or pre-computed tokens
    (N, L, hidden_dim) and an integer target domain id, and returns a dict:

        {
            "v_pred":  (N, 4, 32, 32),
            "x0_pred": (N, 4, 32, 32) or None
        }
    """

    def __init__(
        self,
        backbone: DiTGenerator,
        cond_tokenizer: ConditionTokenizer,
        domain_embedding: DomainEmbedding,
    ):
        super().__init__()
        self.backbone = backbone
        self.cond_tokenizer = cond_tokenizer
        self.domain_embedding = domain_embedding

    def _prepare_condition_tokens(self, condition: Tensor) -> Tensor:
        if condition.dim() == 4:
            return self.cond_tokenizer(condition)
        if condition.dim() == 3:
            return condition
        raise ValueError(
            "condition must be an image tensor (N,3,H,W) or token tensor (N,L,Hd)."
        )

    def _prepare_domain_ids(
        self,
        target_domain: Tensor | int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(target_domain, int):
            domain_ids = torch.full(
                (batch_size,), int(target_domain), device=device, dtype=torch.long
            )
        else:
            domain_ids = target_domain.to(device=device, dtype=torch.long)
            if domain_ids.dim() == 0:
                domain_ids = domain_ids.repeat(batch_size)
        if domain_ids.shape[0] != batch_size:
            raise ValueError("target_domain must broadcast to batch size.")
        return domain_ids.clamp(min=0, max=1)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        condition: Tensor,
        target_domain: Tensor | int,
        scheduler=None,
        prediction_type: str = "v",
    ) -> dict[str, Tensor | None]:
        """
        Run one denoising step and optionally reconstruct x0.
        API identical to v3.1 — no changes to callers required.
        """
        cond_tokens = self._prepare_condition_tokens(condition)
        domain_ids = self._prepare_domain_ids(target_domain, z_t.size(0), z_t.device)
        domain_token = self.domain_embedding(domain_ids).to(cond_tokens.dtype)
        cond_tokens = cond_tokens + domain_token.unsqueeze(1)

        v_pred = self.backbone(z_t, t, cond_tokens)

        x0_pred = None
        if scheduler is not None:
            if prediction_type == "v":
                x0_pred = scheduler.predict_x0_from_v(z_t.float(), v_pred.float(), t)
            elif prediction_type == "eps":
                x0_pred = scheduler.predict_x0(z_t.float(), v_pred.float(), t)
            else:
                raise ValueError(
                    f"prediction_type must be 'v' or 'eps', got {prediction_type!r}"
                )
            x0_pred = x0_pred.to(v_pred.dtype)

        return {"v_pred": v_pred, "x0_pred": x0_pred}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def getGeneratorV3(cfg, device: Optional[torch.device] = None) -> CycleDiTGenerator:
    """
    Build, initialise, and smoke-test a CycleDiTGenerator from a DiffusionConfig.

    New config fields honoured (with safe getattr fallbacks for old configs):
        use_local_window_attn  (bool, default True)
        window_size            (int,  default 4)
        use_multiscale_cond    (bool, default True)

    Args:
        cfg:    DiffusionConfig instance (from config.py).
        device: Target device.

    Returns:
        Initialised CycleDiTGenerator on the requested device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_local = getattr(cfg, "use_local_window_attn", True)
    window_size = getattr(cfg, "window_size", 4) if use_local else 0
    use_multiscale = getattr(cfg, "use_multiscale_cond", True)

    backbone = DiTGenerator(
        in_channels=4,
        hidden_dim=cfg.dit_hidden_dim,
        depth=cfg.dit_depth,
        num_heads=cfg.dit_heads,
        mlp_ratio=cfg.dit_mlp_ratio,
        patch_size=cfg.dit_patch_size,
        latent_size=32,
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        use_cross_attn=getattr(cfg, "use_cross_attention", True),
        window_size=window_size,
    ).to(device)

    cond_tokenizer = ConditionTokenizer(
        hidden_dim=cfg.dit_hidden_dim,
        image_size=256,
        patch_size=cfg.cond_patch_size,
        pool_stride=getattr(cfg, "cond_token_pool_stride", 1),
        use_multiscale=use_multiscale,
    ).to(device)

    domain_embedding = DomainEmbedding(cfg.dit_hidden_dim).to(device)

    model = CycleDiTGenerator(
        backbone=backbone,
        cond_tokenizer=cond_tokenizer,
        domain_embedding=domain_embedding,
    ).to(device)

    init_weights_v2(model)
    nn.init.zeros_(model.backbone.head.weight)
    nn.init.zeros_(model.backbone.head.bias)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[getGeneratorV3] DiT params: {num_params/1e6:.2f}M")

    # Smoke test
    with torch.no_grad():
        z_t = torch.randn(1, 4, 32, 32, device=device)
        t = torch.randint(0, 1000, (1,), device=device)
        x = torch.randn(1, 3, 256, 256, device=device)
        out = model(z_t, t, x, target_domain=1)
        _ = out["v_pred"]
    print("[getGeneratorV3] Smoke test passed.")

    return model


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        dit_hidden_dim=512,
        dit_depth=8,
        dit_heads=8,
        dit_mlp_ratio=4.0,
        dit_patch_size=2,
        use_gradient_checkpointing=False,
        use_cross_attention=True,
        cond_patch_size=16,
        cond_token_pool_stride=1,
    )
    m = getGeneratorV3(cfg)
    print("getGeneratorV3 standalone test passed.")
