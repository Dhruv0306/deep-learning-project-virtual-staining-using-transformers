"""
DiT generator components for the v3 latent diffusion pipeline.

Component structure:
    1) PatchEmbed
    2) TimestepEmbedding
    3) ConditionTokenizer
    4) DiTBlock (adaLN-Zero + cross-attention)
    5) DiTGenerator
    6) getGeneratorV3 factory

Core tensor flow in DiTGenerator.forward:
    z_t:(N,4,32,32) -> tokens:(N,L,Hd) -> blocks ->
    noise prediction latents:(N,4,32,32)
where L=(32/patch_size)^2 and Hd=hidden_dim.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model_v2.generator import _get_2d_sincos_pos_embed, init_weights_v2


class PatchEmbed(nn.Module):
    """
    Patchify latent and project to token embeddings.

    Shape flow:
        input  z: (N, C, H, W)
        output t: (N, (H/p)*(W/p), hidden_dim)
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
        self.proj = nn.Linear(in_channels * patch_size * patch_size, hidden_dim)

    def forward(self, z: Tensor) -> Tensor:
        n, c, h, w = z.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError("Latent H/W must be divisible by patch_size.")
        z = z.view(n, c, h // p, p, w // p, p)
        z = z.permute(0, 2, 4, 1, 3, 5).contiguous()
        z = z.view(n, (h // p) * (w // p), c * p * p)
        return self.proj(z)


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding followed by an MLP.
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
            emb = torch.nn.functional.pad(emb, (0, self.freq_dim - emb.shape[1]))
        return emb

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self._sincos(t))


class ConditionTokenizer(nn.Module):
    """
    Patchify conditioning image and project to token embeddings.

    Shape flow for 256x256 input and patch_size=16:
        (N,3,256,256) -> (N,hidden_dim,16,16) -> (N,256,hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        image_size: int = 256,
        patch_size: int = 16,
        pool_stride: int = 1,
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
        self.proj = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        if self.pool_stride > 1:
            h, w = x.shape[-2:]
            if h % self.pool_stride != 0 or w % self.pool_stride != 0:
                raise ValueError(
                    "Condition token grid must be divisible by pool_stride."
                )
            x = F.avg_pool2d(x, kernel_size=self.pool_stride, stride=self.pool_stride)
        n, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        pos = _get_2d_sincos_pos_embed(
            self.hidden_dim,
            h,
            w,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens + pos.unsqueeze(0)


class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        if use_cross_attn:
            self.norm_xc = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.norm_c = nn.LayerNorm(hidden_dim, elementwise_affine=False)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, batch_first=True
            )
        else:
            self.norm_xc = None
            self.norm_c = None
            self.cross_attn = None
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim))
        ada_linear = self.adaLN[1]
        if isinstance(ada_linear, nn.Linear):
            nn.init.zeros_(ada_linear.weight)
            nn.init.zeros_(ada_linear.bias)

    def forward(self, tokens: Tensor, cond_tokens: Tensor, cond: Tensor) -> Tensor:
        params = self.adaLN(cond)
        (
            gamma1,
            beta1,
            alpha1,
            gamma_x,
            beta_x,
            alpha_x,
            gamma2,
            beta2,
            alpha2,
        ) = params.chunk(9, dim=1)

        x = self.norm1(tokens)
        x = x * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        tokens = tokens + alpha1.unsqueeze(1) * attn_out

        if self.use_cross_attn:
            assert self.norm_xc is not None
            assert self.norm_c is not None
            assert self.cross_attn is not None
            x = self.norm_xc(tokens)
            x = x * (1 + gamma_x.unsqueeze(1)) + beta_x.unsqueeze(1)
            c = self.norm_c(cond_tokens)
            cross_out, _ = self.cross_attn(x, c, c)
            tokens = tokens + alpha_x.unsqueeze(1) * cross_out

        x = self.norm2(tokens)
        x = x * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        mlp_out = self.mlp(x)
        tokens = tokens + alpha2.unsqueeze(1) * mlp_out
        return tokens


class DiTGenerator(nn.Module):
    """
    Diffusion Transformer backbone.

    Forward inputs:
        z_t: (N,4,32,32)
        t:   (N,)
        c:   (N,Lc,hidden_dim)

    Forward output:
        eps_pred: (N,4,32,32)
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_cross_attn = use_cross_attn

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
        tokens = self.patch_embed(z_t)
        tokens = tokens + self._pos_embed(tokens.device, tokens.dtype)
        cond_global = c.mean(dim=1)
        cond = self.time_embed(t) + cond_global
        for block in self.blocks:
            if (
                self.use_gradient_checkpointing
                and self.training
                and isinstance(tokens, torch.Tensor)
                and tokens.requires_grad
            ):
                tokens = checkpoint(block, tokens, c, cond, use_reentrant=False)
            else:
                tokens = block(tokens, c, cond)
        tokens = self.head(tokens)
        return self.unpatchify(tokens)


class DomainEmbedding(nn.Module):
    """
    Learned domain embedding for conditional A/B translation.

    Domain ids:
        0 -> domain A (Un-Stained)
        1 -> domain B (Stained)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(2, hidden_dim)

    def forward(self, domain_ids: Tensor) -> Tensor:
        return self.embed(domain_ids)


class CycleDiTGenerator(nn.Module):
    """
    Phase-0 CycleDiT wrapper:
      - internal condition tokenizer
      - learned domain embedding
      - DiT backbone

    Forward contract returns:
      {
        "v_pred": (N,4,32,32),
        "x0_pred": (N,4,32,32) or None
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
        # Condition can be a raw image (N,3,H,W) or precomputed tokens (N,L,Hd).
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
        cond_tokens = self._prepare_condition_tokens(condition)
        domain_ids = self._prepare_domain_ids(target_domain, z_t.size(0), z_t.device)

        # Input-level domain conditioning (v3.1 can extend to per-block hooks).
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


def getGeneratorV3(cfg, device: Optional[torch.device] = None) -> CycleDiTGenerator:
    """
    Factory for DiTGenerator based on DiffusionConfig.

    Returns a model configured for latent size 32x32 and channel count 4,
    then runs a smoke test on tensors shaped:
        z_t:(1,4,32,32), t:(1,), c:(1,dit_hidden_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ).to(device)

    cond_tokenizer = ConditionTokenizer(
        hidden_dim=cfg.dit_hidden_dim,
        image_size=256,
        patch_size=cfg.cond_patch_size,
        pool_stride=getattr(cfg, "cond_token_pool_stride", 1),
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

    return model
