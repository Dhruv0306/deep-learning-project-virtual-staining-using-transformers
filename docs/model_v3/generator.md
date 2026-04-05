# model_v3/generator.py — v3.2 DiT Generator

Source of truth: `../../model_v3/generator.py`

This module defines the CycleDiT generator stack used by the v3 diffusion pipeline.

## Public Components

1. `PatchEmbed` — overlapping two-conv stem
2. `TimestepEmbedding`
3. `ConditionTokenizer` — multi-scale fusion
4. `_LocalWindowAttention` — window-partitioned local SA (internal)
5. `DiTBlock` — local + global SA, cross-attention, adaLN-Zero (12-chunk)
6. `DiTGenerator` — alternating cross-attention backbone
7. `DomainEmbedding`
8. `CycleDiTGenerator`
9. `getGeneratorV3`

---

## PatchEmbed

Overlapping two-conv stem (DeiT-III style) replacing the single large-stride projection.

Shape flow:

```
(N, C, H, W)
  -> Conv2d(C, hidden//2, 3×3, stride=1) + GroupNorm + SiLU   # full-res
  -> Conv2d(hidden//2, hidden, patch_size×patch_size, stride=patch_size)  # downsample
  -> flatten + transpose
  -> (N, (H/p)*(W/p), hidden_dim)
```

Benefit: smoother gradients near patch boundaries compared to a single strided projection.

Args:

- `in_channels` — latent channels (4 for SD VAE)
- `patch_size` — spatial patch size
- `hidden_dim` — output token dimension
- `latent_size` — spatial side length (H == W assumed)

---

## TimestepEmbedding

Sinusoidal timestep features passed through a two-layer SiLU MLP.

Output: `(N, hidden_dim)`

---

## ConditionTokenizer

Multi-scale conditioning tokenizer fusing full-resolution and 2× downsampled branches.

Shape flow (256×256 input, patch_size=16):

```
full-scale:   (N, 3, 256, 256) -> Conv2d(stride=16) -> (N, hidden, 16, 16) -> (N, 256, hidden)
coarse-scale: avg_pool2d(×2) -> Conv2d(stride=16) -> adaptive_avg_pool -> (N, 256, hidden)
fused:        gate * fine + (1-gate) * coarse  [gate = sigmoid(learnable scalar)]
+ 2-D sincos positional embeddings
-> (N, 256, hidden_dim)
```

Args:

- `hidden_dim`, `image_size`, `patch_size`, `pool_stride`
- `use_multiscale` — if `False`, only the full-scale branch is used

---

## _LocalWindowAttention

Window-based local self-attention on a flat token sequence.

1. Reshape tokens to `(N, grid, grid, D)`.
2. Partition into non-overlapping `window_size × window_size` windows.
3. Apply `nn.MultiheadAttention` within each window.
4. Reverse partition back to `(N, L, D)`.

Skipped silently when `grid % window_size != 0`.

---

## DiTBlock

Enhanced Transformer block with four sub-layers and 12-chunk adaLN-Zero.

Sub-layers:

| # | Sub-layer | adaLN chunks |
|---|---|---|
| 1 | Global self-attention (full sequence) | γ1, β1, α1 |
| 2 | Local window self-attention (gated) | γ_l, β_l, α_l |
| 3 | Cross-attention to condition tokens (optional) | γ_x, β_x, α_x |
| 4 | Feed-forward MLP (GELU + intermediate LayerNorm) | γ2, β2, α2 |

Key details:

- Local SA combined with global SA via `sigmoid(local_gate)` — gate initialised to 0.
- MLP uses GELU (was SiLU) and an intermediate `LayerNorm` for depth stability.
- adaLN uses 12 chunks when local SA is enabled, 9 otherwise.
- adaLN linear layer zero-initialised so blocks start as identity.

Args:

- `hidden_dim`, `num_heads`, `mlp_ratio`
- `use_cross_attn` — enable cross-attention sub-layer
- `window_size` — local window size in tokens (0 = disabled)
- `latent_grid` — token grid side length (needed for window partitioning)

---

## DiTGenerator

DiT backbone with alternating cross-attention strategy.

Alternating strategy:

- Odd-indexed blocks receive full condition tokens (rich texture).
- Even-indexed blocks receive mean-pooled condition summary (lower cost).

Forward inputs:

- `z_t` — noisy latent `(N, 4, latent_size, latent_size)`
- `t` — integer timesteps `(N,)`
- `c` — condition tokens `(N, Lc, hidden_dim)`

Forward flow:

1. `PatchEmbed` → tokens `(N, L, hidden_dim)`
2. Add 2-D sincos positional embeddings
3. `TimestepEmbedding(t)` + `c.mean(dim=1)` → conditioning vector
4. Precompute pooled condition summary for even-indexed blocks
5. Run `depth` DiTBlocks with alternating full / pooled condition tokens
6. Linear head → unpatchify → `(N, 4, latent_size, latent_size)`

Output: noise or v-prediction `(N, 4, latent_size, latent_size)`

Args:

- `in_channels`, `hidden_dim`, `depth`, `num_heads`, `mlp_ratio`
- `patch_size`, `latent_size`
- `use_gradient_checkpointing` — recompute ViT activations during backward
- `use_cross_attn` — enable cross-attention in all blocks
- `window_size` — local attention window size (0 = disabled)

---

## DomainEmbedding

Learned 2-entry embedding added to condition tokens to bias translation direction.

- Domain `0` → unstained (domain A)
- Domain `1` → H&E stained (domain B)

---

## CycleDiTGenerator

Top-level wrapper. Public API unchanged from v3.1.

Inputs:

- `z_t` — noisy latent
- `t` — timestep
- `condition` — raw image `(N, 3, H, W)` or pre-computed tokens `(N, L, Hd)`
- `target_domain` — int or tensor
- `scheduler` — optional diffusion scheduler for x0 reconstruction
- `prediction_type` — `"v"` or `"eps"`

Behavior:

1. Tokenize raw conditioning image if needed.
2. Add domain embedding to condition tokens.
3. Call DiT backbone.
4. Optionally reconstruct `x0_pred` via scheduler.

Output: `{"v_pred": (N,4,32,32), "x0_pred": (N,4,32,32) | None}`

---

## getGeneratorV3

Builds, initialises, and smoke-tests a `CycleDiTGenerator` from a `DiffusionConfig`.

New config fields (with safe `getattr` fallbacks for old configs):

| Field | Default | Purpose |
|---|---|---|
| `use_local_window_attn` | `True` | Enable local window SA in DiTBlocks |
| `window_size` | `4` | Local attention window size in tokens |
| `use_multiscale_cond` | `True` | Multi-scale fusion in ConditionTokenizer |

Existing config fields:

- `dit_hidden_dim`, `dit_depth`, `dit_heads`, `dit_mlp_ratio`, `dit_patch_size`
- `cond_patch_size`, `cond_token_pool_stride`
- `use_cross_attention`, `use_gradient_checkpointing`

Smoke test: forward pass with `z_t (1,4,32,32)`, `t (1,)`, `x (1,3,256,256)`.

Returns: initialised `CycleDiTGenerator` on the requested device.
