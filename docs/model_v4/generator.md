# model_v4/generator.py — v4.2

Source of truth: `../../model_v4/generator.py`

Provides two generator variants for the v4 CUT + Transformer pipeline and a
shared factory function.

## Public Components

1. `SEBlock`
2. `SpatialSelfAttention`
3. `ResnetBlock` (SE-gated)
4. `ResnetGenerator` (SE + bottleneck attention)
5. `EnhancedTransformerBlock` (pre-norm + DW-Conv)
6. `TextureRefinementHead`
7. `TransformerGeneratorV4`
8. `init_weights_v4`
9. `getGeneratorV4`

---

## SEBlock

Squeeze-and-Excitation channel attention gate.

Flow:

```
(N, C, H, W)
  -> AdaptiveAvgPool2d(1) -> Flatten
  -> Linear(C, C//reduction) -> ReLU
  -> Linear(C//reduction, C) -> Sigmoid
  -> reshape to (N, C, 1, 1) -> multiply input
```

Args: `channels`, `reduction` (default 8, min bottleneck 4).

---

## SpatialSelfAttention

Lightweight spatial self-attention for CNN feature maps at the ResNet bottleneck.

Flow:

```
(N, C, H, W)
  -> GroupNorm -> flatten to (N, H*W, C)
  -> MultiheadAttention(heads=4)
  -> reshape back + residual
  -> (N, C, H, W)
```

Used once after all ResNet blocks to capture long-range structural relationships.

---

## ResnetBlock

Enhanced residual block with SE channel attention.

Flow:

```
x -> ReflectionPad + Conv3×3 + IN + ReLU [+ Dropout] + ReflectionPad + Conv3×3 + IN
  -> SEBlock  (channel recalibration)
  -> + x  (identity residual)
```

Args: `channels`, `dropout`, `se_reduction` (0 = disable SE).

---

## ResnetGenerator

Enhanced ResNet encoder–residual–decoder generator.

Architecture:

```
Input
  -> in_conv:  ReflectionPad + Conv7×7 + IN + ReLU          (c1 channels)
  -> down1:    Conv3×3 stride-2 + IN + ReLU                  (c2 channels)
  -> down2:    Conv3×3 stride-2 + IN + ReLU                  (c3 channels)
  -> res_blocks: num_res_blocks × ResnetBlock (SE-gated)
  -> bottleneck_attn: SpatialSelfAttention                   ← NEW in v4.2
  -> up1:      Upsample(×2) + ReflectionPad + Conv3×3 + IN + ReLU
  -> up2:      Upsample(×2) + ReflectionPad + Conv3×3 + IN + ReLU
  -> out_conv: ReflectionPad + Conv7×7 + Tanh
```

Feature extraction for PatchNCE:

- `_encode(x)` → `(f0, f1, f2, f3)` where `f3` includes bottleneck attention
- `encode_features(x, nce_layers)` → selected feature maps
- `forward(x, return_features=True, nce_layers=...)` → `(output, feature_list)`

Args: `input_nc`, `output_nc`, `base_channels`, `num_res_blocks`, `dropout`, `se_reduction`.

---

## EnhancedTransformerBlock

Transformer block with pre-norm and a parallel depth-wise CNN branch.

Sub-layers:

1. Pre-LayerNorm → MultiheadAttention (global SA)
2. Pre-LayerNorm → DW-Conv1d branch (local texture, `kernel=3, groups=dim`)
   — combined with SA via `sigmoid(local_gate)` (gate init = 0 → pure attn initially)
3. Pre-LayerNorm → MLP (GELU, expansion `mlp_ratio`)

Residual: `x = x + attn_out + gate * dw_out` then `x = x + mlp(norm(x))`.

Args: `dim`, `num_heads`, `mlp_ratio`, `dropout`.

---

## TextureRefinementHead

Lightweight DW-Conv + PW-Conv head placed between the last up-block and the
output convolution.

Flow:

```
x -> ReflectionPad + DWConv3×3 + IN + ReLU
  -> PWConv1×1 + IN + ReLU
  -> + x  (residual)
```

Sharpens high-frequency texture detail with minimal parameter overhead.

---

## TransformerGeneratorV4

Enhanced Transformer-encoder + CNN-decoder generator.

Flow:

```
Input
  -> PatchEmbed -> tokens (B, N, embed_dim)
  -> + 2-D sincos positional embeddings
  -> depth × EnhancedTransformerBlock (pre-norm + DW-Conv)
  -> LayerNorm
  -> reshape to (B, embed_dim, H/p, W/p)
  -> proj: Conv1×1 -> (B, base_channels*4, H/p, W/p)
  -> log2(patch_size) × up_block (Upsample + ReflectionPad + Conv3×3 + IN + ReLU)
  -> TextureRefinementHead
  -> out_conv: ReflectionPad + Conv7×7 + Tanh
```

Feature extraction methods:

- `encode_features(x, nce_layers)` — block-level token maps (consistent with `forward`)
- `encode_features_multiscale(x, nce_layers)` — block maps + decoder up-block maps for richer NCE supervision
- `forward(x, return_features=True, nce_layers=...)` → `(output, block_feature_maps)`

Args: `input_nc`, `output_nc`, `image_size`, `patch_size`, `embed_dim`, `depth`,
`num_heads`, `mlp_ratio`, `dropout`, `base_channels`, `use_gradient_checkpointing`.

---

## init_weights_v4

Weight initialisation policy:

- `Conv2d` / `ConvTranspose2d`: Normal(0, 0.02)
- `InstanceNorm2d` scale: Normal(1, 0.02), bias = 0
- `Linear`: Xavier uniform, bias = 0

---

## getGeneratorV4

Build, initialise, and smoke-test a v4 generator.

Selects `TransformerGeneratorV4` when `use_transformer_encoder=True`, otherwise `ResnetGenerator`.

New arg in v4.2:

- `se_reduction` (int, default 8) — SE bottleneck reduction for `ResnetGenerator`; set to 0 to disable SE gates.

All other args unchanged from v4.1 (API-compatible).

Smoke test: forward pass with `(1, input_nc, 256, 256)`, prints output shape.

Returns: initialised `ResnetGenerator` or `TransformerGeneratorV4`.
