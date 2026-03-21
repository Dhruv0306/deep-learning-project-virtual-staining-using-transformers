# `uvcgan_v2_generator.py` — v2 Generator

**Model:** True UVCGAN v2  
**Role:** Defines `ViTUNetGeneratorV2`, used for both translation directions — `G_AB` (unstained→stained) and `G_BA` (stained→unstained). A ground-up redesign of the v1 generator with residual encoder blocks, LayerScale Transformer blocks, 1×1 skip merges, cross-domain skip fusion, and optional gradient checkpointing.

---

## Architecture Overview

```
Input Image (N, 3, 256, 256)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                          ENCODER                             │
│                                                              │
│  enc_in   ReflPad(3)→Conv(3→64,k=7)→IN→ReLU                 │
│           → e0  (N,  64, 256, 256)                           │
│  down1    DownBlock(64→128)                                  │
│  res_enc1 2×ResidualConvBlock(128) → e1  (N, 128, 128, 128) │
│  down2    DownBlock(128→256)                                 │
│  res_enc2 2×ResidualConvBlock(256) → e2  (N, 256,  64,  64) │
│  down3    DownBlock(256→512)                                 │
│  res_enc3 2×ResidualConvBlock(512) → e3  (N, 512,  32,  32) │
│  down4    DownBlock(512→512)       → (N, 512,  16,  16)     │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                        BOTTLENECK                            │
│                                                              │
│  res_bot  ResidualConvBlock(512)       (N, 512, 16, 16)      │
│  vit      PixelwiseViTV2(dim=512)                            │
│    ├─ flatten:  (N,512,16,16) → (N, 256, 512) tokens        │
│    ├─ + 2D sincos positional embedding                       │
│    ├─ × vit_depth LayerScaleTransformerBlocks                │
│    │    (optional: each block gradient-checkpointed)         │
│    └─ reshape: (N, 256, 512) → (N, 512, 16, 16)             │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  [if use_cross_domain=True]
  CrossDomainFusion applied to e0, e1, e2, e3
  (each fuses with corresponding skip from paired generator)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                         DECODER                              │
│                                                              │
│  up1    UpBlock(512→512)                                     │
│  dec1   cat([up1, e3])→Conv1×1(1024→512)→IN→ReLU            │
│  up2    UpBlock(512→256)                                     │
│  dec2   cat([up2, e2])→Conv1×1(512→256) →IN→ReLU            │
│  up3    UpBlock(256→128)                                     │
│  dec3   cat([up3, e1])→Conv1×1(256→128) →IN→ReLU            │
│  up4    UpBlock(128→64)                                      │
│  dec4   cat([up4, e0])→Conv1×1(128→64)  →IN→ReLU            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
  ReflectionPad2d(3) → Conv2d(64→3, k=7) → Tanh
        │
        ▼
Output Image (N, 3, 256, 256)   range [−1, 1]
```

---

## Cross-Domain Skip Fusion

When `use_cross_domain=True`, both generators run simultaneously and each fuses its own skip features with the other generator's skip features at matching spatial levels. This is the defining feature of true UVCGAN.

```
 G_AB processing real_A           G_BA processing real_B
 ──────────────────────           ──────────────────────
 encode(real_A)                   encode(real_B)
   → e0_AB, e1_AB,                  → e0_BA, e1_BA,
     e2_AB, e3_AB, b_AB               e2_BA, e3_BA, b_BA

 fuse1(e3_AB, e3_BA.detach())     fuse1(e3_BA, e3_AB.detach())
 fuse2(e2_AB, e2_BA.detach())     fuse2(e2_BA, e2_AB.detach())
 fuse3(e1_AB, e1_BA.detach())     fuse3(e1_BA, e1_AB.detach())
 fuse4(e0_AB, e0_BA.detach())     fuse4(e0_BA, e0_AB.detach())

 decode(b_AB, fused e0..e3)       decode(b_BA, fused e0..e3)
   → fake_B                         → fake_A
```

The `.detach()` on the other generator's features prevents their gradients from flowing through the fusion layer into the other generator's backward pass, keeping both generators' training independent.

---

## Classes

### `LayerScaleTransformerBlock`

A Transformer block where each residual branch is scaled by a **learnable per-channel vector** (not a scalar) initialised to a small value. This is an improvement over v1's ReZero: the scale is per-channel, giving the network more expressive control over which feature dimensions contribute to each residual.

| Parameter | Default | Description |
|---|---|---|
| `dim` | — | Token embedding dimension (= bottleneck channels) |
| `num_heads` | 8 | Number of parallel attention heads |
| `mlp_ratio` | 4.0 | MLP hidden size = `dim × mlp_ratio` |
| `dropout` | 0.0 | Dropout probability |
| `init_values` | 1e-4 | Initial value for LayerScale parameters. Near-zero means block starts as near-identity. |

| Attribute | Type | Description |
|---|---|---|
| `norm1` | `LayerNorm` | Normalises tokens before attention |
| `attn` | `MultiheadAttention` | Self-attention across all spatial tokens |
| `norm2` | `LayerNorm` | Normalises tokens before MLP |
| `mlp` | `Sequential` | `Linear(dim→dim×ratio) → GELU → Dropout → Linear(→dim) → Dropout` |
| `gamma_attn` | `nn.Parameter` shape `(dim,)` | Per-channel scale on attention residual, init=`init_values` |
| `gamma_ffn` | `nn.Parameter` shape `(dim,)` | Per-channel scale on MLP residual, init=`init_values` |

**Data flow:**
```
x ──► norm1 ──► attn ──► × gamma_attn ──► + x ──► norm2 ──► mlp ──► × gamma_ffn ──► + x ──► output
```

**LayerScale vs ReZero (v1):** ReZero uses one scalar per block. LayerScale uses one scalar per feature channel per block, allowing different channels to grow at different rates. This is more stable for deeper ViT stacks (vit_depth > 2).

---

### `PixelwiseViTV2`

Same reshape logic as v1's `PixelwiseViT` but uses `LayerScaleTransformerBlock` and adds optional gradient checkpointing per block.

| Parameter | Default | Description |
|---|---|---|
| `dim` | — | Bottleneck channel dimension |
| `depth` | 4 | Number of Transformer blocks |
| `num_heads` | 8 | Attention heads per block |
| `mlp_ratio` | 4.0 | MLP expansion ratio |
| `dropout` | 0.0 | Dropout probability |
| `init_values` | 1e-4 | LayerScale initialisation value |
| `use_gradient_checkpointing` | `False` | If `True`, wraps each block with `torch.utils.checkpoint`. Recomputes activations during backward instead of storing them. Saves ~1–2 GB VRAM at ~20% slower backward. Recommended for 8 GB GPUs. |

**Gradient checkpointing detail:** When enabled and `tokens.requires_grad=True` (training mode), each block call becomes:
```python
tokens = grad_checkpoint(block, tokens, use_reentrant=False)
```
`use_reentrant=False` is the modern API, compatible with `torch.compile` and nested checkpointing. The result is cast to `torch.Tensor` via `tcast` to satisfy Pylance's type checker (since `grad_checkpoint` is typed as returning `Any | None` in older stubs).

---

### `ResidualConvBlock`

A two-convolution residual block. The output is `x + conv_path(x)` — the block only learns the residual correction, not the full mapping. This is easier to optimise and allows gradient flow through the identity shortcut.

| Parameter | Default | Description |
|---|---|---|
| `channels` | — | Input and output channels (same — residual blocks preserve channel count) |
| `dropout` | 0.0 | Optional dropout inserted between the two convolutions |

**Structure:**
```
x ──► ReflPad(1) ──► Conv(k=3,bias=False) ──► IN ──► ReLU
    ──► [Dropout] ──► ReflPad(1) ──► Conv(k=3,bias=False) ──► IN
    ──► + x ──► output
```

**Why reflection padding?** Zero padding introduces artificial boundary artefacts because the network sees a hard edge that doesn't exist in natural images. Reflection padding mirrors the image content at the border, producing smoother outputs near the image edges.

---

### `DownBlock`

Strided 4×4 convolution (stride=2) + InstanceNorm + ReLU. Halves `(H, W) → (H/2, W/2)`.

| Parameter | Description |
|---|---|
| `in_channels` | Input channels |
| `out_channels` | Output channels |

---

### `UpBlock`

Nearest-neighbour 2× upsample + ReflectionPad(1) + 3×3 Conv + InstanceNorm + ReLU. Doubles `(H, W) → (H×2, W×2)` without checkerboard artefacts.

| Parameter | Description |
|---|---|
| `in_channels` | Input channels |
| `out_channels` | Output channels |

---

### `CrossDomainFusion`

Fuses encoder skip features from both generators at a matching spatial level.

| Parameter | Description |
|---|---|
| `channels` | Number of channels on this skip level |

**`forward(feat_self, feat_other)`**

| Parameter | Description |
|---|---|
| `feat_self` | Skip features from this generator `(N, C, H, W)` |
| `feat_other` | Skip features from the paired generator — detached inside `forward` |

```
input:  cat([feat_self, feat_other.detach()], dim=1)   → (N, 2C, H, W)
output: Conv1×1(2C→C) → IN → ReLU                      → (N, C, H, W)
```

---

### `ViTUNetGeneratorV2`

Top-level v2 generator module.

| Constructor Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input channels |
| `output_nc` | 3 | Output channels |
| `base_channels` | 64 | Channels at encoder level 1 (128/256/512 at deeper levels) |
| `vit_depth` | 4 | ViT Transformer block count |
| `vit_heads` | 8 | Attention heads per block |
| `vit_mlp_ratio` | 4.0 | MLP expansion ratio |
| `vit_dropout` | 0.0 | ViT dropout probability |
| `layerscale_init` | 1e-4 | Initial value for LayerScale parameters |
| `use_cross_domain` | `True` | Allocate `CrossDomainFusion` layers for all 4 skip levels |
| `use_gradient_checkpointing` | `False` | Enable ViT-block gradient checkpointing |

**Methods:**

| Method | Description |
|---|---|
| `encode(x)` | Runs encoder + bottleneck. Returns `(e0, e1, e2, e3, bottleneck)`. |
| `decode(b, e0, e1, e2, e3)` | Runs decoder. Returns output image tensor. |
| `forward(x)` | Standard path: `encode` then `decode`. Used for identity loss, cycle reconstruction, and inference. |
| `forward_with_cross_domain(x, other_skips)` | Forward with cross-domain fusion. `other_skips` = `(oe0, oe1, oe2, oe3)` from paired generator's `get_skip_features`. |
| `get_skip_features(x)` | Runs encoder only, returns `(e0, e1, e2, e3)` without decoding. Used to obtain one generator's skips to pass to the other. |
| `_encode_segment(x)` | Internal helper. Runs just the conv encoder (without bottleneck). Used as the unit for gradient checkpointing in `encode`. |

---

## Functions

### `_get_1d_sincos_pos_embed(embed_dim, pos)`

Builds 1D sine-cosine positional embeddings. Returns `(N, embed_dim)`. See `generator.py` docs for formula details.

### `_get_2d_sincos_pos_embed(embed_dim, height, width, device, dtype)`

Builds 2D positional embeddings by combining independent row and column embeddings. Returns `(H×W, embed_dim)`.

### `init_weights_v2(net)`

Improved weight initialisation matched to each layer type:

| Module type | Initialisation | Rationale |
|---|---|---|
| `Conv2d`, `ConvTranspose2d` | Kaiming normal, `fan_out`, nonlinearity=`relu` | Correct variance for ReLU networks |
| `Linear` | Xavier uniform, bias=0 | Correct variance for attention/MLP |
| `InstanceNorm2d`, `LayerNorm` | Weight=1, bias=0 | Standard normalisation init |

Kaiming normal is better than the `N(0, 0.02)` used in v1 — it scales the initialisation variance based on the number of output connections and the activation function.

### `getGeneratorsV2(...)`

Factory function. Creates two `ViTUNetGeneratorV2` instances, applies `init_weights_v2`, runs a smoke-test, and returns them.

| Parameter | Default | Description |
|---|---|---|
| `base_channels` | 64 | Feature channels at the first encoder level |
| `vit_depth` | 4 | ViT depth |
| `vit_heads` | 8 | Attention heads |
| `vit_mlp_ratio` | 4.0 | MLP ratio |
| `vit_dropout` | 0.0 | Dropout |
| `layerscale_init` | 1e-4 | LayerScale init |
| `use_cross_domain` | `True` | Enable cross-domain fusion |
| `use_gradient_checkpointing` | `False` | Enable ViT gradient checkpointing |

**Returns:** `(G_AB, G_BA)` — both on the available device.
