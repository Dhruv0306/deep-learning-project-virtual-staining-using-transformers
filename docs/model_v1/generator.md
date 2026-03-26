# `model_v1/generator.py` — v1 Generator

Source of truth: `../../model_v1/generator.py`

**Model:** Hybrid UVCGAN + CycleGAN (v1)  
**Role:** Defines the generator used in both translation directions — `G_AB` (unstained→stained) and `G_BA` (stained→unstained).

---

## Architecture Overview

The generator is a **U-Net** backbone with a **PixelwiseViT bottleneck**. The U-Net provides spatially precise skip connections at each resolution level; the ViT processes the compressed bottleneck as a flat sequence of spatial tokens, enabling global long-range attention across the full image.

```
Input Image (N, 3, 256, 256)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                        ENCODER                           │
│                                                          │
│  enc1   ConvBlock(3→64)         → e1  (N, 64,  256, 256)│
│  down1  DownsampleBlock(64→128) → d1  (N, 128, 128, 128)│
│  enc2   ConvBlock(128→128)      → e2  (N, 128, 128, 128)│
│  down2  DownsampleBlock(128→256)→ d2  (N, 256,  64,  64)│
│  enc3   ConvBlock(256→256)      → e3  (N, 256,  64,  64)│
│  down3  DownsampleBlock(256→512)→ d3  (N, 512,  32,  32)│
│  enc4   ConvBlock(512→512)      → e4  (N, 512,  32,  32)│
│  down4  DownsampleBlock(512→512)→ d4  (N, 512,  16,  16)│
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                      BOTTLENECK                          │
│                                                          │
│  bottleneck  ConvBlock(512→512)    (N, 512, 16, 16)      │
│  vit         PixelwiseViT(dim=512)                       │
│    ├─ flatten:  (N,512,16,16) → (N, 256, 512) tokens    │
│    ├─ + 2D sincos positional embedding                   │
│    ├─ × vit_depth ReZeroTransformerBlocks                │
│    └─ reshape:  (N, 256, 512) → (N, 512, 16, 16)        │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                       DECODER                            │
│                                                          │
│  up1  UpsampleBlock(512→512)                             │
│  dec1 ConvBlock(512+512→512)  ◄── skip from e4           │
│  up2  UpsampleBlock(512→256)                             │
│  dec2 ConvBlock(256+256→256)  ◄── skip from e3           │
│  up3  UpsampleBlock(256→128)                             │
│  dec3 ConvBlock(128+128→128)  ◄── skip from e2           │
│  up4  UpsampleBlock(128→64)                              │
│  dec4 ConvBlock(64+64→64)     ◄── skip from e1           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
  ReflectionPad2d(3) → Conv2d(64→3, k=7) → Tanh
        │
        ▼
Output Image (N, 3, 256, 256)   range [−1, 1]
```

---

## Classes

### `ReZeroTransformerBlock`

A standard Transformer block where each residual branch is gated by a **learnable scalar initialised to 0**. At initialisation both scalars are zero, making the whole block an identity function. The network learns to grow the residuals gradually, which stabilises early training by preventing the ViT from destabilising the U-Net signal at the start.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | — | Token embedding dimension (= bottleneck channels, typically 512) |
| `num_heads` | `int` | 8 | Number of parallel attention heads |
| `mlp_ratio` | `float` | 4.0 | MLP hidden size = `dim × mlp_ratio` |
| `dropout` | `float` | 0.0 | Dropout applied inside attention and MLP |

| Attribute | Type | Description |
|---|---|---|
| `norm1` | `LayerNorm` | Normalises tokens before the attention sub-layer |
| `attn` | `MultiheadAttention` | Self-attention — every token attends to every other token |
| `norm2` | `LayerNorm` | Normalises tokens before the MLP sub-layer |
| `mlp` | `Sequential` | `Linear(dim→dim×ratio) → GELU → Dropout → Linear(→dim) → Dropout` |
| `alpha_attn` | `nn.Parameter` | Scalar, init=0.0. Multiplies the attention residual branch |
| `alpha_ffn` | `nn.Parameter` | Scalar, init=0.0. Multiplies the MLP residual branch |

**Data flow:**
```
x ──► norm1 ──► attn ──► × alpha_attn ──► + x
                                            │
                                            ▼
                              norm2 ──► mlp ──► × alpha_ffn ──► + x ──► output
```

---

### `PixelwiseViT`

Wraps `depth` `ReZeroTransformerBlock` instances and handles the reshape between the 4D spatial feature map `(N, C, H, W)` and the 3D token sequence `(N, H×W, C)`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dim` | `int` | — | Channel dimension of the bottleneck feature map |
| `depth` | `int` | 4 | Number of stacked Transformer blocks |
| `num_heads` | `int` | 8 | Attention heads per block |
| `mlp_ratio` | `float` | 4.0 | MLP expansion ratio |
| `dropout` | `float` | 0.0 | Dropout probability |

**Data flow:**
```
(N, C, H, W)
     │  flatten spatial dims (H×W), then transpose → (N, H×W, C)
     ▼
(N, H×W, C) ← each spatial position becomes one token
     │  add 2D sincos positional embedding  (shape: H×W, C)
     ▼
(N, H×W, C)
     │  pass through depth × ReZeroTransformerBlock
     ▼
(N, H×W, C)
     │  transpose back, then reshape → (N, C, H, W)
     ▼
(N, C, H, W)
```

---

### `ConvBlock`

Basic encoder/decoder building block. A single 3×3 convolution with padding=1 (preserves spatial size), InstanceNorm, and ReLU.

| Parameter | Description |
|---|---|
| `in_channels` | Input feature channels |
| `out_channels` | Output feature channels |

---

### `DownsampleBlock`

Strided 4×4 convolution (stride=2) + InstanceNorm + ReLU. Halves spatial resolution `(H, W) → (H/2, W/2)`.

| Parameter | Description |
|---|---|
| `in_channels` | Input channels |
| `out_channels` | Output channels (typically 2× input) |

---

### `UpsampleBlock`

Nearest-neighbour 2× upsample followed by a 3×3 conv + InstanceNorm + ReLU. Doubles spatial resolution. Nearest-neighbour upsampling is preferred over transposed convolutions to avoid checkerboard artefacts.

| Parameter | Description |
|---|---|
| `in_channels` | Input channels |
| `out_channels` | Output channels (typically ½ input) |

---

### `ViTUNetGenerator`

Top-level generator module. Assembles all blocks into the complete encode → bottleneck → decode pipeline with skip connections.

| Constructor Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input image channels |
| `output_nc` | 3 | Output image channels |
| `base_channels` | 64 | Channels at encoder level 1. Levels 2/3/4 use ×2/×4/×8 = 128/256/512 |
| `vit_depth` | 4 | Number of ReZero Transformer blocks in the ViT bottleneck |
| `vit_heads` | 8 | Attention heads per Transformer block |
| `vit_mlp_ratio` | 4.0 | MLP expansion ratio inside each Transformer block |
| `vit_dropout` | 0.0 | Dropout inside Transformer blocks |

**Skip connections:** At each decoder level `i`, the upsampled tensor is concatenated with encoder output `e_i` along the channel dimension, then passed through a `ConvBlock` to halve the channel count back. This is the U-Net skip mechanism that preserves fine spatial detail lost during downsampling.

---

## Functions

### `_get_1d_sincos_pos_embed(embed_dim, pos)`

Builds 1D sine-cosine positional embeddings for a sequence of positions.

| Parameter | Description |
|---|---|
| `embed_dim` | Must be even. First half of output uses sin, second half uses cos. |
| `pos` | 1D tensor of integer position indices, shape `(N,)` |

**Formula:** For position `p` and frequency index `k`:
- `sin(p / 10000^(2k / embed_dim))`
- `cos(p / 10000^(2k / embed_dim))`

Returns shape `(N, embed_dim)`.

---

### `_get_2d_sincos_pos_embed(embed_dim, height, width, device, dtype)`

Builds 2D positional embeddings for an `(H, W)` spatial grid. Constructs independent 1D embeddings for rows and columns using `embed_dim/2` dimensions each, then concatenates them.

Returns shape `(H×W, embed_dim)` — one embedding vector per spatial position.

---

### `init_weights(net)`

Applies standard GAN weight initialisation in-place to all submodules of `net`:

| Module type | Initialisation |
|---|---|
| `Conv2d`, `ConvTranspose2d` | `Normal(mean=0, std=0.02)` |
| `Linear` | Xavier uniform, bias=0 |
| `InstanceNorm2d` | Weight `Normal(mean=1, std=0.02)`, bias=0 |

---

### `getGenerators()`

Factory. Creates two `ViTUNetGenerator` instances with default parameters, applies `init_weights` to both, runs a smoke-test forward pass to verify output shapes, and returns them.

**Returns:** `(G_AB, G_BA)` — both on CUDA if available, otherwise CPU.

