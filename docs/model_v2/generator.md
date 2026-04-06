# model_v2/generator.py - v2 Generator

Source of truth: ../../model_v2/generator.py

Model: True UVCGAN v2
Role: U-Net plus ViT generator with LayerScale, gated cross-domain skip fusion, buffered positional embeddings, and optional gradient checkpointing.

---

## Component Structure

1. Positional embedding helpers
2. LayerScaleTransformerBlock
3. PixelwiseViTV2
4. ResidualConvBlock
5. DownBlock
6. UpBlock
7. SkipMerge
8. CrossDomainFusion
9. ViTUNetGeneratorV2
10. init_weights_v2
11. getGeneratorsV2

---

## 1) Positional embedding helpers

### _get_1d_sincos_pos_embed

Input:
- embed_dim: even integer
- pos: (L,)

Dataflow:
1. build half-dimension frequency vector: omega (embed_dim/2,)
2. outer-multiply pos and omega: (L, embed_dim/2)
3. concatenate sin and cos parts

Output:
- pe1d: (L, embed_dim)

### _get_2d_sincos_pos_embed

Input:
- embed_dim, height=H, width=W, device, dtype

Dataflow:
1. build row index grid: (H, W)
2. build col index grid: (H, W)
3. flatten rows and cols to length H*W
4. row embedding via _get_1d_sincos_pos_embed: (H*W, embed_dim/2)
5. col embedding via _get_1d_sincos_pos_embed: (H*W, embed_dim/2)
6. concatenate along dim=1

Output:
- pe2d: (H*W, embed_dim)

---

## 2) LayerScaleTransformerBlock

Pre-LN Transformer block with per-channel LayerScale scalars on both residual branches.

Input:
- tokens x: (N, L, C)

Dataflow:
1. norm1(x): (N, L, C)
2. MultiheadAttention(normed, normed, normed): attn_out (N, L, C)
3. x = x + gamma_attn * attn_out  [LayerScale on attention branch]
4. norm2(x): (N, L, C)
5. MLP(normed): (N, L, C)
6. x = x + gamma_ffn * mlp_out   [LayerScale on MLP branch]

Output:
- (N, L, C)

Notes:
- gamma_attn and gamma_ffn are learnable (C,) parameters, init=init_values (default 1e-4)
- Pre-LN ordering: LayerNorm applied before attention/MLP, not after

---

## 3) PixelwiseViTV2

Pixelwise ViT with buffered 2-D sincos positional embeddings and optional gradient checkpointing.

Input:
- feature map x: (N, C, H, W)

Dataflow:
1. flatten spatial and transpose: (N, C, H, W) -> (N, H*W, C)
2. add positional embedding from buffer (recomputed only if spatial size changes):
   - (N, H*W, C)
3. pass through `depth` LayerScaleTransformerBlocks (shape unchanged)
4. transpose and reshape back: (N, H*W, C) -> (N, C, H, W)

Output:
- (N, C, H, W)

Notes:
- positional embedding is pre-computed at construction for `spatial_size × spatial_size` and stored as a non-trainable buffer
- if input spatial size differs at runtime, the embedding is recomputed on the fly
- when `use_gradient_checkpointing=True`, each block is wrapped with `torch.utils.checkpoint` to recompute activations during backward (~30-40% VRAM saving)

---

## 4) ResidualConvBlock

Two 3×3 reflection-padded convolutions with a learnable per-channel gate on the residual branch.

Input:
- x: (N, C, H, W)

Dataflow:
1. ReflectionPad2d(1)
2. Conv C->C, kernel=3
3. InstanceNorm2d
4. ReLU
5. optional Dropout
6. ReflectionPad2d(1)
7. Conv C->C, kernel=3
8. InstanceNorm2d
9. x = x + gate.view(1,-1,1,1) * block(x)  [per-channel gate, init=1.0]

Output:
- (N, C, H, W)

Notes:
- `gate` is a learnable (C,) parameter initialised to `gate_init` (default 1.0)
- gate allows the network to suppress residual blocks that are not contributing useful features

---

## 5) DownBlock

Strided encoder block: halves spatial resolution.

Input:
- x: (N, Cin, H, W)

Dataflow:
1. Conv2d(Cin->Cout, kernel=4, stride=2, pad=1)
2. InstanceNorm2d
3. ReLU

Output:
- (N, Cout, H/2, W/2)

---

## 6) UpBlock

Decoder block: doubles spatial resolution without checkerboard artefacts.

Input:
- x: (N, Cin, H, W)

Dataflow:
1. Upsample(scale_factor=2, mode="nearest"): (N, Cin, 2H, 2W)
2. ReflectionPad2d(1)
3. Conv2d(Cin->Cout, kernel=3)
4. InstanceNorm2d
5. ReLU

Output:
- (N, Cout, 2H, 2W)

---

## 7) SkipMerge

Skip-connection merge using a 3×3 conv (replaces the previous 1×1 merge).

Inputs:
- upsampled: (N, C_up, H, W)
- skip: (N, C_skip, H, W)

Dataflow:
1. torch.cat([upsampled, skip], dim=1): (N, C_up+C_skip, H, W)
2. ReflectionPad2d(1)
3. Conv2d(C_up+C_skip -> out_channels, kernel=3)
4. InstanceNorm2d
5. ReLU

Output:
- (N, out_channels, H, W)

Notes:
- 3×3 kernel integrates local spatial context from both branches; 1×1 cannot

---

## 8) CrossDomainFusion

Gated cross-domain skip-connection fusion (UVCGAN §3.2).

Inputs:
- feat_self: (N, C, H, W)  — current generator's skip features
- feat_other: (N, C, H, W) — paired generator's skip features

Dataflow:
1. feat_other.detach()  [no cross-generator gradient flow]
2. torch.cat([feat_self, feat_other], dim=1): (N, 2C, H, W)
3. Conv2d(2C->C, kernel=1): (N, C, H, W)
4. InstanceNorm2d
5. ReLU -> fused: (N, C, H, W)
6. gate = sigmoid(gate_logit)  [scalar, init sigmoid(0)=0.5]
7. out = gate * fused + (1 - gate) * feat_self

Output:
- (N, C, H, W)

Notes:
- gate_logit is a learnable scalar parameter initialised to 0 (sigmoid → 0.5)
- at the start of training both domains contribute equally; the model learns to suppress unhelpful cross-domain features

---

## 9) ViTUNetGeneratorV2

True UVCGAN v2 generator: U-Net + ViT bottleneck with LayerScale, gated cross-domain skip fusion, and buffered positional embeddings.

Default channels when base_channels=64:
- c1=64, c2=128, c3=256, c4=512

### Encoder dataflow for x: (N, 3, 256, 256)

1. enc_in (7×7 reflect-pad conv, IN, ReLU):
   - (N, 3, 256, 256) -> e0: (N, 64, 256, 256)
2. down1 + res_enc1 (2× ResidualConvBlock):
   - (N, 64, 256, 256) -> e1: (N, 128, 128, 128)
3. down2 + res_enc2 (2× ResidualConvBlock):
   - (N, 128, 128, 128) -> e2: (N, 256, 64, 64)
4. down3 + res_enc3 (3× ResidualConvBlock — extra block at deepest encoder level):
   - (N, 256, 64, 64) -> e3: (N, 512, 32, 32)
5. down4:
   - (N, 512, 32, 32) -> (N, 512, 16, 16)

### Bottleneck dataflow

6. res_bot_pre (ResidualConvBlock):
   - (N, 512, 16, 16) -> (N, 512, 16, 16)
7. vit (PixelwiseViTV2):
   - (N, 512, 16, 16) -> (N, 512, 16, 16)
8. res_bot_post (ResidualConvBlock — wraps ViT on the output side):
   - (N, 512, 16, 16) -> b: (N, 512, 16, 16)

### Decoder dataflow (3×3 SkipMerge)

9. up1 + merge1 with e3:
   - up1(b): (N, 512, 32, 32)
   - SkipMerge(cat[up1, e3]): (N, 1024, 32, 32) -> (N, 512, 32, 32)
10. up2 + merge2 with e2:
    - up2: (N, 256, 64, 64)
    - SkipMerge(cat[up2, e2]): (N, 512, 64, 64) -> (N, 256, 64, 64)
11. up3 + merge3 with e1:
    - up3: (N, 128, 128, 128)
    - SkipMerge(cat[up3, e1]): (N, 256, 128, 128) -> (N, 128, 128, 128)
12. up4 + merge4 with e0:
    - up4: (N, 64, 256, 256)
    - SkipMerge(cat[up4, e0]): (N, 128, 256, 256) -> (N, 64, 256, 256)
13. out_conv (7×7 reflect-pad conv + Tanh):
    - (N, 64, 256, 256) -> y: (N, 3, 256, 256)

Forward returns:
- y in range [-1, 1]

### Cross-domain forward dataflow

Additional inputs:
- other_skips = (oe0, oe1, oe2, oe3) from paired generator's get_skip_features()

Fusion points (gated CrossDomainFusion):
- fuse4(e0, oe0) -> (N, 64, 256, 256)
- fuse3(e1, oe1) -> (N, 128, 128, 128)
- fuse2(e2, oe2) -> (N, 256, 64, 64)
- fuse1(e3, oe3) -> (N, 512, 32, 32)

Then standard decode path with same output shape.

Raises RuntimeError if constructed with use_cross_domain=False.

### Public API shapes

- encode(x) -> (e0, e1, e2, e3, b)
- get_skip_features(x) -> (e0, e1, e2, e3)
- decode(b, e0, e1, e2, e3) -> y
- forward(x) -> y
- forward_with_cross_domain(x, other_skips) -> y

---

## 10) init_weights_v2

No dataflow transform at runtime.

Initialisation mapping:
- Conv2d / ConvTranspose2d: Kaiming normal (fan_out, ReLU)
- Linear (ViT MLP / projections): truncated normal std=0.02 (ViT/BERT standard)
- InstanceNorm2d / LayerNorm: weight=1, bias=0
- LayerScale gamma_attn / gamma_ffn and ResidualConvBlock gate: left at their __init__ values (not overwritten)

---

## 11) getGeneratorsV2

Factory dataflow:

1. instantiate G_AB and G_BA with matching hyperparameters
2. apply init_weights_v2 to both
3. smoke test (no_grad):
   - x: (1, 3, 256, 256)
   - y_AB: (1, 3, 256, 256)
   - y_BA: (1, 3, 256, 256)
4. print parameter count

Returns:
- (G_AB, G_BA) — both on active device
