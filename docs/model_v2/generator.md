# model_v2/generator.py - v2 Generator

Source of truth: ../../model_v2/generator.py

Model: True UVCGAN v2
Role: U-Net plus ViT generator with optional cross-domain fusion and optional gradient checkpointing.

---

## Component Structure

1. Positional embedding helpers
2. LayerScaleTransformerBlock
3. PixelwiseViTV2
4. ResidualConvBlock
5. DownBlock
6. UpBlock
7. CrossDomainFusion
8. ViTUNetGeneratorV2
9. init_weights_v2
10. getGeneratorsV2

---

## 1) Positional embedding helpers

### _get_1d_sincos_pos_embed

Input:
- embed_dim: even integer
- pos: (L,)

Dataflow:
1. build half-dimension frequency vector
2. outer-like multiply pos and frequencies
3. concatenate sin and cos parts

Output:
- pe1d: (L, embed_dim)

### _get_2d_sincos_pos_embed

Input:
- embed_dim, height=H, width=W

Dataflow:
1. build row index grid: (H, W)
2. build col index grid: (H, W)
3. flatten rows and cols to length H*W
4. row embedding: (H*W, embed_dim/2)
5. col embedding: (H*W, embed_dim/2)
6. concatenate

Output:
- pe2d: (H*W, embed_dim)

---

## 2) LayerScaleTransformerBlock

Input:
- tokens x: (N, L, C)

Dataflow:
1. norm1(x): (N, L, C)
2. self-attention: (N, L, C)
3. scale by gamma_attn: (N, L, C)
4. residual add with x: (N, L, C)
5. norm2: (N, L, C)
6. MLP: (N, L, C)
7. scale by gamma_ffn: (N, L, C)
8. residual add: (N, L, C)

Output:
- same shape (N, L, C)

---

## 3) PixelwiseViTV2

Input:
- feature map x: (N, C, H, W)

Dataflow:
1. flatten spatial and transpose:
   - (N, C, H, W) -> (N, H*W, C)
2. build positional embedding:
   - pe: (H*W, C)
3. add positional embedding:
   - (N, H*W, C)
4. pass through depth transformer blocks
   - shape unchanged each block
5. transpose and reshape back:
   - (N, H*W, C) -> (N, C, H, W)

Output:
- (N, C, H, W)

Optional path:
- when gradient checkpointing is enabled, each block is recomputed on backward

---

## 4) ResidualConvBlock

Input:
- x: (N, C, H, W)

Dataflow:
1. reflection pad
2. conv C->C
3. instance norm
4. relu
5. optional dropout
6. reflection pad
7. conv C->C
8. instance norm
9. add original input

Output:
- (N, C, H, W)

---

## 5) DownBlock

Input:
- x: (N, Cin, H, W)

Dataflow:
1. conv kernel=4 stride=2 pad=1, Cin->Cout
2. instance norm
3. relu

Output:
- (N, Cout, H/2, W/2)

---

## 6) UpBlock

Input:
- x: (N, Cin, H, W)

Dataflow:
1. nearest upsample x2
   - (N, Cin, 2H, 2W)
2. reflection pad
3. conv Cin->Cout
4. instance norm
5. relu

Output:
- (N, Cout, 2H, 2W)

---

## 7) CrossDomainFusion

Inputs:
- feat_self: (N, C, H, W)
- feat_other: (N, C, H, W)

Dataflow:
1. detach feat_other
2. concatenate along channel axis
   - (N, 2C, H, W)
3. conv 1x1: 2C -> C
4. instance norm
5. relu

Output:
- fused: (N, C, H, W)

---

## 8) ViTUNetGeneratorV2

Default channels when base_channels=64:
- c1=64, c2=128, c3=256, c4=512

### Encoder dataflow for x: (N, 3, 256, 256)

1. enc_in:
   - (N, 3, 256, 256) -> e0: (N, 64, 256, 256)
2. down1 + res_enc1:
   - (N, 64, 256, 256) -> e1: (N, 128, 128, 128)
3. down2 + res_enc2:
   - (N, 128, 128, 128) -> e2: (N, 256, 64, 64)
4. down3 + res_enc3:
   - (N, 256, 64, 64) -> e3: (N, 512, 32, 32)
5. down4:
   - (N, 512, 32, 32) -> (N, 512, 16, 16)

### Bottleneck dataflow

6. res_bot:
   - (N, 512, 16, 16) -> (N, 512, 16, 16)
7. vit:
   - (N, 512, 16, 16) -> b: (N, 512, 16, 16)

### Decoder dataflow

8. up1 and merge with e3:
   - up1(b): (N, 512, 32, 32)
   - concat with e3: (N, 1024, 32, 32)
   - dec1_merge: (N, 512, 32, 32)
9. up2 and merge with e2:
   - up2: (N, 256, 64, 64)
   - concat: (N, 512, 64, 64)
   - dec2_merge: (N, 256, 64, 64)
10. up3 and merge with e1:
   - up3: (N, 128, 128, 128)
   - concat: (N, 256, 128, 128)
   - dec3_merge: (N, 128, 128, 128)
11. up4 and merge with e0:
   - up4: (N, 64, 256, 256)
   - concat: (N, 128, 256, 256)
   - dec4_merge: (N, 64, 256, 256)
12. output head:
   - out_conv -> y: (N, 3, 256, 256)

Forward returns:
- y in range [-1, 1]

### Cross-domain forward dataflow

Additional inputs:
- other_skips = (oe0, oe1, oe2, oe3)

Fusion points:
- e0 with oe0 -> (N,64,256,256)
- e1 with oe1 -> (N,128,128,128)
- e2 with oe2 -> (N,256,64,64)
- e3 with oe3 -> (N,512,32,32)

Then standard decode path with same output shape.

### Public API shapes

- encode(x) -> (e0, e1, e2, e3, b)
- get_skip_features(x) -> (e0, e1, e2, e3)
- decode(b, e0, e1, e2, e3) -> y
- forward(x) -> y
- forward_with_cross_domain(x, other_skips) -> y

---

## 9) init_weights_v2

No dataflow transform at runtime.

Initialization mapping:
- Conv and ConvTranspose weights
- Linear weights and bias
- InstanceNorm2d and LayerNorm affine parameters

---

## 10) getGeneratorsV2

Factory dataflow:

1. instantiate G_AB and G_BA
2. apply init_weights_v2
3. smoke test input:
   - x: (1, 3, 256, 256)
4. smoke test outputs:
   - y_AB: (1, 3, 256, 256)
   - y_BA: (1, 3, 256, 256)

Returns:
- (G_AB, G_BA)
