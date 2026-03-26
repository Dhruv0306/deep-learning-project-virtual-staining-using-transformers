# model_v1/generator.py - v1 Generator

Source of truth: ../../model_v1/generator.py

Model: Hybrid CycleGAN/UVCGAN v1
Role: Defines the image-to-image generator used in both directions (G_AB and G_BA).

---

## High-Level Design

The v1 generator is a U-Net style encoder-decoder with a transformer bottleneck:

- local feature extraction and reconstruction via convolutional down/up blocks
- global context modeling via PixelwiseViT at the bottleneck
- skip connections from encoder to decoder to preserve high-frequency structure
- tanh output activation for normalized image range [-1, 1]

---

## Building Blocks

### ConvBlock

Pattern:

- Conv2d(kernel=3, padding=1, bias=False)
- InstanceNorm2d
- ReLU

Used in both encoder and decoder refinement stages. Preserves spatial size.

### DownsampleBlock

Pattern:

- Conv2d(kernel=4, stride=2, padding=1, bias=False)
- InstanceNorm2d
- ReLU

Halves spatial resolution while increasing channel capacity.

### UpsampleBlock

Pattern:

- Upsample(scale_factor=2, mode="nearest")
- Conv2d(kernel=3, padding=1, bias=False)
- InstanceNorm2d
- ReLU

Doubles spatial resolution. Nearest-neighbor upsampling is used before convolution.

---

## Transformer Components

### ReZeroTransformerBlock

A transformer layer with learnable residual scaling:

- LayerNorm -> MultiheadAttention
- LayerNorm -> MLP
- residual branches multiplied by learnable scalars alpha_attn and alpha_ffn
- both residual scales start at 0.0 (identity-like initialization)

This helps stabilize training by allowing residual contributions to grow gradually.

### PixelwiseViT

Applies transformer blocks to flattened spatial tokens.

Flow:

1. reshape feature map (N, C, H, W) -> tokens (N, HxW, C)
2. add fixed 2D sine/cosine positional embeddings
3. apply stacked ReZeroTransformerBlock modules
4. reshape back to (N, C, H, W)

---

## Positional Embedding Helpers

### _get_1d_sincos_pos_embed(embed_dim, pos)

Creates 1D sin/cos embeddings for a vector of positions.

### _get_2d_sincos_pos_embed(embed_dim, height, width, device, dtype)

Creates 2D embeddings for an HxW grid by combining row and column embeddings.

These are deterministic (not learned) and generated at runtime in forward().

---

## Main Model

### ViTUNetGenerator

Default channel layout:

- base = 64
- c1=64, c2=128, c3=256, c4=512

Encoder path:

- enc1 -> down1 -> enc2 -> down2 -> enc3 -> down3 -> enc4 -> down4

Bottleneck:

- ConvBlock(c4 -> c4)
- PixelwiseViT(dim=c4, depth=vit_depth, heads=vit_heads)

Decoder path with skip concatenation:

- up1 + e4 -> dec1
- up2 + e3 -> dec2
- up3 + e2 -> dec3
- up4 + e1 -> dec4

Output head:

- ReflectionPad2d(3)
- Conv2d(c1 -> output_nc, kernel=7)
- Tanh

Input/output range expectation:

- input tensors are expected in normalized training range
- output is bounded by tanh to [-1, 1]

---

## Weight Initialization

### init_weights(net)

Applies module-wise initialization:

- Conv2d / ConvTranspose2d: Normal(0.0, 0.02)
- Linear: Xavier uniform, bias 0
- InstanceNorm2d: weight Normal(1.0, 0.02), bias 0

This routine is used for both generators and discriminators via factory setup.

---

## Factory Function

### getGenerators()

Builds and initializes two generators for bidirectional translation:

- G_AB: domain A -> domain B
- G_BA: domain B -> domain A

Steps:

1. select device
2. instantiate two ViTUNetGenerator models
3. apply init_weights on both
4. run smoke-test forward pass on random 256x256 tensor
5. print output shapes and return pair

This function is consumed directly by model_v1/training_loop.py.
