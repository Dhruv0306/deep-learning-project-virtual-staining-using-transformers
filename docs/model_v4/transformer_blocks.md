# model_v4/transformer_blocks.py

Source of truth: ../../model_v4/transformer_blocks.py

Utility components used by the v4 Transformer encoder generator.

## Positional Embedding Utilities

## _get_1d_sincos_pos_embed(embed_dim, pos)

- requires even `embed_dim`
- returns concatenated sine and cosine embeddings with shape `(N, embed_dim)`

## _get_2d_sincos_pos_embed(embed_dim, height, width, device, dtype)

- requires even `embed_dim`
- builds row and column embeddings and concatenates them
- returns shape `(height * width, embed_dim)`

## PatchEmbed

Purpose:

- converts image tensor `(B, C, H, W)` into patch tokens

Implementation:

- single `Conv2d` with `kernel_size=patch_size`, `stride=patch_size`
- validates `image_size % patch_size == 0`

Forward output:

- `tokens`: `(B, N, embed_dim)`
- `grid`: `(H', W')`

## TransformerBlock

Pre-norm Transformer block:

1. `LayerNorm`
2. `MultiheadAttention`
3. residual add
4. `LayerNorm`
5. MLP (`Linear -> GELU -> Dropout -> Linear -> Dropout`)
6. residual add

Arguments:

- `dim`
- `num_heads`
- `mlp_ratio`
- `dropout`

Exports:

- `PatchEmbed`
- `TransformerBlock`
- `_get_2d_sincos_pos_embed`
