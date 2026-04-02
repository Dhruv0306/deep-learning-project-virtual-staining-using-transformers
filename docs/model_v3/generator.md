# model_v3/generator.py - v3 DiT Generator

Source of truth: ../../model_v3/generator.py

This module defines the CycleDiT generator stack used by v3 diffusion.
The generator combines noisy latent patch embedding, timestep conditioning,
raw-image tokenization, and learned domain biasing.

## Public Components

1. `PatchEmbed`
2. `TimestepEmbedding`
3. `ConditionTokenizer`
4. `DiTBlock`
5. `DiTGenerator`
6. `DomainEmbedding`
7. `CycleDiTGenerator`
8. `getGeneratorV3`

## PatchEmbed

Input:

- `z`: `(N, 4, 32, 32)` by default
- `patch_size`: latent patch size, usually `2`

Behavior:

1. split the latent map into non-overlapping patches
2. flatten each patch to a token vector
3. project patch vectors into `hidden_dim`

Output:

- token sequence of shape `(N, L, hidden_dim)` where `L = (H / p) * (W / p)`

## TimestepEmbedding

Input:

- scalar or batch timestep tensor

Behavior:

1. build sinusoidal timestep features
2. pass them through a two-layer SiLU MLP

Output:

- `(N, hidden_dim)` timestep embedding

## ConditionTokenizer

Input:

- conditioning image tensor `(N, 3, 256, 256)`

Behavior:

1. patchify with `Conv2d(kernel=stride=cond_patch_size)`
2. optionally average-pool token grid using `cond_token_pool_stride`
3. flatten to tokens
4. add 2-D sine/cosine positional embeddings

Output:

- conditioning tokens `(N, Lc, hidden_dim)`

## DiTBlock

Each block uses adaLN-Zero conditioning from the combined timestep and
domain embedding.

Inputs:

- latent tokens `(N, L, Hd)`
- condition tokens `(N, Lc, Hd)`
- combined conditioning vector `(N, Hd)`

Flow:

1. adaptive LayerNorm modulation on self-attention
2. optional cross-attention to condition tokens
3. adaptive LayerNorm modulation on the MLP branch
4. residual updates gated by learned alpha parameters

## DiTGenerator

Forward inputs:

- `z_t`: `(N, 4, 32, 32)` noisy latent
- `t`: diffusion timestep tensor
- `c`: condition tokens `(N, Lc, hidden_dim)`

Forward flow:

1. patch embed the latent
2. add 2-D positional embeddings
3. compute timestep embedding
4. add pooled condition tokens to form the conditioning vector
5. run the Transformer block stack
6. project tokens back to latent patches
7. unpatchify to `(N, 4, 32, 32)`

Output:

- `v_pred` or `eps_pred` latent tensor `(N, 4, 32, 32)`

## DomainEmbedding

Purpose:

- learned target-domain bias for A/B translation

Domain ids:

- `0` for domain A
- `1` for domain B

## CycleDiTGenerator

This is the top-level v3 model wrapper.

Inputs:

- noisy latent `z_t`
- timestep `t`
- conditioning image or precomputed tokens
- target domain id
- optional diffusion scheduler
- prediction type (`"v"` or `"eps"`)

Behavior:

1. tokenize raw conditioning images when needed
2. add the learned domain embedding to the conditioning tokens
3. call the DiT backbone
4. optionally reconstruct `x0_pred` using the scheduler

Output:

- dictionary with `v_pred` and optional `x0_pred`

## getGeneratorV3

Builds and smoke-tests a `CycleDiTGenerator` from a diffusion config.

Important config values:

- `dit_hidden_dim`
- `dit_depth`
- `dit_heads`
- `dit_mlp_ratio`
- `dit_patch_size`
- `cond_patch_size`
- `cond_token_pool_stride`
- `use_cross_attention`
- `use_gradient_checkpointing`

The smoke test runs a dummy forward pass with:

- `z_t`: `(1, 4, 32, 32)`
- `t`: `(1,)`
- `x`: `(1, 3, 256, 256)`

Return:

- initialized `CycleDiTGenerator`
