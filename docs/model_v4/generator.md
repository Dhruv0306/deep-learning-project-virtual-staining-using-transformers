# model_v4/generator.py

Source of truth: ../../model_v4/generator.py

This module provides two generator implementations used by v4:

- `ResnetGenerator`: CNN encoder-residual-decoder baseline
- `TransformerGeneratorV4`: patch-embedding Transformer encoder + CNN decoder

It also provides `getGeneratorV4(...)` to build, initialize, and optionally
smoke-test either variant.

## Key Classes

## ResnetBlock

- two 3x3 conv layers with reflection padding and InstanceNorm
- optional dropout between convs
- residual output: `x + block(x)`

## ResnetGenerator

Pipeline:

1. `in_conv`: 7x7 reflect-pad conv
2. `down1`, `down2`: two stride-2 downsamples
3. `res_blocks`: `num_res_blocks` bottleneck residual blocks
4. `up1`, `up2`: nearest-neighbor upsample + conv
5. `out_conv`: 7x7 reflect-pad conv + `Tanh`

Feature extraction support:

- `_encode(x)` returns multi-level features `(f0, f1, f2, f3)`
- `encode_features(x, nce_layers)` returns selected feature maps for PatchNCE
- `forward(..., return_features=True)` returns `(output, selected_features)`

## TransformerGeneratorV4

High-level flow:

1. patchify image with `PatchEmbed`
2. add 2-D sin/cos positional embedding
3. run Transformer blocks (`TransformerBlock` list)
4. optional gradient checkpointing per block
5. normalize tokens and reshape to spatial map
6. 1x1 projection to decoder channels
7. upsample decoder stack to image resolution
8. output conv + `Tanh`

Important methods:

- `_encode_tokens(...)`: returns final tokens, token grid, and optional
  intermediate token features
- `_tokens_to_map(...)`: reshapes `(B, N, C)` token sequence into `(B, C, H, W)`
- `encode_features(...)`: collects NCE feature maps from selected Transformer
  blocks

## Initialization and Factory

## init_weights_v4(net)

Weight policy:

- Conv / ConvTranspose: Normal(0, 0.02)
- InstanceNorm scale: Normal(1, 0.02), bias 0
- Linear: Xavier uniform, bias 0

## getGeneratorV4(...)

Behavior:

- auto-selects device when not provided
- chooses Transformer generator when `use_transformer_encoder=True`
- otherwise builds ResNet generator
- applies `init_weights_v4`
- optional smoke test with random `(1, input_nc, 256, 256)` tensor

Returns:

- initialized `ResnetGenerator` or `TransformerGeneratorV4`
