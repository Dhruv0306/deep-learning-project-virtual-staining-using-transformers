# model_v2/discriminator.py - v2 Discriminator

Source of truth: ../../model_v2/discriminator.py

Model: True UVCGAN v2
Role: Multi-scale PatchGAN discriminator with optional spectral normalization.

---

## Component Structure

1. _conv_block
2. SpectralNormDiscriminator
3. MultiScaleDiscriminator
4. getDiscriminatorsV2

---

## 1) _conv_block

Purpose:
- builds one reusable conv stage for the discriminator

Dataflow with shapes:

Input:
- x: (N, Cin, H, W)

Step 1:
- Conv2d(Cin -> Cout, kernel=4 by default, stride and padding configurable)
- Output: (N, Cout, Hout, Wout)

Step 2:
- optional InstanceNorm2d(Cout)
- Output: (N, Cout, Hout, Wout)

Step 3:
- LeakyReLU(0.2)
- Output: (N, Cout, Hout, Wout)

Returns:
- Sequential block preserving channel and spatial shape from step 1

---

## 2) SpectralNormDiscriminator

Purpose:
- single-scale PatchGAN logit predictor

Typical config:
- input_nc=3, base_channels=64, n_layers=3

Dataflow example for input 256 by 256:

Input:
- x0: (N, 3, 256, 256)

Block 1 (no instance norm):
- Conv 3 -> 64, kernel=4, stride=2, pad=1
- x1: (N, 64, 128, 128)

Block 2:
- Conv 64 -> 128, stride=2
- x2: (N, 128, 64, 64)

Block 3:
- Conv 128 -> 256, stride=2
- x3: (N, 256, 32, 32)

Stride-1 refinement block:
- Conv 256 -> 512, stride=1
- x4: (N, 512, 31, 31)

Output layer:
- Conv 512 -> 1, stride=1
- y: (N, 1, 30, 30)

Forward output:
- patch logits map y

Notes:
- spectral normalization wraps conv layers when enabled
- spatial map values are patch-level realism scores

---

## 3) MultiScaleDiscriminator

Purpose:
- evaluate realism at multiple spatial scales

Internal components:
- ModuleList of per-scale SpectralNormDiscriminator modules
- AvgPool2d(kernel=3, stride=2, padding=1, count_include_pad=False)

Dataflow with num_scales=3 and input 256 by 256:

Scale 0:
- in0: (N, 3, 256, 256)
- out0 from discriminator: (N, 1, 30, 30)

Downsample:
- in1: (N, 3, 128, 128)

Scale 1:
- out1: (N, 1, 14, 14)

Downsample:
- in2: (N, 3, 64, 64)

Scale 2:
- out2: (N, 1, 6, 6)

Return:
- list [out0, out1, out2]

---

## 4) getDiscriminatorsV2

Purpose:
- factory that builds D_A and D_B

Factory dataflow:

1. choose device
2. instantiate D_A and D_B as MultiScaleDiscriminator
3. apply init_weights
4. smoke test input:
   - x: (1, 3, 256, 256)
5. smoke test outputs per model:
   - list of shapes typically [(1,1,30,30), (1,1,14,14), (1,1,6,6)]

Return:
- D_A, D_B

---

## Training Integration

In the v2 training loop:
- D_A consumes real_A and fake_A batches
- D_B consumes real_B and fake_B batches
- each returns either one map per scale
- losses reduce multi-scale outputs by averaging per-scale objectives
- gradient penalty uses interpolated inputs and sums multi-scale outputs to a scalar before gradient computation
