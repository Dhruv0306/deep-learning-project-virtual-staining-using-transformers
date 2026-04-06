# model_v2/discriminator.py - v2 Discriminator

Source of truth: ../../model_v2/discriminator.py

Model: True UVCGAN v2
Role: Multi-scale spectral-norm PatchGAN with a residual shortcut, learnable per-scale weights, and an FPN lateral connection.

---

## Component Structure

1. _conv_block
2. _match_spatial_size
3. SpectralNormDiscriminator
4. MultiScaleDiscriminator
5. getDiscriminatorsV2

---

## 1) _conv_block

Builds one reusable conv stage for the discriminator.

Input:
- x: (N, Cin, H, W)

Dataflow:
1. optional ReflectionPad2d(1) when reflect_pad=True and kernel_size=3 (avoids border artefacts)
2. Conv2d(Cin -> Cout, configurable kernel/stride/padding)
   - wrapped with spectral_norm when use_spectral=True
3. optional InstanceNorm2d(Cout) when use_norm=True
4. LeakyReLU(slope)
   - slope=0.2 in early layers, 0.1 in deeper layers

Output:
- (N, Cout, Hout, Wout)

---

## 2) _match_spatial_size

Aligns a tensor's spatial dimensions to a target (H, W) for the residual shortcut.

Input:
- x: (N, C, h, w)
- target_hw: (target_h, target_w)

Dataflow:
- if already matching: return x unchanged
- if larger: center-crop to target
- if smaller: symmetric zero-pad to target

Output:
- (N, C, target_h, target_w)

---

## 3) SpectralNormDiscriminator

Single-scale PatchGAN with spectral normalisation and a residual shortcut at the penultimate stride-1 layer.

The shortcut improves gradient flow to early feature-extraction layers when the generator is strong. A spectral-normalised 1×1 conv projects channels when dimensions differ.

Typical config: input_nc=3, base_channels=64, n_layers=3

Dataflow for input (N, 3, 256, 256):

Strided downsampling (self.down):
- Block 0 (no IN, slope=0.2): Conv 3->64, stride=2  -> (N, 64, 128, 128)
- Block 1 (IN, slope=0.2):    Conv 64->128, stride=2 -> (N, 128, 64, 64)
- Block 2 (IN, slope=0.1):    Conv 128->256, stride=2 -> (N, 256, 32, 32)

Penultimate stride-1 layer + residual shortcut:
- penultimate (IN, slope=0.1): Conv 256->512, stride=1 -> (N, 512, 31, 31)
- shortcut: SN 1×1 Conv 256->512, _match_spatial_size -> (N, 512, 31, 31)
- feat = penultimate + LeakyReLU(shortcut, 0.1)

Output layer (no norm, no activation):
- Conv 512->1, stride=1 -> (N, 1, 30, 30)

forward() returns:
- patch logit map (N, 1, 30, 30)

forward_with_intermediates() returns:
- (post_down_feat, logit_map) — post_down used by MultiScaleDiscriminator for FPN

Notes:
- spectral_norm wraps all conv layers when use_spectral_norm=True
- smaller LeakyReLU slope (0.1) in deeper layers reduces dead-neuron risk under tight spectral norm

---

## 4) MultiScaleDiscriminator

Multi-scale discriminator with learnable per-scale weights (softmax-normalised) and an FPN lateral connection from the finest to the coarsest scale.

Learnable scale weights allow the model to emphasise whichever spatial scale is most informative. For H&E histology the finest scale typically carries most stain texture signal.

FPN lateral: finest-scale post-down features are spatially aligned and added to the coarsest-scale features before that discriminator's penultimate layer, letting global reasoning be informed by fine-scale detail.

Internal components:
- ModuleList of `num_scales` SpectralNormDiscriminator instances
- AvgPool2d(kernel=3, stride=2, padding=1, count_include_pad=False) for coarser inputs
- scale_logweights: learnable (num_scales,) parameter, softmax-normalised in forward
- fpn_lateral: SN 1×1 Conv aligning finest-scale channels to coarsest-scale channels

Dataflow with num_scales=3 and input (N, 3, 256, 256):

Scale 0 (finest) — forward_with_intermediates:
- in0: (N, 3, 256, 256)
- post_down0: (N, 256, 32, 32)  [saved for FPN]
- out0: (N, 1, 30, 30)  weighted by softmax(scale_logweights)[0]

Downsample:
- in1: (N, 3, 128, 128)

Scale 1 (middle) — standard forward:
- out1: (N, 1, 14, 14)  weighted by softmax(scale_logweights)[1]

Downsample:
- in2: (N, 3, 64, 64)

Scale 2 (coarsest) — FPN-enriched forward:
- forward_with_intermediates -> post_down2: (N, 256, 8, 8)
- fine_aligned = adaptive_avg_pool2d(fpn_lateral(post_down0), (8,8))
- enriched = post_down2 + fine_aligned
- re-run penultimate + shortcut on enriched
- out2: (N, 1, 6, 6)  weighted by softmax(scale_logweights)[2]

Return:
- list [weighted_out0, weighted_out1, weighted_out2]

---

## 5) getDiscriminatorsV2

Factory that builds D_A and D_B.

Dataflow:
1. choose device (CUDA if available)
2. instantiate D_A and D_B as MultiScaleDiscriminator with matching kwargs
3. apply init_weights to both
4. smoke test (no_grad):
   - x: (1, 3, 256, 256)
   - out_A / out_B: list of weighted logit tensors per scale
5. print D_A parameter count

Return:
- (D_A, D_B)

---

## Training Integration

In the v2 training loop:
- D_A consumes real_A and fake_A batches
- D_B consumes real_B and fake_B batches
- each discriminator returns a list of per-scale weighted logit maps
- UVCGANLoss.discriminator_loss averages per-scale LSGAN objectives
- gradient penalty uses interpolated inputs; multi-scale outputs are summed to a scalar before gradient computation
- GP is always computed in float32 (autocast disabled inside UVCGANLoss)
