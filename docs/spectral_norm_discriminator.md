# `spectral_norm_discriminator.py` — v2 Discriminator

**Model:** True UVCGAN v2  
**Role:** Multi-scale spectral-norm PatchGAN discriminator. Two instances are created — `D_A` for domain A (unstained) and `D_B` for domain B (stained).

---

## Architecture Overview

The v2 discriminator wraps multiple independent `SpectralNormDiscriminator` instances, each operating on a progressively downsampled version of the input. This **multi-scale design** allows discrimination at different spatial frequencies simultaneously: fine-scale discriminators catch local texture artefacts, while coarse-scale discriminators catch global structural inconsistencies.

```
Input Image (N, 3, 256, 256)
        │
        ├──────────────────────────────────────────────────┐
        │ Scale 0 (finest)                                 │
        │ SpectralNormDiscriminator on original 256×256    │
        │ → logit map (N, 1, 30, 30)                      │
        │                                                  │
        ▼ AvgPool2d(k=3, s=2, p=1)                         │
(N, 3, 128, 128)                                           │
        │                                                  │
        ├──────────────────────────────────────────────────┤
        │ Scale 1 (medium)                                 │
        │ SpectralNormDiscriminator on 128×128             │
        │ → logit map (N, 1, 14, 14)                      │
        │                                                  │
        ▼ AvgPool2d(k=3, s=2, p=1)                         │
(N, 3, 64, 64)                                             │
        │                                                  │
        ├──────────────────────────────────────────────────┤
        │ Scale 2 (coarsest)                               │
        │ SpectralNormDiscriminator on 64×64               │
        │ → logit map (N, 1, 6, 6)                        │
        └──────────────────────────────────────────────────┘

Output: list of 3 logit maps [(N,1,30,30), (N,1,14,14), (N,1,6,6)]
```

---

## Single-Scale Discriminator Architecture (`SpectralNormDiscriminator`)

Each scale uses the same PatchGAN structure as v1 but with **spectral normalisation** on every convolutional layer:

```
Input (N, 3, H, W)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: SN-Conv(3→64, k=4, s=2, p=1) + LeakyReLU(0.2)     │
│ No InstanceNorm on first layer                              │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: SN-Conv(64→128, k=4, s=2, p=1) + IN + LeakyReLU   │
├─────────────────────────────────────────────────────────────┤
│ [repeat for n_layers-1 total strided layers,                │
│  channels capped at base_channels×8 = 512]                  │
├─────────────────────────────────────────────────────────────┤
│ Stride-1 layer: SN-Conv(→512, k=4, s=1, p=1) + IN + LReLU │
├─────────────────────────────────────────────────────────────┤
│ Output: SN-Conv(512→1, k=4, s=1, p=1)   ← raw logits       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Logit map (N, 1, H', W')
```

---

## What Spectral Normalisation Does

Spectral normalisation divides each weight matrix `W` by its largest singular value `σ(W)`:

```
W_SN = W / σ(W)
```

This constrains the Lipschitz constant of each linear layer to 1, which in turn bounds the discriminator's overall Lipschitz constant. A Lipschitz-bounded discriminator cannot grow arbitrarily strong relative to the generator, stabilising the adversarial training dynamic. This is especially important when combined with the gradient penalty: both together enforce a smooth, well-behaved discriminator loss landscape.

---

## Classes

### `SpectralNormDiscriminator`

Single-scale PatchGAN with optional spectral normalisation on all layers.

| Constructor Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input image channels |
| `base_channels` | 64 | Channels after the first conv layer |
| `n_layers` | 3 | Number of strided downsampling layers (layer count excludes the stride-1 layer and output layer) |
| `use_spectral_norm` | `True` | Wrap every conv with spectral normalisation |

| Attribute | Description |
|---|---|
| `model` | `nn.Sequential` of all conv blocks |

**Channel progression** (with `base_channels=64`, `n_layers=3`):

| Layer | In → Out channels | Stride | Notes |
|---|---|---|---|
| 1 | 3 → 64 | 2 | No IN |
| 2 | 64 → 128 | 2 | With IN |
| 3 | 128 → 256 | 2 | With IN |
| stride-1 | 256 → 512 | 1 | With IN, increases receptive field |
| output | 512 → 1 | 1 | No IN, no activation |

Channels are capped at `base_channels × 8 = 512`.

**`forward(x)`** — returns logit map `(N, 1, H', W')`.

---

### `MultiScaleDiscriminator`

Wraps `num_scales` independent `SpectralNormDiscriminator` instances, applying each to a progressively downsampled version of the input.

| Constructor Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input channels |
| `base_channels` | 64 | Channels at the finest scale |
| `n_layers` | 3 | Strided layers per scale |
| `num_scales` | 3 | Number of spatial scales to discriminate at. In `get_8gb_config()` this is reduced to 2 to save ~0.4 GB VRAM. |
| `use_spectral_norm` | `True` | Spectral normalisation on all convolutions |

| Attribute | Description |
|---|---|
| `discriminators` | `nn.ModuleList` of `num_scales` `SpectralNormDiscriminator` instances |
| `downsample` | `nn.AvgPool2d(k=3, s=2, p=1)` — used to produce coarser inputs. `count_include_pad=False` avoids border artefacts. |

**`forward(x)`**

```python
outputs = []
for disc in self.discriminators:
    outputs.append(disc(x))   # discriminate at current resolution
    x = self.downsample(x)    # halve resolution for next scale
return outputs
```

Returns a list of logit maps, one per scale, from finest to coarsest.

---

## Functions

### `_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_norm, use_spectral)`

Helper that builds a single discriminator layer: conv → (optional spectral norm) → (optional InstanceNorm) → LeakyReLU(0.2).

| Parameter | Default | Description |
|---|---|---|
| `in_channels` | — | Input channels |
| `out_channels` | — | Output channels |
| `kernel_size` | 4 | Convolution kernel size |
| `stride` | 2 | Convolution stride |
| `padding` | 1 | Convolution padding |
| `use_norm` | `True` | Apply InstanceNorm2d after conv. When `True`, `bias=False` on the conv. |
| `use_spectral` | `True` | Wrap conv with `spectral_norm` |

Returns: `nn.Sequential`.

---

### `getDiscriminatorsV2(...)`

Factory function. Creates two `MultiScaleDiscriminator` instances, applies `init_weights` from `generator.py` to both, runs a smoke-test, and returns them.

| Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input channels |
| `base_channels` | 64 | Channels at finest scale |
| `n_layers` | 3 | Strided layers per scale |
| `num_scales` | 3 | Number of spatial scales |
| `use_spectral_norm` | `True` | Apply spectral normalisation |

**Returns:** `(D_A, D_B)` — both on the available device.

---

## How Multi-Scale Output Is Used in Loss Computation

The `UVCGANLoss` (in `advanced_losses.py`) handles the list of logit maps returned by `MultiScaleDiscriminator`. For discriminator loss, LSGAN is applied independently at each scale and averaged:

```python
# real_outputs and fake_outputs are both lists of (N, 1, H_i, W_i) tensors
def _lsgan_disc_loss(self, real_outputs, fake_outputs):
    losses = [_single(r, f) for r, f in zip(real_outputs, fake_outputs)]
    return torch.stack(losses).mean()
```

For the gradient penalty, all scale outputs are summed into a single scalar before computing gradients, so the penalty enforces Lipschitz continuity across all scales simultaneously:

```python
pred_scalar = torch.stack([p.sum() for p in pred]).sum()
```
