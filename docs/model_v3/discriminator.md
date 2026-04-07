# model_v3/discriminator.py — v3.2 ProjectionDiscriminator

Source of truth: `../../model_v3/discriminator.py`

This module defines the dual v3 discriminators used for adversarial training.
Each discriminator combines three complementary branches that cover different
frequency bands and spatial scales of the H&E staining signal.

---

## Public Components

| Symbol | Kind | Description |
|---|---|---|
| `SpectralNormDiscriminator` | class | Re-exported from `model_v2.discriminator` |
| `MinibatchStdDev` | class | ProGAN diversity layer |
| `LocalPatchBranchWithMBStd` | class | Branch 1 — texture + diversity |
| `GlobalDiscriminatorBranch` | class | Branch 2 — full-image layout |
| `FFTDiscriminatorBranch` | class | Branch 3 — frequency domain |
| `ProjectionDiscriminator` | class | Three-branch composite discriminator |
| `getDiscriminatorsV3` | function | Factory returning `(D_A, D_B)` |

---

## `_sn_conv` (module-private helper)

```python
_sn_conv(in_ch, out_ch, kernel_size=4, stride=2, padding=1) -> nn.Module
```

Returns a `spectral_norm`-wrapped `Conv2d` without bias. Used throughout all
three branches for consistent spectral normalisation.

---

## `MinibatchStdDev`

ProGAN/StyleGAN minibatch standard-deviation layer.

Splits the batch into groups of `group_size`, computes the mean std-dev across
each group, and appends it as `num_features` extra channels. Gives the
discriminator a within-batch diversity signal, penalising mode-collapsed
generators that produce repetitive textures.

**Args**

- `group_size` — samples per group; clamped to `min(group_size, N)` at runtime
- `num_features` — summary statistics appended per spatial location (1 in the
  original ProGAN paper)

**Output shape** `(N, C + num_features, H, W)`

**Forward internals**

```
x: (N, C, H, W)
  reshape → (g, N//g, f, C//f, H, W)   # split into groups
  var(dim=0) + sqrt                     # std per group
  mean(feature/spatial dims)            # scalar per (group, f)
  repeat(g, ...)                        # tile back to full batch
  cat([x, stats])                       # append as extra channel(s)
```

---

## `LocalPatchBranchWithMBStd`

Wraps `SpectralNormDiscriminator` with `MinibatchStdDev` inserted before the
final scoring convolution to penalise texture repetition.

**Construction**

1. Build `SpectralNormDiscriminator` body (`down` + `penultimate` sub-modules).
2. Probe the body on CPU with `torch.inference_mode()` to measure the true
   pre-final channel count — avoids brittle static arithmetic and is compatible
   with both the legacy `self.model` Sequential layout and the current
   `down`/`penultimate`/`out_conv` split.
3. Attach `MinibatchStdDev` (appends 1 channel).
4. Add a spectral-norm `Conv2d(actual_ch + 1, 1, 4×4, stride=1)` final layer.

**Forward**

```
x → body.down → body.penultimate → MinibatchStdDev → final_conv → patch logit map
```

The body's own `out_conv` and shortcut are bypassed: the shortcut spatial size
differs from `penultimate` output (4×4 kernel shrinks by 1 px), so adding them
would cause a shape mismatch.

**Args**

- `input_nc`, `base_channels`, `n_layers`, `use_spectral_norm`
- `group_size` — minibatch std-dev group size

---

## `GlobalDiscriminatorBranch`

Full-image receptive field branch with self-attention on the 4×4 feature map.

Self-attention over the 16 spatial tokens lets the discriminator reason about
long-range co-occurrences (e.g., stain balance across the whole tissue section)
that a purely convolutional path would miss.

**Shape flow for 256×256 inputs**

```
(N,   3, 256, 256)  stride-4 conv + LReLU
(N,  64,  64,  64)  stride-4 conv + IN + LReLU
(N, 128,  16,  16)  stride-4 conv + IN + LReLU
(N, 256,   4,   4)  flatten → (N, 16, 256) tokens
                    MultiheadAttention(heads=4) + LayerNorm (residual)
                    reshape → (N, 256, 4, 4)
(N,   1,   1,   1)  4×4 spectral-norm conv head
(N,   1)            view
```

**Args**

- `input_nc` — input image channels (default 3)
- `base_channels` — feature width of the first conv (default 64)

---

## `FFTDiscriminatorBranch`

Color-aware frequency-domain branch.

Processes grayscale + per-channel (R, G, B) log-magnitude FFT spectra
(4 channels total). H&E staining artifacts often differ strongly between the
hematoxylin (blue) and eosin (pink/red) channels, so the per-channel spectra
give the discriminator more discriminative power than grayscale alone.

**Shape flow**

```
(N, 3, H, W)
  → grayscale + R/G/B rfft2 → log1p(|·|)   → (N, 4, H, W//2+1)
  → normalize per-sample (÷ mean, clamped to ≥ 1e-6)
  → 4-layer spectral-norm CNN                → (N, c*4, ~8, ~8)
  → global average pool                      → (N, c*4)
  → spectral-norm Linear head                → (N, 1)
```

**`_log_magnitude(x)` (static)**

Applies `torch.fft.rfft2(norm="ortho")` and returns `log1p(|spectrum|)`.
Input `(N, C, H, W)` → output `(N, C, H, W//2+1)`.

**Args**

- `base_channels` — feature channels for the CNN (default 32, kept modest)

---

## `ProjectionDiscriminator`

Three-branch composite discriminator with learnable branch weights.

| Branch | Class | Signal |
|---|---|---|
| Local | `LocalPatchBranchWithMBStd` | Mid-frequency texture, stain granularity, diversity |
| Global | `GlobalDiscriminatorBranch` | Full-image color balance, tissue layout |
| FFT | `FFTDiscriminatorBranch` | Periodic decode artifacts, per-channel stain imbalance |

**Learnable weights**

`branch_logweights` is an `nn.Parameter` of shape `(n_branches,)`, initialised
to 0 (equal softmax weighting). At forward time, `softmax(branch_logweights)`
is applied so weights sum to 1, letting training automatically up-weight the
most informative branch.

**Forward**

Returns a list of weighted logit tensors, one per enabled branch. The LSGAN
helpers average across the list, making this equivalent to a learned
weighted-average adversarial loss.

**Args**

- `input_nc`, `base_channels`, `n_layers`
- `global_base_channels`, `fft_base_channels`
- `use_spectral_norm`
- `use_local`, `use_global`, `use_fft` — toggle individual branches; disabled
  branches add no parameters
- `mbstd_group_size` — minibatch std-dev group size

---

## `getDiscriminatorsV3`

Factory that builds and initialises two `ProjectionDiscriminator` instances
`(D_A, D_B)`.

**Args**

| Arg | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input image channels |
| `base_channels` | 64 | Local branch feature width |
| `n_layers` | 3 | Strided layers in local branch |
| `global_base_channels` | 64 | Global branch feature width |
| `fft_base_channels` | 32 | FFT branch feature width |
| `use_spectral_norm` | `True` | Spectral norm on all Conv2d |
| `use_local` | `True` | Enable local + MinibatchStdDev branch |
| `use_global` | `True` | Enable global + self-attention branch |
| `use_fft` | `True` | Enable color-aware FFT branch |
| `mbstd_group_size` | 4 | Minibatch std-dev group size |
| `device` | auto | Target device (CUDA if available, else CPU) |

**Behaviour**

1. Builds two independent `ProjectionDiscriminator` instances and moves them to
   `device`.
2. Applies `init_weights` (from `model_v1.generator`) to both.
3. Runs a smoke test under `torch.inference_mode()` with a random
   `(max(mbstd_group_size, 2), input_nc, 256, 256)` tensor to verify shapes
   and catch construction errors early.
4. Prints output shapes and parameter counts for D_A and D_B.

**Returns** `(D_A, D_B)` — initialised, on device, in train mode.
