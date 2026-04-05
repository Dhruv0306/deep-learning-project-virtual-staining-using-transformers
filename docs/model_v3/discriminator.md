# model_v3/discriminator.py — v3.2 ProjectionDiscriminator

Source of truth: `../../model_v3/discriminator.py`

This module defines the dual v3 discriminators used for adversarial training.
Each discriminator combines local, global, and frequency-domain signals.

## Public Components

1. `MinibatchStdDev`
2. `LocalPatchBranchWithMBStd`
3. `GlobalDiscriminatorBranch`
4. `FFTDiscriminatorBranch`
5. `ProjectionDiscriminator`
6. `getDiscriminatorsV3`

`SpectralNormDiscriminator` is re-exported from `model_v2.discriminator`.

---

## MinibatchStdDev

ProGAN/StyleGAN minibatch standard-deviation layer.

Computes the mean std-dev across a group of `group_size` samples and appends
it as an extra channel. Gives the discriminator a diversity signal, penalising
mode-dropped generators that produce repetitive textures.

Args:

- `group_size` — samples per group (clamped to `min(group_size, N)`)
- `num_features` — number of summary statistics appended (1 in original)

Output: `(N, C+num_features, H, W)`

---

## LocalPatchBranchWithMBStd

Wraps `SpectralNormDiscriminator` with `MinibatchStdDev` inserted before the
final scoring convolution.

Construction:

1. Build `SpectralNormDiscriminator` body.
2. Dry-run a CPU probe through all-but-last body layers to determine the true
   pre-final channel count (avoids brittle static arithmetic).
3. Append `MinibatchStdDev` (adds 1 channel).
4. Add a spectral-norm `Conv2d(actual_ch + 1, 1, 4×4)` final layer.

Forward:

1. Run all body layers except the last.
2. Apply `MinibatchStdDev`.
3. Apply `final_conv` → patch logit map.

Args:

- `input_nc`, `base_channels`, `n_layers`, `use_spectral_norm`
- `group_size` — minibatch std-dev group size

---

## GlobalDiscriminatorBranch

Full-image receptive field branch with self-attention on the 4×4 feature map.

Shape flow for 256×256 inputs:

```
(N, 3, 256, 256)
  -> _sn_conv(stride=4) + LeakyReLU          -> (N, 64,  64, 64)
  -> _sn_conv(stride=4) + IN + LeakyReLU     -> (N, 128, 16, 16)
  -> _sn_conv(stride=4) + IN + LeakyReLU     -> (N, 256,  4,  4)
  -> flatten to (N, 16, 256) tokens
  -> MultiheadAttention(heads=4) + LayerNorm  # global layout dependencies
  -> reshape back to (N, 256, 4, 4)
  -> _sn_conv head (4×4, stride=1)            -> (N, 1, 1, 1)
  -> view -> (N, 1)
```

Enhancement over v3.1: the self-attention layer allows the discriminator to
reason about long-range spatial co-occurrences (e.g., stain balance across tissue).

---

## FFTDiscriminatorBranch

Color-aware frequency-domain branch.

Enhancement over v3.1: processes grayscale + per-channel (R, G, B) FFT
magnitudes (4 channels total). H&E staining artifacts often differ strongly
between hematoxylin (blue) and eosin (pink/red) channels.

Shape flow:

```
(N, 3, H, W)
  -> grayscale + R/G/B rfft2 -> log1p(|·|)   -> (N, 4, H, W//2+1)
  -> normalize per-sample (÷ mean, clamped)
  -> 4-layer spectral-norm CNN                -> (N, c*4, ~8, ~8)
  -> global avg pool                          -> (N, c*4)
  -> spectral-norm Linear head               -> (N, 1)
```

Args:

- `base_channels` — feature channels (kept modest for efficiency)

---

## ProjectionDiscriminator

Three-branch composite discriminator with learnable branch weights.

Branches:

| Branch | Class | Purpose |
|---|---|---|
| Local | `LocalPatchBranchWithMBStd` | Texture, stain granularity, diversity penalty |
| Global | `GlobalDiscriminatorBranch` | Color balance, tissue layout |
| FFT | `FFTDiscriminatorBranch` | Periodic decode artifacts, channel imbalance |

Learnable weights:

- `branch_logweights` — `nn.Parameter` of shape `(n_branches,)`, initialised to 0.
- Applied as `softmax(branch_logweights)` so weights sum to 1.
- Allows training to automatically up-weight the most informative branch.

Forward output: list of weighted logit tensors, one per enabled branch.
The LSGAN helpers average across the list, making this equivalent to a
weighted average loss.

Args:

- `input_nc`, `base_channels`, `n_layers`
- `global_base_channels`, `fft_base_channels`
- `use_spectral_norm`
- `use_local`, `use_global`, `use_fft` — toggle individual branches
- `mbstd_group_size` — minibatch std-dev group size

---

## getDiscriminatorsV3

Factory that builds and initialises two `ProjectionDiscriminator` instances `(D_A, D_B)`.

Args:

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
| `device` | auto | Target device |

Smoke test: random `(max(mbstd_group_size,2), 3, 256, 256)` forward pass.
Prints output shapes and parameter counts for D_A and D_B.

Returns: `(D_A, D_B)` — initialised, moved to device, in train mode.
