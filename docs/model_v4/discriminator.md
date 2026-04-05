# model_v4/discriminator.py — v4.2

Source of truth: `../../model_v4/discriminator.py`

Enhanced multi-scale PatchGAN discriminator for the v4 CUT + Transformer pipeline.

## Public Components

1. `MinibatchStdDev`
2. `PatchGANDiscriminator`
3. `init_weights_v4`
4. `getDiscriminatorV4`

`_sn_conv_block` is an internal helper.

---

## _sn_conv_block (internal)

Single spectral-norm Conv2d → [InstanceNorm2d] → LeakyReLU(0.2) block.

- Spectral norm applied to all convolutions (new in v4.2).
- First layer disables InstanceNorm per PatchGAN convention (`use_norm=False`).

---

## MinibatchStdDev

Minibatch standard-deviation channel appended before the final score head.

Computes average std-dev across groups of `group_size` samples and tiles it as
an extra channel. Gives the discriminator a diversity signal to detect
mode-collapsed generators producing repetitive stain patterns.

Args: `group_size` (default 4, clamped to batch size).

Output: `(N, C+1, H, W)`

---

## PatchGANDiscriminator

Enhanced multi-scale PatchGAN with spectral norm, auxiliary head, and MinibatchStdDev.

Architecture (n_layers=3 example):

```
Input (N, 3, 256, 256)
  -> down_layers[0]: SN-Conv(stride=2, no IN)
  -> down_layers[1]: SN-Conv(stride=2, IN)   <- aux head tapped here (n_layers//2 = 1)
       -> aux_head: SN-Conv(4×4, stride=1) -> aux score map
  -> down_layers[2]: SN-Conv(stride=2, IN)
  -> penultimate:    SN-Conv(stride=1, IN)
  -> MinibatchStdDev                          <- appends 1 diversity channel
  -> final_conv:     SN-Conv(4×4, stride=1) -> main score map
```

The auxiliary head tap point is determined by a CPU dry-run probe to avoid
brittle static channel arithmetic.

Forward methods:

- `forward(x)` — returns a single merged score map `(N, 1, H', W')`.
  The aux map is bilinearly interpolated to match main's spatial size, then
  averaged: `0.5 * (main + aux)`. Compatible with existing LSGAN helpers.
- `forward_multiscale(x)` — returns `(main, aux)` separately for callers
  that want per-scale supervision.

Args:

| Arg | Default | Description |
|---|---|---|
| `input_nc` | 3 | Input image channels |
| `base_channels` | 64 | Feature width of first conv |
| `n_layers` | 3 | Strided downsampling layers |
| `mbstd_group` | 4 | MinibatchStdDev group size (0 = disable) |

---

## init_weights_v4

Weight initialisation policy:

- `Conv2d` / `ConvTranspose2d`: Normal(0, 0.02)
- `InstanceNorm2d` scale: Normal(1, 0.02), bias = 0

---

## getDiscriminatorV4

Build, initialise, and smoke-test a `PatchGANDiscriminator`.

New arg in v4.2:

- `mbstd_group` (int, default 4) — MinibatchStdDev group size; 0 = disabled.

Smoke test verifies three output shapes:

```
[getDiscriminatorV4] merged output: (N, 1, H', W')
[getDiscriminatorV4] main   output: (N, 1, H', W')
[getDiscriminatorV4] aux    output: (N, 1, H'', W'')
```

Returns: initialised `PatchGANDiscriminator` on the requested device.
