# model_v3/discriminator.py - v3 ProjectionDiscriminator

Source of truth: ../../model_v3/discriminator.py

Role: Dual discriminators (D_A, D_B) for adversarial training. Each discriminator uses a three-branch architecture (local PatchGAN + global CNN + spectral FFT) optimized for DiT-generated images.

---

## Design Philosophy: Why Not Reuse v2?

The v2 multi-scale PatchGAN was designed for CNN generators. For DiT-generated images, three failure modes emerge:

### 1) Global Structure Inconsistency
- V2's 70×70 receptive field covers only ~7% of a 256×256 image
- DiT outputs can be locally correct but globally incoherent (wrong stain distribution, colour cast, tissue morphology mismatch)
- **Solution**: Add a dedicated global branch with 100% receptive field

### 2) VAE Decode Artifacts
- Frozen SD VAE (8× compression) introduces periodic high-frequency ringing
- Standard PatchGAN is blind to periodic patterns (no frequency-domain sensitivity)
- **Solution**: Add spectral FFT branch to catch periodic artifacts directly in frequency domain

### 3) Over-Smoothness in Early Training
- Diffusion models minimize MSE in latent space → tend toward blurry outputs
- PatchGAN gives weak gradient signal because blurred images fool it
- **Solution**: Global branch provides complementary low-frequency signal

---

## Architecture Overview

`ProjectionDiscriminator` = Three parallel branches, outputs summed before LSGAN loss:

```
Input: (N, 3, 256, 256) [image in [-1, 1]]
  ├─ Branch 1: Local (SpectralNormDiscriminator)      → (N, 1, 8, 8) spatial logits
  ├─ Branch 2: Global (GlobalDiscriminatorBranch)     → (N, 1) scalar logit
  └─ Branch 3: Spectral (FFTDiscriminatorBranch)      → (N, 1) scalar logit
         ↓
[All branches' outputs handled by LSGAN loss (list-aware)]
```

---

## Branch 1: Local PatchGAN

**Reused from v2**: `SpectralNormDiscriminator`

**Purpose**: Capture fine texture, local stain granularity, high-frequency details.

**Architecture** (n_layers=3):

```
Input: (N, 3, 256, 256)
↓ Conv(3→64, stride=2)      SpectralNorm, LeakyReLU → (N,  64, 128, 128)
↓ Conv(64→128, stride=2)    SpectralNorm, IN,  LeakyReLU → (N, 128,  64,  64)
↓ Conv(128→256, stride=2)   SpectralNorm, IN,  LeakyReLU → (N, 256,  32,  32)
↓ Conv(256→256, stride=1)   SpectralNorm, IN,  LeakyReLU → (N, 256,  32,  32)
↓ Conv(256→1, stride=1)     SpectralNorm                 → (N,   1,  32,  32)
```

**Receptive field**: ~70×70 at finest scale (covers ~7% of 256×256 image)

**Output shape**: (N, 1, 8, 8) spatial logits (or similar, depends on pooling)

---

## Branch 2: Global Discriminator

**Class**: `GlobalDiscriminatorBranch`

**Purpose**: Judge entire image globally (100% receptive field).

**Architecture**:

```
Input: (N, 3, 256, 256)
↓ Conv(3→64, stride=4, k=4, p=0)    SpectralNorm, LeakyReLU → (N,  64,  64,  64)
↓ Conv(64→128, stride=4, k=4, p=0)  SpectralNorm, IN,  LeakyReLU → (N, 128,  16,  16)
↓ Conv(128→256, stride=4, k=4, p=0) SpectralNorm, IN,  LeakyReLU → (N, 256,   4,   4)
↓ Conv(256→1, stride=1, k=4, p=0)   SpectralNorm (bias=True)         → (N,   1,   1,   1)
↓ Squeeze
Output: (N, 1) scalar per image
```

**Key Design Choices**:

- **Stride=4**: Aggressive downsampling to reach 1×1 in fewer layers
- **No padding**: Reduces spatial size more quickly
- **No IN on first layer**: Standard PatchGAN practice
- **Single scalar output**: Forces discriminator to judge overall image statistics (colour distribution, tissue layout)

**Receptive field**: 100% (covers entire 256×256 image)

---

## Branch 3: Spectral (FFT) Discriminator

**Class**: `FFTDiscriminatorBranch`

**Purpose**: Detect periodic VAE decode artifacts in frequency domain.

**Forward Dataflow**:

```
Step 1: Convert to grayscale (float32)
  Input: (N, 3, H, W)
  Gray = 0.299*R + 0.587*G + 0.114*B  → (N, 1, 256, 256)

Step 2: Compute 2D real FFT
  gray: (N, 1, 256, 256) ──rfft2──> complex (N, 1, 256, 129)
                                       ↓ (W//2+1 = 129)

Step 3: Log-magnitude
  log_mag = log(1 + |FFT|)  → (N, 1, 256, 129)

Step 4: Normalize per-sample
  mean = log_mag.mean(dim=[2,3], keepdim=True)
  log_mag = log_mag / mean  [brightness-invariant]

Step 5: CNN on frequency map
  (N, 1, 256, 129) ──CNN──> (N, 1) scalar logit
```

**CNN Architecture** (base_channels=32):

```
Input: (N, 1, 256, 129)
↓ Conv(1→32, stride=2, k=4, p=1)   SpectralNorm, LeakyReLU → (N,  32, 128,  65)
↓ Conv(32→64, stride=2, k=4, p=1)  SpectralNorm, IN,  LeakyReLU → (N,  64,  64,  33)
↓ Conv(64→128, stride=2, k=4, p=1) SpectralNorm, IN,  LeakyReLU → (N, 128,  32,  17)
↓ Conv(128→128, stride=2, k=4, p=1) SpectralNorm, IN, LeakyReLU → (N, 128,  16,   9)
↓ Global Average Pool                                             → (N, 128)
↓ Linear(128→1)                      SpectralNorm                 → (N, 1)
```

**Key Design Choices**:

- **Float32 FFT**: Always computed in float32 for numerical stability
- **Log-magnitude** (log₁ₚ, not log): Avoids log(0) on zero-frequency components
- **Grayscale**: Preserves luminance (where most frequency structure lives)
- **Normalized**: Per-sample normalization makes it brightness-invariant
- **Lighter CNN**: base_channels=32 (vs 64 for local/global) because frequency map has less spatial information

**Why FFT?**:

- Periodic ringing at specific spatial frequencies from VAE 8× compression is invisible to PatchGAN
- FFT magnitude is shift-invariant: catches artifacts regardless of position
- Log-scale accounts for the 1/f energy distribution in natural images

---

## Integration with Loss Functions

**Output Format**:

```python
discriminator_logits = [local_logits, global_logit, spectral_logit]
# local_logits:    (N, 1, 8, 8) spatial map
# global_logit:    (N, 1)       scalar
# spectral_logit:  (N, 1)       scalar
```

**Loss Compatibility**:

The existing `_lsgan_disc_loss` and `_lsgan_gen_loss` (in model_v3/losses.py) already handle list-of-tensors output:

```python
def _lsgan_disc_loss(real_outputs, fake_outputs):
    if isinstance(real_outputs, (list, tuple)):
        return torch.stack([...for r, f in zip(...)...]).mean()
    return _single(real_outputs, fake_outputs)
```

**No changes needed** in the loss module — the multi-branch format is transparent.

---

## VRAM Overhead

- Local branch: ~same as v2 multi-scale (single scale)
- Global branch: lightweight (3 conv layers)
- Spectral branch: FFT is O(n log n), frequency map (N, 1, H, 129) is immediately downsampled by CNN

**Total**: Slightly higher than v2 single-scale, significantly lower than v2 three-scale.

---

## Factory Function

```python
D_A, D_B = getDiscriminatorsV3(
    input_nc=3,
    base_channels=64,
    n_layers=3,
    global_base_channels=64,
    fft_base_channels=32,
    use_local=True,
    use_global=True,
    use_fft=True,
    device=torch.device("cuda")
)
```

**Returns**: Two identical `ProjectionDiscriminator` instances (one per domain).

---

## Configuration Parameters (from DiffusionConfig)

| Parameter | Role | Default |
|-----------|------|---------|
| `disc_base_channels` | Local branch feature channels | 64 |
| `disc_n_layers` | Local branch strided conv layers | 3 |
| `disc_global_base_channels` | Global branch feature channels | 64 |
| `disc_fft_base_channels` | Spectral branch feature channels | 32 |
| `disc_use_local` | Enable local PatchGAN branch | true |
| `disc_use_global` | Enable global CNN branch | true |
| `disc_use_fft` | Enable spectral FFT branch | true |
| `use_r1_penalty` | Enable R1 regularization on D | true |
| `r1_gamma` | R1 penalty strength | 100 |
| `r1_interval` | Apply R1 every N steps | 16 |
| `adaptive_d_update` | Skip D update if loss < threshold | false |
| `adaptive_d_loss_threshold` | Threshold for adaptive update | 0.5 |

---

## Training Integration

**Step Order** (per batch):

1. Generator forward → produces fake images
2. Disable discriminator gradients
3. Generator backward
4. Generator step
5. Enable discriminator gradients
6. Discriminator forward (real + fake from replay buffer)
7. Optional R1 penalty on real images
8. Discriminator backward
9. Discriminator step

**Adversarial Loss** (LSGAN):

```
L_D = MSE(D(real), 1) + MSE(D(fake), 0)
L_G = MSE(D(fake_G), 1)
```

All three branch logits are averaged by the loss function.

---

## Reference

- **v2 MultiScaleDiscriminator**: model_v2/discriminator.py
- **LSGAN losses**: model_v3/losses.py (_lsgan_disc_loss, _lsgan_gen_loss)
- **Training loop**: model_v3/training_loop.py (discriminator steps)
