# `advanced_losses.py` — V2 Loss Functions

Source of truth: `../advanced_losses.py`

**Model:** True UVCGAN v2  
**Role:** Defines all loss terms used during v2 training. The composite `UVCGANLoss` class computes every loss component for each training step — generator and discriminator — using LSGAN as the adversarial objective and a paper-aligned one-sided gradient penalty.

---

## Loss Formula

```
Generator total loss:
L_G = L_GAN_AB + L_GAN_BA
    + λ_cycle   × (L_cycle_A     + L_cycle_B)
    + λ_idt     × (L_idt_A       + L_idt_B)       ← λ_idt decays after epoch 50%
    + λ_cyc_p   × (L_cyc_p_A    + L_cyc_p_B)
    + λ_idt_p   × (L_idt_p_A    + L_idt_p_B)
    + λ_spectral × (L_spec_AB   + L_spec_BA)       ← 0.0 by default
    + λ_contrast × (L_cont_AB   + L_cont_BA)       ← 0.0 by default

Discriminator loss (per domain):
L_D = 0.5 × (L_real + L_fake) + λ_gp × GP
```

---

## Class: `VGGPerceptualLossV2`

Perceptual loss using **four** VGG19 feature levels. V1's `VGGPerceptualLoss` uses three levels; v2 adds `relu4_4` for higher-level semantic matching.

### `__init__(resize_to=128, weights=(1.0, 1.0, 1.0, 1.0))`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `resize_to` | int | 128 | Images are bilinearly resized to this resolution before VGG. Reduces VRAM and compute. 8 GB config uses 64 |
| `weights` | tuple | (1,1,1,1) | Per-level loss weights `(w1, w2, w3, w4)` |

**VGG19 feature slices:**

| Attribute | VGG19 layers | Feature name | What it captures |
|---|---|---|---|
| `slice1` | 0–3 | `relu1_2` | Low-level edges and colours |
| `slice2` | 4–8 | `relu2_2` | Mid-level textures |
| `slice3` | 9–17 | `relu3_4` | Higher-level patterns |
| `slice4` | 18–26 | `relu4_4` | Semantic structures |

All VGG parameters are frozen (`requires_grad=False`).

**Registered buffers:**
- `mean` — ImageNet mean `[0.485, 0.456, 0.406]` shaped `(1, 3, 1, 1)`
- `std` — ImageNet std `[0.229, 0.224, 0.225]` shaped `(1, 3, 1, 1)`

Buffers are used instead of plain tensors so they automatically move to the correct device when `.to(device)` is called on the module.

### `_normalize(x)`

Applies ImageNet channel normalisation expected by VGG19:
```
x_vgg = (x - mean) / std
```

### `_extract(x)`

Passes a normalised image through all four frozen slices and returns `(h1, h2, h3, h4)`.

### `forward(x, y)`

**Data flow:**
```
x (generated), y (target)
    │
    ▼ bilinear resize to resize_to × resize_to (if needed)
    ▼ _normalize (ImageNet stats)
    ▼ _extract → (h1_x, h2_x, h3_x, h4_x)
    ▼ _extract → (h1_y, h2_y, h3_y, h4_y)
    ▼
loss = w1×L1(h1_x,h1_y) + w2×L1(h2_x,h2_y) + w3×L1(h3_x,h3_y) + w4×L1(h4_x,h4_y)
```

Returns a scalar. Grayscale inputs are expanded to 3 channels before being passed to VGG.

---

## Class: `SpectralLoss`

Frequency-domain loss that penalises differences in the power spectrum of generated vs real images.

### `forward(x, y)`

```
x, y
    │
    ▼ torch.fft.rfft2(..., norm="ortho")   [2D real FFT]
    ▼ torch.log1p(abs(...))                [log-magnitude spectrum]
    ▼ F.l1_loss(...)                       [L1 distance in frequency space]
```

`rfft2` computes the 2D real FFT, which is more efficient than the full complex FFT for real-valued inputs. `log1p` compresses the dynamic range so that large low-frequency components do not dominate the loss over smaller high-frequency details.

This loss is **disabled by default** (`lambda_spectral=0.0`). Enable it once GAN training is stable to encourage generated images to match the stain's characteristic frequency signature.

---

## Class: `ContrastiveLoss`

NT-Xent (Normalised Temperature-scaled Cross Entropy) contrastive loss for domain alignment. Encourages the bottleneck representation of `G_AB(real_A)` to be closer to real domain B features than to real domain A features in a projected embedding space.

### `__init__(in_features=512, proj_dim=128, temperature=0.07)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `in_features` | int | 512 | Bottleneck channel count after global avg pooling (= `base_channels × 8` = 512 for default config) |
| `proj_dim` | int | 128 | Output dimension of the projection head |
| `temperature` | float | 0.07 | Softmax temperature. Lower = sharper similarity distribution, harder negatives |

**Internal attribute:**
- `projection` — `nn.Sequential`: `Linear(512→512) → ReLU → Linear(512→128)`. Maps pooled bottleneck features to a normalised embedding space where cosine similarity is computed

### `_project(x)`

If `x` is a 4D feature map `(N, C, H, W)`, applies global average pooling first to get `(N, C)`. Then passes through `projection` and L2-normalises the output. Returns a unit-norm embedding vector per sample.

### `forward(anchor, positive, negative)`

```
anchor   = bottleneck of G_AB(real_A)     ← the translated feature
positive = bottleneck of G_AB(real_B)     ← a real target domain B feature
negative = bottleneck of G_BA(real_A)     ← a real source domain A feature

z_a = _project(anchor)      ← unit-norm embedding
z_p = _project(positive)
z_n = _project(negative)

sim_pos = dot(z_a, z_p) / temperature    ← similarity to positive
sim_neg = dot(z_a, z_n) / temperature    ← similarity to negative

logits = stack([sim_pos, sim_neg], dim=1)   ← shape (N, 2)
labels = zeros (class 0 = positive is correct)
loss   = cross_entropy(logits, labels)
```

This loss is **disabled by default** (`lambda_contrastive=0.0`). Only instantiated when `lambda_contrastive > 0.0`.

---

## Class: `LSGANGradientPenalty`

Paper-aligned one-sided gradient penalty. This is the key algorithmic difference from v1's standard two-sided WGAN-GP.

### Standard WGAN-GP vs UVCGAN v2

**Standard WGAN-GP (two-sided, Gulrajani 2017):**
```
GP = E[ (‖∇D(x̂)‖₂ - 1)² ]
```
Penalises gradients both above AND below 1. Forces D to be exactly 1-Lipschitz — appropriate for Wasserstein distance but unnecessarily restrictive for LSGAN.

**UVCGAN v2 (one-sided, Prokopenko 2023):**
```
GP = E[ max(0, ‖∇D(x̂)‖₂ - γ)² ] / γ²       γ = 100
```
Only penalises gradients that **exceed** γ. Allows D to have small-norm gradients near real data, which is natural and desirable for LSGAN. The `γ²` normalisation keeps the penalty scale independent of the choice of γ, so `lambda_gp=0.1` works without retuning.

**Class variable:**
- `GAMMA = 100.0` — paper value. Do not change this without also retuning `lambda_gp`

### `gradient_penalty(D, real, fake)`

| Argument | Type | Description |
|---|---|---|
| `D` | `nn.Module` | Discriminator to penalise (D_A or D_B), must be in train mode |
| `real` | Tensor `(N, C, H, W)` | Real image batch — detached and cast to float32 internally |
| `fake` | Tensor `(N, C, H, W)` | Generated image batch — detached and cast to float32 internally |

**Steps:**
1. Cast both inputs to float32 and detach from any computation graph
2. Sample `ε ~ Uniform(0,1)` of shape `(N, 1, 1, 1)`
3. Create interpolated samples: `interp = ε×real + (1-ε)×fake`, set `requires_grad=True`
4. Run `interp` through `D` to get `pred`
5. For multi-scale D: `pred_scalar = torch.stack([p.sum() for p in pred]).sum()` — `torch.stack` is used instead of Python `sum()` to avoid the result being typed as `int | Any` by Pylance
6. For single-scale D: `pred_scalar = pred.sum()`
7. Compute `grads = autograd.grad(pred_scalar, interp, create_graph=True)`
8. `grad_norms = grads.view(N, -1).norm(2, dim=1)` — per-sample gradient L2 norms, shape `(N,)`
9. `penalty = F.relu(grad_norms - GAMMA)² / GAMMA²` — one-sided hinge, averaged over batch

**Must always be called with autocast DISABLED.** `UVCGANLoss.discriminator_loss` wraps the call in `torch.autocast(device_type=..., enabled=False)` to enforce this. Reason: `torch.autograd.grad` on float16 tensors produces NaN gradients because the Jacobian computation exceeds float16 numerical range.

---

## Class: `UVCGANLoss`

Composite loss manager for the v2 training loop. Owns all loss criteria, the gradient penalty, and the replay buffers.

### `__init__(...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda_cycle` | float | 10.0 | Cycle-consistency weight |
| `lambda_identity` | float | 5.0 | Identity weight before decay |
| `lambda_cycle_perceptual` | float | 0.1 | VGG perceptual weight on cycle outputs |
| `lambda_identity_perceptual` | float | 0.05 | VGG perceptual weight on identity outputs |
| `lambda_gp` | float | 0.1 | Gradient penalty weight. Paper value — much smaller than WGAN-GP's 10.0 because the γ²-normalised penalty has a different scale |
| `lambda_contrastive` | float | 0.0 | NT-Xent contrastive weight (0 = disabled) |
| `lambda_spectral` | float | 0.0 | Spectral frequency loss weight (0 = disabled) |
| `perceptual_resize` | int | 128 | VGG input resolution |
| `contrastive_temperature` | float | 0.07 | NT-Xent temperature |
| `lsgan_real_label` | float | 0.9 | Label-smoothing target for real samples. Prevents the discriminator from becoming over-confident |
| `identity_decay_start` | float | 0.5 | Fraction of training after which identity weight begins decaying |
| `identity_decay_rate` | float | 0.997 | Per-epoch multiplicative decay applied after `identity_decay_start` |
| `device` | torch.device | auto | Device for VGG and contrastive models |

**Internal attributes:**

| Attribute | Type | Description |
|---|---|---|
| `criterion_cycle` | `nn.L1Loss` | Cycle-consistency |
| `criterion_identity` | `nn.L1Loss` | Identity |
| `criterion_GAN` | `nn.MSELoss` | LSGAN adversarial |
| `criterion_perceptual` | `VGGPerceptualLossV2` | 4-level VGG perceptual |
| `criterion_spectral` | `SpectralLoss` | Frequency-domain (always instantiated, gated by lambda) |
| `gp` | `LSGANGradientPenalty` | One-sided gradient penalty |
| `criterion_contrastive` | `ContrastiveLoss` or `None` | Only instantiated when `lambda_contrastive > 0.0` |
| `fake_A_buffer` | `ReplayBuffer(50)` | Stores past fake A images for D_A stabilisation |
| `fake_B_buffer` | `ReplayBuffer(50)` | Stores past fake B images for D_B stabilisation |

---

### `get_identity_lambda(epoch, total_epochs)`

Returns the effective identity loss weight for the current epoch.

```
epoch ≤ identity_decay_start × total:
    return lambda_identity

epoch > identity_decay_start × total:
    return lambda_identity × identity_decay_rate ^ (epoch - identity_decay_start × total)
```

With defaults (`identity_decay_start=0.5`, `identity_decay_rate=0.997`, `total=200`):
- Epochs 0–100: full weight of 5.0
- Epochs 100–200: exponential decay, e.g. at epoch 150 → `5.0 × 0.997^50 ≈ 4.27`

---

### `_lsgan_disc_loss(real_outputs, fake_outputs)`

LSGAN discriminator loss. Handles both single-scale (Tensor) and multi-scale (list of Tensors) outputs from the discriminator.

```
For each scale (or for the single output):
    loss_real = MSE(real_out, lsgan_real_label × ones)
    loss_fake = MSE(fake_out, zeros)
    scale_loss = 0.5 × (loss_real + loss_fake)

Total = mean(scale_losses)
```

Uses `torch.stack([...]).mean()` for multi-scale to ensure the result is typed as `Tensor` rather than relying on Python's `sum()`.

---

### `_lsgan_gen_loss(disc_fake_outputs)`

LSGAN generator loss. Pushes the discriminator's outputs on fake images toward 1 (the target for real images).

```
For each scale (or for the single output):
    MSE(disc_fake_out, ones)

Total = mean across scales
```

---

### `generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, total_epochs)`

Computes the full generator loss for one training step. Returns `(loss_G, fake_A, fake_B)`.

**Cross-domain fusion availability check:**
```python
use_cross = (
    G_AB.use_cross_domain and G_BA.use_cross_domain
    and hasattr(G_AB, 'forward_with_cross_domain')
    and hasattr(G_BA, 'forward_with_cross_domain')
    and hasattr(G_AB, 'get_skip_features')
    and hasattr(G_BA, 'get_skip_features')
)
```

**Full data flow:**
```
real_A, real_B
    │
    ├── Identity (always standard forward — no cross-domain needed)
    │   idt_A = G_BA(real_A)   loss_idt_A = L1(idt_A, real_A) × λ_idt(epoch)
    │   idt_B = G_AB(real_B)   loss_idt_B = L1(idt_B, real_B) × λ_idt(epoch)
    │
    ├── Translation
    │   IF use_cross:
    │       skips_A = G_AB.get_skip_features(real_A)   ← encoder skips only, no decode
    │       skips_B = G_BA.get_skip_features(real_B)
    │       fake_B  = G_AB.forward_with_cross_domain(real_A, skips_B)
    │       fake_A  = G_BA.forward_with_cross_domain(real_B, skips_A)
    │   ELSE:
    │       fake_B = G_AB(real_A)
    │       fake_A = G_BA(real_B)
    │
    ├── LSGAN adversarial
    │   loss_GAN_AB = _lsgan_gen_loss(D_B(fake_B))
    │   loss_GAN_BA = _lsgan_gen_loss(D_A(fake_A))
    │
    ├── Cycle-consistency
    │   rec_A = G_BA(fake_B)   loss_cycle_A = L1(rec_A, real_A) × λ_cycle
    │   rec_B = G_AB(fake_A)   loss_cycle_B = L1(rec_B, real_B) × λ_cycle
    │
    ├── Perceptual cycle
    │   loss_cyc_p_A = VGG(rec_A, real_A.detach()) × λ_cyc_p
    │   loss_cyc_p_B = VGG(rec_B, real_B.detach()) × λ_cyc_p
    │
    ├── Perceptual identity
    │   loss_idt_p_A = VGG(idt_A, real_A.detach()) × λ_idt_p
    │   loss_idt_p_B = VGG(idt_B, real_B.detach()) × λ_idt_p
    │
    ├── Spectral (if λ_spectral > 0)
    │   loss_spectral = (SpectralLoss(fake_B, real_B)
    │                  + SpectralLoss(fake_A, real_A)) × λ_spectral
    │
    └── Contrastive (if criterion_contrastive is not None and λ_contrast > 0)
        Uses G_AB.encode() and G_BA.encode() to obtain bottleneck features:
        _, _, _, _, bot_AB = G_AB.encode(real_A)
        _, _, _, _, bot_B  = G_AB.encode(real_B)
        _, _, _, _, bot_BA = G_BA.encode(real_B)
        _, _, _, _, bot_A  = G_BA.encode(real_A)
        loss_contrastive = (ContrastiveLoss(bot_AB, bot_B, bot_A)
                           + ContrastiveLoss(bot_BA, bot_A, bot_B)) × λ_contrast

L_G = sum of all active terms
Returns: (loss_G, fake_A, fake_B)
```

**Why are `fake_A` and `fake_B` returned?** The discriminator step follows immediately in the training loop. Returning the already-computed fakes avoids re-running the generators, saving one full forward pass per step.

**Why `real_A.detach()` in perceptual terms?** The VGG network receives the real image as a target. It does not need to contribute gradients back into the real image (which has no parameters), so detaching prevents unnecessary gradient computation.

---

### `discriminator_loss(D, real, fake, replay_buffer=None)`

Computes the LSGAN discriminator loss plus the one-sided gradient penalty for one domain.

**Data flow:**
```
fake → replay_buffer.push_and_pop(fake.detach()) → fake_buf
       (or fake.detach() if no replay buffer)

real_out = D(real)
fake_out = D(fake_buf)
loss_D   = _lsgan_disc_loss(real_out, fake_out)

if lambda_gp > 0:
    with torch.autocast(device_type=..., enabled=False):   ← always float32
        gp = self.gp.gradient_penalty(D, real, fake)
    loss_D = loss_D + lambda_gp × gp.to(loss_D.dtype)

return loss_D
```

**Why `fake.detach()` before replay buffer?** The buffer stores tensors for future training steps. Detaching prevents the buffer from holding references to old computation graphs, which would cause memory leaks and incorrect gradients if those graphs were later used.

**Why cast GP back with `.to(loss_D.dtype)`?** The GP is computed in float32 but `loss_D` may be float16 under AMP. Casting the GP to match ensures the addition doesn't silently upcast or cause type errors.

---

## Loss Term Reference

| Term | Formula | Default weight | Notes |
|---|---|---|---|
| Adversarial G | `MSE(D(fake), ones)`, avg over scales | 1.0 | LSGAN |
| Adversarial D real | `MSE(D(real), 0.9×ones)` | 0.5× | Label smoothing |
| Adversarial D fake | `MSE(D(fake), zeros)` | 0.5× | Uses replay buffer |
| One-sided GP | `E[max(0, ‖∇D‖₂-100)²] / 100²` | λ=0.1 | Float32, outside autocast |
| Cycle | `L1(G_BA(G_AB(A)), A)` | λ=10.0 | Both directions |
| Identity | `L1(G_BA(A), A)` | λ=5.0 → decays | Decays from epoch 50% onward |
| Perceptual cycle | VGG19 L1, 4 levels | λ=0.1 | On reconstructed images |
| Perceptual identity | VGG19 L1, 4 levels | λ=0.05 | On identity outputs |
| Spectral | `L1(log\|FFT(fake)\|, log\|FFT(real)\|)` | λ=0.0 | Enable once stable |
| Contrastive | NT-Xent on projected bottleneck features | λ=0.0 | Enable once stable |
