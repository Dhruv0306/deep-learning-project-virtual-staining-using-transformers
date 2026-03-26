# `model_v1/losses.py` — v1 Loss Functions

**Model:** Hybrid UVCGAN + CycleGAN (v1)  
**Role:** Defines the composite loss used to train both generators and both discriminators.

---

## Loss Structure Overview

```
Generator total loss:

  L_G = L_GAN_AB + L_GAN_BA
      + λ_cycle    × (L_cycle_A    + L_cycle_B)
      + λ_identity × (L_idt_A      + L_idt_B)      ← decays after 50% of training
      + λ_cyc_perc × (L_cyc_perc_A + L_cyc_perc_B)
      + λ_idt_perc × (L_idt_perc_A + L_idt_perc_B)


Discriminator loss (computed independently for D_A and D_B):

  L_D = 0.5 × (L_real + L_fake) + λ_gp × GP
```

---

## Data Flow Through `generator_loss`

```
real_A ──► G_BA ──► idt_A    L_idt_A     = L1(idt_A, real_A)      × λ_idt
real_B ──► G_AB ──► idt_B    L_idt_B     = L1(idt_B, real_B)      × λ_idt

real_A ──► G_AB ──► fake_B ──► D_B ──► pred_fake_B
                               L_GAN_AB  = MSE(pred_fake_B, 1)

real_B ──► G_BA ──► fake_A ──► D_A ──► pred_fake_A
                               L_GAN_BA  = MSE(pred_fake_A, 1)

fake_B ──► G_BA ──► rec_A    L_cycle_A  = L1(rec_A, real_A)       × λ_cycle
fake_A ──► G_AB ──► rec_B    L_cycle_B  = L1(rec_B, real_B)       × λ_cycle

VGG(rec_A, real_A) ──► L_cyc_perc_A   × λ_cyc_perc
VGG(rec_B, real_B) ──► L_cyc_perc_B   × λ_cyc_perc
VGG(idt_A, real_A) ──► L_idt_perc_A   × λ_idt_perc
VGG(idt_B, real_B) ──► L_idt_perc_B   × λ_idt_perc

returns: loss_G (sum of all terms), fake_A, fake_B
```

---

## Classes

### `VGGPerceptualLoss`

Measures perceptual similarity between two images using frozen VGG19 feature activations at three levels. Images that look similar to a human produce similar intermediate VGG representations even if their pixel values differ, making this a good structural similarity measure.

| Constructor Parameter | Default | Description |
|---|---|---|
| `resize_to` | 128 | Both images are resized to this resolution before VGG. Lower = less VRAM and faster, at some loss of fine detail in the perceptual signal. `None` skips resizing. |

| Attribute | Type | Description |
|---|---|---|
| `slice1` | `nn.Sequential` | VGG19 layers 0–3, producing relu1_2 activations |
| `slice2` | `nn.Sequential` | VGG19 layers 4–8, producing relu2_2 activations |
| `slice3` | `nn.Sequential` | VGG19 layers 9–17, producing relu3_4 activations |
| `mean` | buffer | ImageNet mean `[0.485, 0.456, 0.406]` reshaped to `(1,3,1,1)` |
| `std` | buffer | ImageNet std `[0.229, 0.224, 0.225]` reshaped to `(1,3,1,1)` |

All VGG parameters have `requires_grad=False` — the VGG19 weights are never updated.

**Three feature levels and what they capture:**

| Level | VGG layer | Captures |
|---|---|---|
| relu1_2 | After 2nd conv | Low-level: edges, colours, fine texture |
| relu2_2 | After 4th conv | Mid-level: patterns, local structures |
| relu3_4 | After 8th conv | High-level: object parts, spatial layout |

**`forward(x, y)`** — computes the sum of L1 losses at all three levels:
```
loss = L1(slice1(norm(x)), slice1(norm(y)))
     + L1(slice2(slice1(norm(x))), ...)
     + L1(slice3(slice2(slice1(norm(x)))), ...)
```

**`extract_features(x)`** — runs `x` through the three slices sequentially and returns `(h1, h2, h3)` feature maps.

**`normalize(x)`** — applies `(x - mean) / std`. Required because VGG19 was trained on ImageNet-normalised inputs; passing unnormalised images would produce meaningless features.

---

### `CycleGANLoss`

Composite loss class. Owns all loss criteria, replay buffers, and the identity decay schedule. Exposes `generator_loss` and `discriminator_loss` methods called from the training loop.

| Constructor Parameter | Default | Description |
|---|---|---|
| `lambda_cycle` | 10.0 | Cycle-consistency weight. Enforces that G_BA(G_AB(A)) ≈ A. High value = strong content preservation. |
| `lambda_identity` | 5.0 | Identity loss weight. Enforces G_BA(A) ≈ A (mapping within the same domain is near-identity). Helps preserve image colour and structure. |
| `lambda_cycle_perceptual` | 0.1 | VGG19 perceptual weight on cycle-reconstructed images. |
| `lambda_identity_perceptual` | 0.05 | VGG19 perceptual weight on identity-mapped images. |
| `lambda_gp` | 10.0 | Two-sided gradient penalty weight. Set to 0 to disable. |
| `perceptual_resize` | 128 | VGG19 input resolution. |
| `device` | auto | Torch device. Auto-detected if `None`. |

| Attribute | Type | Description |
|---|---|---|
| `criterion_GAN` | `nn.MSELoss` | LSGAN adversarial loss |
| `criterion_cycle` | `nn.L1Loss` | Cycle-consistency |
| `criterion_identity` | `nn.L1Loss` | Identity mapping |
| `criterion_perceptual` | `VGGPerceptualLoss` | Perceptual loss |
| `fake_A_buffer` | `ReplayBuffer(50)` | Stores past fake domain-A images |
| `fake_B_buffer` | `ReplayBuffer(50)` | Stores past fake domain-B images |

---

#### `get_identity_lambda(epoch, total_epochs)`

Returns the effective identity loss weight at a given epoch. Held constant for the first 50% of training, then decays exponentially:

```
if epoch ≤ 0.5 × total_epochs:
    return lambda_identity
else:
    return lambda_identity × 0.997^(epoch − 0.5×total_epochs)
```

Decaying the identity loss in later training allows the generator to focus more on domain translation once the basic structure is preserved.

---

#### `gradient_penalty(D, real, fake)`

Computes the two-sided WGAN gradient penalty on interpolated samples between real and fake images:

```
ε ~ Uniform(0, 1)   shape (N, 1, 1, 1)
x̂ = ε × real + (1−ε) × fake     ← random interpolation
penalty = E[ (‖∇_x̂ D(x̂)‖₂ − 1)² ]
```

This enforces the 1-Lipschitz constraint on D, preventing its gradient norms from growing unboundedly. Always computed in float32 regardless of AMP state to avoid numerical instability.

| Parameter | Description |
|---|---|
| `D` | The discriminator to compute the penalty for |
| `real` | Real image batch |
| `fake` | Generated fake image batch (detached internally) |

---

#### `generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, total_epochs)`

Computes the complete generator loss for one training step.

| Parameter | Description |
|---|---|
| `real_A` | Real unstained images `(N, 3, H, W)` |
| `real_B` | Real stained images `(N, 3, H, W)` |
| `G_AB` | Generator: unstained → stained |
| `G_BA` | Generator: stained → unstained |
| `D_A` | Discriminator for domain A |
| `D_B` | Discriminator for domain B |
| `epoch` | Current epoch (0-indexed), used for identity decay |
| `total_epochs` | Total training epochs, used for identity decay |

**Returns:** `(loss_G, fake_A, fake_B)` — `fake_A` and `fake_B` are returned so the training loop can pass them to the discriminator step without re-running the generators.

---

#### `discriminator_loss(D, real, fake, replay_buffer=None)`

Computes the LSGAN discriminator loss for a single domain.

| Parameter | Description |
|---|---|
| `D` | Discriminator for this domain |
| `real` | Real images from this domain |
| `fake` | Newly generated fakes (from the generator step) |
| `replay_buffer` | Optional `ReplayBuffer`. Pass `loss_fn.fake_A_buffer` for D_A, `fake_B_buffer` for D_B. |

**Internal steps:**
```
1. fake_buf = replay_buffer.push_and_pop(fake)   ← mix old and new fakes

2. pred_real = D(real)
   loss_real  = MSE(pred_real,  0.97 × ones)     ← label smoothing

3. pred_fake = D(fake_buf.detach())
   loss_fake  = MSE(pred_fake, zeros)

4. loss_D = (loss_real + loss_fake) × 0.5

5. if lambda_gp > 0:
       loss_D += lambda_gp × gradient_penalty(D, real, fake)
```

The `0.97` real target is **one-sided label smoothing** — it prevents the discriminator from becoming overconfident on real images, which keeps its gradients informative for the generator.

