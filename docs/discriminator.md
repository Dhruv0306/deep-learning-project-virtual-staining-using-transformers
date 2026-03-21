# `discriminator.py` — v1 Discriminator

**Model:** Hybrid UVCGAN + CycleGAN (v1)  
**Role:** Classifies image patches as real or fake. Two instances are created — `D_A` judges domain A (unstained) images, `D_B` judges domain B (stained) images.

---

## Architecture Overview

The v1 discriminator is a **PatchGAN** — a fully convolutional network producing a spatial grid of real/fake logits rather than a single scalar. Each value in the output grid corresponds to a **70×70 receptive field** patch of the 256×256 input. This design encourages the generator to produce locally realistic textures rather than just globally plausible images.

```
Input Image (N, 3, 256, 256)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Conv(3→64, k=4, s=2, p=1) + LeakyReLU(0.2)       │
│  Output:  (N, 64, 128, 128)                                 │
│  Note: No normalisation on first layer                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Conv(64→128, k=4, s=2, p=1) + IN + LeakyReLU(0.2)│
│  Output:  (N, 128, 64, 64)                                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Conv(128→256, k=4, s=2, p=1) + IN + LeakyReLU(0.2│
│  Output:  (N, 256, 32, 32)                                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Conv(256→512, k=4, s=1, p=1) + IN + LeakyReLU(0.2│ ← stride=1
│  Output:  (N, 512, 31, 31)                                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Conv(512→1, k=4, s=1, p=1)                        │ ← logit output
│  Output:  (N, 1, 30, 30)                                    │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
Logit map (N, 1, 30, 30)
Each value = real/fake score for one 70×70 patch of the input.
Higher = more likely real. No activation (raw logits).
```

---

## Design Choices Explained

**Why no normalisation on layer 1?**  
Normalising the very first layer removes low-level statistics (mean colour, intensity range) that are important for the discriminator to judge whether an image looks like it belongs to the correct domain. All subsequent layers use InstanceNorm.

**Why InstanceNorm instead of BatchNorm?**  
InstanceNorm normalises per-image rather than per-batch. This makes the discriminator robust to small batch sizes and better at capturing per-image style variations — important for histology where staining intensity varies across slides.

**Why LeakyReLU(0.2) instead of ReLU?**  
Standard ReLU kills negative activations, which can cause dead neurons in discriminators. LeakyReLU preserves a small gradient (slope 0.2) for negative values, keeping all neurons active throughout training.

**Why stride-1 at layer 4?**  
The last strided layer uses stride=1 to increase the receptive field without further halving spatial resolution. This provides a larger patch context for the final logit without losing spatial resolution in the output map.

---

## Classes

### `PatchDiscriminator`

The single-scale PatchGAN discriminator used in v1.

| Constructor Parameter | Default | Description |
|---|---|---|
| `input_nc` | 3 | Number of input image channels |

| Attribute | Description |
|---|---|
| `model` | `nn.Sequential` containing all 5 convolutional layers |

**`forward(x)`**

| Parameter | Shape | Description |
|---|---|---|
| `x` | `(N, input_nc, H, W)` | Input image tensor |

Returns: logit map `(N, 1, H', W')`. Each value is an unbounded real/fake score for one patch. Used with LSGAN loss (MSE against 0.97 for real, 0 for fake).

---

## Functions

### `getDiscriminators()`

Factory function. Creates two `PatchDiscriminator` instances (`D_A` and `D_B`), applies `init_weights` from `generator.py` to both, runs a smoke-test forward pass to verify output shapes, and returns them.

**Returns:** `(D_A, D_B)` — both on CUDA if available, otherwise CPU.

---

## How the Discriminator Fits Into Training

During each training step the discriminator is updated after the generator step:

```
# Step 1: Update generators (D frozen)
loss_G, fake_A, fake_B = loss_fn.generator_loss(real_A, real_B, G_AB, G_BA, D_A, D_B, ...)

# Step 2: Update D_A
loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A, loss_fn.fake_A_buffer)

# Step 3: Update D_B
loss_D_B = loss_fn.discriminator_loss(D_B, real_B, fake_B, loss_fn.fake_B_buffer)
```

The discriminator receives a mix of fresh fakes and buffered fakes (via `ReplayBuffer`) to avoid oscillating between the current and previous generator outputs.
