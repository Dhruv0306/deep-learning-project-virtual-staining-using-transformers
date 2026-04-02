# Model V4 Dataflow Pipeline

Model v4 is the CUT-style unpaired translation pipeline for virtual staining. It keeps the adversarial A <-> B setup, adds a Transformer encoder generator option, and trains with PatchNCE contrastive loss over sampled feature patches.

## What Lives Here

- [generator.md](generator.md) - ResNet baseline and Transformer encoder generator
- [discriminator.md](discriminator.md) - PatchGAN discriminator implementation
- [transformer_blocks.md](transformer_blocks.md) - patch embedding and Transformer utilities
- [patch_sampler.md](patch_sampler.md) - shared spatial patch sampling for PatchNCE
- [nce_loss.md](nce_loss.md) - contrastive PatchNCE objective
- [training_loop.md](training_loop.md) - full CUT + Transformer optimization loop

## End-to-End Dataflow

```
real_A -> G_AB -> fake_B -> D_B / PatchNCE / identity
real_B -> G_BA -> fake_A -> D_A / PatchNCE / identity

real_A / real_B
    -> shared data loader
    -> normalized image batch
    -> generator forward pass
    -> feature maps returned from encoder layers
    -> patch sampler reuses shared spatial ids
    -> PatchNCE aligns real and generated patches
    -> discriminator evaluates real and fake images
    -> weighted losses drive optimizer step
    -> EMA generator updated for validation and test export
```

## Training Pipeline

### 1. Input preparation

Each batch contains an unpaired sample from domain A and domain B. The dataloader normalizes both domains to `[-1, 1]`, matching the `Tanh` output of the generators.

### 2. Bidirectional translation path

The main forward pass runs both directions every iteration:

- `G_AB(real_A) -> fake_B`
- `G_BA(real_B) -> fake_A`

These translated images are then reused for adversarial loss, optional identity loss, validation visualization, and the cycle-style reconstructions used during evaluation.

### 3. Feature alignment path

When the Transformer generator is enabled, intermediate encoder features are exposed to the PatchNCE branch. The patch sampler picks matching spatial ids from real and generated feature maps so the loss can compare corresponding local structures rather than global image statistics.

### 4. Loss path

The total generator objective combines:

- LSGAN adversarial loss
- PatchNCE contrastive loss
- identity loss on same-domain inputs

The discriminator objective is the standard LSGAN real-vs-fake score-map loss.

### 5. Validation path

Validation uses the EMA copies of `G_AB` and `G_BA` when enabled. It reports SSIM, PSNR, and optional FID for both directions, then saves side-by-side comparison grids for visual inspection.

## Model Map

| File | Role in the pipeline |
|---|---|
| `generator.py` | Builds the CNN and Transformer generator variants |
| `transformer_blocks.py` | Provides patch embedding and Transformer block primitives |
| `patch_sampler.py` | Samples aligned spatial patches for contrastive learning |
| `nce_loss.py` | Computes PatchNCE over matched patch features |
| `discriminator.py` | Scores realism with a PatchGAN discriminator |
| `training_loop.py` | Coordinates GAN, NCE, identity, EMA, and validation steps |

## Recommended Reading Order

1. Start with [training_loop.md](training_loop.md) to see the full v4 optimization flow.
2. Read [generator.md](generator.md) and [transformer_blocks.md](transformer_blocks.md) for the model backbone.
3. Read [patch_sampler.md](patch_sampler.md) and [nce_loss.md](nce_loss.md) to see how PatchNCE is formed.
4. Read [discriminator.md](discriminator.md) for the adversarial branch.