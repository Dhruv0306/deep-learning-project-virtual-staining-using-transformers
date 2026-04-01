# CUT + Transformer (v4)

---

# Project Overview

This project implements a lightweight unpaired image-to-image translation model based on:

* Contrastive learning (CUT paradigm)
* Transformer-based generator encoder
* Patch-based discriminator (GAN)

---

## Objectives

* Support unpaired datasets
* Be VRAM-efficient (<= 8GB), prioritize lightweight design
* Replace CNN encoder with Transformer backbone
* Remove expensive cycle consistency (v3)

---

# High-Level Architecture

```
Input Image (A)
   ->
Generator (Transformer Encoder + CNN Decoder)
   ->
Fake Image (B)
   ->
 +-------------------+
 | GAN Loss (D)      |
 +-------------------+
 +-------------------------------+
 | PatchNCE Contrastive Loss     |
 +-------------------------------+
```

---

# Project Structure

```
model_v4/
 |-- generator.py            # Transformer-based generator
 |-- discriminator.py        # PatchGAN discriminator
 |-- transformer_blocks.py   # ViT / Swin blocks
 |-- patch_sampler.py        # Patch extraction for NCE
 |-- nce_loss.py             # PatchNCE loss
 |-- training_loop.py        # Full training pipeline
 |-- inference.py            # Image translation
 |-- utils/
 |   |-- replay_buffer.py
 |   |-- image_utils.py
 |   `-- metrics.py
```

---

# Phase-wise Implementation Plan

---

# Phase 1 — Baseline GAN (No Transformer, No NCE)

## Goal

Establish stable unpaired translation baseline.

---

## Components

### 1. Generator (CNN)

* ResNet / UNet style
* Input -> Conv -> Residual blocks -> Upsample -> Output

### 2. Discriminator

* PatchGAN (70x70)

### 3. Losses

* LSGAN:

```
L_GAN = (D(fake) - 1)^2
L_D   = (D(real) - 1)^2 + (D(fake))^2
```

---

## Deliverables

* `generator.py` (CNN version)
* `discriminator.py`
* `training_loop.py` (GAN only)

---

## Exit Criteria

* Stable GAN training
* No NaNs
* Reasonable visual outputs

---

# Phase 2 — Add CUT (Contrastive Learning)

## Goal

Replace cycle loss with PatchNCE loss.

---

## Components

### 1. Feature Extraction

* Extract intermediate features from generator encoder
* Return a list of features for NCE

### 2. Patch Sampler

* Random spatial sampling from feature maps

### 3. PatchNCE Loss

* InfoNCE loss between:
  * real image features
  * generated image features

### 4. NCE Details (Lightweight Defaults)

* Per-layer MLP projection head (small MLP, e.g., 2 layers)
* L2-normalize projected features
* Temperature `tau` (start with 0.07 or 0.1)
* Positives: same spatial location across real_A and fake_B
* Negatives: other patches in the same batch

### 5. Early Stability Add-ons (Lightweight First)

* Enable mixed precision (AMP) for first full CUT run
* Add replay buffer for discriminator stability

---

## Loss

```
L_G = lambda_GAN * L_GAN + lambda_NCE * L_NCE
```

---

## Deliverables

* `patch_sampler.py`
* `nce_loss.py`
* Modify `generator.py` to expose intermediate features
* Update `training_loop.py`

---

## Exit Criteria

* Cycle loss removed
* Stable training
* Better content preservation

---

# Phase 3 — Transformer Encoder Integration

## Goal

Replace CNN encoder with Transformer, keeping a lightweight footprint.

---

## Components

### 1. Patch Embedding

* Conv -> flatten -> tokens
* Token grid size: H' = H / patch_size, W' = W / patch_size

### 2. Transformer Blocks

* Multi-head attention
* MLP
* LayerNorm

### 3. Feature Hooks

* Extract features from multiple transformer layers
* Convert tokens (B, N, C) to spatial maps (B, C, H', W') for NCE

### 4. Memory Control (Optional)

* Gradient checkpointing for transformer blocks

---

## Generator Structure

```
Input
 -> Patch Embedding
 -> Transformer Encoder
 -> CNN Decoder
 -> Output
```

---

## Deliverables

* `transformer_blocks.py`
* Update `generator.py` (Transformer encoder)
* Ensure NCE works with transformer features
* Optional gradient checkpointing toggle

---

## Exit Criteria

* Transformer forward pass stable
* Features correctly extracted
* No major VRAM spikes

---

# Phase 4 — Optimization & Stability

## Goal

Make model efficient and production-ready.

---

## Additions

### 1. Replay Buffer

* Tune replay behavior and buffer size for stable discriminator updates

### 2. Identity Loss (Optional)

```
L_id = ||G(B) - B||
```

### 3. Mixed Precision (AMP)

* Keep AMP enabled and tune scaler behavior for stability/performance

### 4. EMA Generator

### 5. Learning Rate Schedulers

---

## Deliverables

* Update `training_loop.py`
* Add `utils/replay_buffer.py`

---

## Exit Criteria

* Smooth loss curves
* Stable GAN + NCE balance

---

# Phase 5 — Validation & Metrics

## Goal

Evaluate model properly.

---

## Metrics

* SSIM
* PSNR
* FID (optional)

---

## Visualization

* Real A -> Fake B
* Real B -> Identity
* Patch-level consistency (optional)

---

## Deliverables

* `utils/metrics.py`
* Validation loop

---

## Exit Criteria

* Metrics logged
* Image grids saved

---

# Phase 6 — Inference Pipeline

## Goal

Deployable translation system.

---

## Features

* Single image inference
* Batch inference
* Patch-based inference (if needed)

---

## Deliverables

* `inference.py`

---

# Lightweight Baseline Spec (8GB-safe)

1. Input size: 256x256
2. Patch size: 8
3. Encoder dim: 192
4. Transformer blocks: 4
5. Heads: 4
6. MLP ratio: 2.0
7. NCE layers: 3
8. Patches sampled: 128 initially, then 256 if stable
9. Decoder channels: keep narrow, avoid very wide skip paths
10. Discriminator: 70x70 PatchGAN, standard depth
11. Batch size: 1 to 2
12. AMP: enabled from first full CUT run

---

# Default Configuration (Lightweight Bias)

```
Image size: 256x256
Batch size: 1-2
Encoder dim (channels): 192 (cap 256)
Transformer layers: 4 (cap 6)
Heads: 4
MLP ratio: 2.0
Patch size: 8 (min 8 for default profile)
NCE layers: 3
Patches sampled: 128 -> 256 if stable
AMP: on
```

---

# Training Flow (Final)

```
real_A -> G -> fake_B

GAN loss:
    D(fake_B)

NCE loss:
    features(real_A) vs features(fake_B)

(optional)
identity:
    G(real_B)

Update G

Update D
```

---

# Key Design Rules

* No cycle consistency
* Always compute NCE on encoder features
* Use same generator for feature extraction
* Do NOT detach features for NCE
* Detach only for discriminator

---

# Lightweight Guardrails

* Max transformer depth: 6
* Max encoder dim: 256
* Min patch size: 8 for default profile
* NCE start: 3 layers, 128 patches, temperature 0.07 to 0.1
* Update ratio: start 1 G step : 1 D step, reduce D frequency if unstable

---

# Success Criteria (Lightweight)

1. Peak VRAM under 7.5GB at train time on 256x256
2. Throughput target recorded as images/sec baseline
3. Stability: no NaNs, no discriminator collapse at fixed checkpoints
4. Quality: SSIM/PSNR trend improves over Phase 1 baseline
5. Measure VRAM, throughput, and quality at a fixed cadence (for example every 1 epoch)

---

# Final Outcome

A lightweight, transformer-based unpaired translation model that:

* Runs on 8GB VRAM
* Trains faster than diffusion
* Preserves structure via contrastive learning
