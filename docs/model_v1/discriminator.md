# model_v1/discriminator.py - v1 Discriminator

Source of truth: ../../model_v1/discriminator.py

Model: Hybrid CycleGAN/UVCGAN v1
Role: Provides PatchGAN discriminators for both domains used during adversarial training.

---

## Overview

The v1 discriminator is a single-scale PatchGAN classifier. Instead of producing one scalar real/fake score for the full image, it produces a spatial grid of logits where each location corresponds to a local receptive field in the input image.

Two instances are used in training:

- D_A: judges realism in domain A (unstained)
- D_B: judges realism in domain B (stained)

Both share the same architecture and are initialized independently.

---

## Architecture

PatchDiscriminator is built as a stack of convolutional blocks:

1. Conv2d(input_nc -> 64, kernel=4, stride=2, padding=1)
   - followed by LeakyReLU(0.2)
   - no normalization on first layer

2. Conv2d(64 -> 128, kernel=4, stride=2, padding=1, bias=False)
   - InstanceNorm2d
   - LeakyReLU(0.2)

3. Conv2d(128 -> 256, kernel=4, stride=2, padding=1, bias=False)
   - InstanceNorm2d
   - LeakyReLU(0.2)

4. Conv2d(256 -> 512, kernel=4, stride=1, padding=1, bias=False)
   - InstanceNorm2d
   - LeakyReLU(0.2)

5. Conv2d(512 -> 1, kernel=4, stride=1, padding=1)
   - output logits per spatial patch

For input (N, 3, 256, 256), output is a 1-channel score map (patch-level logits).

---

## Class Reference

### PatchDiscriminator(input_nc=3)

Constructs the PatchGAN network.

Arguments:

- input_nc: number of input channels; defaults to 3 (RGB)

Forward:

- input: tensor shaped (batch, channels, height, width)
- output: patch logit map used by the LSGAN objective in losses.py

---

## Factory Function

### getDiscriminators()

Builds the pair (D_A, D_B) and handles initialization.

Steps:

1. Select device (CUDA if available, else CPU)
2. Instantiate two PatchDiscriminator models
3. Apply init_weights imported from model_v1.generator
4. Run a smoke-test forward pass on random input
5. Print output shapes for quick verification
6. Return (D_A, D_B)

This function is used by model_v1/training_loop.py during setup.

---

## Training Integration Notes

Inside the v1 training loop:

- discriminators are frozen during generator update
- discriminators are unfrozen for their own update steps
- each domain discriminator loss uses replay-buffered fakes
- optional gradient penalty term from CycleGANLoss can be added

The discriminator objective itself is implemented in model_v1/losses.py, not in this file.
