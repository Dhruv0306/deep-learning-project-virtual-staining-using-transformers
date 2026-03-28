# New Structure for Model v3: CycleDiT (Refined)

## Overview
- This variant upgrades to a CycleGAN + Diffusion hybrid with a single conditional DiT generator and two projection discriminators.
- The dataset mode is unpaired, using shared/data_loader.py.
- The training loop remains a single shared loop over real_A (Un-Stained) and real_B (Stained).

## Finalized Architecture

### 1) Generator: Single Conditional DiT (Locked)
- Replace dual-generator design (G_AB and G_BA) with one shared generator:
  - G(z_t, target_domain, noise_seed, t)
- target_domain is explicit direction control:
  - target_domain = B for A -> B
  - target_domain = A for B -> A
- Conditioning is dual:
  - image conditioning from input image/tokenizer
  - learned domain token from nn.Embedding(2, hidden_dim)
- Generator return contract is explicit:
  - return both v prediction and reconstructed x0
  - example return format:
    - {"v_pred": v_pred, "x0_pred": x0_pred}
- Rationale:
  - halves generator memory relative to two-DiT setup
  - improves shared representation learning
  - reduces optimization instability between two independent generators

### 2) Discriminators: Keep ProjectionDiscriminator (Locked)
- Keep two discriminators:
  - D_A for Un-Stained domain
  - D_B for Stained domain
- Each discriminator keeps three branches:
  - Local branch: patch-level realism and texture
  - Global branch: full-image structure and color balance
  - FFT branch: periodic artifact detection from VAE decode
- Output format remains list-of-logits across branches, averaged by LSGAN utilities.

### 3) Prediction Type and Diffusion Objective
- prediction_type is fixed to v-parameterization (prediction_type = "v").
- Latent noise construction is explicit:
  - z_t = sqrt(alpha_t) * z + sqrt(1 - alpha_t) * epsilon
- Denoising loss is computed only on primary passes:
  - fake_B primary pass from real_A with target_domain = B
  - fake_A primary pass from real_B with target_domain = A
- No denoising loss on cycle and identity passes.
- Predicted x0 reconstruction is explicit:
  - x0 is reconstructed from v prediction using the scheduler reconstruction formula for v-parameterization.
  - all non-denoising losses consume this reconstructed x0.

## Cycle and Identity Redefinition

### 4) Deterministic Cycle Path (Locked)
- Cycle consistency uses deterministic DDIM shortcut with fixed timestep and noise reuse.
- Per sample in batch:
  - sample (epsilon, t) once
  - fake_B = G(real_A, B, epsilon, t)
  - rec_A = G(fake_B, A, same epsilon, same t)
  - fake_A = G(real_B, A, epsilon', t')
  - rec_B = G(fake_A, B, same epsilon', same t')
- Cycle sampler settings:
  - DDIM steps: 10 to 20 (default: 10)
  - eta = 0 (deterministic)
- Gradient control rule:
  - detach intermediate outputs before reverse-cycle call to avoid heavy second-order dependency.
  - examples:
    - rec_A = G(fake_B.detach(), A, same epsilon, same t)
    - rec_B = G(fake_A.detach(), B, same epsilon', same t')
- This makes cycle mapping less noisy and more learnable for diffusion models.

### 5) Identity Strategy (Locked)
- Identity stays enabled but light:
  - idt_A = G(real_A, A, epsilon = 0, t = 0)
  - idt_B = G(real_B, B, epsilon = 0, t = 0)
- Identity schedule:
  - start with low weight and decay quickly to 0
  - do not keep strong identity throughout training
- Identity path must be deterministic or near-deterministic:
  - preferred: t = 0
  - fallback: very small t (for example t = 1)

## Losses and Weights (Locked)

Let:
- fake_B = G(real_A, B, epsilon, t)
- fake_A = G(real_B, A, epsilon', t')
- rec_A = G(fake_B.detach(), A, epsilon, t)
- rec_B = G(fake_A.detach(), B, epsilon', t')
- idt_A = G(real_A, A, epsilon = 0, t = 0)
- idt_B = G(real_B, B, epsilon = 0, t = 0)

Use:
- Denoising:
  - L_den = L_den_A_to_B + L_den_B_to_A
- Adversarial generator loss (LSGAN):
  - L_adv_G = GAN(D_B(fake_B), real) + GAN(D_A(fake_A), real)
- Discriminator losses (LSGAN):
  - L_D_B = GAN(D_B(real_B), real) + GAN(D_B(fake_B.detach()), fake)
  - L_D_A = GAN(D_A(real_A), real) + GAN(D_A(fake_A.detach()), fake)
- Cycle:
  - L_cyc = ||rec_A - real_A||_1 + ||rec_B - real_B||_1
- Identity:
  - L_id = ||idt_A - real_A||_1 + ||idt_B - real_B||_1
- Adversarial input rule:
  - Apply GAN losses only on fully denoised outputs (effective t = 0 outputs), never on intermediate noisy states.
  - All non-denoising losses (adversarial, cycle, identity) operate on predicted x0 reconstructions, not multi-step sampled outputs.
  - Before feeding x0_pred to discriminators, clamp or scale to valid latent/image range.
  - Default safety rule: x0_pred = x0_pred.clamp(-1, 1) (or equivalent normalization consistent with VAE pipeline).

Example v-parameterization reconstruction (scheduler-equivalent form):
- x0_pred = sqrt(alpha_t) * z_t - sqrt(1 - alpha_t) * v_pred
- implementation must use scheduler-native coefficients and reconstruction utilities to avoid mismatch.

Total generator loss:
- L_G = lambda_denoising * L_den + lambda_adv * L_adv_G + lambda_cycle * L_cyc + lambda_identity(t) * L_id

Locked weights:
- lambda_denoising = 1.0
- lambda_adv = 0.5
- lambda_cycle = 5.0
- lambda_identity_start = 1.0
- lambda_identity_end = 0.0

Identity decay schedule:
- fast linear decay from start to configured early-stop epoch ratio (for example 0.3 * E)
- remains 0 for the rest of training

## Per-Batch Training Order
1. Load real_A and real_B from unpaired loader.
2. Encode to latents with frozen VAE under no_grad.
3. Build primary denoising passes in both directions.
4. Build deterministic cycle passes with fixed noise reuse and short DDIM path.
5. Build identity passes in same-domain mode.
6. Generator step (freeze D_A and D_B):
   - compute L_den, L_adv_G, L_cyc, L_id
   - update optimizer_G
7. Discriminator A step:
   - train on real_A and replay-buffered fake_A
   - update optimizer_D_A
8. Discriminator B step:
   - train on real_B and replay-buffered fake_B
   - update optimizer_D_B
9. Log all losses and current lambda_identity.
10. Print first batch, every 50 batches, and last batch.

## Noise Sampling Policy
- Timestep sampling:
  - t is sampled uniformly from [0, T].
- Noise sampling:
  - epsilon is sampled from N(0, 1).
- Note:
  - importance sampling on t can be added later as an optional upgrade.
  - Log timestep statistics per epoch (histogram or mean/std) for debugging and drift detection.

## Stabilization Features
- Replay buffers are enabled from epoch 1.
- Replay buffer policy:
  - store only generated images (and implicit domain by buffer ownership), not noise or timestep metadata.
- EMA is optional and delayed to Phase C after baseline stability.
- Add R1 gradient penalty for discriminators.
- R1 application rule:
  - apply on real samples only
  - apply every N steps (default N = 16) to control overhead
- Add adaptive discriminator updates:
  - if discriminator overpowers generator, skip D update by explicit threshold rule.
  - default rule:
    - if D_loss < 0.1 then skip D update for that step
  - threshold must be logged each step for reproducibility.
- Generator stability rules:
  - apply generator gradient clipping with default max_norm = 1.0
  - keep scheduler coefficients (alpha_t and related terms) in float32 under AMP to avoid precision drift
- Adversarial warmup rule:
  - ramp lambda_adv from 0.0 to target value (0.5) over early training steps
  - this warmup is recommended to reduce discriminator shock at Phase 1 start

## Config Requirements

Add to DiffusionConfig:
- lambda_denoising: float = 1.0
- lambda_adv_v3: float = 0.5
- lambda_adv_warmup_steps: int = 3000
- lambda_cycle_v3: float = 5.0
- lambda_identity_v3_start: float = 1.0
- lambda_identity_v3_end: float = 0.0
- identity_decay_end_ratio: float = 0.3
- cycle_ddim_steps: int = 10
- cycle_ddim_eta: float = 0.0
- use_r1_penalty: bool = True
- r1_gamma: float = 10.0
- r1_interval: int = 16
- adaptive_d_update: bool = True
- adaptive_d_loss_threshold: float = 0.1
- grad_clip_norm_g: float = 1.0

Discriminator config switches:
- disc_use_local: bool = True
- disc_use_global: bool = True
- disc_use_fft: bool = True
- disc_base_channels: int = 64
- disc_global_base_channels: int = 64
- disc_fft_base_channels: int = 32
- disc_n_layers: int = 3

8 GB VRAM guidance:
- first disable FFT branch: disc_use_fft = False
- keep global branch on: disc_use_global = True
- expected discriminator-side saving: about 200 MB

## File-Level Implementation Plan
- model_v3/generator.py
  - refactor API to forward(x, target_domain, noise_seed=None, t_override=None, mode="primary|cycle|identity")
  - generator operates on VAE latents z, not raw pixel images
  - preferred API shape:
    - forward(z_t, target_domain, noise=None, t=None, mode="primary|cycle|identity")
  - return both v_pred and x0_pred from each forward call
  - add domain embedding and fuse with existing condition tokens
  - lock domain-conditioning placement:
    - inject domain conditioning at input token level and expose hook for per-block conditioning extension
  - explicitly pass shared (epsilon, t) for cycle forward pairs
  - force deterministic identity mode at t = 0 (or tiny t fallback)
  - add single shared getGeneratorV3() factory
- model_v3/discriminator.py
  - keep ProjectionDiscriminator as implemented
  - keep getDiscriminatorsV3() with branch toggles from config
- model_v3/losses.py
  - add CycleDiTLoss for denoising, adversarial, cycle, identity, optional R1
- model_v3/training_loop.py
  - rewrite for single-generator + two-discriminator flow
  - add deterministic cycle path with fixed (epsilon, t) reuse
  - add fast identity decay and adaptive D update support
  - apply adversarial loss only on fully denoised outputs
  - clamp or normalize x0_pred before discriminator forward
  - construct z_t explicitly from z, epsilon, and scheduler alpha_t terms
  - keep scheduler alpha_t terms in float32 even when AMP is enabled
  - apply generator grad clipping each optimizer_G step
  - apply adversarial warmup schedule before fixed lambda_adv
  - apply R1 on real samples every configured interval
  - log timestep distribution statistics each epoch
  - integrate replay buffers
- model_v3/history_utils.py
  - update CSV schema to include:
    - Loss_G, Loss_D_A, Loss_D_B, Loss_denoising, Loss_cycle, Loss_identity, Loss_adv, Loss_R1
- config.py
  - add all v3 generator, loss, cycle-sampler, stabilizer, and discriminator keys listed above
- model_v3/data_loader.py
  - remove from v3 training path and use shared/data_loader.py directly

## Validation, Logging, and Early Stopping
- Keep current checkpointing, CSV, tensorboard, validation, and test image export flows.
- Track both domain metrics:
  - Domain B: SSIM, PSNR, FID (real_B vs fake_B)
  - Domain A: SSIM, PSNR, FID (real_A vs fake_A)
- Early stopping score:
  - mean(SSIM_A, SSIM_B) with optional tie-break on FID trend

## Risks and Mitigations
- Risk: cycle still noisy if DDIM path is too short.
  - Mitigation: keep default cycle_ddim_steps at 10 and only reduce after stability.
- Risk: discriminator dominates early.
  - Mitigation: adaptive D update, lower lambda_adv, or temporary R1 increase.
- Risk: identity interferes with translation if kept too long.
  - Mitigation: fast identity decay to zero by early training ratio.
- Risk: VRAM pressure.
  - Mitigation: disable FFT first, then reduce discriminator channels, then batch size/accumulation.
- Risk: FFT branch numerical instability under AMP.
  - Mitigation: keep FFT in float32 and cast back to native dtype after spectral features.

## Acceptance Criteria
- Training reaches at least 20 epochs without NaN or divergence.
- Both A -> B and B -> A outputs are visually plausible.
- Cycle reconstructions preserve structural content.
- Loss curves remain stable and non-oscillatory after warmup.
- Checkpointing, tensorboard, CSV, validation, and test exports function end to end.

## Immediate Next Chunk
- Implement minimal stable version with:
  - single conditional generator with domain embedding
  - dual ProjectionDiscriminators
  - deterministic cycle path with shared (epsilon, t)
  - denoising + adversarial + cycle + fast-decay identity
  - replay buffers + optional R1

## Rollout Phases (Execution Order)
- Phase 0 (critical bootstrap)
  - implement generator API and domain conditioning
  - train primary denoising only for 1 to 2 epochs (no cycle, no adversarial)
- Phase 1
  - enable adversarial training with dual discriminators
  - keep cycle disabled and verify GAN stability
- Phase 2
  - enable deterministic cycle path and identity schedule
  - run full objective after Phase 0 and Phase 1 are stable
