# model_v2/losses.py - v2 Losses

Source of truth: ../../model_v2/losses.py

Model: True UVCGAN v2
Role: Composite loss system with explicit modular components.

---

## Component Structure

1. VGGPerceptualLossV2
2. SpectralLoss
3. ContrastiveLoss
4. LSGANGradientPenalty
5. UVCGANLoss

---

## 1) VGGPerceptualLossV2

Input:
- x: (N, C, H, W), expected in [-1, 1]
- y: (N, C, H, W), expected in [-1, 1]

Dataflow:

1. cast to float32
   - shapes unchanged
2. if C=1, repeat to 3 channels
   - (N,1,H,W) -> (N,3,H,W)
3. optional resize to R by R
   - (N,3,H,W) -> (N,3,R,R)
4. range conversion
   - [-1,1] -> [0,1]
5. clamp to [0,1]
6. ImageNet normalization
   - shape unchanged
7. extract VGG features:
   - h1 from slice1: (N, C1, H1, W1)
   - h2 from slice2: (N, C2, H2, W2)
   - h3 from slice3: (N, C3, H3, W3)
   - h4 from slice4: (N, C4, H4, W4)
8. compute weighted L1 across each pair level and sum

Output:
- scalar tensor loss

---

## 2) SpectralLoss

Input:
- x: (N, C, H, W)
- y: (N, C, H, W)

Dataflow:

1. FFT:
   - x_f = rfft2(x)
   - y_f = rfft2(y)
   - shapes become complex frequency tensors with last dimension reduced by rfft
2. magnitude:
   - abs(x_f), abs(y_f)
3. compression:
   - log1p of magnitudes
4. L1 distance between transformed spectra

Output:
- scalar tensor loss

---

## 3) ContrastiveLoss

Inputs:
- anchor: (N, C) or (N, C, H, W)
- positive: (N, C) or (N, C, H, W)
- negative: (N, C) or (N, C, H, W)

Projection dataflow per input:

1. if 4D, global average pooling
   - (N,C,H,W) -> (N,C)
2. projection MLP
   - (N,C) -> (N,proj_dim)
3. L2 normalization along feature dimension
   - shape unchanged (N,proj_dim)

NT-Xent dataflow:

4. similarity logits:
   - sim_pos: (N,)
   - sim_neg: (N,)
5. stack logits:
   - (N,2)
6. labels:
   - zeros of shape (N,)
7. cross entropy

Output:
- scalar tensor loss

---

## 4) LSGANGradientPenalty

Inputs:
- D: discriminator module
- real: (N, C, H, W)
- fake: (N, C, H, W)

Dataflow:

1. cast and detach real/fake to float32
2. sample epsilon:
   - eps: (N,1,1,1)
3. interpolate:
   - interp = eps*real + (1-eps)*fake
   - interp: (N,C,H,W), requires_grad=True
4. discriminator forward:
   - pred is either
     - tensor: (N,1,h,w), or
     - list of per-scale tensors
5. scalar reduction for autograd:
   - single-scale: pred.sum() -> scalar
   - multi-scale: sum over each scale map then sum scales -> scalar
6. gradient wrt interp:
   - grads: (N,C,H,W)
7. flatten per sample:
   - grads.view(N,-1)
8. norm per sample:
   - grad_norms: (N,)
9. one-sided penalty:
   - relu(grad_norms - gamma)^2 / gamma^2
10. mean over batch

Output:
- scalar gradient penalty

---

## 5) UVCGANLoss

Internal components:
- criterion_cycle
- criterion_identity
- criterion_GAN
- criterion_perceptual
- criterion_spectral
- optional criterion_contrastive
- gp utility
- replay buffers

### 5.1 get_identity_lambda

Input:
- epoch, total_epochs

Output:
- scalar float lambda value

### 5.2 _lsgan_disc_loss

Input options:
- single-scale tensors or multi-scale list outputs

Single-scale dataflow:
- real_out: (N,1,h,w)
- fake_out: (N,1,h,w)
- compute real/fake MSE and average

Multi-scale dataflow:
- iterate each pair of scale outputs
- compute per-scale scalar loss
- stack and mean

Output:
- scalar tensor

### 5.3 _lsgan_gen_loss

Input options:
- single-scale tensor or list of tensors

Dataflow:
- MSE to ones target per scale
- average across scales when list input

Output:
- scalar tensor

### 5.4 generator_loss

Inputs:
- real_A: (N,3,H,W)
- real_B: (N,3,H,W)
- G_AB, G_BA, D_A, D_B
- epoch, total_epochs

Dataflow sections with shapes:

1. identity:
   - idt_A = G_BA(real_A): (N,3,H,W)
   - idt_B = G_AB(real_B): (N,3,H,W)
2. translation:
   - fake_B from real_A: (N,3,H,W)
   - fake_A from real_B: (N,3,H,W)
   - optionally cross-domain skip path internally
3. adversarial generator loss:
   - D_B(fake_B) -> tensor or list of tensors
   - D_A(fake_A) -> tensor or list of tensors
   - reduce to scalar losses
4. cycle:
   - rec_A = G_BA(fake_B): (N,3,H,W)
   - rec_B = G_AB(fake_A): (N,3,H,W)
5. perceptual:
   - VGG-based scalar losses on rec and identity outputs
6. optional spectral:
   - spectral scalar losses on fake vs real
7. optional contrastive:
   - encode bottlenecks and compute NT-Xent scalars
8. total sum

Outputs:
- loss_G: scalar
- fake_A: (N,3,H,W)
- fake_B: (N,3,H,W)

### 5.5 discriminator_loss

Inputs:
- D, real, fake, optional replay buffer

Dataflow:
1. buffer selection:
   - fake_buf: (N,3,H,W)
2. forward:
   - real_out and fake_out as tensor or list outputs
3. adversarial discriminator reduction to scalar
4. optional GP scalar addition (float32 path)

Output:
- scalar discriminator loss
