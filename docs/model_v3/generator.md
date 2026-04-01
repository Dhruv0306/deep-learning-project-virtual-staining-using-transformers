# model_v3/generator.py - v3 DiT Generator

Source of truth: ../../model_v3/generator.py

Role: Defines the CycleDiT generator stack for v3 diffusion with image-token conditioning and domain embedding.

---

## Component Structure

1. PatchEmbed
2. TimestepEmbedding
3. ConditionTokenizer
4. DiTBlock
5. DiTGenerator (backbone)
6. DomainEmbedding
7. CycleDiTGenerator
8. getGeneratorV3

---

## 1) PatchEmbed

Input:
- z: (N, C, H, W), default C=4, H=W=32
- patch_size = p (default 2)

Dataflow:
1. reshape into patch grid:
   - (N,C,H,W) -> (N,C,H/p,p,W/p,p)
2. permute to patch-major layout
3. flatten each patch
   - (N, H/p*W/p, C*p*p)
4. linear projection
   - (N, H/p*W/p, hidden_dim)

Output:
- tokens: (N, L, hidden_dim), L=(H/p)*(W/p)

For H=W=32 and p=2:
- L = 16*16 = 256
- output: (N,256,hidden_dim)

---

## 2) TimestepEmbedding

Input:
- t: (N,) or scalar

Dataflow:
1. sinusoidal embedding
   - (N,) -> (N,freq_dim)
2. MLP
   - Linear(freq_dim->hidden_dim)
   - SiLU
   - Linear(hidden_dim->hidden_dim)

Output:
- t_emb: (N,hidden_dim)

---

## 3) ConditionTokenizer

Input:
- x: (N,3,256,256)

Dataflow:
1. patch projection Conv2d with kernel=stride=cond_patch_size
2. optional token-grid pooling via cond_token_pool_stride
3. flatten to token sequence
4. add 2D sin/cos positional embeddings

Output:
- c_tokens: (N,Lc,hidden_dim)

---

## 4) DiTBlock

Inputs:
- tokens: (N,L,Hd)
- cond: (N,Hd)

Dataflow:
1. adaLN params from cond:
   - linear output: (N,6*Hd)
   - split into gamma1,beta1,alpha1,gamma2,beta2,alpha2 each (N,Hd)
2. norm1(tokens): (N,L,Hd)
3. apply modulation with gamma1/beta1: (N,L,Hd)
4. self-attention: (N,L,Hd)
5. residual update with alpha1 gate: (N,L,Hd)
6. norm2 + modulation with gamma2/beta2: (N,L,Hd)
7. MLP: (N,L,Hd)
8. residual update with alpha2 gate: (N,L,Hd)

Output:
- tokens_out: (N,L,Hd)

---

## 5) DiTGenerator (Backbone)

Inputs to forward:
- z_t: (N,4,32,32)
- t: (N,)
- c: (N,Lc,hidden_dim)

### Forward dataflow

1. patch embed:
   - z_t -> tokens: (N,256,Hd)
2. positional embedding add:
   - pos: (1,256,Hd)
   - tokens: (N,256,Hd)
3. timestep embedding:
   - t -> t_emb: (N,Hd)
4. condition combine:
   - cond_global = mean(c, dim=1): (N,Hd)
   - cond = t_emb + cond_global: (N,Hd)
5. transformer stack:
   - each block keeps shape (N,256,Hd)
6. output head linear:
   - (N,256,Hd) -> (N,256, p*p*4)
   - with p=2 => (N,256,16)
7. unpatchify:
   - (N,256,16) -> (N,4,32,32)

Output:
- v_pred/eps_pred tensor: (N,4,32,32)

### unpatchify dataflow

Input:
- tokens: (N,L, p*p*C)

Reshape path:
1. (N, h, w, p, p, C)
2. permute to channel-first
3. merge patch axes

Output:
- latent map: (N,C,h*p,w*p)

---

## 6) DomainEmbedding

Purpose:
- learned domain token for target domain conditioning

Input:
- target_domain ids (0 for A, 1 for B)

Output:
- domain embedding: (N,hidden_dim)

---

## 7) CycleDiTGenerator

Purpose:
- wrapper around DiT backbone with:
   - ConditionTokenizer
   - DomainEmbedding
   - optional x0 reconstruction via scheduler

Forward input:
- z_t: (N,4,32,32)
- t: (N,)
- condition: image (N,3,256,256) or tokens (N,Lc,Hd)
- target_domain: int or tensor in {0,1}
- scheduler: optional DDPMScheduler
- prediction_type: v or eps

Forward output:
- dict with:
   - v_pred: (N,4,32,32)
   - x0_pred: (N,4,32,32) or None

---

## 8) getGeneratorV3

Purpose:
- factory that builds and initializes CycleDiTGenerator

Dataflow:
1. build DiT backbone from diffusion config
2. build ConditionTokenizer and DomainEmbedding
3. compose CycleDiTGenerator
4. apply init_weights_v2
5. zero-initialize DiT output head
6. smoke test:
   - z_t: (1,4,32,32)
   - t:   (1,)
    - x:   (1,3,256,256)
    - output["v_pred"]: (1,4,32,32)

Return:
- CycleDiTGenerator instance
