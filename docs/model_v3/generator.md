# model_v3/generator.py - v3 DiT Generator

Source of truth: ../../model_v3/generator.py

Role: Defines latent-space DiT backbone and conditioning path for v3 diffusion.

---

## Component Structure

1. PatchEmbed
2. TimestepEmbedding
3. ConditionEncoder
4. DiTBlock
5. DiTGenerator
6. getGeneratorV3

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

## 3) ConditionEncoder

Input:
- x: (N,3,256,256)

Dataflow:
1. Conv stack:
   - 3->64, stride2  : (N,64,128,128)
   - 64->128, stride2: (N,128,64,64)
   - 128->256,stride2: (N,256,32,32)
   - 256->512,stride2: (N,512,16,16)
2. adaptive avg pool
   - (N,512,1,1)
3. flatten
   - (N,512)
4. linear projection
   - (N,hidden_dim)

Output:
- c: (N,hidden_dim)

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

## 5) DiTGenerator

Inputs to forward:
- z_t: (N,4,32,32)
- t: (N,)
- c: (N,hidden_dim)

### Forward dataflow

1. patch embed:
   - z_t -> tokens: (N,256,Hd)
2. positional embedding add:
   - pos: (1,256,Hd)
   - tokens: (N,256,Hd)
3. timestep embedding:
   - t -> t_emb: (N,Hd)
4. condition combine:
   - cond = t_emb + c: (N,Hd)
5. transformer stack:
   - each block keeps shape (N,256,Hd)
6. output head linear:
   - (N,256,Hd) -> (N,256, p*p*4)
   - with p=2 => (N,256,16)
7. unpatchify:
   - (N,256,16) -> (N,4,32,32)

Output:
- eps_pred: (N,4,32,32)

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

## 6) getGeneratorV3

Purpose:
- factory that reads config and returns initialized DiTGenerator

Dataflow:
1. build model from diffusion config
2. apply init_weights_v2
3. zero-initialize output head
4. smoke test:
   - z_t: (1,4,32,32)
   - t:   (1,)
   - c:   (1,hidden_dim)
   - output: (1,4,32,32)

Return:
- DiTGenerator instance
