# model_v3/vae_wrapper.py - VAE Wrapper

Source of truth: ../../model_v3/vae_wrapper.py

Role: Wraps AutoencoderKL encode/decode with Stable Diffusion latent scaling.

---

## Component Structure

1. VAEWrapper.__init__
2. VAEWrapper.encode
3. VAEWrapper.decode

---

## 1) __init__

Input:
- model_id string

Dataflow:
1. load AutoencoderKL from pretrained source
2. freeze parameters (requires_grad=False)
3. set eval mode
4. set latent_scale = 0.18215

Output state:
- ready wrapper with frozen VAE and scaling rule

---

## 2) encode

Input:
- x: (N,3,H,W), expected range [-1,1]

Dataflow:
1. clamp input to [-1,1]
2. VAE encode distribution
3. sample latent
4. scale latent by 0.18215

Shape transitions (typical H=W=256):
- (N,3,256,256) -> latent sample (N,4,32,32) -> scaled (N,4,32,32)

Output:
- z: (N,4,H/8,W/8)

---

## 3) decode

Input:
- z: (N,4,h,w), expected scaled latent

Dataflow:
1. unscale latent by dividing 0.18215
2. cast to float32
3. VAE decode
4. clamp output to [-1,1]

Shape transitions (typical h=w=32):
- (N,4,32,32) -> decoded (N,3,256,256) -> clamped (N,3,256,256)

Output:
- img: (N,3,h*8,w*8)

---

## Notes

- Wrapper keeps VAE frozen by default.
- Gradients can still flow through encode/decode unless caller wraps calls with no_grad.
- Module raises informative ImportError when diffusers is unavailable.
