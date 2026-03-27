# model_v3/noise_scheduler.py - DDPM Scheduler and DDIM Sampler

Source of truth: ../../model_v3/noise_scheduler.py

Role: Forward noising utilities and reverse latent sampling.

---

## Component Structure

1. _linear_beta_schedule
2. _cosine_beta_schedule
3. DDPMScheduler
4. DDIMSampler

---

## 1) _linear_beta_schedule

Input:
- num_timesteps = T

Dataflow:
- linspace from 1e-4 to 2e-2 length T

Output:
- betas: (T,)

---

## 2) _cosine_beta_schedule

Input:
- num_timesteps = T
- offset s

Dataflow:
1. build t grid length T+1
2. cosine cumulative profile f
3. convert adjacent ratios to betas
4. clamp max to 0.999

Output:
- betas: (T,)

---

## 3) DDPMScheduler

Buffers after initialization:
- betas: (T,)
- alphas: (T,)
- alphas_cumprod: (T,)
- sqrt_alphas_cumprod: (T,)
- sqrt_one_minus_alphas_cumprod: (T,)

### _extract(arr, t, x_shape)

Input:
- arr: (T,)
- t: (N,)
- x_shape: usually (N,C,H,W)

Dataflow:
1. gather arr at timestep indices -> (N,)
2. unsqueeze until broadcast rank matches x_shape

Output:
- broadcast tensor typically (N,1,1,1)

### add_noise(x0, noise, t)

Input:
- x0: (N,C,H,W)
- noise: (N,C,H,W)
- t: (N,)

Dataflow:
1. coeff1 = sqrt_alpha_bar(t): (N,1,1,1)
2. coeff2 = sqrt(1-alpha_bar(t)): (N,1,1,1)
3. x_t = coeff1*x0 + coeff2*noise

Output:
- x_t: (N,C,H,W)

### predict_x0(x_t, eps_pred, t)

Input:
- x_t: (N,C,H,W)
- eps_pred: (N,C,H,W)
- t: (N,)

Dataflow:
- x0_pred = (x_t - sqrt(1-alpha_bar_t)*eps_pred) / sqrt(alpha_bar_t)

Output:
- x0_pred: (N,C,H,W)

### get_alpha_bar(t)

Input:
- t scalar or (N,)

Output:
- alpha_bar values broadcastable to query shape

---

## 4) DDIMSampler

Inputs to sample:
- model: predicts eps
- condition: (N,Hd)
- shape: (N,4,32,32)
- num_steps
- eta

### Dataflow per sampling run

1. initialize latent noise:
   - z_t: shape argument, usually (N,4,32,32)
2. build timestep sequence length num_steps
3. for each timestep:
   - t_batch: (N,)
   - eps_pred = model(z_t, t_batch, condition): (N,4,32,32)
   - z0_pred = predict_x0(...): (N,4,32,32)
   - compute alpha_bar_t and alpha_bar_prev scalars
   - compute sigma scalar (depends on eta)
   - compute direction term dir_xt: (N,4,32,32)
   - update z_t: (N,4,32,32)
   - optional noise add when eta>0
4. return final z_t

Output:
- sampled latent z0-like tensor: (N,4,32,32)
