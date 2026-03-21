# `training_loop_v2.py` — V2 Training Loop

**Model:** True UVCGAN v2  
**Entry point:** `train_v2()`  
**Role:** Orchestrates the complete v2 training process — data loading, model construction, loss computation, optimiser stepping, LR scheduling, validation, checkpointing, and early stopping. Everything is driven by a `UVCGANConfig` object.

---

## Full Training Flow

```
train_v2(epoch_size, num_epochs, model_dir, val_dir, test_size, cfg)
    │
    ├── Build data loaders  (getDataLoader)
    ├── Build models        (getGeneratorsV2, getDiscriminatorsV2)
    ├── Build loss          (UVCGANLoss)
    ├── Build optimisers    (3× Adam — G, D_A, D_B)
    ├── Build LR schedulers (3× LambdaLR)
    ├── Set up TensorBoard, CSV, directories
    │
    └── for epoch in range(num_epochs):
            │
            ├── for batch in train_loader:
            │       │
            │       ├── [Discriminator step × n_critic]
            │       │     G frozen, D trainable
            │       │     fake_B = G_AB(real_A)  [no_grad]
            │       │     fake_A = G_BA(real_B)  [no_grad]
            │       │     loss_D_A = discriminator_loss(D_A, real_A, fake_A, buffer)
            │       │     loss_D_B = discriminator_loss(D_B, real_B, fake_B, buffer)
            │       │     backward + clip + step D_A, D_B
            │       │
            │       └── [Generator step with gradient accumulation]
            │             D frozen, G trainable
            │             zero_grad every accumulate_grads batches
            │             loss_G, fake_A, fake_B = generator_loss(...)
            │             backward (loss_G / accumulate_grads)
            │             clip + step G every accumulate_grads batches
            │
            ├── Log epoch losses + LR to TensorBoard
            ├── Step LR schedulers
            ├── Every 5 epochs: flush history CSV
            ├── Every 20 epochs: save checkpoint
            │
                ├── After validation_warmup_epochs:
                │       run_validation each epoch (save comparison images)
                │
                └── Every early_stopping_interval (after early_stopping_warmup):
                    calculate_metrics (SSIM, PSNR, FID)
                    check early stopping (SSIM + divergence)
    │
    ├── Final metrics on test set
    ├── Final test inference + image export
    └── Save final checkpoint
```

---

## Function: `_make_lr_lambda(warmup, decay_start, total)`

Returns a `lr_lambda` callable for `LambdaLR`.

| Parameter | Type | Description |
|---|---|---|
| `warmup` | int | Epochs to ramp from 0 to 1 |
| `decay_start` | int | Epoch where linear decay begins |
| `total` | int | Total training epochs |

**Schedule:**
```
epoch ∈ [0, warmup):        max(1e-8, epoch / warmup)       ← ramp up
epoch ∈ [warmup, decay_start): 1.0                           ← constant
epoch ∈ [decay_start, total):  max(0, 1 - (epoch-decay_start)/(total-decay_start))  ← decay
```

The `1e-8` floor during warm-up prevents the LR from being exactly 0 on epoch 0, which would give zero gradients before any learning has occurred.

---

## Function: `_global_grad_norm(parameters)`

Computes the global gradient L2 norm across all parameters.

```python
grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
return float(torch.norm(torch.stack([g.norm() for g in grads])))
```

Used for TensorBoard logging (`Diagnostics/GradNorm_G`) to detect gradient explosion early. Returns 0.0 when no gradients exist (e.g. before the first backward).

---

## Function: `train_v2(...)`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | int or None | None | Overrides `cfg.training.epoch_size` |
| `num_epochs` | int or None | None | Overrides `cfg.training.num_epochs` |
| `model_dir` | str or None | None | Overrides `cfg.model_dir` |
| `val_dir` | str or None | None | Overrides `cfg.val_dir` |
| `test_size` | int or None | None | Overrides `cfg.training.test_size` |
| `cfg` | `UVCGANConfig` or None | None | Full config. `get_default_config(2)` used if None |

**Returns:** `(history, G_AB, G_BA, D_A, D_B)`

---

### Model Construction

```python
G_AB, G_BA = getGeneratorsV2(
    base_channels, vit_depth, vit_heads, vit_mlp_ratio,
    vit_dropout, layerscale_init, use_cross_domain,
    use_gradient_checkpointing   ← from GeneratorConfig
)
D_A, D_B = getDiscriminatorsV2(
    input_nc, base_channels, n_layers, num_scales, use_spectral_norm
)
```

Both generators and both discriminators are moved to the device (CUDA if available).

---

### Optimiser Setup

Three independent Adam optimisers share the same `lr`, `beta1`, `beta2`:

| Optimiser | Parameters | Description |
|---|---|---|
| `optimizer_G` | All G_AB + G_BA params | Updated once per `accumulate_grads` batches |
| `optimizer_D_A` | All D_A params | Updated once per discriminator step |
| `optimizer_D_B` | All D_B params | Updated once per discriminator step |

Three corresponding `LambdaLR` schedulers step once per epoch.

---

### AMP (Automatic Mixed Precision)

```python
use_amp = tcfg.use_amp and device.type == "cuda"
scaler  = GradScaler("cuda", enabled=use_amp)
```

When enabled, generator forward passes run in float16 (saving ~50% activation memory), while the GradScaler rescales gradients to prevent underflow.

**The gradient penalty always runs in float32** regardless of `use_amp` — `UVCGANLoss.discriminator_loss` wraps the GP call in `torch.autocast(enabled=False)` internally. This is critical: float16 precision is insufficient for the `torch.autograd.grad` call in the penalty computation and produces NaN gradients.

---

### Per-Batch Training Step

#### Discriminator step (`n_critic` times)

```python
# Freeze G so it doesn't accumulate gradients
for p in G_AB.parameters(): p.requires_grad_(False)
for p in G_BA.parameters(): p.requires_grad_(False)

with torch.no_grad():
    fake_B_d = G_AB(real_A)    # generate fakes for D training
    fake_A_d = G_BA(real_B)

# D_A update
optimizer_D_A.zero_grad()
loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A_d, fake_A_buffer)
loss_D_A.backward()
clip_grad_norm_(D_A.parameters(), grad_clip_norm)
optimizer_D_A.step()

# D_B update (same pattern)
```

The fakes are generated with `torch.no_grad()` during the discriminator step — the generator's computation graph is not needed here, saving memory and compute.

#### Generator step (with gradient accumulation)

```python
# Freeze D
for p in D_A.parameters(): p.requires_grad_(False)
for p in D_B.parameters(): p.requires_grad_(False)

# Zero grad only at start of accumulation window
if (i - 1) % accumulate == 0:
    optimizer_G.zero_grad(set_to_none=True)

with autocast("cuda", enabled=use_amp):
    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
    )
    loss_G_scaled = loss_G / accumulate    # scale for accumulation

scaler.scale(loss_G_scaled).backward()

# Step only at end of accumulation window
if i % accumulate == 0 or i == len(train_loader):
    scaler.unscale_(optimizer_G)
    clip_grad_norm_(G_params, grad_clip_norm)
    scaler.step(optimizer_G)
    scaler.update()
```

**Why divide by `accumulate`?** Without scaling, accumulating `K` batches would produce gradients `K×` larger than a single batch. Dividing each loss by `K` before backward makes the accumulated gradient equal to the average gradient over the `K` batches — equivalent to training on a batch of size `batch_size × K`.

**Why `set_to_none=True`?** Setting gradients to `None` instead of zero is slightly faster (avoids a memset) and uses less memory since `None` tensors don't occupy memory.

---

### Epoch-Level Logging

At the end of each epoch:

| TensorBoard scalar | Description |
|---|---|
| `Loss/Generator` | Average generator loss across all batches |
| `Loss/Discriminator_A` | Average D_A loss |
| `Loss/Discriminator_B` | Average D_B loss |
| `Diagnostics/GradNorm_G` | Average generator gradient L2 norm |
| `LR/Generator` | Current learning rate after scheduler step |
| `LR/Discriminator_A` | Current D_A learning rate |
| `LR/Discriminator_B` | Current D_B learning rate |

---

### Validation and Early Stopping

Validation image export and validation metrics are scheduled separately.

Validation images start only after `validation_warmup_epochs`, then run every epoch:

```python
if (epoch + 1) > tcfg.validation_warmup_epochs:
    run_validation(...)
```

Validation metrics and early stopping are checked on `early_stopping_interval`:

```python
if (
    (epoch + 1) % tcfg.early_stopping_interval == 0
    and epoch + 1 >= tcfg.early_stopping_warmup
):
    val_metrics = calculate_metrics(...)
    avg_ssim = (val_metrics.get("ssim_A", 0.0) + val_metrics.get("ssim_B", 0.0)) / 2.0
    should_stop = early_stopping(ssim=avg_ssim, losses={...})
```

This keeps image monitoring frequent after warmup while controlling metric/early-stopping overhead with a separate interval.

| TensorBoard scalar | Description |
|---|---|
| `Validation/ssim_A` | SSIM between real_A and fake_A |
| `Validation/ssim_B` | SSIM between real_B and fake_B |
| `Validation/psnr_A` | PSNR for domain A |
| `Validation/psnr_B` | PSNR for domain B |
| `EarlyStopping/ssim` | Average SSIM used for early stopping |
| `EarlyStopping/counter` | Steps without SSIM improvement |
| `EarlyStopping/divergence_counter` | Consecutive divergence checks |

---

### Checkpointing

Every 20 epochs:
```
checkpoint_epoch_N.pth  ← G_AB, G_BA, D_A, D_B state dicts + all optimiser states
```

At end of training:
```
final_checkpoint_epoch_N.pth  ← same structure
```

The checkpoint always saves all four models and three optimisers so training can be resumed from any checkpoint.

---

### Output Directory Structure

```
model_dir/
    checkpoint_epoch_20.pth
    checkpoint_epoch_40.pth
    ...
    final_checkpoint_epoch_N.pth
    training_history.csv
    training_history.png
    tensorboard_logs/
    validation_images/
        image_1_A.png    ← Real A | Fake B | Rec A | Real B
        image_1_B.png    ← Real B | Fake A | Rec B | Real A
        ...
    test_images/
```
