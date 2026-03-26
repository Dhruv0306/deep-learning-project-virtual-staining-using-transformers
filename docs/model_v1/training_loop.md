# `model_v1/training_loop.py` — V1 Training Loop

**Model:** Hybrid UVCGAN + CycleGAN (v1)  
**Entry point:** `train()`  
**Role:** Orchestrates the complete v1 training process. Unlike v2, all hyperparameters are hardcoded inside the function rather than read from a config object.

---

## Full Training Flow

```
train(epoch_size, num_epochs, model_dir, val_dir, test_size)
    │
    ├── Build data loaders  (getDataLoader, epoch_size=3000 default)
    ├── Build models        (getGenerators, getDiscriminators)
    ├── Build loss          (CycleGANLoss)
    ├── Build optimisers    (3× Adam)
    ├── Build LR schedulers (3× LambdaLR, linear decay from epoch 100)
    ├── Set up TensorBoard, CSV, directories
    │
    └── for epoch in range(num_epochs):
            │
            ├── for batch in train_loader:
            │       │
            │       ├── [Generator step]
            │       │     D frozen
            │       │     loss_G, fake_A, fake_B = generator_loss(...)
            │       │     backward + step G
            │       │
            │       └── [Discriminator steps]
            │             G frozen
            │             loss_D_A = discriminator_loss(D_A, real_A, fake_A)
            │             loss_D_B = discriminator_loss(D_B, real_B, fake_B)
            │             backward + step D_A, D_B
            │
            ├── Log losses to TensorBoard and CSV
            ├── Step LR schedulers
            ├── Every 5 epochs: flush CSV
            ├── Every 20 epochs: save checkpoint
            │
                ├── Every epoch:
                │       run_validation    (save comparison images)
                │
                └── Every 10 epochs:
                    calculate_metrics (SSIM, PSNR, FID)
                    check early stopping (enabled from epoch 80)
    │
    ├── Final metrics + test inference
    └── Save final checkpoint
```

---

## Key Differences from V2

| Aspect | V1 | V2 |
|---|---|---|
| Hyperparameters | Hardcoded in function | Read from `UVCGANConfig` |
| Training order | Generator step first, then discriminator | Discriminator first, then generator |
| Gradient accumulation | Not supported | Supported via `accumulate_grads` |
| LR schedule | Simple linear decay from epoch 100 | Warm-up ramp + constant + linear decay |
| Gradient clipping | Not applied | Applied to G and D separately |
| Validation | Validation images every epoch; metrics every 10 epochs; early stopping enabled from epoch 80 | Validation images start after `validation_warmup_epochs`; metrics/early-stopping checks run by `early_stopping_interval` |
| AMP for discriminator | Yes (autocast enabled) | GP always float32 (autocast disabled for D loss) |

---

## Hardcoded Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `lr` | 0.0002 | Learning rate for all optimisers |
| `beta1` | 0.5 | Adam β₁ |
| `beta2` | 0.999 | Adam β₂ (implicit) |
| `lambda_cycle` | 10.0 | Cycle-consistency weight |
| `lambda_identity` | 5.0 | Identity weight |
| `lambda_cycle_perceptual` | 0.2 | Perceptual cycle weight |
| `lambda_identity_perceptual` | 0.1 | Perceptual identity weight |
| `lambda_gp` | 10.0 | Gradient penalty weight |
| `perceptual_resize` | 160 | VGG input resolution |
| LR decay start | epoch 100 | Linear decay from 100 to `num_epochs` |
| Early stopping check interval | 10 epochs | |
| Early stopping warmup | 80 epochs | |
| Early stopping patience | 40 epochs | |
| Divergence threshold | 5.0× | |
| Divergence patience | 2 checks | |
| Checkpoint interval | 20 epochs | |

---

## Function: `train(...)`

### Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `epoch_size` | int or None | None | Samples per epoch (3000 if None) |
| `num_epochs` | int or None | None | Training epochs (200 if None) |
| `model_dir` | str or None | None | Output directory |
| `val_dir` | str or None | None | Validation images directory |
| `test_size` | float or None | None | Number of test samples to export |

**Returns:** `(history, G_AB, G_BA, D_A, D_B)`

---

### LR Schedule

```python
lr_lambda = lambda epoch: 1.0 - max(0, epoch - 100) / 100
```

| Epoch range | LR multiplier |
|---|---|
| 0–100 | 1.0 (constant) |
| 100–200 | linear decay from 1.0 to 0.0 |

No warm-up phase. If `num_epochs` ≠ 200, the decay rate changes proportionally.

---

### Per-Batch Training Step

#### Generator step

```python
# Freeze discriminators
for p in D_A.parameters(): p.requires_grad_(False)
for p in D_B.parameters(): p.requires_grad_(False)

optimizer_G.zero_grad(set_to_none=True)

with autocast("cuda", enabled=use_amp):
    loss_G, fake_A, fake_B = loss_fn.generator_loss(
        real_A, real_B, G_AB, G_BA, D_A, D_B, epoch, num_epochs
    )

scaler.scale(loss_G).backward()
scaler.step(optimizer_G)
scaler.update()
```

**Note:** v1 does the generator step before the discriminator step. v2 reverses this order (discriminator first) which is theoretically more correct — the discriminator should be reasonably converged before the generator receives its adversarial gradient signal.

#### Discriminator steps

```python
# Re-enable discriminators
for p in D_A.parameters(): p.requires_grad_(True)
for p in D_B.parameters(): p.requires_grad_(True)

optimizer_D_A.zero_grad(set_to_none=True)
with autocast("cuda", enabled=use_amp):
    loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A, fake_A_buffer)
scaler.scale(loss_D_A).backward()
scaler.step(optimizer_D_A)
scaler.update()

# Same for D_B with fake_B
```

**Important:** `fake_A` and `fake_B` are reused from the generator step without re-running the generators. This is more efficient but means the discriminator sees the same fakes in both the G and D steps within one batch — v2 generates fresh fakes with `torch.no_grad()` for the D step to avoid this.

---

### Validation and Early Stopping

Validation image export runs every epoch:

```python
save_dir = os.path.join(val_dir, f"epoch_{epoch+1}")
run_validation(...)
```

Validation metrics are computed every 10 epochs, and early stopping is only evaluated after epoch 80:

```python
if (epoch + 1) % early_stopping_check_interval == 0:
    avg_metrics = calculate_metrics(...)
    ...
    if (epoch + 1) >= early_stopping_warmup_epochs:
        should_stop = early_stopping(avg_ssim, tracked_losses)
        if should_stop:
            break
```

So in v1 you will see validation images from epoch 1, but early-stopping decisions start at epoch 80.

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
    test_images/
```

