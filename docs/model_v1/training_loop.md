# model_v1/training_loop.py - v1 Training Loop

Source of truth: ../../model_v1/training_loop.py

Model: Hybrid CycleGAN/UVCGAN v1
Primary entrypoint: train(...)
Compatibility entrypoint: train_v1(...)

---

## Purpose

This file orchestrates full v1 training:

- data loading
- model construction
- loss setup
- optimizer and scheduler setup
- mixed-precision training
- checkpointing
- validation image export
- metric computation
- early stopping
- final testing and artifact persistence

---

## Public Functions

### train(epoch_size=None, num_epochs=None, model_dir=None, val_dir=None, test_size=None)

Main v1 training function.

Arguments:

- epoch_size: max samples per epoch (defaults to 3000)
- num_epochs: number of epochs (defaults to 200)
- model_dir: output directory for checkpoints/logs
- val_dir: directory for validation image exports
- test_size: number of test samples to save in final testing

Returns:

- history
- G_AB
- G_BA
- D_A
- D_B

### train_v1(*args, **kwargs)

Thin alias that forwards to train(...), used for import consistency in top-level scripts.

---

## Runtime Setup

Environment/performance setup:

- sets TF_ENABLE_ONEDNN_OPTS=0 for consistency across systems
- enables cuDNN benchmark mode
- enables TF32 for CUDA matmul and cuDNN when available

Data:

- train_loader and test_loader from shared.data_loader.getDataLoader

Models:

- generators from model_v1.generator.getGenerators
- discriminators from model_v1.discriminator.getDiscriminators

Loss:

- CycleGANLoss with v1-specific hardcoded lambdas

AMP:

- autocast and GradScaler enabled only when device is CUDA

Logging:

- TensorBoard SummaryWriter in model_dir/tensorboard_logs
- rolling CSV persistence via shared.history_utils helpers

---

## Optimization and Schedulers

Optimizers:

- optimizer_G: AdamW over G_AB + G_BA parameters (Transformer bottleneck)
- optimizer_D_A: Adam over D_A
- optimizer_D_B: Adam over D_B

Hardcoded optimizer params:

- lr = 2e-4
- betas = (0.5, 0.999)
- weight_decay = 0.01 (generator only)

Schedulers:

- LambdaLR on all three optimizers
- schedule factor: 1.0 - max(0, epoch - 100) / 100
- effect: constant until epoch 100, then linear decay

---

## Per-Epoch Training Flow

For each epoch:

1. switch all models to train mode
2. iterate over train_loader batches
3. for each batch:
   - move A and B tensors to device
   - generator step
   - discriminator A step
   - discriminator B step
   - accumulate scalar losses and save batch history
4. log epoch-mean losses to TensorBoard
5. flush history CSV every 5 epochs
6. save checkpoint every 20 epochs
7. step all LR schedulers
8. run validation image generation
9. periodically compute metrics and early-stopping decision

---

## Per-Batch Update Order

### 1) Generator Step

- freeze D_A and D_B params (requires_grad=False)
- zero optimizer_G grads
- compute loss_G, fake_A, fake_B via loss_fn.generator_loss
- backward with scaler
- optimizer_G step
- scaler update

### 2) Discriminator Steps

- unfreeze D_A and D_B params

D_A update:

- zero optimizer_D_A grads
- compute loss_D_A = loss_fn.discriminator_loss(D_A, real_A, fake_A, fake_A_buffer)
- backward with scaler
- optimizer_D_A step
- scaler update

D_B update:

- zero optimizer_D_B grads
- compute loss_D_B = loss_fn.discriminator_loss(D_B, real_B, fake_B, fake_B_buffer)
- backward with scaler
- optimizer_D_B step
- scaler update

---

## Validation, Metrics, Early Stopping

Validation images:

- run_validation executes every epoch
- outputs saved in val_dir/epoch_{k}

Metrics and stopping checks:

- every 10 epochs, calculate_metrics is called
- average SSIM from both domains is used as primary stopping score
- tracked losses: mean G, D_A, D_B for the epoch
- early stopping checks begin only after warmup epoch 80

EarlyStopping configuration in this file:

- check interval: 10 epochs
- patience: 40 epochs converted to check-count internally
- divergence threshold: 5.0
- divergence patience: 2

If early stopping triggers, loop exits and stopped_epoch is recorded.

---

## Checkpointing and Artifacts

Periodic checkpoint every 20 epochs includes:

- epoch number
- G_AB, G_BA, D_A, D_B state_dicts
- optimizer_G, optimizer_D_A, optimizer_D_B states

Final phase after training loop:

1. compute final metrics
2. run test-image export through shared.testing.run_testing
3. save final checkpoint as final_checkpoint_epoch_{stopped_epoch}.pth
4. flush remaining history to CSV
5. reload history from CSV for consistent return format
6. close TensorBoard writer

---

## Default Directory Layout

Within model_dir:

- tensorboard_logs/
- validation_images/
- test_images/
- training_history.csv
- checkpoint_epoch_*.pth
- final_checkpoint_epoch_*.pth

If model_dir is not provided, a default path under data/E_Staining_DermaRepo/H_E-Staining_dataset/models is used.

---

## Notes

- v1 hyperparameters are hardcoded here, unlike v2/v3 config-driven loops.
- test_size is cast to int before being passed to run_testing.
- history is temporarily accumulated in-memory and periodically flushed to CSV to reduce memory growth.
