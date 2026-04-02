# model_v4/training_loop.py

Source of truth: ../../model_v4/training_loop.py

v4 training loop for CUT + Transformer using unpaired A/B image domains.

## Objective

Train bidirectional generators and discriminators with:

- LSGAN adversarial loss
- PatchNCE contrastive loss on encoder features
- identity regularization
- optional EMA, replay buffer, AMP, accumulation, and LR schedule

## Main Public Function

## train_v4(epoch_size=None, num_epochs=None, model_dir=None, val_dir=None, cfg=None)

Returns:

- `(history, G_AB, G_BA, D_A, D_B)`

Key setup stages:

1. apply runtime overrides onto `V4Config`
2. build loaders via `shared.data_loader.getDataLoader`
3. construct `G_AB`, `G_BA`, `D_A`, `D_B`
4. create optimizers and optional LambdaLR schedulers
5. initialize AMP scaler, EMA models, PatchSampler, PatchNCELoss,
   MetricsCalculator, replay buffers, TensorBoard writer

## Helper Functions

- `_set_requires_grad(module, flag)`
- `_global_grad_norm(parameters)`
- `_make_lr_lambda(warmup, decay_start, total)`
- `_lsgan_gen_loss(pred_fake)`
- `_lsgan_disc_loss(pred_real, pred_fake)`
- `_run_validation_v4(...)`

## Per-Batch Training Flow

For each batch (`real_A`, `real_B`):

1. Discriminator updates
   - freeze generators
   - generate fakes under `no_grad`
   - compute LSGAN D losses for A and B
   - optionally use replay-buffer fakes
   - backward + grad clipping + optimizer step per discriminator

2. Generator update
   - freeze discriminators, unfreeze generators
   - forward `G_AB` and `G_BA` with optional feature returns
   - GAN generator losses from `D_B(fake_B)` and `D_A(fake_A)`
   - if enabled, compute PatchNCE:
     - sample real patches and patch ids
     - sample fake patches using same ids
     - compute `PatchNCELoss` both directions and average
   - if enabled, compute identity loss on opposite-domain inputs
   - total:
     - `lambda_gan * loss_G_gan + lambda_nce * loss_nce + lambda_identity * loss_id`
   - apply gradient accumulation and optional AMP scaling
   - clip and step when accumulation window closes
   - update EMA generators when enabled

3. Logging
   - store per-batch losses in `history[epoch][batch]`
   - write epoch metrics and LR values to TensorBoard

## Validation, Checkpoints, and Final Test

- periodic checkpoints every `save_every` epochs
- validation runs once `epoch >= validation_every`
- validation/test use EMA generators when enabled
- final checkpoint is always saved as `final_checkpoint.pth`
- final test export is written under `test_images`

Checkpoint payload includes:

- model weights (raw and EMA)
- optimizer states
- LR scheduler states
- epoch number
