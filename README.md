# UVCGAN-Style CycleGAN for Histology Stain/Unstain Translation

This project trains a CycleGAN-style pipeline with a **UVCGAN generator (U-Net + ViT)** to translate between **unstained** and **stained** histology tissue images. It includes:
- Dataset preprocessing to create 256x256 patches (with background/tissue filtering)
- Training with validation, metrics, early stopping, and TensorBoard
- Inference to stain or unstain whole images by patching and reconstruction

## Dataset
This project uses the **E-stainind DermaRepo H&E staining dataset** with **unstained** and **stained** image domains. The raw images should be organized into `Un_Stained` and `C_Stained` folders, and preprocessing will create `trainA/trainB/testA/testB` patch datasets from them.

All rights for the dataset are held by the original owners and licensors of the dataset.

## Project Layout

### Entry Points
- `trainModel.py`: Training entry point (prompts for epoch size, epochs, test size)
- `app.py`: Inference script for stain/unstain translation
- `preprocess_data.py`: Patch extraction, tissue/background filtering, and train/test split
- `unzip.py`: Helper script to extract the dataset ZIP archive into the expected directory layout

### Configuration
- `config.py`: Centralized configuration manager; defines `UVCGANConfig` dataclasses for generator, discriminator, loss, training, and data hyperparameters; supports both v1 and v2 model versions via `model_version`

### Training Loops
- `training_loop.py`: v1 training loop — full training loop, logging, validation, testing, and checkpoints
- `training_loop_v2.py`: v2 training loop — improved loop with gradient clipping, warm-up + linear LR decay, multi-scale discriminator support, and extended TensorBoard logging

### Models
- `generator.py`: UVCGAN v1 generator (U-Net + ViT bottleneck) and weight initializer
- `uvcgan_v2_generator.py`: True UVCGAN v2 generator with LayerScale, cross-domain feature sharing, and improved weight initialization
- `discriminator.py`: PatchGAN discriminator (v1)
- `spectral_norm_discriminator.py`: Spectral-norm multi-scale discriminator for v2

### Losses
- `losses.py`: v1 composite loss — VGG19 perceptual loss, LSGAN GAN loss, cycle-consistency, and identity terms
- `advanced_losses.py`: v2 advanced losses — adds WGAN-GP gradient penalty, NT-Xent contrastive domain-alignment loss, and frequency-domain spectral loss

### Data
- `data_loader.py`: Unpaired dataset loader and augmentation transforms

### Utilities
- `EarlyStopping.py`: Early stopping based on SSIM improvements and loss-divergence detection
- `replay_buffer.py`: Fixed-size replay buffer that mixes old and new fake samples to stabilize discriminator training
- `metrics.py`: SSIM, PSNR, and FID metrics using InceptionV3 features
- `validation.py`: Per-epoch validation — runs generators over the validation set, computes metrics, and saves comparison images
- `testing.py`: End-of-training testing — runs inference over the test set and writes comparison images to disk
- `history_utils.py`: Training history visualization and CSV persistence helpers

## Data Layout (Expected)
Place your dataset under `data\E_Staining_DermaRepo\H_E-Staining_dataset`:

```
data/
  E_Staining_DermaRepo/
    H_E-Staining_dataset/
      Un_Stained/
      C_Stained/
      trainA/
      trainB/
      testA/
      testB/
      models_YYYY_MM_DD_HH_MM_SS/
```

Notes:
- `Un_Stained` and `C_Stained` hold the original whole-slide images.
- `trainA/trainB/testA/testB` are created by `preprocess_data.py`.

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

GPU is optional. If you want CUDA acceleration, install a CUDA-compatible PyTorch build.

## Preprocess the Dataset
This step extracts 256x256 patches, applies tissue/background filtering, and creates the CycleGAN-style folders:

```bash
python preprocess_data.py
```

Outputs are written to:
`data\E_Staining_DermaRepo\H_E-Staining_dataset\trainA|trainB|testA|testB`

Filtering defaults (configurable in `preprocess_data.py`):
- `tissue_threshold=0.1` (minimum tissue fraction to keep)
- `background_keep_ratio=0.1` (keep ~10% background patches)
- `white_thresh=220`, `sat_thresh=0.05`

## Train the Model
Run the training entry point:

```bash
python trainModel.py
```

You will be prompted for:
- `Epoch Size`: number of samples per epoch
- `Number of Epochs`
- `Test Size`: number of test samples to export during testing

Hyperparameters are managed in `config.py`. Edit `UVCGANConfig` (or use `get_default_config(model_version=1|2)`) to tune the generator, discriminator, loss weights, learning rate schedule, and data settings before launching a run.

The script selects the training loop based on `model_version`:
- **v1** (`training_loop.py`): original CycleGAN-style pipeline with a single PatchGAN discriminator per domain and LSGAN losses
- **v2** (`training_loop_v2.py`): improved pipeline with WGAN-GP, multi-scale spectral-norm discriminators, gradient clipping, warm-up + linear LR decay, contrastive and spectral loss terms, and richer TensorBoard diagnostics

Training artifacts:
- Checkpoints and logs under `data\E_Staining_DermaRepo\H_E-Staining_dataset\models_...`
- Validation images per epoch under `validation_images`
- Test images under `test_images`
- TensorBoard logs under `tensorboard_logs`
- Training history CSV under the model directory (written by `history_utils.py`)

Training highlights (v2):
- UVCGAN v2 generator (U-Net + ViT, LayerScale, cross-domain sharing) with multi-scale spectral-norm discriminators
- WGAN-GP + cycle + identity + VGG19 perceptual + contrastive + spectral losses
- Gradient clipping and warm-up/linear-decay LR schedule
- Replay buffer for discriminator stabilization
- Mixed precision (AMP) when CUDA is available
- Early stopping monitored by SSIM improvement and loss-divergence detection

## Monitor with TensorBoard
From the model directory:

```bash
tensorboard --logdir data\E_Staining_DermaRepo\H_E-Staining_dataset\models_...\tensorboard_logs
```

## Inference (Stain / Unstain)
`app.py` loads a checkpoint and translates images by patching and reconstruction.

```bash
python app.py
```

By default, it expects a checkpoint path and example images inside the dataset tree.
Update the checkpoint path in `app.py` to point at your trained model:

- `data\E_Staining_DermaRepo\H_E-Staining_dataset\models_YYYY_MM_DD_HH_MM_SS\final_checkpoint_epoch_XXX.pth`

Outputs:
- `data\reconstructed_stained_output.png`
- `data\reconstructed_unstained_output.png`

Note: Checkpoints trained with older ResNet generators are not compatible with the current UVCGAN generator.

## Metrics
Validation includes:
- SSIM
- PSNR
- FID (on a small subset for speed)

Metrics are logged to TensorBoard and printed during training.

## Notes
- Paths are currently Windows-style (`\`). If you run on Linux/macOS, update paths accordingly.
- Patching size is 256; adjust in `preprocess_data.py` and `app.py` together if you change it.
