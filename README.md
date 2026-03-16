# UVCGAN-Style CycleGAN for Histology Stain/Unstain Translation

This project trains a CycleGAN-style pipeline with a **UVCGAN generator (U-Net + ViT)** to translate between **unstained** and **stained** histology tissue images. It includes:
- Dataset preprocessing to create 256x256 patches (with background/tissue filtering)
- Training with validation, metrics, early stopping, and TensorBoard
- Inference to stain or unstain whole images by patching and reconstruction

## Dataset
This project uses the **E-stainind DermaRepo H&E staining dataset** with **unstained** and **stained** image domains. The raw images should be organized into `Un_Stained` and `C_Stained` folders, and preprocessing will create `trainA/trainB/testA/testB` patch datasets from them.

All rights for the dataset are held by the original owners and licensors of the dataset.

## Project Layout
- `trainModel.py`: Training entry point (prompts for epoch size, epochs, test size)
- `training_loop.py`: Full training loop, logging, validation, testing, checkpoints
- `preprocess_data.py`: Patch extraction, tissue/background filtering, and train/test split
- `app.py`: Inference script for stain/unstain translation
- `generator.py` / `discriminator.py`: UVCGAN generator (U-Net + ViT) and PatchGAN discriminator
- `data_loader.py`: Unpaired dataset loader and transforms

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

Training artifacts:
- Checkpoints and logs under `data\E_Staining_DermaRepo\H_E-Staining_dataset\models_...`
- Validation images per epoch under `validation_images`
- Test images under `test_images`
- TensorBoard logs under `tensorboard_logs`

Training highlights:
- UVCGAN generator (U-Net + ViT) with PatchGAN discriminators
- LSGAN + cycle + identity + perceptual losses
- Gradient penalty on discriminators
- Mixed precision (AMP) when CUDA is available
- Perceptual loss computed at 128x128 for lower memory
- Channels-last memory format on CUDA for better performance

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
