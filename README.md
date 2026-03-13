# CycleGAN for Histology Stain/Unstain Translation

This project trains a CycleGAN to translate between **unstained** and **stained** histology tissue images. It includes:
- Dataset preprocessing to create 256x256 patches
- Training with validation, metrics, and early stopping
- Inference to stain or unstain whole images by patching and reconstruction

## Dataset
This project uses the **E-stainind DermaRepo H&E staining dataset** with **unstained** and **stained** image domains. The raw images should be organized into `Un_Stained` and `C_Stained` folders, and preprocessing will create `trainA/trainB/testA/testB` patch datasets from them.

All rights for the dataset are held by the original owners and licensors of the dataset.

## Project Layout
- `trainModel.py`: Training entry point (prompts for epoch size, epochs, test size)
- `training_loop.py`: Full training loop, logging, validation, testing, checkpoints
- `preprocess_data.py`: Patch extraction and train/test split
- `app.py`: Inference script for stain/unstain translation
- `generator.py` / `discriminator.py`: CycleGAN models (ResNet generator, PatchGAN discriminator)
- `data_loader.py`: Unpaired dataset loader and transforms

## Data Layout (Expected)
Place your dataset under `data\E_Staining_DermaRepo\H_E-Staining_dataset`:

```
data
└─ E_Staining_DermaRepo
   └─ H_E-Staining_dataset
      ├─ Un_Stained
      ├─ C_Stained
      ├─ trainA
      ├─ trainB
      ├─ testA
      ├─ testB
      └─ models_YYYY_MM_DD_HH_MM_SS
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
This step extracts 256x256 patches and creates the CycleGAN-style folders:

```bash
python preprocess_data.py
```

Outputs are written to:
`data\E_Staining_DermaRepo\H_E-Staining_dataset\trainA|trainB|testA|testB`

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

## Metrics
Validation includes:
- SSIM
- PSNR
- FID (on a small subset for speed)

Metrics are logged to TensorBoard and printed during training.

## Notes
- Paths are currently Windows-style (`\`). If you run on Linux/macOS, update paths accordingly.
- Patching size is 256; adjust in `preprocess_data.py` and `app.py` together if you change it.
