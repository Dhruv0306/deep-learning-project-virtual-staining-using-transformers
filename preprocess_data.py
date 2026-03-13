"""
Preprocessing script for building CycleGAN training/test datasets.

Splits whole-slide images into fixed-size patches and writes them into
trainA/trainB/testA/testB folders.
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Allow very large whole-slide images without PIL raising a DecompressionBombWarning.
# This is common for whole-slide pathology images, which can be gigapixels.
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)


def extract_patches_pil(img, patch_size=256, stride=256):
    """
    Extract patches from a PIL image.

    Args:
        img (PIL.Image.Image): Source image.
        patch_size (int): Square patch size.
        stride (int): Stride between patches; stride == patch_size is non-overlap.

    Returns:
        list[PIL.Image.Image]: List of patch images.
    """
    # PIL uses (width, height) ordering.
    img_w, img_h = img.size
    patches = []

    # Slide a window across the image to collect tiles.
    for top in range(0, img_h - patch_size + 1, stride):
        for left in range(0, img_w - patch_size + 1, stride):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patches.append(patch)

    return patches


def split_filenames(file_list, train_ratio=0.8, seed=42):
    """
    Split a list of filenames into train and test subsets.

    Args:
        file_list (list[str]): Filenames to split.
        train_ratio (float): Fraction to keep for training.
        seed (int): RNG seed for reproducibility.

    Returns:
        tuple: (train_files, test_files)
    """
    file_list = sorted(file_list)
    np.random.seed(seed)
    np.random.shuffle(file_list)

    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]


def save_patches(image_path, save_dir, patch_size=256):
    """
    Load an image, extract patches, and write them as PNGs.

    Args:
        image_path (str): Path to the source image.
        save_dir (str): Directory to save patches.
        patch_size (int): Square patch size.
    """
    # Convert to RGB to guarantee 3-channel patches for CycleGAN-style datasets.
    img = Image.open(image_path).convert("RGB")
    # Extract tiles directly from the full-resolution slide.
    patches = extract_patches_pil(img, patch_size)

    base = os.path.splitext(os.path.basename(image_path))[0]
    print(f"Processing Patch for image {base}")

    for i, patch in enumerate(patches):
        # Keep the patch as 8-bit RGB; normalization is handled during data loading.
        patch = patch.resize((patch_size, patch_size), Image.Resampling.BICUBIC)
        patch.save(os.path.join(save_dir, f"{base}_{i}.png"))


def main():
    """
    Entry point for preprocessing and dataset split generation.
    """
    # Dataset root directory.
    DATASET_DIR = "data\\E_Staining_DermaRepo\\H_E-Staining_dataset"
    print(f'Dataset Dir "{DATASET_DIR}"')

    # Source folders for unstained and stained images.
    UNSTAINED_DIR = DATASET_DIR + "\\Un_Stained"
    print(f'Unstained Images Dir "{UNSTAINED_DIR}"')

    STAINED_DIR = DATASET_DIR + "\\C_Stained"
    print(f'Stained Images Dir "{STAINED_DIR}"')

    # Inspect one sample unstained image for sanity check.
    # This visualization is optional and can be removed for headless runs.
    img_path = os.path.join(UNSTAINED_DIR, os.listdir(UNSTAINED_DIR)[0])

    with Image.open(img_path) as img:
        plt.figure(figsize=(10, 20))
        plt.imshow(img)
        plt.title("Sample Unstained Tissue Image")
        plt.axis("off")

    # CycleGAN-style output folders.
    os.makedirs(f"{DATASET_DIR}\\trainA", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}\\trainB", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}\\testA", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}\\testB", exist_ok=True)

    print(f"Saving Unstained Images Patch")
    print(f"Splitting images in train test list")
    unstained_files = [
        f
        for f in os.listdir(UNSTAINED_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    trainA_files, testA_files = split_filenames(unstained_files)

    print(f"Saving Unstained TRAIN patches")
    for img_name in trainA_files:
        # Unstained images become domain A.
        save_patches(os.path.join(UNSTAINED_DIR, img_name), f"{DATASET_DIR}\\trainA")
    print(f"Saved Unstained Train Patch")

    print(f"Saving Unstained TEST patches")
    for img_name in testA_files:
        # Unstained images become domain A.
        save_patches(os.path.join(UNSTAINED_DIR, img_name), f"{DATASET_DIR}\\testA")
    print(f"Saved Unstained TEST Patch")

    print(f"Saving Stained Images Patch")
    print(f"Splitting images in train test list")
    stained_files = [
        f
        for f in os.listdir(STAINED_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    trainB_files, testB_files = split_filenames(stained_files)

    print(f"Saving Stained TRAIN patches")
    for img_name in trainB_files:
        # Stained images become domain B.
        save_patches(os.path.join(STAINED_DIR, img_name), f"{DATASET_DIR}\\trainB")
    print(f"Saved Stained Train Patch")

    print(f"Saving Stained TEST patches")
    for img_name in testB_files:
        # Stained images become domain B.
        save_patches(os.path.join(STAINED_DIR, img_name), f"{DATASET_DIR}\\testB")
    print(f"Saved Stained TEST Patch")


if __name__ == "__main__":
    main()
