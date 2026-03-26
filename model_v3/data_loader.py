"""
V3-specific data loader helpers for paired image translation.

DiT diffusion uses paired (A, B) samples, so this module builds paired
train/test datasets from matching filenames in trainA/trainB and testA/testB.
"""

import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class PairedImageDataset(Dataset):
    """
    Paired dataset for aligned A/B domains.

    Pairs are formed by matching filenames across the two directories.
    """

    def __init__(self, dir_A, dir_B, transform=None, epoch_size=None, strict_pairs=True):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transform = transform
        self.epoch_size = epoch_size
        self.strict_pairs = strict_pairs

        images_A = sorted(os.listdir(dir_A))
        images_B = sorted(os.listdir(dir_B))

        if strict_pairs:
            common = sorted(set(images_A).intersection(images_B))
            if not common:
                raise ValueError(
                    "No paired filenames found between directories. "
                    "Ensure trainA/trainB (or testA/testB) share matching filenames."
                )
            self.pairs: List[Tuple[str, str]] = [(name, name) for name in common]
        else:
            min_len = min(len(images_A), len(images_B))
            if min_len == 0:
                raise ValueError("Empty paired directories for A/B data.")
            self.pairs = list(zip(images_A[:min_len], images_B[:min_len]))

    def __len__(self):
        if self.epoch_size is not None:
            return self.epoch_size
        return len(self.pairs)

    def __getitem__(self, idx):
        name_A, name_B = self.pairs[idx % len(self.pairs)]
        path_A = os.path.join(self.dir_A, name_A)
        path_B = os.path.join(self.dir_B, name_B)

        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {"A": image_A, "B": image_B}


def getDataLoaderV3(
    epoch_size=None,
    image_size=256,
    batch_size=4,
    num_workers=4,
    strict_pairs=True,
):
    """
    Create and return paired train/test data loaders for v3 diffusion.

    Args:
        epoch_size (int | None): Fixed number of samples per epoch.
        image_size (int): Resize size.
        batch_size (int): Batch size.
        num_workers (int): Data loader workers.
        strict_pairs (bool): Require matching filenames across A/B dirs.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    print(f"torch version: {torch.__version__}")
    print("Checking GPU available:")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    data_root = os.path.join("data", "E_Staining_DermaRepo", "H_E-Staining_dataset")
    trainA = os.path.join(data_root, "trainA")
    trainB = os.path.join(data_root, "trainB")
    testA = os.path.join(data_root, "testA")
    testB = os.path.join(data_root, "testB")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = PairedImageDataset(
        dir_A=trainA,
        dir_B=trainB,
        transform=transform,
        epoch_size=epoch_size,
        strict_pairs=strict_pairs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    test_dataset = PairedImageDataset(
        dir_A=testA,
        dir_B=testB,
        transform=transform,
        strict_pairs=strict_pairs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    batchTrain = next(iter(train_loader))
    print("Train dataloader batch shape test:")
    print(f"BatchA or trainA : {batchTrain['A'].shape}")
    print(f"BatchB or trainB : {batchTrain['B'].shape}")

    batchTest = next(iter(test_loader))
    print("Test dataloader batch shape test:")
    print(f"BatchA or testA : {batchTest['A'].shape}")
    print(f"BatchB or testB : {batchTest['B'].shape}")

    return train_loader, test_loader
