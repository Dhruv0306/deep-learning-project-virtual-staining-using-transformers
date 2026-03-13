"""
Data loading utilities for CycleGAN training.

Provides:
- denormalize: convert tensors from [-1, 1] to [0, 1] for visualization
- UnpairedImageDataset: unpaired sampling across two domains
- getDataLoader: builds train/test loaders with standard transforms
"""

# Imports
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Function to denormalize tensors from [-1,1] back to [0,1] range for visualization

def denormalize(t):
    """
    Denormalize tensors from [-1, 1] to [0, 1] for visualization.

    Args:
        t (torch.Tensor): Input tensor in [-1, 1] range.

    Returns:
        torch.Tensor: Denormalized tensor in [0, 1] range.
    """
    # Reverse the normalization: (x - 0.5) / 0.5 becomes x * 0.5 + 0.5.
    return (t * 0.5 + 0.5).clamp(0, 1)


# Custom dataset class for handling unpaired image datasets used in CycleGAN training
# Unlike paired datasets, this allows training with images from two domains that do not correspond 1:1
class UnpairedImageDataset(Dataset):
    """
    Unpaired dataset for two image domains (A and B).

    Domain A is sampled sequentially; domain B is sampled randomly to avoid
    fixed pairings and encourage diverse combinations.
    """

    def __init__(self, dir_A, dir_B, transform=None, epoch_size=None):
        """
        Initialize the UnpairedImageDataset.

        Args:
            dir_A (str): Directory containing images from domain A.
            dir_B (str): Directory containing images from domain B.
            transform (callable, optional): Transform to apply to images.
            epoch_size (int, optional): Fixed epoch size to control iterations.
        """
        # Store directory paths for both image domains.
        self.dir_A = dir_A
        self.dir_B = dir_B

        # Get sorted list of image filenames from both directories.
        # Sorting ensures consistent ordering across runs.
        self.images_A = sorted(os.listdir(dir_A))
        self.images_B = sorted(os.listdir(dir_B))

        # Store transformation pipeline and optional epoch size.
        self.transform = transform
        self.epoch_size = epoch_size

    def __len__(self):
        """
        Return dataset length for unpaired sampling.

        Returns:
            int: epoch_size if provided, else max(len(A), len(B)).
        """
        # If epoch_size is specified, use it to control training iterations.
        if self.epoch_size is not None:
            return self.epoch_size

        # CycleGAN uses the max length so the smaller domain does not end training early.
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        """
        Retrieve one unpaired sample.

        Args:
            idx (int): Index for domain A (domain B is random).

        Returns:
            dict: {'A': image_A, 'B': image_B}
        """
        # Import random here to avoid global import overhead.
        import random

        # Sequential sampling for domain A with wraparound using modulo.
        img_A = self.images_A[idx % len(self.images_A)]

        # Random sampling for domain B to increase pairing diversity.
        img_B = random.choice(self.images_B)

        # Construct full file paths.
        path_A = os.path.join(self.dir_A, img_A)
        path_B = os.path.join(self.dir_B, img_B)

        # Load images and ensure RGB format (handles grayscale conversion).
        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")

        # Apply transformation pipeline if provided (resize, normalize, etc.).
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return {"A": image_A, "B": image_B}


# Data Loader Main Function
# Comprehensive function to set up the entire data loading pipeline for CycleGAN training

def getDataLoader(epoch_size=None):
    """
    Create and return train and test data loaders for CycleGAN training.

    Args:
        epoch_size (int, optional): Fixed number of samples per epoch.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # System diagnostics for GPU availability.
    print(f"torch version: {torch.__version__}")
    print(f"Checking GPU available: ")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Dataset path configuration (Windows-style paths).
    trainA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainA"  # Domain A
    trainB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\trainB"  # Domain B
    testA = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testA"  # Domain A
    testB = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\testB"  # Domain B

    # Image transformation pipeline (CycleGAN standard).
    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    # Training dataset and loader.
    train_dataset = UnpairedImageDataset(
        dir_A=trainA,
        dir_B=trainB,
        transform=transform,
        epoch_size=epoch_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    # Testing dataset and loader.
    test_dataset = UnpairedImageDataset(
        dir_A=testA,
        dir_B=testB,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    # Sanity checks for batch shapes.
    batchTrain = next(iter(train_loader))

    print(f"Train dataloader batch shape test: ")
    print(
        f"BatchA or trainA : {batchTrain['A'].shape}"
    )  # Expected: [batch_size, 3, 256, 256]
    print(
        f"BatchB or trainB : {batchTrain['B'].shape}"
    )  # Expected: [batch_size, 3, 256, 256]

    batchTest = next(iter(test_loader))

    print(f"Test dataloader batch shape test: ")
    print(f"BatchA or testA : {batchTest['A'].shape}")  # Expected: [1, 3, 256, 256]
    print(f"BatchB or testB : {batchTest['B'].shape}")  # Expected: [1, 3, 256, 256]

    return train_loader, test_loader


# Main execution block - demonstration and testing when script is run directly
if __name__ == "__main__":
    # Initialize data loaders with specified epoch size for controlled training.
    train_loader, test_loader = getDataLoader(epoch_size=3000)

    # Training Data Visualization
    batchTrain = next(iter(train_loader))

    A = denormalize(batchTrain["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTrain["B"][0]).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")
    plt.axis("off")
    plt.show()

    # Testing Data Visualization
    batchTest = next(iter(test_loader))

    A = denormalize(batchTest["A"][0]).permute(1, 2, 0).cpu().numpy()
    B = denormalize(batchTest["B"][0]).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.title(f"Un-Stained (A)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.title("C-Stained (B)")
    plt.axis("off")
    plt.show()
