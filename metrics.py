"""
Metrics for evaluating CycleGAN outputs.

Includes SSIM, PSNR, and FID utilities with an InceptionV3 feature extractor.
"""

# Imports
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from PIL import Image


class MetricsCalculator:
    """
    Metrics calculator for image translation quality and distribution alignment.

    Supports SSIM, PSNR, and FID calculations.
    """

    def __init__(self, device=None):
        """
        Initialize the metrics calculator.

        Args:
            device (torch.device | None): Device to run computations on.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Initialize InceptionV3 for FID calculation.
        self.inception_model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
        )
        # Remove final classification layer to get feature embeddings.
        setattr(self.inception_model, "fc", torch.nn.Identity())
        self.inception_model.eval().to(self.device)

        # Preprocessing for InceptionV3 (ImageNet normalization).
        self.inception_transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM (Structural Similarity Index) between two images.

        Args:
            img1, img2 (torch.Tensor | np.ndarray): Images to compare.

        Returns:
            float: SSIM value in [-1, 1] (higher is better).
        """
        # Convert to numpy and ensure proper format.
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()

        # Handle batch dimension.
        if len(img1.shape) == 4:
            ssim_values = []
            for i in range(img1.shape[0]):
                im1 = np.transpose(img1[i], (1, 2, 0))
                im2 = np.transpose(img2[i], (1, 2, 0))
                ssim_val = ssim(im1, im2, channel_axis=-1, data_range=2.0)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)

        # Single image case.
        im1 = np.transpose(img1, (1, 2, 0))
        im2 = np.transpose(img2, (1, 2, 0))
        return ssim(im1, im2, channel_axis=-1, data_range=2.0)

    def calculate_psnr(self, img1, img2):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

        Args:
            img1, img2 (torch.Tensor): Images to compare.

        Returns:
            float: PSNR value in dB (higher is better).
        """
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float("inf")
        # Assuming pixel values in range [-1, 1], max pixel value is 2.
        return 20 * torch.log10(2.0 / torch.sqrt(mse)).item()

    def get_inception_features(self, images):
        """
        Extract feature vectors from InceptionV3.

        Args:
            images (torch.Tensor): Batch of images in [-1, 1] range.

        Returns:
            numpy.ndarray: Feature vectors from InceptionV3.
        """
        with torch.no_grad():
            # Denormalize from [-1, 1] to [0, 1], then normalize for Inception.
            images = (images + 1) / 2.0
            images = self.inception_transform(images)
            features = self.inception_model(images)
        return features.cpu().numpy()

    def calculate_fid(self, real_images, fake_images):
        """
        Calculate FID (Frechet Inception Distance) between real and fake images.

        Args:
            real_images (torch.Tensor): Batch of real images.
            fake_images (torch.Tensor): Batch of generated images.

        Returns:
            float: FID score (lower is better).
        """
        real_features = self.get_inception_features(real_images)
        fake_features = self.get_inception_features(fake_images)

        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

        diff = mu1 - mu2
        try:
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except:
            # Add small regularization to diagonal if matrix is singular.
            eps = 1e-6
            sigma1_reg = sigma1 + eps * np.eye(sigma1.shape[0])
            sigma2_reg = sigma2 + eps * np.eye(sigma2.shape[0])
            covmean, _ = linalg.sqrtm(sigma1_reg.dot(sigma2_reg), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def evaluate_batch(self, real_A, real_B, fake_A, fake_B):
        """
        Evaluate a batch of images with SSIM and PSNR.

        Args:
            real_A, real_B (torch.Tensor): Real images from domains A and B.
            fake_A, fake_B (torch.Tensor): Generated images for domains A and B.

        Returns:
            dict: Dictionary containing calculated metrics.
        """
        metrics = {}

        metrics["ssim_A"] = self.calculate_ssim(real_A, fake_A)
        metrics["ssim_B"] = self.calculate_ssim(real_B, fake_B)

        metrics["psnr_A"] = self.calculate_psnr(real_A, fake_A)
        metrics["psnr_B"] = self.calculate_psnr(real_B, fake_B)

        return metrics

    def evaluate_fid(self, real_images, fake_images):
        """
        Wrapper for FID computation.

        Args:
            real_images (torch.Tensor): Real image batch.
            fake_images (torch.Tensor): Generated image batch.

        Returns:
            float: FID score.
        """
        return self.calculate_fid(real_images, fake_images)
