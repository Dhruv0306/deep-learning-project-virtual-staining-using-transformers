"""
Patch sampler for CUT-style PatchNCE (v4).

Samples spatial locations from feature maps and returns patch tensors for
contrastive learning.  Patch indices can be reused to align positives
between real and generated features.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch


class PatchSampler:
    """
    Random spatial patch sampler for NCE features.

    Args:
        num_patches: Default number of patches to sample per feature map.
    """

    def __init__(self, num_patches: int = 128):
        self.num_patches = num_patches

    def _sample_ids(self, b: int, hw: int, num_patches: int, device) -> torch.Tensor:
        """
        Sample random spatial indices for one feature map.

        If *num_patches* >= *hw*, all positions are returned in order.

        Args:
            b:           Batch size.
            hw:          Total number of spatial positions (H * W).
            num_patches: Number of positions to sample.
            device:      Device for the output index tensor.

        Returns:
            LongTensor of shape (B, num_patches) with sampled indices.
        """
        if num_patches <= 0 or num_patches >= hw:
            ids = torch.arange(hw, device=device)
            ids = ids.unsqueeze(0).repeat(b, 1)
            return ids
        return torch.randint(0, hw, (b, num_patches), device=device)

    def sample(
        self,
        features: Iterable[torch.Tensor],
        num_patches: int | None = None,
        patch_ids: List[torch.Tensor] | None = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample patches from each feature map.

        Args:
            features: Iterable of tensors shaped (B, C, H, W).
            num_patches: Number of patches per feature map.
            patch_ids: Optional list of precomputed indices per feature map.

        Returns:
            tuple(list[Tensor], list[Tensor]):
                - sampled features list, each of shape (B, N, C)
                - patch ids list, each of shape (B, N)
        """
        if num_patches is None:
            num_patches = self.num_patches

        sampled: List[torch.Tensor] = []
        out_ids: List[torch.Tensor] = []

        feats = list(features)
        for i, feat in enumerate(feats):
            if feat.dim() != 4:
                raise ValueError("Expected feature map with shape (B, C, H, W).")
            b, c, h, w = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            hw = h * w

            if patch_ids is None:
                ids = self._sample_ids(b, hw, num_patches, feat.device)
            else:
                ids = patch_ids[i]
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0).repeat(b, 1)
                if ids.shape[0] != b:
                    raise ValueError("patch_ids batch dimension must match features.")

            ids = ids.clamp(min=0, max=hw - 1)
            idx = ids.unsqueeze(-1).expand(-1, -1, c)
            patches = feat_flat.gather(1, idx)

            sampled.append(patches)
            out_ids.append(ids)

        return sampled, out_ids


__all__ = ["PatchSampler"]
