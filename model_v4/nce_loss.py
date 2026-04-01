"""
PatchNCE loss for CUT-style contrastive learning (v4).

Implements InfoNCE over sampled patches with per-layer projection heads.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNCELoss(nn.Module):
    """
    PatchNCE contrastive loss with per-layer MLP projection.

    Args:
        temperature: Softmax temperature for InfoNCE.
        proj_dim: Output dimension for the projection MLPs.
    """

    def __init__(self, temperature: float = 0.07, proj_dim: int = 256):
        super().__init__()
        self.temperature = temperature
        self.proj_dim = proj_dim
        self.projections = nn.ModuleDict()

    def _get_projector(
        self, key: str, in_dim: int, device: torch.device, dtype: torch.dtype
    ) -> nn.Module:
        """
        Lazily create and cache a 2-layer MLP projection head.

        A separate projector is maintained for each feature layer (keyed by
        *key*) so that different layers can learn independent projections.

        Args:
            key:    String identifier for the layer (e.g. "0", "1").
            in_dim: Input feature dimension.
            device: Device on which to place the projector.
            dtype:  Dtype for the projector parameters.

        Returns:
            nn.Sequential MLP projector.
        """
        if key not in self.projections:
            proj = nn.Sequential(
                nn.Linear(in_dim, self.proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_dim, self.proj_dim),
            )
            proj = proj.to(device=device, dtype=dtype)
            self.projections[key] = proj
        return self.projections[key]

    def _layer_loss(self, feat_q: torch.Tensor, feat_k: torch.Tensor, key: str):
        """
        Compute InfoNCE loss for a single feature layer.

        Queries and keys are projected, L2-normalised, and compared via a
        cross-entropy objective where each query's positive is the key at the
        same spatial position.

        Args:
            feat_q: Query features of shape (B, N, C).
            feat_k: Key features of shape (B, N, C).
            key:    Layer identifier used to look up the projector.

        Returns:
            Scalar cross-entropy loss.
        """
        b, n, c = feat_q.shape
        proj = self._get_projector(key, c, feat_q.device, feat_q.dtype)

        q = proj(feat_q)
        k = proj(feat_k)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        q = q.reshape(b * n, -1)
        k = k.reshape(b * n, -1)

        logits = torch.matmul(q, k.t()) / max(1e-8, self.temperature)
        labels = torch.arange(b * n, device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        feats_q: Iterable[torch.Tensor],
        feats_k: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute average PatchNCE loss across all feature layers.

        Args:
            feats_q: Iterable of sampled query features (B, N, C).
            feats_k: Iterable of sampled key features (B, N, C).

        Returns:
            torch.Tensor: Scalar PatchNCE loss.
        """
        losses: List[torch.Tensor] = []
        for i, (fq, fk) in enumerate(zip(feats_q, feats_k)):
            if fq.numel() == 0 or fk.numel() == 0:
                continue
            losses.append(self._layer_loss(fq, fk, key=str(i)))

        if not losses:
            device = (
                next(iter(feats_q)).device
                if isinstance(feats_q, list) and feats_q
                else torch.device("cpu")
            )
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()


__all__ = ["PatchNCELoss"]
