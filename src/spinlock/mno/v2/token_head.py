"""Learned token prediction head — bypasses frozen VQ encoder.

The frozen VQ encoder creates a gradient bottleneck: token CE gradients must
flow through frozen pyramid encoder + projectors (trained for noise invariance).
This head provides a direct gradient path: CE → learned head → MNO trajectory.

Architecture:
    pred_trajectory [B, T, C, H, W]
        → Spatial encoder (shared per-frame Conv2D, stride-heavy): → [B, T, d_spatial]
        → Temporal encoder (Conv1D over frame embeddings): → [B, d_temporal]
        → Per-quantizer classification heads (linear): → {qkey: [B, K_l]} logits
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class TokenPredictionHead(nn.Module):
    """Learned head: trajectory → token logits, bypassing frozen VQ encoder.

    ~295K params for d_spatial=64, d_temporal=128, 90 quantizers.
    """

    def __init__(
        self,
        in_channels: int,
        quantizer_vocab_sizes: Dict[str, int],
        d_spatial: int = 64,
        d_temporal: int = 128,
    ) -> None:
        super().__init__()

        # Spatial encoder: per-frame CNN, stride-heavy 128→32→8→4→pool(1)
        # GroupNorm instead of BatchNorm: B=1 training makes BN statistics
        # meaningless, and constant-trajectory inputs cause BN to produce NaN.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(min(8, 16), 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2),
            nn.GroupNorm(min(8, 32), 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, d_spatial, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, d_spatial), d_spatial),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # Temporal encoder: Conv1D over frame embedding sequence
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(d_spatial, d_temporal, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, d_temporal), d_temporal),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_temporal, d_temporal, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, d_temporal), d_temporal),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

        # Per-quantizer classification heads
        self.heads = nn.ModuleDict({
            qkey: nn.Linear(d_temporal, vocab_size)
            for qkey, vocab_size in sorted(quantizer_vocab_sizes.items())
        })

    def forward(self, trajectory: Tensor) -> Dict[str, Tensor]:
        """Map trajectory to per-quantizer token logits.

        Args:
            trajectory: [B, T, C, H, W] predicted trajectory (IC excluded).

        Returns:
            {quantizer_key: [B, K_l]} logits for each temporal quantizer.
        """
        B, T, C, H, W = trajectory.shape

        # Spatial: merge B and T, run shared CNN, reshape back
        x = trajectory.reshape(B * T, C, H, W)           # [B*T, C, H, W]
        x = self.spatial_encoder(x)                       # [B*T, d_spatial, 1, 1]
        x = x.squeeze(-1).squeeze(-1)                     # [B*T, d_spatial]
        x = x.reshape(B, T, -1)                           # [B, T, d_spatial]

        # Temporal: Conv1D expects [B, C, T]
        x = x.transpose(1, 2)                             # [B, d_spatial, T]
        x = self.temporal_encoder(x)                      # [B, d_temporal, 1]
        x = x.squeeze(-1)                                 # [B, d_temporal]

        # Per-quantizer logits
        return {qkey: head(x) for qkey, head in self.heads.items()}

    @classmethod
    def from_vq_adapter(
        cls,
        vq_adapter: nn.Module,
        in_channels: int,
        d_spatial: int = 64,
        d_temporal: int = 128,
    ) -> "TokenPredictionHead":
        """Factory: extract vocab sizes from frozen VQ model quantizers.

        Args:
            vq_adapter: VQCoherenceAdapter with .model.quantizers dict.
            in_channels: Number of trajectory channels (e.g. 3 for Lenia).
            d_spatial: Spatial encoder output dim.
            d_temporal: Temporal encoder output dim.

        Returns:
            TokenPredictionHead with auto-detected vocab sizes.
        """
        vocab_sizes: Dict[str, int] = {}
        for qkey, quantizer in vq_adapter.model.quantizers.items():
            if qkey.startswith("temporal_"):
                vocab_sizes[qkey] = quantizer.embedding.num_embeddings
        if not vocab_sizes:
            raise ValueError(
                "No temporal quantizers found in VQ adapter — "
                "cannot build token prediction head"
            )
        return cls(
            in_channels=in_channels,
            quantizer_vocab_sizes=vocab_sizes,
            d_spatial=d_spatial,
            d_temporal=d_temporal,
        )
