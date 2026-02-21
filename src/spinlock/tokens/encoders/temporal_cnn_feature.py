"""Per-frame 2D CNN for learned temporal feature extraction.

Replaces the hand-crafted TemporalFeatureOrchestrator (spatial means, stds,
FFT, etc.) with a learnable ResNet-3 applied independently to each trajectory
frame. This provides dense, spatially-specific gradients instead of the
diffuse ~1/HW gradients from summary statistics.

Architecture:
    [B, T, C, H, W] → reshape [B*T, C, H, W]
                     → InitialCNNEncoder (ResNet-3)
                     → [B*T, D_per_frame]
                     → reshape [B, T, D_per_frame]

Reuses InitialCNNEncoder via composition — no code duplication.
"""

import torch
import torch.nn as nn

from spinlock.features.initial.cnn_encoder import InitialCNNEncoder


class TemporalCNNFeatureEncoder(nn.Module):
    """Per-frame 2D CNN for learned temporal feature extraction.

    Processes each trajectory frame independently with shared weights.
    The same InitialCNNEncoder (ResNet-3) is applied to every timestep,
    producing per-frame feature vectors that preserve temporal ordering.

    Args:
        in_channels: Number of input channels per frame (auto-detected from data)
        embedding_dim: Output dimension per frame (should be num_groups * group_dim)
    """

    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.frame_encoder = InitialCNNEncoder(
            embedding_dim=embedding_dim,
            in_channels=in_channels,
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each trajectory frame independently.

        Args:
            x: Trajectory tensor [B, T, C, H, W]

        Returns:
            Per-frame features [B, T, embedding_dim]
        """
        B, T, C, H, W = x.shape
        # Merge batch and time → process all frames in one CNN pass
        features = self.frame_encoder(x.reshape(B * T, C, H, W))  # [B*T, D]
        return features.reshape(B, T, self.embedding_dim)  # [B, T, D]
