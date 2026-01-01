"""1D CNN encoder for Temporal Dynamics (TD) features.

ResNet-1D architecture adapted from IC's ResNet-3 (2D spatial → 1D temporal).
"""

import torch
import torch.nn as nn
from typing import Literal
from .base import BaseEncoder


class ResidualBlock1D(nn.Module):
    """1D Residual block for temporal encoding.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        stride: Stride for temporal downsampling (1 or 2)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input [B, in_channels, T]

        Returns:
            Output [B, out_channels, T']
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class TemporalCNNEncoder(BaseEncoder):
    """1D CNN encoder for Temporal Dynamics sequences.

    ResNet-1D architecture for temporal encoding. Processes time series
    [B, T, D_in] → temporal embeddings [B, D_out].

    Architecture:
        - Input: [B, T, D_in] time series (typically T=500, D_in~56)
        - Stage 1: Conv1d(D_in→32, k=7, s=2) + BN + ReLU + MaxPool
        - Stage 2: ResBlock1D(32→64, s=2)
        - Stage 3: ResBlock1D(64→128, s=2)
        - Stage 4: ResBlock1D(128→256, s=2)
        - Output: AdaptiveAvgPool1d + Linear(256→embedding_dim) + BN

    Temporal progression:
        T=500 → 250 (conv s=2) → 125 (maxpool s=2) → 62 (res s=2)
        → 31 (res s=2) → 16 (res s=2) → 8 (res s=2) → 1 (global pool)

    Receptive field: ~43 timesteps (8.6% of 500-timestep sequence)

    Args:
        input_dim: Input feature dimension per timestep (D_in)
        embedding_dim: Output embedding dimension (default: 64)
        architecture: Architecture variant (only 'resnet1d_3' supported)

    Example:
        >>> encoder = TemporalCNNEncoder(input_dim=56, embedding_dim=64)
        >>> sequences = torch.randn(32, 500, 56)  # [batch, time, features]
        >>> embeddings = encoder(sequences)       # [batch, 64]
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        architecture: Literal["resnet1d_3"] = "resnet1d_3",
    ):
        super().__init__()

        if architecture != "resnet1d_3":
            raise ValueError(f"Only 'resnet1d_3' architecture supported, got: {architecture}")

        self._input_dim = input_dim
        self._embedding_dim = embedding_dim

        # Stage 1: Initial temporal compression
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        # T=500 → 250 → 125 (after conv) → 62 (after maxpool)

        # Stage 2: First residual block
        self.stage2 = ResidualBlock1D(32, 64, stride=2)
        # T=62 → 31

        # Stage 3: Second residual block
        self.stage3 = ResidualBlock1D(64, 128, stride=2)
        # T=31 → 16

        # Stage 4: Third residual block
        self.stage4 = ResidualBlock1D(128, 256, stride=2)
        # T=16 → 8

        # Global temporal aggregation
        self.gap = nn.AdaptiveAvgPool1d(1)
        # T=8 → 1

        # Final projection to embedding space
        self.fc = nn.Linear(256, embedding_dim)
        self.bn_final = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal sequences to embeddings.

        Args:
            x: Input sequences [batch_size, T, input_dim]
               Expected: [B, 500, 56] for spinlock TD features

        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        # Transpose to [B, D_in, T] for Conv1D
        x = x.transpose(1, 2)  # [B, input_dim, T]

        # Residual stages
        x = self.stage1(x)  # [B, 32, 62]
        x = self.stage2(x)  # [B, 64, 31]
        x = self.stage3(x)  # [B, 128, 16]
        x = self.stage4(x)  # [B, 256, 8]

        # Global pooling
        x = self.gap(x)  # [B, 256, 1]
        x = x.view(x.size(0), -1)  # [B, 256]

        # Final projection
        x = self.fc(x)  # [B, embedding_dim]
        x = self.bn_final(x)  # [B, embedding_dim]

        return x

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self._embedding_dim
