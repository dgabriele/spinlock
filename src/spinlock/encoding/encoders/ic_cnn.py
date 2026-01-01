"""CNN encoder for Initial Condition (IC) spatial processing.

ResNet-3 architecture adapted for 128×128 IC grids.
"""

import torch
import torch.nn as nn
from typing import Literal
from .base import BaseEncoder


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        stride: Stride for downsampling (1 or 2)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input [B, in_channels, H, W]

        Returns:
            Output [B, out_channels, H', W']
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


class ICCNNEncoder(BaseEncoder):
    """CNN encoder for Initial Condition spatial features.

    ResNet-3 architecture optimized for 128×128 IC grids. Extracts learned
    spatial representations from raw IC data for VQ-VAE tokenization.

    Architecture:
        - Input: [B, 1, 128, 128] IC grids
        - Stage 1: Conv(1→32, k=7, s=2) + BN + ReLU + MaxPool
        - Stage 2: ResBlock(32→64, s=2)
        - Stage 3: ResBlock(64→128, s=2)
        - Stage 4: ResBlock(128→256, s=2)
        - Output: AdaptiveAvgPool + Linear(256→embedding_dim) + BN

    Args:
        embedding_dim: Output embedding dimension (default: 28)
        in_channels: Input channels (default: 1 for grayscale ICs)
        architecture: Architecture variant (only 'resnet3' supported)

    Example:
        >>> encoder = ICCNNEncoder(embedding_dim=28)
        >>> ics = torch.randn(32, 1, 128, 128)  # [batch, channels, H, W]
        >>> embeddings = encoder(ics)           # [batch, 28]
    """

    def __init__(
        self,
        embedding_dim: int = 28,
        in_channels: int = 1,
        architecture: Literal['resnet3'] = 'resnet3'
    ):
        super().__init__()

        if architecture != 'resnet3':
            raise ValueError(f"Only 'resnet3' architecture supported, got: {architecture}")

        self._embedding_dim = embedding_dim
        self.in_channels = in_channels

        # Stage 1: Initial convolution
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # After stage1: [B, 32, 32, 32]

        # Stage 2: First residual block
        self.stage2 = ResidualBlock(32, 64, stride=2)
        # After stage2: [B, 64, 16, 16]

        # Stage 3: Second residual block
        self.stage3 = ResidualBlock(64, 128, stride=2)
        # After stage3: [B, 128, 8, 8]

        # Stage 4: Third residual block
        self.stage4 = ResidualBlock(128, 256, stride=2)
        # After stage4: [B, 256, 4, 4]

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # After GAP: [B, 256, 1, 1] → [B, 256]

        # Final projection to embedding space
        self.fc = nn.Linear(256, embedding_dim)
        self.bn_final = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode IC spatial data to embedding.

        Args:
            x: Input IC grids [batch_size, in_channels, H, W]
               Expected: [B, 1, 128, 128]

        Returns:
            Embeddings [batch_size, embedding_dim]
        """
        # Residual stages
        x = self.stage1(x)  # [B, 32, 32, 32]
        x = self.stage2(x)  # [B, 64, 16, 16]
        x = self.stage3(x)  # [B, 128, 8, 8]
        x = self.stage4(x)  # [B, 256, 4, 4]

        # Global pooling
        x = self.gap(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]

        # Final projection
        x = self.fc(x)  # [B, embedding_dim]
        x = self.bn_final(x)  # [B, embedding_dim]

        return x

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self._embedding_dim
