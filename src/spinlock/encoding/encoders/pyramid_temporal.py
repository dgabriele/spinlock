"""Multi-resolution pyramid encoder for temporal features.

Uses a shared ResNet-1D backbone (reusing ResidualBlock1D from temporal_cnn.py)
with per-level projection heads to produce multi-scale temporal embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Literal

from .base import BaseEncoder
from .temporal_cnn import ResidualBlock1D
from ..temporal_pyramid import TemporalPyramid


class PyramidTemporalEncoder(BaseEncoder):
    """Multi-resolution temporal encoder with shared backbone.

    Processes temporal input at multiple resolutions (pyramid levels) through
    a shared ResNet-1D backbone, then projects each level to a different
    embedding dimension via per-level heads.

    Input:  [B, T, D_in]
    Output: [B, sum(level_dims)]  (e.g., [B, 320] for [32, 64, 96, 128])

    Architecture:
        TemporalPyramid → shared backbone (per level) → per-level head → concat

    The shared backbone is:
        Stage 1: Conv1d(D_in→32, k=7, s=2) + BN + ReLU + MaxPool
        Stage 2: ResBlock1D(32→64, s=2)
        Stage 3: ResBlock1D(64→128, s=2)
        Stage 4: ResBlock1D(128→256, s=2)
        GAP → [B, 256]

    Each per-level head: Linear(256→level_dim) + BatchNorm1d(level_dim)

    Attributes:
        output_dims_per_level: List[int] of per-level output dimensions.
            Used by train_vqvae.py to detect pyramid encoder and split
            features into per-level families.

    Args:
        input_dim: Per-timestep feature dimension (D_in)
        level_dims: Output dimensions for each pyramid level.
            Default: [32, 64, 96, 128] (total 320D)
        downsample_factors: Temporal downsampling factors for pyramid.
            Default: [1, 2, 4, 8] (full, half, quarter, eighth resolution)
        architecture: Backbone architecture variant.
            Only 'resnet1d_3' supported.

    Example:
        >>> encoder = PyramidTemporalEncoder(input_dim=345, level_dims=[32, 64, 96, 128])
        >>> x = torch.randn(16, 256, 345)  # [B, T, D_in]
        >>> out = encoder(x)               # [B, 320]
        >>> out.shape
        torch.Size([16, 320])
    """

    def __init__(
        self,
        input_dim: int,
        level_dims: List[int] = [32, 64, 96, 128],
        downsample_factors: List[int] = [1, 2, 4, 8],
        architecture: Literal["resnet1d_3"] = "resnet1d_3",
    ):
        super().__init__()

        if architecture != "resnet1d_3":
            raise ValueError(f"Only 'resnet1d_3' architecture supported, got: {architecture}")

        if len(level_dims) != len(downsample_factors):
            raise ValueError(
                f"level_dims ({len(level_dims)}) must match "
                f"downsample_factors ({len(downsample_factors)})"
            )

        self._input_dim = input_dim
        self.output_dims_per_level = list(level_dims)
        self._output_dim = sum(level_dims)

        # Multi-resolution temporal pyramid
        self.pyramid = TemporalPyramid(downsample_factors)

        # Shared backbone (same architecture as TemporalCNNEncoder)
        self.stage1 = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage2 = ResidualBlock1D(32, 64, stride=2)
        self.stage3 = ResidualBlock1D(64, 128, stride=2)
        self.stage4 = ResidualBlock1D(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Per-level projection heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, dim),
                nn.BatchNorm1d(dim),
            )
            for dim in level_dims
        ])

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared backbone on a single pyramid level.

        Args:
            x: [B, D_in, T_i] (already transposed)

        Returns:
            [B, 256] backbone features
        """
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.gap(h).squeeze(-1)  # [B, 256]
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal input at multiple resolutions.

        Args:
            x: Input sequences [B, T, D_in]

        Returns:
            Concatenated multi-resolution embeddings [B, sum(level_dims)]
        """
        pyramid_levels = self.pyramid(x)  # List of [B, T_i, D_in]

        level_embeddings = []
        for level_input, head in zip(pyramid_levels, self.heads):
            # Transpose for Conv1d: [B, T_i, D_in] → [B, D_in, T_i]
            h = level_input.transpose(1, 2)
            h = self._backbone(h)   # [B, 256]
            h = head(h)             # [B, level_dim]
            level_embeddings.append(h)

        return torch.cat(level_embeddings, dim=1)  # [B, sum(level_dims)]

    def forward_per_level(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return per-level embeddings separately.

        Useful for per-level naming, auxiliary losses, or diagnostics.

        Args:
            x: Input sequences [B, T, D_in]

        Returns:
            List of [B, level_dim] tensors, one per pyramid level
        """
        pyramid_levels = self.pyramid(x)

        level_embeddings = []
        for level_input, head in zip(pyramid_levels, self.heads):
            h = level_input.transpose(1, 2)
            h = self._backbone(h)
            h = head(h)
            level_embeddings.append(h)

        return level_embeddings

    @property
    def output_dim(self) -> int:
        """Total output dimension (sum of all level dims)."""
        return self._output_dim
