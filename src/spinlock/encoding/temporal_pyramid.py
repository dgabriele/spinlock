"""Multi-resolution temporal downsampling pyramid.

Produces multiple temporal resolutions from a single input sequence
using adaptive average pooling along the time axis.
"""

import torch
import torch.nn as nn
from typing import List


class TemporalPyramid(nn.Module):
    """Multi-resolution temporal downsampling.

    Takes [B, T, D] and produces a list of [B, T_i, D] tensors at
    decreasing temporal resolutions via nn.AdaptiveAvgPool1d.

    Args:
        downsample_factors: List of temporal downsampling factors.
            Factor 1 = full resolution, 2 = half, 4 = quarter, etc.
            Default: [1, 2, 4, 8]

    Example:
        >>> pyramid = TemporalPyramid([1, 2, 4, 8])
        >>> x = torch.randn(32, 256, 64)  # [B, T, D]
        >>> levels = pyramid(x)
        >>> [l.shape[1] for l in levels]  # [256, 128, 64, 32]
    """

    def __init__(self, downsample_factors: List[int] = [1, 2, 4, 8]):
        super().__init__()
        if not downsample_factors:
            raise ValueError("downsample_factors must be non-empty")
        for f in downsample_factors:
            if f < 1:
                raise ValueError(f"Downsample factor must be >= 1, got {f}")
        self.downsample_factors = downsample_factors

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Produce multi-resolution temporal pyramid.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            List of [B, T_i, D] tensors, one per downsample factor.
            T_i = max(1, T // factor) for each factor.
        """
        B, T, D = x.shape
        # Transpose to [B, D, T] for pooling along temporal axis
        x_t = x.transpose(1, 2)  # [B, D, T]

        levels = []
        for factor in self.downsample_factors:
            target_len = max(1, T // factor)
            if target_len == T:
                # No downsampling needed, use original
                levels.append(x)
            else:
                pool = nn.AdaptiveAvgPool1d(target_len)
                pooled = pool(x_t)  # [B, D, target_len]
                levels.append(pooled.transpose(1, 2))  # [B, target_len, D]

        return levels
