"""Temporal mean pooling encoder for aggregating per-timestep features."""

import torch
import torch.nn as nn
from .base import BaseEncoder


class TemporalMeanEncoder(BaseEncoder):
    """Simple temporal aggregation via mean pooling across time dimension.

    Aggregates [N, T, D] → [N, D] by averaging over timesteps. Preserves all
    feature dimensions without compression, unlike TemporalCNNEncoder which
    learns a compressed representation.

    Use this when you want to preserve all raw temporal feature dimensions for
    clustering/categorization without learned compression that might create
    degenerate representations.

    Args:
        input_dim: Input feature dimension D (output_dim = input_dim)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate temporal features via mean pooling.

        Args:
            x: Input tensor [batch_size, timesteps, input_dim]

        Returns:
            Aggregated tensor [batch_size, input_dim]
        """
        if x.dim() != 3:
            raise ValueError(
                f"TemporalMeanEncoder expects 3D input [N, T, D], got shape {x.shape}"
            )

        # Mean over time dimension (dim=1)
        return x.mean(dim=1)

    @property
    def output_dim(self) -> int:
        """Output dimension (same as input - no compression)."""
        return self._input_dim
