"""Identity encoder (pass-through, no transformation)."""

import torch
import torch.nn as nn
from .base import BaseEncoder


class IdentityEncoder(BaseEncoder):
    """Identity encoder that passes input through without transformation.

    Useful when features are already in the correct format for VQ-VAE.

    Args:
        input_dim: Input feature dimension (equals output_dim)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through unchanged.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Same tensor [batch_size, input_dim]
        """
        return x

    @property
    def output_dim(self) -> int:
        """Output dimension (same as input)."""
        return self._input_dim
