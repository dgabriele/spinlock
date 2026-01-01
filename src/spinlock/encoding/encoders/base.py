"""Base encoder class for VQ-VAE feature family processing."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for VQ-VAE encoders.

    Encoders transform raw features or data into embeddings suitable for
    VQ-VAE tokenization. Each feature family can use a different encoder.

    Subclasses must implement:
    - forward(x): Transform input to embedding
    - output_dim property: Output embedding dimension
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input to embedding.

        Args:
            x: Input tensor (family-specific shape)

        Returns:
            Embedding tensor [batch_size, output_dim]
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output embedding dimension."""
        pass
