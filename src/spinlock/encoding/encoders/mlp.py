"""MLP encoder for feature processing.

Used for SDF (trajectory) features and other pre-processed feature vectors.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseEncoder


class MLPEncoder(BaseEncoder):
    """Multi-layer perceptron encoder for vector features.

    Standard feedforward architecture with configurable hidden layers.
    Used for SDF features and other pre-computed feature vectors.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions (e.g., [256, 128])
        output_dim: Final output dimension (embedding size)
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: "relu")
        batch_norm: Whether to use batch normalization (default: True)

    Example:
        >>> encoder = MLPEncoder(
        ...     input_dim=221,  # SDF features
        ...     hidden_dims=[256, 128],
        ...     output_dim=64
        ... )
        >>> x = torch.randn(32, 221)  # [batch, features]
        >>> embeddings = encoder(x)   # [batch, 64]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_norm: bool = True
    ):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Default hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(act_fn())

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final projection to output_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        # Optional final batch norm
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode feature vector to embedding.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Embeddings [batch_size, output_dim]
        """
        return self.encoder(x)

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self._output_dim
