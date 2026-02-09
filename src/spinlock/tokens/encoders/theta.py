"""
Encoder for operator parameters (theta).

This module provides neural network encoders that map continuous operator
parameters to dense embedding spaces suitable for vector quantization.
"""

import torch
import torch.nn as nn
from typing import Optional


class ThetaMLPEncoder(nn.Module):
    """
    MLP encoder for operator parameters (theta).

    Encodes 14D continuous parameters in [0,1] to dense embedding space
    for vector quantization.

    Architecture:
        Input [B, param_dim] → Linear(param_dim, hidden_dim) → LayerNorm → ReLU → Dropout
                              → Linear(hidden_dim, output_dim) → LayerNorm
                              → Output [B, output_dim]

    Args:
        param_dim: Dimensionality of input parameters (default: 14)
        hidden_dim: Hidden layer size (default: 64)
        output_dim: Output embedding size (default: 32)
        dropout: Dropout probability (default: 0.1)
        use_layer_norm: Whether to apply LayerNorm (default: True)
    """

    def __init__(
        self,
        param_dim: int = 14,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layer 1: param_dim → hidden_dim
        self.layer1 = nn.Linear(param_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Layer 2: hidden_dim → output_dim
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.norm2 = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Encode parameters to embedding space.

        Args:
            theta: [B, param_dim] parameter vectors in [0,1]

        Returns:
            [B, output_dim] encoded parameters
        """
        # Layer 1
        x = self.layer1(theta)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Layer 2
        x = self.layer2(x)
        x = self.norm2(x)

        return x

    def __repr__(self) -> str:
        return (
            f"ThetaMLPEncoder(param_dim={self.param_dim}, "
            f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim})"
        )
