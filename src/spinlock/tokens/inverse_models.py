"""
Inverse decoders for VQTokenizer reconstruction.

This module provides neural network decoders that map from encoded feature spaces
back to the original continuous spaces:
  - ThetaInverseMLP: Encoded theta [B, 32] → actual parameters [B, 14]
  - InitialInverseCNN: Encoded initial [B, 426] → spatial grids [B, 3, 64, 64]

These inverse models enable the VQTokenizer to properly decode tokens back to
(theta, ICs) for use in CNOReplayer rollouts and sampling pipelines.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ThetaInverseMLP(nn.Module):
    """
    Inverse decoder: encoded theta features → actual operator parameters.

    Maps from the encoded theta space [B, encoded_dim] back to actual
    continuous parameters [B, param_dim] in [0,1] range.

    Architecture:
        Input [B, encoded_dim] → Linear(encoded_dim, 64) → LayerNorm → ReLU → Dropout
                                → Linear(64, param_dim) → Sigmoid
                                → Output [B, param_dim] in [0,1]

    Args:
        encoded_dim: Dimension of encoded theta features (default: 32)
        param_dim: Dimension of actual parameters (default: 14)
        hidden_dim: Hidden layer size (default: 64)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        encoded_dim: int = 32,
        param_dim: int = 14,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # Build MLP: encoded_dim → hidden_dim → param_dim
        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, param_dim),
            nn.Sigmoid(),  # Ensure output in [0,1]
        )

    def forward(self, theta_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded theta to actual parameters.

        Args:
            theta_encoded: [B, encoded_dim] encoded theta features

        Returns:
            [B, param_dim] actual parameters in [0,1]
        """
        return self.net(theta_encoded)

    def __repr__(self) -> str:
        return (
            f"ThetaInverseMLP(encoded_dim={self.encoded_dim}, "
            f"param_dim={self.param_dim}, hidden_dim={self.hidden_dim})"
        )


class InitialInverseCNN(nn.Module):
    """
    Inverse decoder: encoded initial features → spatial initial conditions.

    Maps from the encoded initial space [B, encoded_dim] back to spatial
    initial condition grids [B, channels, spatial_size, spatial_size].

    Architecture (reverse of InitialHybridEncoder's CNN):
        Input [B, encoded_dim] → Linear → Reshape [B, 256, 8, 8]
                               → ConvTranspose2d 8x8 → 16x16
                               → ConvTranspose2d 16x16 → 32x32
                               → ConvTranspose2d 32x32 → 64x64
                               → Output [B, channels, 64, 64]

    Args:
        encoded_dim: Dimension of encoded initial features (default: 426)
        channels: Number of output channels (default: 3)
        spatial_size: Spatial dimension of output grid (default: 64)
    """

    def __init__(
        self,
        encoded_dim: int = 426,
        channels: int = 3,
        spatial_size: int = 64,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.channels = channels
        self.spatial_size = spatial_size

        # Validate spatial_size is power of 2 and >= 32
        assert spatial_size in [32, 64, 128], f"spatial_size must be 32, 64, or 128, got {spatial_size}"

        # Compute starting spatial size (8x8 for 64x64 output)
        self.start_size = spatial_size // 8

        # Project encoded features to spatial starting point
        self.project = nn.Linear(encoded_dim, 256 * self.start_size * self.start_size)

        # Build upsampling decoder (reverse of encoder)
        layers = []

        # 8x8 → 16x16
        layers.extend([
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ])

        # 16x16 → 32x32
        layers.extend([
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ])

        # 32x32 → 64x64
        layers.extend([
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
        ])

        self.decoder = nn.Sequential(*layers)

    def forward(self, initial_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded initial features to spatial grids.

        Args:
            initial_encoded: [B, encoded_dim] encoded initial features

        Returns:
            [B, channels, spatial_size, spatial_size] spatial initial conditions
        """
        batch_size = initial_encoded.shape[0]

        # Project to spatial features
        x = self.project(initial_encoded)  # [B, 256*start_size*start_size]

        # Reshape to spatial tensor
        x = x.view(batch_size, 256, self.start_size, self.start_size)  # [B, 256, 8, 8]

        # Upsample to target resolution
        x = self.decoder(x)  # [B, channels, spatial_size, spatial_size]

        return x

    def __repr__(self) -> str:
        return (
            f"InitialInverseCNN(encoded_dim={self.encoded_dim}, "
            f"channels={self.channels}, spatial_size={self.spatial_size})"
        )
