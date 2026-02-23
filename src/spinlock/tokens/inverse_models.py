"""
Inverse decoders for VQTokenizer reconstruction.

This module provides neural network decoders that map from encoded feature spaces
back to the original continuous spaces:
  - ThetaInverseMLP: Encoded theta [B, 32] → actual parameters [B, 14]
  - InitialInverseCNN: Encoded initial [B, D] → spatial grids [B, C, 64, 64]
  - TemporalInverseMLP: Encoded temporal [B, 1920] → synthetic CNN features [B, T_rt, 240]

These inverse models enable the VQTokenizer to properly decode tokens back to
(theta, ICs, temporal features) for roundtrip symbolic self-consistency:
tokens → decode → re-encode → same tokens.
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


class _ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN residual block (pre-activation style)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class InitialInverseCNN(nn.Module):
    """
    Inverse decoder: encoded initial features → spatial initial conditions.

    Maps from the encoded initial space [B, encoded_dim] back to spatial
    initial condition grids [B, channels, spatial_size, spatial_size].

    Architecture:
        Input [B, encoded_dim] → Linear → Reshape [B, 128, 16, 16]
                               → ResBlock(128)
                               → Upsample(2×) + Conv 128→64 → ResBlock(64)
                               → Upsample(2×) + Conv 64→channels
                               → Sigmoid
                               → Output [B, channels, 64, 64]

    Improvements over the original ConvTranspose2d decoder:
      - Bilinear upsample + conv eliminates checkerboard artifacts
      - Residual blocks add refinement capacity at each resolution
      - 16×16 starting resolution reduces upsampling burden (2 stages vs 3)
      - Sigmoid output matches [0,1] target range (Lenia ICs/states)

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

        # Start from 16×16 (spatial_size // 4) instead of 8×8
        self.start_size = spatial_size // 4
        self.start_channels = 128

        # Project encoded features to spatial starting point via bottleneck
        spatial_dim = self.start_channels * self.start_size * self.start_size
        bottleneck = min(512, encoded_dim)
        self.project = nn.Sequential(
            nn.Linear(encoded_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, spatial_dim),
            nn.ReLU(inplace=True),
        )

        # 16×16: residual refinement at starting resolution
        self.res0 = _ResBlock(128)

        # 16×16 → 32×32: bilinear upsample + conv + residual
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = _ResBlock(64)

        # 32×32 → 64×64: bilinear upsample + conv to output channels
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, initial_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded initial features to spatial grids.

        Args:
            initial_encoded: [B, encoded_dim] encoded initial features

        Returns:
            [B, channels, spatial_size, spatial_size] spatial initial conditions
        """
        B = initial_encoded.shape[0]

        # Project to spatial features [B, 128, 16, 16]
        x = self.project(initial_encoded)
        x = x.view(B, self.start_channels, self.start_size, self.start_size)

        # Refine at 16×16
        x = self.res0(x)

        # 16×16 → 32×32
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1_conv(x)
        x = self.res1(x)

        # 32×32 → 64×64
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2_conv(x)

        # Sigmoid: ICs are in [0, 1]
        return torch.sigmoid(x)

    def __repr__(self) -> str:
        return (
            f"InitialInverseCNN(encoded_dim={self.encoded_dim}, "
            f"channels={self.channels}, spatial_size={self.spatial_size})"
        )


class TemporalInverseMLP(nn.Module):
    """
    Inverse decoder: encoded temporal features → synthetic CNN feature space.

    Maps from the concatenated pyramid-encoded temporal features [B, encoded_dim]
    to a synthetic trajectory in CNN feature space [B, T_rt, cnn_dim]. The output
    is re-encoded through the REAL PyramidTemporalEncoders during roundtrip loss,
    testing the full decode → re-encode → quantize cycle for symbolic self-consistency.

    The bottleneck hidden layer forces cross-group information sharing: all 30
    temporal groups must jointly reconstruct a coherent CNN feature trajectory.

    Architecture:
        Input [B, encoded_dim] → Linear → LayerNorm → ReLU → Dropout
                                → Linear (bottleneck) → LayerNorm → ReLU → Dropout
                                → Linear → reshape → Output [B, T_rt, cnn_dim]

    Args:
        encoded_dim: Total encoded temporal dimension (e.g. 1920 = 30 groups × 64)
        cnn_dim: CNN output dimension per frame (e.g. 240 = 30 groups × 8)
        roundtrip_timesteps: Number of synthetic timesteps to generate (default: 8)
        hidden_dim: Hidden layer size (default: 512)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        encoded_dim: int,
        cnn_dim: int,
        roundtrip_timesteps: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoded_dim = encoded_dim
        self.cnn_dim = cnn_dim
        self.roundtrip_timesteps = roundtrip_timesteps
        self.hidden_dim = hidden_dim

        output_dim = roundtrip_timesteps * cnn_dim

        self.net = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, temporal_encoded: torch.Tensor) -> torch.Tensor:
        """
        Decode encoded temporal features to synthetic CNN feature trajectory.

        Args:
            temporal_encoded: [B, encoded_dim] concatenated pyramid-encoded temporal

        Returns:
            [B, T_rt, cnn_dim] synthetic CNN features for roundtrip re-encoding
        """
        B = temporal_encoded.shape[0]
        flat = self.net(temporal_encoded)
        return flat.view(B, self.roundtrip_timesteps, self.cnn_dim)

    def __repr__(self) -> str:
        return (
            f"TemporalInverseMLP(encoded_dim={self.encoded_dim}, "
            f"cnn_dim={self.cnn_dim}, T_rt={self.roundtrip_timesteps}, "
            f"hidden_dim={self.hidden_dim})"
        )
