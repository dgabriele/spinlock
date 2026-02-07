"""Projection module to align NOA U-AFNO latents with VQ-VAE embedding space.

This module projects NOA's internal continuous representations (bottleneck features
from U-AFNO spectral mixing) to the VQ-VAE's discrete behavioral embedding space.

Architecture:
    Input: NOA bottleneck features [B, C_noa, H, W]
    1. AdaptiveAvgPool2d → [B, C_noa, 1, 1]
    2. Flatten → [B, C_noa]
    3. MLP: Linear(C_noa → hidden) + BatchNorm + ReLU + Dropout
    4. Linear(hidden → C_vq)
    Output: Projected latents [B, C_vq] matching VQ-VAE latent dimension

The projection learns to map NOA's physics-native features (spectral modes,
multi-scale decompositions) to VQ-VAE's behavioral embeddings (discrete patterns
discovered from feature statistics), providing a meaningful gradient signal for
aligning the two representation spaces.
"""

import torch
import torch.nn as nn


class LatentProjector(nn.Module):
    """Projects NOA U-AFNO bottleneck features to VQ-VAE latent space.

    Args:
        noa_latent_dim: Dimension of NOA bottleneck features (inferred at runtime)
        vq_latent_dim: Dimension of VQ-VAE pre-quantization latents (inferred at runtime)
        hidden_dim: Hidden layer dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        noa_latent_dim: int,
        vq_latent_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.noa_latent_dim = noa_latent_dim
        self.vq_latent_dim = vq_latent_dim
        self.hidden_dim = hidden_dim

        # Spatial pooling: [B, C, H, W] → [B, C, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # MLP projection
        self.projection = nn.Sequential(
            nn.Linear(noa_latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vq_latent_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bottleneck_features: torch.Tensor) -> torch.Tensor:
        """Project bottleneck features to VQ latent space.

        Args:
            bottleneck_features: [B, C, H, W] NOA bottleneck from U-AFNO

        Returns:
            Projected latents [B, vq_latent_dim]
        """
        # Pool spatial dimensions
        x = self.pool(bottleneck_features)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]

        # Project to VQ space
        x = self.projection(x)  # [B, vq_latent_dim]

        return x

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  noa_latent_dim={self.noa_latent_dim},\n"
            f"  vq_latent_dim={self.vq_latent_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_parameters={self.num_parameters:,}\n"
            f")"
        )
