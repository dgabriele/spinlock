"""
Initial Conditions Feature Extractor.

Extracts statistical, structural, and energy features from initial conditions
designed to preserve natural variance and provide reconstruction-relevant information.

Design principles:
- Preserve natural variance (no per-feature normalization/clamping)
- Focus on statistical/structural info, not pattern complexity
- Output reconstruction-relevant features for VQ-VAE decoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class InitialConditionsFeatureExtractor(nn.Module):
    """
    Extract statistical and structural features from initial conditions.

    This extractor focuses on reconstruction-relevant features rather than
    pattern complexity metrics. It preserves natural variance without
    per-feature normalization.

    For C=3 channels, outputs 66 features:
    - Distributional: 33 features (C*11: mean, std, min, max, median, 4 percentiles, skew, kurt)
    - Spatial: 24 features (C*8: gradients, Laplacian, center of mass)
    - Energy: 9 features (C*2 norms + C*(C-1)/2 cross-correlations = 6 + 3)
    """

    def __init__(self, device: Optional[str] = None, include_spatial: bool = False):
        """
        Initialize the feature extractor.

        Args:
            device: Device to run computations on ('cuda' or 'cpu')
            include_spatial: Include spatial features (gradients, Laplacian, CoM).
                           Default False because these have collapsed variance on small grids.
        """
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.include_spatial = include_spatial

    def _skewness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute skewness along last dimension.

        Args:
            x: [B, C, N] tensor

        Returns:
            [B, C] skewness values
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        z = (x - mean) / (std + 1e-8)
        return (z ** 3).mean(dim=-1)

    def _kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute kurtosis along last dimension.

        Args:
            x: [B, C, N] tensor

        Returns:
            [B, C] kurtosis values (excess kurtosis, 0 for normal distribution)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        z = (x - mean) / (std + 1e-8)
        return (z ** 4).mean(dim=-1) - 3.0

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute discrete Laplacian (curvature measure).

        Args:
            x: [B, C, H, W] tensor

        Returns:
            [B, C, H-2, W-2] Laplacian
        """
        # Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        laplacian = (
            x[:, :, 1:-1, 2:]      # right
            + x[:, :, 1:-1, :-2]   # left
            + x[:, :, 2:, 1:-1]    # down
            + x[:, :, :-2, 1:-1]   # up
            - 4 * x[:, :, 1:-1, 1:-1]  # center
        )
        return laplacian

    def _center_of_mass(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute center of mass for each channel.

        Args:
            x: [B, C, H, W] tensor

        Returns:
            com_x: [B, C] x-coordinate of center of mass
            com_y: [B, C] y-coordinate of center of mass
        """
        B, C, H, W = x.shape

        # Create coordinate grids
        y_coords = torch.arange(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        x_coords = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)

        # Use absolute values as weights (mass)
        weights = torch.abs(x)
        total_mass = weights.sum(dim=(-2, -1), keepdim=True) + 1e-8

        # Weighted average of coordinates
        com_y = (weights * y_coords).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)
        com_x = (weights * x_coords).sum(dim=(-2, -1)) / total_mass.squeeze(-1).squeeze(-1)

        return com_x, com_y

    def _correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation coefficient between two tensors.

        Args:
            x: [B, N] tensor
            y: [B, N] tensor

        Returns:
            [B] correlation coefficients
        """
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)

        numerator = (x_centered * y_centered).sum(dim=-1)
        denominator = (
            torch.sqrt((x_centered ** 2).sum(dim=-1)) *
            torch.sqrt((y_centered ** 2).sum(dim=-1))
        )

        return numerator / (denominator + 1e-8)

    def extract_distributional(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Extract per-channel distribution statistics.

        Captures information about IC magnitude, scale, and distribution shape
        without normalization, preserving natural variance.

        Args:
            ic: [B, C, H, W] initial condition

        Returns:
            [B, C*11] features containing:
            - mean, std, min, max, median (5 features)
            - 5th, 25th, 75th, 95th percentiles (4 features)
            - skewness, kurtosis (2 features)
        """
        B, C, H, W = ic.shape
        ic_flat = ic.view(B, C, -1)  # [B, C, H*W]

        features = []

        # Basic statistics (NO normalization - preserve natural variance!)
        features.append(ic_flat.mean(dim=-1))      # [B, C]
        features.append(ic_flat.std(dim=-1))       # [B, C]
        features.append(ic_flat.min(dim=-1).values)  # [B, C]
        features.append(ic_flat.max(dim=-1).values)  # [B, C]
        features.append(ic_flat.median(dim=-1).values)  # [B, C]

        # Percentiles
        for pct in [0.05, 0.25, 0.75, 0.95]:
            features.append(torch.quantile(ic_flat, pct, dim=-1))  # [B, C]

        # Higher moments (preserve scale!)
        features.append(self._skewness(ic_flat))   # [B, C]
        features.append(self._kurtosis(ic_flat))   # [B, C]

        # Stack and flatten: [B, C*10]
        return torch.stack(features, dim=-1).reshape(B, -1)

    def extract_spatial(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial structure features.

        Captures gradient information, curvature, and spatial distribution
        that help the decoder understand boundary conditions and spatial structure.

        Args:
            ic: [B, C, H, W] initial condition

        Returns:
            [B, C*8] features containing:
            - x-gradient mean/std, y-gradient mean/std
            - Laplacian mean/std
            - Center of mass (normalized x, y coordinates)
        """
        B, C, H, W = ic.shape

        features = []

        # Spatial gradients
        grad_x = ic[:, :, :, 1:] - ic[:, :, :, :-1]  # [B, C, H, W-1]
        grad_y = ic[:, :, 1:, :] - ic[:, :, :-1, :]  # [B, C, H-1, W]

        features.append(grad_x.mean(dim=(-2, -1)))  # [B, C]
        features.append(grad_x.std(dim=(-2, -1)))   # [B, C]
        features.append(grad_y.mean(dim=(-2, -1)))  # [B, C]
        features.append(grad_y.std(dim=(-2, -1)))   # [B, C]

        # Laplacian (curvature)
        laplacian = self._laplacian(ic)  # [B, C, H-2, W-2]
        features.append(laplacian.mean(dim=(-2, -1)))  # [B, C]
        features.append(laplacian.std(dim=(-2, -1)))   # [B, C]

        # Center of mass (normalized to [0,1])
        com_x, com_y = self._center_of_mass(ic)  # [B, C], [B, C]
        features.append(com_x / W)  # [B, C]
        features.append(com_y / H)  # [B, C]

        # Stack and flatten: [B, C*8]
        return torch.stack(features, dim=-1).reshape(B, -1)

    def extract_energy(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Extract energy and conservation features.

        Captures physics-relevant quantities like norms and cross-channel
        correlations that inform the decoder about energy scales and
        channel interactions.

        Args:
            ic: [B, C, H, W] initial condition

        Returns:
            [B, C*2+3] features (for C=3) containing:
            - L2 norm, L1 norm per channel
            - Cross-channel correlations (3 for C=3 channels)
        """
        B, C, H, W = ic.shape
        ic_flat = ic.view(B, C, -1)  # [B, C, H*W]

        features = []

        # Norms per channel
        features.append(torch.norm(ic_flat, p=2, dim=-1))  # L2 norm [B, C]
        features.append(torch.norm(ic_flat, p=1, dim=-1))  # L1 norm [B, C]

        # Stack per-channel features: [B, C*2]
        per_channel_feats = torch.stack(features, dim=-1).reshape(B, -1)

        # Cross-channel correlations
        cross_channel_feats = []
        if C >= 2:
            # Compute correlations for all channel pairs
            for i in range(C):
                for j in range(i + 1, C):
                    corr = self._correlation(ic_flat[:, i], ic_flat[:, j])  # [B]
                    cross_channel_feats.append(corr.unsqueeze(-1))

        if cross_channel_feats:
            cross_channel_feats = torch.cat(cross_channel_feats, dim=-1)  # [B, C*(C-1)/2]
            return torch.cat([per_channel_feats, cross_channel_feats], dim=-1)
        else:
            return per_channel_feats

    def extract_all(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Extract all reconstruction-relevant features.

        Combines distributional, spatial (optional), and energy features without
        per-feature normalization to preserve natural variance.

        Args:
            ic: [B, C, H, W] initial condition

        Returns:
            [B, D] features where:
            - For C=3 with spatial: D = 33 + 24 + 9 = 66
            - For C=3 without spatial: D = 33 + 9 = 42
            - Distributional: C*11
            - Spatial (optional): C*8
            - Energy: C*2 + C*(C-1)/2
        """
        distributional = self.extract_distributional(ic)  # [B, C*11]
        energy = self.extract_energy(ic)                  # [B, C*2 + C*(C-1)/2]

        if self.include_spatial:
            spatial = self.extract_spatial(ic)            # [B, C*8]
            return torch.cat([distributional, spatial, energy], dim=-1)
        else:
            return torch.cat([distributional, energy], dim=-1)

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract all features.

        Args:
            ic: [B, C, H, W] or [B, M, C, H, W] initial condition

        Returns:
            [B, D] or [B, M, D] feature vector
        """
        # Handle both [B, C, H, W] and [B, M, C, H, W] inputs
        original_shape = ic.shape
        if len(original_shape) == 5:
            # [B, M, C, H, W] - batch with realizations
            B, M, C, H, W = original_shape
            ic_reshaped = ic.view(B * M, C, H, W)
            features = self.extract_all(ic_reshaped)  # [B*M, D]
            return features.view(B, M, -1)  # [B, M, D]
        else:
            # [B, C, H, W] - simple batch
            return self.extract_all(ic)
