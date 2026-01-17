"""Simple feature extraction for Phase 2 validation.

Provides lightweight feature extraction from MNO states for VQ-VAE tokenization
during autonomous episodes. This is a simplified version for Phase 2 validation.

For production use, the full NOAFeatureExtractor should be used.
"""

import torch
from torch import Tensor
from typing import Dict, Any
import torch.nn.functional as F


class SimpleFeatureExtractor:
    """Extract simple statistical features from MNO states.

    Computes basic spatial statistics that can be used for VQ-VAE tokenization
    during episode validation. This avoids the complexity of the full feature
    extraction pipeline for Phase 2 testing.

    Features extracted:
    - Spatial statistics (mean, std, min, max per channel)
    - Gradient statistics (spatial derivatives)
    - Spectral features (FFT-based)
    - Global pooling features

    Total dimension: ~100-200 features (configurable)
    """

    def __init__(
        self,
        spatial_stats: bool = True,
        gradient_stats: bool = True,
        spectral_stats: bool = True,
        pooling_stats: bool = True,
    ):
        """Initialize simple feature extractor.

        Args:
            spatial_stats: Extract spatial statistics
            gradient_stats: Extract gradient statistics
            spectral_stats: Extract spectral (FFT) statistics
            pooling_stats: Extract multi-scale pooling statistics
        """
        self.spatial_stats = spatial_stats
        self.gradient_stats = gradient_stats
        self.spectral_stats = spectral_stats
        self.pooling_stats = pooling_stats

    def __call__(self, state: Tensor) -> Tensor:
        """Extract features from MNO state.

        Args:
            state: MNO state [B, C, H, W]

        Returns:
            Feature vector [B, D] where D depends on enabled statistics
        """
        B, C, H, W = state.shape
        features = []

        # 1. Spatial statistics
        if self.spatial_stats:
            # Per-channel statistics
            for c in range(C):
                channel = state[:, c:c+1, :, :]  # [B, 1, H, W]
                features.extend([
                    channel.mean(dim=(-2, -1)),  # Mean
                    channel.std(dim=(-2, -1)),  # Std
                    channel.min(dim=-1)[0].min(dim=-1)[0],  # Min
                    channel.max(dim=-1)[0].max(dim=-1)[0],  # Max
                ])

        # 2. Gradient statistics (spatial derivatives)
        if self.gradient_stats:
            for c in range(C):
                channel = state[:, c, :, :]  # [B, H, W]

                # Horizontal gradient
                grad_x = torch.diff(channel, dim=-1)  # [B, H, W-1]
                features.extend([
                    grad_x.abs().mean(dim=(-2, -1)),
                    grad_x.abs().max(dim=-1)[0].max(dim=-1)[0],
                ])

                # Vertical gradient
                grad_y = torch.diff(channel, dim=-2)  # [B, H-1, W]
                features.extend([
                    grad_y.abs().mean(dim=(-2, -1)),
                    grad_y.abs().max(dim=-1)[0].max(dim=-1)[0],
                ])

        # 3. Spectral features (FFT-based)
        if self.spectral_stats:
            for c in range(C):
                channel = state[:, c, :, :]  # [B, H, W]

                # 2D FFT
                fft = torch.fft.rfft2(channel)  # [B, H, W//2+1]
                magnitude = torch.abs(fft)

                # Power spectrum statistics
                features.extend([
                    magnitude.mean(dim=(-2, -1)),
                    magnitude.std(dim=(-2, -1)),
                    magnitude.max(dim=-1)[0].max(dim=-1)[0],
                ])

                # Low-frequency power (0-25% of spectrum)
                H_freq, W_freq = magnitude.shape[-2:]
                low_freq = magnitude[:, :H_freq//4, :W_freq//4]
                features.append(low_freq.mean(dim=(-2, -1)))

                # High-frequency power (75-100% of spectrum)
                high_freq = magnitude[:, 3*H_freq//4:, 3*W_freq//4:]
                features.append(high_freq.mean(dim=(-2, -1)))

        # 4. Multi-scale pooling statistics
        if self.pooling_stats:
            for c in range(C):
                channel = state[:, c:c+1, :, :]  # [B, 1, H, W]

                # Different pooling scales
                for kernel_size in [2, 4, 8]:
                    if kernel_size <= min(H, W):
                        pooled = F.avg_pool2d(channel, kernel_size=kernel_size, stride=kernel_size)
                        features.extend([
                            pooled.mean(dim=(-2, -1)),
                            pooled.std(dim=(-2, -1)),
                        ])

        # Stack all features
        feature_tensor = torch.stack(features, dim=1)  # [B, D]

        return feature_tensor

    def get_feature_dim(self, num_channels: int = 2, H: int = 64, W: int = 64) -> int:
        """Get total feature dimensionality.

        Args:
            num_channels: Number of MNO state channels
            H, W: Spatial dimensions

        Returns:
            Total feature dimension
        """
        dim = 0

        if self.spatial_stats:
            dim += num_channels * 4  # mean, std, min, max per channel

        if self.gradient_stats:
            dim += num_channels * 4  # 2 gradients × 2 stats per channel

        if self.spectral_stats:
            dim += num_channels * 5  # 3 overall + 2 frequency bands per channel

        if self.pooling_stats:
            # Count valid pooling scales
            valid_scales = [k for k in [2, 4, 8] if k <= min(H, W)]
            dim += num_channels * len(valid_scales) * 2  # mean, std per scale

        return dim


def create_default_feature_extractor() -> SimpleFeatureExtractor:
    """Create default feature extractor for Phase 2 validation.

    Returns:
        Configured SimpleFeatureExtractor
    """
    return SimpleFeatureExtractor(
        spatial_stats=True,
        gradient_stats=True,
        spectral_stats=True,
        pooling_stats=True,
    )


def extract_features_for_vqvae(state: Tensor) -> Tensor:
    """Convenience function to extract features for VQ-VAE.

    Args:
        state: MNO state [B, C, H, W]

    Returns:
        Feature vector [B, D]
    """
    extractor = create_default_feature_extractor()
    return extractor(state)
