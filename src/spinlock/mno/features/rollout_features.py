"""Rollout feature extraction for parameter sensitivity.

This module provides a reusable feature extractor that computes statistical
summaries from rollouts. These features are used by multiple loss components:
- Parameter reconstruction (predicting params from rollouts)
- Contrastive learning (embedding rollouts for similarity)
- Sensitivity regularization (measuring parameter influence)

The extractor computes 32-dimensional feature vectors covering:
- Temporal statistics (8D): mean, std, min, max, skewness, kurtosis, trend, energy
- Spatial statistics (8D): mean, std, min, max, gradient magnitude, laplacian, correlation
- Spectral features (16D): FFT-based frequency content in temporal/spatial domains

Design principles:
- DRY: Single implementation reused across all parameter sensitivity components
- Functional: Pure computation, no learnable parameters (statistical features)
- Efficient: Vectorized operations, minimal memory overhead
"""

import torch
import torch.nn as nn
import torch.fft as fft


class RolloutFeatureExtractor(nn.Module):
    """Extract statistical summary features from rollouts.

    This is a reusable component (DRY principle) used by:
    - ParameterReconstructor: predicts params from rollout features
    - ContrastiveSimilarity: embeds rollouts for contrastive learning
    - Sensitivity metrics: measures how parameter changes affect features

    Features are purely statistical (no learned parameters), making them
    interpretable and robust across different training stages.

    Attributes:
        feature_dim: Total feature dimensionality (32)
        temporal_dim: Temporal statistics dimensionality (8)
        spatial_dim: Spatial statistics dimensionality (8)
        spectral_dim: Spectral features dimensionality (16)
    """

    def __init__(self):
        """Initialize feature extractor (no learnable parameters)."""
        super().__init__()
        self.feature_dim = 32
        self.temporal_dim = 8
        self.spatial_dim = 8
        self.spectral_dim = 16

    def forward(self, rollout: torch.Tensor) -> torch.Tensor:
        """Extract 32D feature vector from rollout.

        Args:
            rollout: Tensor of shape [B, T, C, H, W] or [B, M, T, C, H, W]
                    B: batch size
                    M: number of realizations (optional)
                    T: time steps
                    C: channels
                    H, W: spatial dimensions

        Returns:
            features: Tensor of shape [B, 32] containing:
                - features[:, 0:8]: temporal statistics
                - features[:, 8:16]: spatial statistics
                - features[:, 16:32]: spectral features
        """
        # Handle both [B, T, C, H, W] and [B, M, T, C, H, W]
        if rollout.ndim == 6:
            # Average over realizations
            rollout = rollout.mean(dim=1)  # [B, T, C, H, W]

        B = rollout.shape[0]
        device = rollout.device

        # Extract temporal, spatial, and spectral features
        temporal_feats = self._extract_temporal_features(rollout)  # [B, 8]
        spatial_feats = self._extract_spatial_features(rollout)    # [B, 8]
        spectral_feats = self._extract_spectral_features(rollout)  # [B, 16]

        # Concatenate all features
        features = torch.cat([temporal_feats, spatial_feats, spectral_feats], dim=1)  # [B, 32]

        return features

    def _extract_temporal_features(self, rollout: torch.Tensor) -> torch.Tensor:
        """Extract temporal evolution statistics.

        Computes statistics over time dimension to capture dynamics:
        - Mean/std/min/max: Overall temporal variation
        - Skewness/kurtosis: Shape of temporal distribution
        - Trend: Linear trend coefficient
        - Energy: Temporal energy (sum of squares)

        Args:
            rollout: [B, T, C, H, W]

        Returns:
            temporal_features: [B, 8]
        """
        B, T = rollout.shape[0], rollout.shape[1]

        # Flatten spatial dimensions: [B, T, C*H*W]
        rollout_flat = rollout.flatten(start_dim=2)  # [B, T, C*H*W]

        # Aggregate across spatial dimensions: [B, T]
        temporal_series = rollout_flat.mean(dim=2)

        # Basic statistics
        mean = temporal_series.mean(dim=1)  # [B]
        std = temporal_series.std(dim=1)    # [B]
        min_val = temporal_series.min(dim=1)[0]  # [B]
        max_val = temporal_series.max(dim=1)[0]  # [B]

        # Higher-order moments
        centered = temporal_series - mean.unsqueeze(1)
        skewness = (centered ** 3).mean(dim=1) / (std ** 3 + 1e-8)  # [B]
        kurtosis = (centered ** 4).mean(dim=1) / (std ** 4 + 1e-8)  # [B]

        # Linear trend (correlation with time)
        # Compute correlation between temporal series and linear time for each batch element
        time = torch.linspace(0, 1, T, device=rollout.device)  # [T]
        trend_list = []
        for b in range(B):
            # Stack [2, T] and compute correlation
            stacked = torch.stack([temporal_series[b], time])  # [2, T]
            corr_matrix = torch.corrcoef(stacked)  # [2, 2]
            trend_list.append(corr_matrix[0, 1])  # Correlation between series and time
        trend = torch.stack(trend_list)  # [B]

        # Temporal energy
        energy = (temporal_series ** 2).mean(dim=1)  # [B]

        return torch.stack([mean, std, min_val, max_val, skewness, kurtosis, trend, energy], dim=1)  # [B, 8]

    def _extract_spatial_features(self, rollout: torch.Tensor) -> torch.Tensor:
        """Extract spatial structure statistics.

        Computes statistics over spatial dimensions to capture patterns:
        - Mean/std/min/max: Spatial field statistics
        - Gradient magnitude: Spatial variation strength
        - Laplacian: Curvature/smoothness
        - Spatial autocorrelation: Structured patterns

        Args:
            rollout: [B, T, C, H, W]

        Returns:
            spatial_features: [B, 8]
        """
        B = rollout.shape[0]

        # Average over time: [B, C, H, W]
        spatial_field = rollout.mean(dim=1)

        # Basic statistics
        mean = spatial_field.mean(dim=[1, 2, 3])  # [B]
        std = spatial_field.std(dim=[1, 2, 3])    # [B]
        min_val = spatial_field.flatten(start_dim=1).min(dim=1)[0]  # [B]
        max_val = spatial_field.flatten(start_dim=1).max(dim=1)[0]  # [B]

        # Gradient magnitude (spatial variation)
        dx = spatial_field[:, :, :, 1:] - spatial_field[:, :, :, :-1]  # [B, C, H, W-1]
        dy = spatial_field[:, :, 1:, :] - spatial_field[:, :, :-1, :]  # [B, C, H-1, W]
        # Align to common [H-1, W-1] region
        dx_aligned = dx[:, :, :-1, :]  # [B, C, H-1, W-1]
        dy_aligned = dy[:, :, :, :-1]  # [B, C, H-1, W-1]
        grad_mag = (dx_aligned ** 2 + dy_aligned ** 2).sqrt().mean(dim=[1, 2, 3])  # [B]

        # Laplacian (curvature/smoothness)
        laplacian = (
            spatial_field[:, :, 2:, 1:-1] + spatial_field[:, :, :-2, 1:-1] +
            spatial_field[:, :, 1:-1, 2:] + spatial_field[:, :, 1:-1, :-2] -
            4 * spatial_field[:, :, 1:-1, 1:-1]
        ).abs().mean(dim=[1, 2, 3])  # [B]

        # Spatial autocorrelation (structured patterns)
        # Compute correlation between field and shifted version
        shifted = torch.roll(spatial_field, shifts=1, dims=-1)
        correlation = (spatial_field * shifted).mean(dim=[1, 2, 3])  # [B]

        # Spatial energy
        energy = (spatial_field ** 2).mean(dim=[1, 2, 3])  # [B]

        return torch.stack([mean, std, min_val, max_val, grad_mag, laplacian, correlation, energy], dim=1)  # [B, 8]

    def _extract_spectral_features(self, rollout: torch.Tensor) -> torch.Tensor:
        """Extract frequency domain features.

        Computes FFT-based features to capture periodic patterns:
        - Temporal spectrum (8D): Frequency content of temporal evolution
        - Spatial spectrum (8D): Frequency content of spatial patterns

        Args:
            rollout: [B, T, C, H, W]

        Returns:
            spectral_features: [B, 16]
        """
        B, T = rollout.shape[0], rollout.shape[1]

        # Temporal FFT: Analyze frequency content of time series
        # Average over spatial dimensions first
        temporal_series = rollout.mean(dim=[2, 3, 4])  # [B, T]
        temporal_fft = fft.rfft(temporal_series, dim=1)  # [B, T//2+1]
        temporal_power = temporal_fft.abs() ** 2  # [B, T//2+1]

        # Extract temporal spectral features (8 bins)
        temporal_bins = self._bin_spectrum(temporal_power, num_bins=8)  # [B, 8]

        # Spatial FFT: Analyze frequency content of spatial patterns
        # Average over time first
        spatial_field = rollout.mean(dim=1)  # [B, C, H, W]
        spatial_fft = fft.rfft2(spatial_field, dim=(-2, -1))  # [B, C, H, W//2+1]
        spatial_power = spatial_fft.abs() ** 2  # [B, C, H, W//2+1]
        spatial_power_flat = spatial_power.mean(dim=1)  # [B, H, W//2+1]

        # Extract spatial spectral features (8 bins)
        spatial_bins = self._bin_spectrum(spatial_power_flat.flatten(start_dim=1), num_bins=8)  # [B, 8]

        return torch.cat([temporal_bins, spatial_bins], dim=1)  # [B, 16]

    def _bin_spectrum(self, power_spectrum: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Bin power spectrum into frequency bands.

        Args:
            power_spectrum: [B, F] where F is number of frequency components
            num_bins: Number of bins to create

        Returns:
            binned: [B, num_bins] with average power in each frequency band
        """
        B, F = power_spectrum.shape
        bin_size = F // num_bins

        # Pad if necessary to make divisible
        if F % num_bins != 0:
            pad_size = num_bins - (F % num_bins)
            power_spectrum = torch.nn.functional.pad(power_spectrum, (0, pad_size))
            F = power_spectrum.shape[1]
            bin_size = F // num_bins

        # Reshape and average within bins
        binned = power_spectrum.view(B, num_bins, bin_size).mean(dim=2)  # [B, num_bins]

        return binned
