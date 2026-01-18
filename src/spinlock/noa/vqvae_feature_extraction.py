"""Feature extraction for VQ-VAE tokenization in autonomous episodes.

This module provides feature extraction that matches the exact features used during
VQ-VAE training. The VQ-VAE's encoder is trained on per-timestep features
with 193 dimensions:
  - 24 spatial features (statistics, gradients, percentiles)
  - 27 spectral features (FFT power, frequency analysis)
  - 12 cross-channel features (correlation, mutual information, eigenvalues)
  - 130 enhanced temporal features (instantaneous dynamics, local temporal,
    local stability, phase space geometry, multi-scale temporal)

For autonomous episode execution, we extract these features from individual MNO
states at each timestep, matching the training distribution exactly.

Architecture:
    The VQ-VAE expects features extracted from the reference CNO distribution.
    During training, features were extracted from full trajectories using
    TemporalFeatureOrchestrator.extract_per_timestep(), which returns [N, T, 193] features.

    For autonomous episodes, we extract features from single states [B, C, H, W]
    by reshaping to [B, 1, 1, C, H, W] and calling extract_per_timestep(),
    which returns [B, 1, 193]. We then squeeze to get [B, 193].

Usage:
    >>> from spinlock.noa.vqvae_feature_extraction import VQVAEFeatureExtractor
    >>>
    >>> extractor = VQVAEFeatureExtractor(device='cuda')
    >>> state = torch.randn(1, 1, 64, 64, device='cuda')  # [B, C, H, W]
    >>> features = extractor.extract(state)  # [B, 193]
"""

import torch
from torch import Tensor
from typing import Optional

from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator
from spinlock.features.temporal.config import (
    TemporalFeatureConfig,
    SpatialConfig,
    SpectralConfig,
    CrossChannelConfig,
    TemporalConfig,
)


class VQVAEFeatureExtractor:
    """Extract features matching VQ-VAE training distribution.

    Extracts per-timestep features (spatial + spectral + cross-channel + enhanced temporal)
    that match the features used to train the VQ-VAE encoder.

    This ensures feature distribution compatibility between:
      - Training: Features from CNO reference rollouts
      - Inference: Features from autonomous MNO states

    Attributes:
        device: Computation device
        extractor: TemporalFeatureOrchestrator configured to match VQ-VAE training
        expected_dim: Expected feature dimension (set from VQ-VAE or 193 default)
    """

    def __init__(self, device: str = "cuda", expected_dim: Optional[int] = None):
        """Initialize VQ-VAE feature extractor.

        Args:
            device: Computation device (cuda/cpu)
            expected_dim: Expected feature dimension (if None, defaults to 193)
        """
        self.device = torch.device(device)
        self._expected_dim = expected_dim if expected_dim is not None else 193

        # Create config matching VQ-VAE training
        # Per-timestep features: spatial + spectral + cross_channel + temporal = 193D
        config = TemporalFeatureConfig(
            # Enable per-timestep features
            spatial=SpatialConfig(
                enabled=True,
                include_percentiles=True,  # Adds 5 percentile features
                include_histogram=False,
            ),
            spectral=SpectralConfig(
                enabled=True,
                num_fft_scales=5,  # Standard 5-scale FFT decomposition
            ),
            cross_channel=CrossChannelConfig(
                enabled=True,  # 12D cross-channel features
            ),
            # Enable enhanced temporal features (130D)
            temporal=TemporalConfig(
                enabled=True,
                window_size=5,
                short_window=5,
                medium_window=20,
                long_window=50,
            ),
        )

        # Initialize extractor
        self.extractor = TemporalFeatureOrchestrator(device=self.device, config=config)

    def extract(self, state: Tensor) -> Tensor:
        """Extract features from MNO state for VQ-VAE tokenization.

        Args:
            state: MNO state [B, C, H, W]

        Returns:
            Features [B, 193] (spatial + spectral + cross_channel + enhanced temporal)

        Raises:
            ValueError: If state has incorrect shape or extracted features
                       don't match expected dimension

        Example:
            >>> extractor = VQVAEFeatureExtractor(device='cuda')
            >>> state = torch.randn(2, 1, 64, 64, device='cuda')
            >>> features = extractor.extract(state)  # [2, 193]
        """
        if state.ndim != 4:
            raise ValueError(
                f"Expected state shape [B, C, H, W], got {state.shape}"
            )

        B, C, H, W = state.shape

        # Reshape to trajectory format: [N=B, M=1, T=1, C, H, W]
        trajectory = state.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C, H, W]

        # Extract per-timestep features: [B, T=1, D]
        with torch.no_grad():
            features = self.extractor.extract_per_timestep(trajectory)

        # Squeeze timestep dimension: [B, 1, D] -> [B, D]
        features = features.squeeze(1)

        # Validate dimension if expected_dim is set
        if self._expected_dim is not None and features.shape[-1] != self._expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self._expected_dim}, "
                f"got {features.shape[-1]}. This indicates a configuration error."
            )

        return features

    def get_feature_dim(self) -> Optional[int]:
        """Get expected feature dimension.

        Returns:
            Feature dimension if set, otherwise None
        """
        return self._expected_dim
