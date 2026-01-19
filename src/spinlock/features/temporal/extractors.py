"""Temporal Feature Orchestrator (v3.0).

Per-timestep-only feature extraction for online perturbation-based NOA.
Extracts 193D features per timestep from trajectory data.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
import numpy as np

from spinlock.features.base import FeatureExtractorBase
from spinlock.features.temporal.spatial import SpatialFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.temporal.temporal import TemporalFeatureExtractor


class TemporalFeatureOrchestrator(FeatureExtractorBase):
    """Per-timestep-only feature orchestrator (v3.0).

    Extracts 193D features per timestep:
    - Spatial: 24D (per-channel statistics)
    - Spectral: 27D (frequency domain features)
    - Cross-channel: 12D (channel interactions)
    - Enhanced temporal: 130D (windowed dynamics)

    This replaces the v2.x SummaryExtractor by focusing exclusively on
    per-timestep features that are computable online.

    Args:
        device: Torch device for computation
        config: TemporalFeatureConfig (optional)

    Example:
        >>> orchestrator = TemporalFeatureOrchestrator(device=torch.device('cuda'))
        >>> trajectories = torch.randn(100, 10, 256, 3, 128, 128)  # [N, M, T, C, H, W]
        >>> features = orchestrator.extract_per_timestep(trajectories)  # [100, 256, 193]
    """

    def __init__(self, device: torch.device, config: Optional[Any] = None):
        """Initialize temporal feature orchestrator.

        Args:
            device: Torch device
            config: TemporalFeatureConfig (optional)
        """
        self.device = device
        self.config = config

        # Initialize sub-extractors
        self.spatial_extractor = SpatialFeatureExtractor(device=device)
        self.spectral_extractor = SpectralFeatureExtractor(device=device)
        self.cross_channel_extractor = CrossChannelFeatureExtractor(device=device)

        # Enhanced temporal extractor with configurable windows
        if config is not None and hasattr(config, 'temporal'):
            self.temporal_extractor = TemporalFeatureExtractor(
                device=device,
                window_size=getattr(config.temporal, 'window_size', 5),
                short_window=getattr(config.temporal, 'short_window', 5),
                medium_window=getattr(config.temporal, 'medium_window', 20),
                long_window=getattr(config.temporal, 'long_window', 50),
            )
        else:
            self.temporal_extractor = TemporalFeatureExtractor(
                device=device,
                window_size=5,
                short_window=5,
                medium_window=20,
                long_window=50,
            )

    def extract_per_timestep(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep features (v3.0 ONLY method).

        Args:
            trajectories: [N, M, T, C, H, W] trajectory tensor
                N: number of operators
                M: number of realizations per operator
                T: number of timesteps
                C: number of channels
                H, W: spatial dimensions

        Returns:
            features: [N, T, 193] per-timestep features

        Note:
            The realization dimension M is averaged out before extraction,
            as we compute features on the mean field across realizations.
        """
        N, M, T, C, H, W = trajectories.shape

        # Average across realizations: [N, M, T, C, H, W] → [N, T, C, H, W]
        fields = trajectories.mean(dim=1)

        # Extract spatial features (24D)
        # The extractors return dicts with features of shape [N, T, C] or [N, T]
        spatial_result = self.spatial_extractor.extract(fields)
        if isinstance(spatial_result, dict):
            # Collect all features and flatten to [N, T, D]
            spatial_features = []
            for key in sorted(spatial_result.keys()):
                feat = spatial_result[key]  # [N, T, C] or [N, T]
                if feat.dim() == 3:
                    # [N, T, C] → keep as is, will be flattened to [N, T, C]
                    spatial_features.append(feat)
                elif feat.dim() == 2:
                    # [N, T] → [N, T, 1]
                    spatial_features.append(feat.unsqueeze(-1))
            # Concatenate: [N, T, D_spatial]
            spatial = torch.cat(spatial_features, dim=-1) if spatial_features else torch.zeros(N, T, 24, device=self.device)
        else:
            spatial = spatial_result
        # Flatten last dimension: [N, T, D_spatial]
        if spatial.dim() > 2:
            spatial = spatial.flatten(start_dim=2)

        # Extract spectral features (27D)
        spectral_result = self.spectral_extractor.extract(fields)
        if isinstance(spectral_result, dict):
            spectral_features = []
            for key in sorted(spectral_result.keys()):
                feat = spectral_result[key]  # [N, T, C] or [N, T]
                if feat.dim() == 3:
                    spectral_features.append(feat)
                elif feat.dim() == 2:
                    spectral_features.append(feat.unsqueeze(-1))
            spectral = torch.cat(spectral_features, dim=-1) if spectral_features else torch.zeros(N, T, 27, device=self.device)
        else:
            spectral = spectral_result
        if spectral.dim() > 2:
            spectral = spectral.flatten(start_dim=2)

        # Extract cross-channel features (12D)
        cross_result = self.cross_channel_extractor.extract(fields)
        if isinstance(cross_result, dict):
            cross_features = []
            for key in sorted(cross_result.keys()):
                feat = cross_result[key]  # [N, T, C] or [N, T]
                if feat.dim() == 3:
                    cross_features.append(feat)
                elif feat.dim() == 2:
                    cross_features.append(feat.unsqueeze(-1))
            cross = torch.cat(cross_features, dim=-1) if cross_features else torch.zeros(N, T, 12, device=self.device)
        else:
            cross = cross_result
        if cross.dim() > 2:
            cross = cross.flatten(start_dim=2)

        # Extract enhanced temporal features (130D)
        # Process timestep by timestep to maintain history
        temporal_features = []
        self.temporal_extractor.reset()  # Reset history buffers

        for t in range(T):
            # Get all samples at timestep t: [N, C, H, W]
            u_t = fields[:, t, :, :, :]

            # Extract temporal features: [N, 130]
            temporal_t = self.temporal_extractor.extract(u_t)

            temporal_features.append(temporal_t)

        # Stack: [T, N, 130] → [N, T, 130]
        temporal = torch.stack(temporal_features, dim=0).transpose(0, 1)

        # Now all features are [N, T, D_i], concatenate along last dimension
        features = torch.cat([spatial, spectral, cross, temporal], dim=-1)

        return features

    def extract_all(
        self,
        batch_outputs: torch.Tensor,
        operators: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """Legacy compatibility adapter for pipeline code.

        In v3.0, only per-timestep features exist. This method adapts the old
        interface to the new v3.0 architecture by returning a dict.

        Args:
            batch_outputs: [N, M, T, C, H, W] trajectories
            operators: Optional operator metadata (unused in v3.0)

        Returns:
            Dictionary with keys:
                - per_timestep: [N, T, ~328] features
                - per_trajectory: [N, M, 0] empty (removed in v3.0)
                - aggregated_mean: [N, 0] empty (removed in v3.0)
        """
        per_timestep = self.extract_per_timestep(batch_outputs)
        N = batch_outputs.shape[0]
        M = batch_outputs.shape[1]

        # Return empty tensors for removed features (for backward compatibility)
        return {
            'per_timestep': per_timestep,
            'per_trajectory': torch.zeros(N, M, 0, device=self.device),
            'aggregated_mean': torch.zeros(N, 0, device=self.device),
        }

    def get_actual_per_timestep_dim(self) -> int:
        """Get actual per-timestep feature dimension by runtime extraction.

        Returns:
            Actual feature dimension (auto-detected)
        """
        # Create a small dummy batch to measure actual dimensions
        # Shape: [N=1, M=2, T=5, C=3, H=64, W=64]
        dummy = torch.zeros(1, 2, 5, 3, 64, 64, device=self.device)

        # Extract to get actual shape
        with torch.no_grad():
            features = self.extract_per_timestep(dummy)

        return features.shape[-1]

    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each component.

        Returns:
            Dictionary mapping component names to ACTUAL dimensions
        """
        total_dim = self.get_actual_per_timestep_dim()

        return {
            'spatial': 105,      # Actual: stats + gradients + histograms
            'spectral': 93,      # Actual: multi-scale FFT features
            'cross_channel': 10, # Actual: pairwise correlations
            'temporal': 120,     # Actual: enhanced temporal dynamics
            'total': total_dim,  # Auto-detected at runtime
        }

    def reset(self):
        """Reset temporal history buffers.

        Call this between operators or when starting a new trajectory.
        """
        self.temporal_extractor.reset()

    # Abstract method implementations

    @property
    def family_name(self) -> str:
        """Feature family name."""
        return "temporal"

    @property
    def version(self) -> str:
        """Feature family version."""
        return "3.0.0"

    def extract_per_trajectory(
        self,
        trajectories: torch.Tensor,
        metadata: Optional[Any] = None
    ) -> Optional[torch.Tensor]:
        """Extract per-trajectory features.

        In v3.0, trajectory-level features are removed (per-timestep only).
        This method returns None for backward compatibility.

        Args:
            trajectories: [N, M, T, C, H, W]
            metadata: Optional metadata

        Returns:
            None (v3.0 has no trajectory-level features)
        """
        return None

    def aggregate_realizations(
        self,
        per_trajectory_features: Optional[torch.Tensor],
        method: str = "mean"
    ) -> Optional[torch.Tensor]:
        """Aggregate realizations.

        In v3.0, trajectory-level features are removed.
        This method returns None for backward compatibility.

        Args:
            per_trajectory_features: [N, M, D_traj] or None
            method: Aggregation method (unused)

        Returns:
            None (v3.0 has no trajectory-level features)
        """
        return None

    def get_feature_registry(self):
        """Get feature registry for v3.0.

        Returns:
            FeatureRegistry with 193D feature mapping
        """
        from spinlock.features.registry import FeatureRegistry

        registry = FeatureRegistry(family_name="temporal")

        # Register features by category
        # Spatial: 24D (8 features x 3 channels)
        for feat in ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'min', 'max', 'range']:
            for ch in ['ch0', 'ch1', 'ch2']:
                registry.register(name=f"spatial_{feat}_{ch}", category="spatial")

        # Spectral: 27D (9 features x 3 channels)
        for feat in ['peak_freq', 'spectral_centroid', 'spectral_spread', 'spectral_flatness',
                     'spectral_rolloff', 'spectral_entropy', 'power_low', 'power_mid', 'power_high']:
            for ch in ['ch0', 'ch1', 'ch2']:
                registry.register(name=f"spectral_{feat}_{ch}", category="spectral")

        # Cross-channel: 12D (4 features x 3 channel pairs)
        for feat in ['correlation', 'covariance', 'mutual_info', 'phase_sync']:
            for pair in ['ch0_ch1', 'ch0_ch2', 'ch1_ch2']:
                registry.register(name=f"cross_channel_{feat}_{pair}", category="cross_channel")

        # Temporal: 130D (enhanced temporal features)
        # Instantaneous (22D)
        for i in range(22):
            registry.register(name=f"temporal_instantaneous_{i}", category="temporal")

        # Local temporal (28D)
        for i in range(28):
            registry.register(name=f"temporal_local_{i}", category="temporal")

        # Local stability (24D)
        for i in range(24):
            registry.register(name=f"temporal_stability_{i}", category="temporal")

        # Phase space (26D)
        for i in range(26):
            registry.register(name=f"temporal_phase_space_{i}", category="temporal")

        # Multi-scale (30D)
        for i in range(30):
            registry.register(name=f"temporal_multiscale_{i}", category="temporal")

        return registry


# Legacy alias for backward compatibility
SummaryExtractor = TemporalFeatureOrchestrator
