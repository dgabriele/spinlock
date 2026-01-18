"""
TEMPORAL Features v3.0.

Per-timestep-only feature extraction for online perturbation-based NOA.

Architecture:
- Spatial statistics (24D): moments, gradients, curvature
- Spectral features (27D): FFT, power spectrum, frequencies
- Cross-channel interactions (12D): correlation, mutual information, eigenvalues
- Enhanced temporal dynamics (130D): instantaneous, local temporal, stability,
  phase space geometry, multi-scale

Total: 193D per-timestep features for online autonomous episodes.

Breaking Changes from v2.x:
- Removed trajectory-level features (causality, invariant_drift, operator_sensitivity, nonlinear)
- Renamed all "Summary*" classes to remove prefix or rename to "Temporal*"
- Per-timestep-only architecture (no aggregation across trajectories)
"""

__version__ = "3.0.0"

from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator
from spinlock.features.temporal.config import (
    TemporalFeatureConfig,
    SpatialConfig,
    SpectralConfig,
    CrossChannelConfig,
    TemporalConfig,
)
from spinlock.features.temporal.spatial import SpatialFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.temporal.cross_channel import CrossChannelFeatureExtractor

# Legacy aliases for backward compatibility
SummaryExtractor = TemporalFeatureOrchestrator
SummaryConfig = TemporalFeatureConfig
SummarySpatialConfig = SpatialConfig
SummarySpectralConfig = SpectralConfig
SummaryCrossChannelConfig = CrossChannelConfig
SummaryTemporalConfig = TemporalConfig

__all__ = [
    "TemporalFeatureOrchestrator",
    "TemporalFeatureConfig",
    "SpatialConfig",
    "SpectralConfig",
    "CrossChannelConfig",
    "TemporalConfig",
    "SpatialFeatureExtractor",
    "SpectralFeatureExtractor",
    "TemporalFeatureExtractor",
    "CrossChannelFeatureExtractor",
    # Legacy aliases
    "SummaryExtractor",
    "SummaryConfig",
    "SummarySpatialConfig",
    "SummarySpectralConfig",
    "SummaryCrossChannelConfig",
    "SummaryTemporalConfig",
]
