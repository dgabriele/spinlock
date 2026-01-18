"""TEMPORAL Features v3.0.

Per-timestep-only feature extraction for online perturbation-based NOA.

This module extracts 193D features per timestep:
- Spatial: 24D (per-channel statistics)
- Spectral: 27D (frequency domain features)
- Cross-channel: 12D (channel interactions)
- Enhanced temporal: 130D (windowed dynamics)

Example:
    >>> from spinlock.features.temporal import TemporalFeatureOrchestrator
    >>> orchestrator = TemporalFeatureOrchestrator(device=torch.device('cuda'))
    >>> trajectories = torch.randn(100, 10, 256, 3, 128, 128)
    >>> features = orchestrator.extract_per_timestep(trajectories)
    >>> features.shape
    torch.Size([100, 256, 193])
"""

__version__ = "3.0.0"

from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator
from spinlock.features.temporal.config import (
    TemporalFeatureConfig,
    SpatialConfig,
    SpectralConfig,
    CrossChannelConfig,
    TemporalConfig,
    SummaryConfig,  # Legacy alias
    SummarySpatialConfig,  # Legacy alias
    SummarySpectralConfig,  # Legacy alias
    SummaryCrossChannelConfig,  # Legacy alias
)
from spinlock.features.temporal.spatial import SpatialFeatureExtractor
from spinlock.features.temporal.spectral import SpectralFeatureExtractor
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.temporal.cross_channel import CrossChannelFeatureExtractor

# Legacy aliases for backward compatibility
SummaryExtractor = TemporalFeatureOrchestrator

__all__ = [
    # v3.0 names (recommended)
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
    # Legacy aliases (for backward compatibility)
    "SummaryExtractor",
    "SummaryConfig",
    "SummarySpatialConfig",
    "SummarySpectralConfig",
    "SummaryCrossChannelConfig",
]
