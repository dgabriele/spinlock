"""Temporal Feature Configuration (v3.0).

Configuration classes for per-timestep-only feature extraction.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpatialConfig:
    """Configuration for spatial feature extraction.

    Attributes:
        enabled: Whether spatial features are enabled
        per_channel: Extract features per channel (vs. aggregated)
    """
    enabled: bool = True
    per_channel: bool = True


@dataclass
class SpectralConfig:
    """Configuration for spectral feature extraction.

    Attributes:
        enabled: Whether spectral features are enabled
        per_channel: Extract features per channel (vs. aggregated)
    """
    enabled: bool = True
    per_channel: bool = True


@dataclass
class CrossChannelConfig:
    """Configuration for cross-channel feature extraction.

    Attributes:
        enabled: Whether cross-channel features are enabled
    """
    enabled: bool = True


@dataclass
class TemporalConfig:
    """Configuration for enhanced temporal dynamics (130D).

    Attributes:
        enabled: Whether temporal features are enabled
        window_size: Primary window size for temporal features
        short_window: Short-term window (5 steps)
        medium_window: Medium-term window (20 steps)
        long_window: Long-term window (50 steps)
    """
    enabled: bool = True
    window_size: int = 5
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 50


@dataclass
class TemporalFeatureConfig:
    """Complete per-timestep feature configuration (v3.0).

    This replaces the v2.x SummaryConfig with a focus on per-timestep-only features.

    Total dimensions: 193D
    - Spatial: 24D
    - Spectral: 27D
    - Cross-channel: 12D
    - Enhanced temporal: 130D

    Attributes:
        spatial: Spatial feature configuration
        spectral: Spectral feature configuration
        cross_channel: Cross-channel feature configuration
        temporal: Enhanced temporal feature configuration
        per_channel: Global per-channel setting (can be overridden by component configs)
        version: Configuration version

    Example:
        >>> config = TemporalFeatureConfig()
        >>> config.temporal.window_size = 10
        >>> config.spatial.enabled = False
    """
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    cross_channel: CrossChannelConfig = field(default_factory=CrossChannelConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    per_channel: bool = True
    version: str = "3.0.0"

    @classmethod
    def from_schema_config(cls, schema_config):
        """Convert from spinlock.config.schema.TemporalFeaturesConfig.

        Args:
            schema_config: Config from YAML schema

        Returns:
            TemporalFeatureConfig instance
        """
        # For now, just use defaults
        # Can be extended to map schema config fields if needed
        return cls()

    def get_total_dims(self) -> int:
        """Get total feature dimensions.

        Returns:
            Total feature dimension (193D for default config)
        """
        dims = 0
        if self.spatial.enabled:
            dims += 24
        if self.spectral.enabled:
            dims += 27
        if self.cross_channel.enabled:
            dims += 12
        if self.temporal.enabled:
            dims += 130
        return dims


# Legacy alias for backward compatibility
SummaryConfig = TemporalFeatureConfig


# Legacy spatial config for compatibility with old code
@dataclass
class SummarySpatialConfig:
    """Legacy spatial config alias."""
    per_channel: bool = True
    enabled: bool = True


# Legacy spectral config
@dataclass
class SummarySpectralConfig:
    """Legacy spectral config alias."""
    per_channel: bool = True
    enabled: bool = True


# Legacy cross-channel config
@dataclass
class SummaryCrossChannelConfig:
    """Legacy cross-channel config alias."""
    enabled: bool = True
