"""Temporal Feature Configuration (v3.1).

Configuration classes for per-timestep-only feature extraction.
Updated with enhanced temporal and spectral features.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from spinlock.features.quantum.config import QuantumConfig


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
        num_fft_scales: Number of FFT scales (for CLI compatibility)
    """
    enabled: bool = True
    per_channel: bool = True
    num_fft_scales: int = 5  # For backward compatibility with CLI


@dataclass
class CrossChannelConfig:
    """Configuration for cross-channel feature extraction.

    Attributes:
        enabled: Whether cross-channel features are enabled
    """
    enabled: bool = True


@dataclass
class TemporalConfig:
    """Configuration for enhanced temporal dynamics (145D).

    v3.1 additions: skewness, kurtosis, multi-lag autocorrelation,
    sample entropy, enstrophy, divergence (+15D from v3.0)

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
class StructuralConfig:
    """Stub config for removed structural features (v2.x legacy)."""
    enabled: bool = False  # Removed in v3.0


@dataclass
class PhysicsConfig:
    """Stub config for removed physics features (v2.x legacy)."""
    enabled: bool = False  # Removed in v3.0


@dataclass
class MorphologicalConfig:
    """Stub config for removed morphological features (v2.x legacy)."""
    enabled: bool = False  # Removed in v3.0


@dataclass
class MultiscaleConfig:
    """Stub config for removed multiscale features (v2.x legacy)."""
    enabled: bool = False  # Removed in v3.0


@dataclass
class TemporalFeatureConfig:
    """Complete per-timestep feature configuration (v3.2).

    This replaces the v2.x SummaryConfig with a focus on per-timestep-only features.

    Feature families (registered categories):
    - spatial: Spatial statistics (per-channel, category="spatial")
    - spectral: Frequency domain features (category="spectral")
    - cross_channel: Channel interaction features (category="cross_channel")
    - temporal: Temporal dynamics features (category="temporal")
      - Includes standard dynamics (145D): windowed stats, autocorr, entropy, etc.
      - Includes quantum features (10D, conditional): purity, coherence, uncertainty

    Total dimensions: ~209D per timestep (v3.1), ~219D with quantum (v3.2)

    Note: Quantum features are a configurable subset of the temporal family, only
    extracted for 2-channel wavefunction data (Re, Im). They are registered with
    category="temporal" for proper organization.

    Attributes:
        spatial: Spatial feature configuration
        spectral: Spectral feature configuration
        cross_channel: Cross-channel feature configuration
        temporal: Enhanced temporal dynamics configuration
        quantum: Quantum feature configuration (subset of temporal family)
        per_channel: Global per-channel setting (can be overridden by component configs)
        version: Configuration version

    Example:
        >>> config = TemporalFeatureConfig()
        >>> config.temporal.window_size = 10
        >>> config.quantum.enabled = True  # Enable quantum subset
        >>> config.spatial.enabled = False
    """
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    cross_channel: CrossChannelConfig = field(default_factory=CrossChannelConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)  # Subset of temporal family (category="temporal")
    per_channel: bool = True
    version: str = "3.1.0"
    channel_indices: Optional[list] = None  # Select specific channels for feature extraction (e.g., [0] for density-only)

    # v2.x legacy stubs (removed features)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    morphological: MorphologicalConfig = field(default_factory=MorphologicalConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)

    # v2.x aggregation settings (not used in v3.0, but needed for backward compat)
    temporal_aggregation: list = field(default_factory=lambda: ["mean"])
    realization_aggregation: list = field(default_factory=lambda: ["mean"])

    # v2.x summary mode (v3.0 uses manual hand-crafted features only)
    summary_mode: str = "manual"  # Options: "manual", "learned", "hybrid"
    learned: Optional[Any] = None  # v2.x learned feature config (not used in v3.0)

    @classmethod
    def from_schema_config(cls, schema_config):
        """Convert from spinlock.config.schema.TemporalFeaturesConfig.

        Args:
            schema_config: Config from YAML schema

        Returns:
            TemporalFeatureConfig instance
        """
        # Extract channel_indices if present
        channel_indices = getattr(schema_config, 'channel_indices', None)

        # For now, use defaults with optional channel_indices override
        return cls(channel_indices=channel_indices)

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

    def estimate_feature_count(self) -> int:
        """Estimate total feature count (for backward compatibility).

        In v3.0, this returns the actual per-timestep feature count.
        For legacy compatibility with CLI that expects this method.

        Returns:
            Estimated feature count (~328D per timestep in practice)
        """
        # Return approximate actual dimensions based on extractors
        # (spatial: ~105D, spectral: ~93D, cross-channel: ~10D, temporal: ~120D)
        dims = 0
        if self.spatial.enabled:
            dims += 105  # Actual from extractor (includes gradients, histograms)
        if self.spectral.enabled:
            dims += 93   # Actual from extractor (multi-scale FFT)
        if self.cross_channel.enabled:
            dims += 10   # Actual from extractor
        if self.temporal.enabled:
            dims += 120  # Actual from enhanced temporal extractor
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
