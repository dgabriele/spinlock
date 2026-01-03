"""
General feature extraction configuration.

Root configuration for the feature extraction pipeline that supports
two sibling feature families:
- TEMPORAL: Per-timestep time series [N, T, D] (spatial, spectral, cross_channel)
- SUMMARY: Aggregated scalars [N, D] (temporal dynamics, causality, invariant_drift)

Example:
    >>> from spinlock.features.config import FeatureExtractionConfig, TemporalConfig
    >>> from spinlock.features.summary.config import SummaryConfig
    >>>
    >>> config = FeatureExtractionConfig(
    ...     input_dataset=Path("datasets/benchmark_10k.h5"),
    ...     temporal=TemporalConfig(enabled=True),
    ...     summary=SummaryConfig()
    ... )
"""

from typing import Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryConfig


class TemporalConfig(BaseModel):
    """
    TEMPORAL feature family configuration.

    TEMPORAL features are per-timestep time series [N, T, D] computed at each
    timestep. Categories: spatial, spectral, cross_channel.

    Attributes:
        enabled: Whether to extract TEMPORAL features
    """
    enabled: bool = Field(
        default=True,
        description="Extract per-timestep time series features [N, T, D]"
    )


class FeatureExtractionConfig(BaseModel):
    """
    Root configuration for feature extraction pipeline.

    Two sibling feature families:
    - TEMPORAL: Per-timestep time series [N, T, D]
    - SUMMARY: Aggregated scalars [N, D]

    Attributes:
        input_dataset: Path to input HDF5 dataset
        output_dataset: Optional separate output path (None writes to input)
        temporal: TEMPORAL feature family configuration (per-timestep)
        summary: SUMMARY feature family configuration (aggregated scalars)
        batch_size: Batch size for extraction
        device: Computation device (cuda or cpu)
        overwrite: Whether to overwrite existing features
        num_workers: Number of parallel workers
        cache_trajectories: Cache full trajectories in memory vs streaming

    Example:
        >>> from spinlock.features.summary.config import SummaryConfig
        >>> config = FeatureExtractionConfig(
        ...     input_dataset=Path("datasets/benchmark_10k.h5"),
        ...     temporal=TemporalConfig(enabled=False),  # Disable TEMPORAL
        ...     summary=SummaryConfig(),
        ...     batch_size=32,
        ...     device="cuda"
        ... )
    """

    # Input/output paths
    input_dataset: Path
    output_dataset: Optional[Path] = None  # If None, writes to input_dataset

    # Feature families to extract (siblings, not nested)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)  # TEMPORAL features (per-timestep)
    summary: Optional['SummaryConfig'] = None  # SUMMARY features (aggregated scalars)

    # Extraction settings
    batch_size: int = Field(default=32, ge=1, le=1000)
    device: Literal["cuda", "cpu"] = "cuda"
    overwrite: bool = False  # Whether to overwrite existing features
    max_samples: Optional[int] = Field(default=None, ge=1)  # Limit number of samples (None=all)

    # Processing options
    num_workers: int = Field(default=4, ge=1)  # For parallel batch processing
    cache_trajectories: bool = True  # Cache full trajectories in memory vs streaming

    @field_validator('input_dataset')
    @classmethod
    def validate_input_exists(cls, v: Path) -> Path:
        """Ensure input dataset path is valid."""
        if not v.suffix == '.h5':
            raise ValueError("input_dataset must be an HDF5 file (.h5)")
        return v

    @field_validator('output_dataset')
    @classmethod
    def validate_output_path(cls, v: Optional[Path], info) -> Optional[Path]:
        """Ensure output path is different from input if specified."""
        if v is not None:
            if not v.suffix == '.h5':
                raise ValueError("output_dataset must be an HDF5 file (.h5)")
            if 'input_dataset' in info.data:
                if v.resolve() == info.data['input_dataset'].resolve():
                    raise ValueError("output_dataset must differ from input_dataset")
        return v

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Ensure batch size is reasonable."""
        if v < 1 or v > 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        return v


# Rebuild model after SummaryConfig is imported to resolve forward references
def _rebuild_model():
    """Rebuild FeatureExtractionConfig after SummaryConfig is defined."""
    try:
        from spinlock.features.summary.config import SummaryConfig  # noqa: F401
        FeatureExtractionConfig.model_rebuild()
    except ImportError:
        # SummaryConfig not yet available (during initial import)
        pass


_rebuild_model()
