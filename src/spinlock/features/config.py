"""
General feature extraction configuration.

Root configuration for the feature extraction pipeline that supports
multiple feature families (SDF, and future families like temporal_series, spatial_tokens).

Example:
    >>> from spinlock.features.config import FeatureExtractionConfig
    >>> from spinlock.features.sdf.config import SDFConfig
    >>>
    >>> config = FeatureExtractionConfig(
    ...     input_dataset=Path("datasets/benchmark_10k.h5"),
    ...     sdf=SDFConfig()
    ... )
"""

from typing import Literal, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFConfig


class FeatureExtractionConfig(BaseModel):
    """
    Root configuration for feature extraction pipeline.

    Supports multiple feature families (SDF is first, more to come).

    Attributes:
        input_dataset: Path to input HDF5 dataset
        output_dataset: Optional separate output path (None writes to input)
        sdf: Summary Descriptor Features configuration
        batch_size: Batch size for extraction
        device: Computation device (cuda or cpu)
        overwrite: Whether to overwrite existing features
        num_workers: Number of parallel workers
        cache_trajectories: Cache full trajectories in memory vs streaming

    Example:
        >>> from spinlock.features.sdf.config import SDFConfig
        >>> config = FeatureExtractionConfig(
        ...     input_dataset=Path("datasets/benchmark_10k.h5"),
        ...     sdf=SDFConfig(),
        ...     batch_size=32,
        ...     device="cuda"
        ... )
    """

    # Input/output paths
    input_dataset: Path
    output_dataset: Optional[Path] = None  # If None, writes to input_dataset

    # Feature families to extract
    sdf: Optional['SDFConfig'] = None  # TYPE_CHECKING import avoids circular dependency
    # Future: temporal_series, spatial_tokens, etc.

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
