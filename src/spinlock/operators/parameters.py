"""
Dataclasses for type-safe operator parameters.

Provides structured, validated parameter representations that replace
Dict[str, Any] for better type safety and introspection.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Literal


@dataclass(frozen=True)
class OperatorParameters:
    """
    Type-safe operator parameters.

    Replaces Dict[str, Any] with explicit fields for type checking
    and IDE support. Includes evolution parameters for temporal dynamics.

    Example:
        ```python
        params = OperatorParameters(
            num_layers=3,
            base_channels=32,
            input_channels=3,
            output_channels=3,
            kernel_size=3,
            activation="gelu",
            normalization="instance",
            dropout_rate=0.1,
            noise_type="gaussian",
            noise_scale=0.05,
            noise_schedule="periodic",
            schedule_period=100,
            spatial_correlation=0.1,
            # Evolution parameters
            update_policy="convex",
            alpha=0.7,
            dt=0.01
        )

        # Type-safe access
        layers = params.num_layers  # int, not Any
        policy = params.update_policy  # str

        # Convert to dict for legacy code
        param_dict = params.to_dict()
        ```
    """

    # Architecture parameters
    num_layers: int
    base_channels: int
    input_channels: int
    output_channels: int
    kernel_size: int
    activation: str
    normalization: str

    # Regularization
    dropout_rate: float = 0.0
    use_batch_norm: bool = False

    # Stochastic parameters
    noise_type: Optional[Literal["gaussian", "dropout", "multiplicative", "laplace"]] = None
    noise_scale: Optional[float] = None
    noise_location: Optional[str] = None
    noise_schedule: Optional[Literal["constant", "annealing", "periodic"]] = None
    schedule_period: Optional[int] = None
    spatial_correlation: Optional[float] = None

    # Grid parameters
    grid_size: int = 64

    # Evolution parameters (temporal dynamics)
    # These define how the operator evolves over time when run autoregressively
    update_policy: str = "convex"  # "autoregressive", "residual", "convex"
    alpha: float = 0.5  # Convex policy: mixing parameter in [0,1]
    dt: float = 0.01    # Residual policy: integration step size

    # Additional metadata
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatorParameters":
        """Create from dictionary with validation."""
        # Extract known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra"}

        # Separate known and unknown fields
        known_data = {k: v for k, v in data.items() if k in known_fields}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}

        if extra_data:
            known_data["extra"] = extra_data

        return cls(**known_data)

    def __getitem__(self, key: str) -> Any:
        """Support dict-like access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Support dict-like .get() for backward compatibility."""
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default


@dataclass(frozen=True)
class SamplingMetrics:
    """
    Type-safe sampling quality metrics.

    Replaces Dict[str, Any] returned by validation functions.

    Example:
        ```python
        metrics = SamplingMetrics(
            discrepancy=0.00001,
            max_correlation=0.002,
            discrepancy_pass=True,
            correlation_pass=True
        )

        # Type-safe access
        if metrics.discrepancy_pass:
            print(f"Quality: {metrics.discrepancy:.6f}")
        ```
    """

    discrepancy: float
    max_correlation: float
    discrepancy_pass: bool
    correlation_pass: bool

    # Optional detailed metrics
    min_correlation: Optional[float] = None
    mean_correlation: Optional[float] = None
    coverage_uniformity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingMetrics":
        """Create from dictionary with validation."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)


@dataclass
class DatasetMetadata:
    """
    Type-safe dataset metadata.

    Replaces metadata dict stored in HDF5 files.

    Example:
        ```python
        metadata = DatasetMetadata(
            creation_date="2025-12-27T19:22:11.217090",
            version="1.0",
            grid_size=64,
            num_parameter_sets=10000,
            num_realizations=10,
            config={...},
            sampling_metrics=SamplingMetrics(...)
        )

        # Type-safe access
        print(f"Created: {metadata.creation_date}")
        print(f"Samples: {metadata.num_parameter_sets}")
        ```
    """

    creation_date: str
    version: str
    grid_size: int
    num_parameter_sets: int
    num_realizations: int

    # Complex nested data (stored as JSON in HDF5)
    config: Dict[str, Any]
    sampling_metrics: Dict[str, Any]  # Can be SamplingMetrics.to_dict()

    # Optional fields
    description: Optional[str] = None
    experiment_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        """Create from HDF5 metadata attributes."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def get_sampling_metrics(self) -> Optional[SamplingMetrics]:
        """Parse sampling_metrics dict into SamplingMetrics dataclass."""
        if self.sampling_metrics:
            try:
                return SamplingMetrics.from_dict(self.sampling_metrics)
            except (TypeError, KeyError):
                return None
        return None


@dataclass
class BatchMetadata:
    """
    Metadata for a batch of generated data.

    Used internally during dataset generation to track batch processing.

    Example:
        ```python
        batch_meta = BatchMetadata(
            batch_idx=5,
            batch_size=100,
            start_idx=500,
            end_idx=600,
            generation_time=2.34,
            inference_time=8.91
        )
        ```
    """

    batch_idx: int
    batch_size: int
    start_idx: int
    end_idx: int

    # Timing information
    generation_time: float = 0.0
    inference_time: float = 0.0
    storage_time: float = 0.0

    # Memory information (optional)
    peak_memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
