"""
Configuration schemas for Spinlock using Pydantic.

Provides type-safe, validated configuration models for all system components:
- Parameter space definitions
- Sampling strategies
- Simulation settings
- Dataset storage

Design principles:
- DRY: Shared base classes for common validation logic
- Modular: Each component has its own config model
- Extensible: Easy to add new parameter types or strategies
"""

from typing import Literal, Union, Any, Iterator, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


# =============================================================================
# Base Parameter Classes (DRY Foundation)
# =============================================================================


class ParameterSpec(BaseModel):
    """Base class for all parameter specifications."""

    type: Literal["integer", "continuous", "choice", "boolean", "array"]
    description: str = ""


class BoundedParameter(ParameterSpec):
    """Shared bounds validation for numeric parameters."""

    bounds: tuple[float, float]

    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] >= v[1]:
            raise ValueError(f"Invalid bounds: {v}. Lower bound must be < upper bound.")
        return v


# =============================================================================
# Specific Parameter Types
# =============================================================================


class IntegerParameter(BoundedParameter):
    """Integer-valued parameter with bounds."""

    type: Literal["integer"] = Field(default="integer", frozen=True)  # type: ignore[assignment]
    bounds: tuple[int, int]  # type: ignore[assignment]


class ContinuousParameter(BoundedParameter):
    """Continuous-valued parameter with optional log-scale sampling."""

    type: Literal["continuous"] = Field(default="continuous", frozen=True)  # type: ignore[assignment]
    log_scale: bool = False


class ChoiceParameter(ParameterSpec):
    """Categorical parameter with discrete choices."""

    type: Literal["choice"] = Field(default="choice", frozen=True)  # type: ignore[assignment]
    choices: list[Union[str, int, float]]

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: list[Any]) -> list[Any]:
        if len(v) < 1:
            raise ValueError("Choice parameter must have at least 1 option")
        return v


class BooleanParameter(ParameterSpec):
    """Boolean parameter."""

    type: Literal["boolean"] = Field(default="boolean", frozen=True)  # type: ignore[assignment]


class ArrayParameter(ParameterSpec):
    """Array of parameters (e.g., channels per layer)."""

    type: Literal["array"] = Field(default="array", frozen=True)  # type: ignore[assignment]
    length: Union[int, str]  # Static int or reference to another param
    element_type: Literal["integer", "continuous", "choice", "boolean"]
    bounds: Union[tuple[float, float], None] = None
    choices: Union[list[Any], None] = None

    @model_validator(mode='after')
    def validate_element_constraints(self) -> 'ArrayParameter':
        if self.element_type in ["integer", "continuous"] and self.bounds is None:
            raise ValueError(f"Array with element_type={self.element_type} requires bounds")
        if self.element_type == "choice" and self.choices is None:
            raise ValueError("Array with element_type=choice requires choices")
        return self


# =============================================================================
# Parameter Space Configuration
# =============================================================================


class ParameterSpace(BaseModel):
    """
    Complete parameter space definition.

    Includes architecture, stochastic, operator, and evolution parameters.
    All parameters are sampled via Sobol sequences and stored in datasets
    for downstream use (e.g., parameter space embeddings, analysis).
    """

    architecture: Dict[str, Union[
        IntegerParameter, ContinuousParameter, ChoiceParameter,
        BooleanParameter, ArrayParameter
    ]]
    stochastic: Dict[str, Union[
        IntegerParameter, ContinuousParameter, ChoiceParameter, BooleanParameter
    ]]
    operator: Dict[str, Union[
        IntegerParameter, ContinuousParameter, ChoiceParameter, BooleanParameter
    ]]
    evolution: Dict[str, Union[
        IntegerParameter, ContinuousParameter, ChoiceParameter, BooleanParameter
    ]] = Field(default_factory=dict)  # Optional for backward compatibility

    @property
    def total_dimensions(self) -> int:
        """
        Calculate total parameter space dimensionality.

        Returns:
            Total number of scalar parameters across all categories
            (architecture, stochastic, operator, evolution).
        """
        total = 0
        for param in self._all_params():
            total += self._count_dimensions(param)
        return total

    def _all_params(self) -> Iterator[ParameterSpec]:
        """DRY: Single iterator over all parameters across all categories."""
        yield from self.architecture.values()
        yield from self.stochastic.values()
        yield from self.operator.values()
        yield from self.evolution.values()  # NEW: Include evolution params

    def _count_dimensions(self, param: ParameterSpec) -> int:
        """Count dimensions for a single parameter."""
        if isinstance(param, ArrayParameter):
            # Array contributes multiple dimensions
            if isinstance(param.length, int):
                return param.length
            # If length is a string reference, we'd resolve it here
            # For now, assume it's been validated elsewhere
            return 1  # Placeholder
        return 1


# =============================================================================
# Sampling Configuration
# =============================================================================


class SobolConfig(BaseModel):
    """Sobol sequence configuration."""

    scramble: bool = True
    scramble_method: Literal["owen", "lms_shift"] = "owen"
    seed: int = Field(default=42, ge=0)


class StratificationConfig(BaseModel):
    """Parameter space stratification configuration."""

    method: Literal["adaptive", "uniform", "variance_based"] = "adaptive"
    num_strata_per_dim: int = Field(default=4, ge=2, le=10)
    min_samples_per_stratum: int = Field(default=10, ge=1)


class RefinementConfig(BaseModel):
    """Adaptive refinement configuration."""

    enabled: bool = True
    pilot_fraction: float = Field(default=0.01, gt=0, le=0.1)
    variance_threshold: float = Field(default=0.1, gt=0)
    max_iterations: int = Field(default=3, ge=1, le=10)


class ValidationConfig(BaseModel):
    """Sample quality validation configuration."""

    check_discrepancy: bool = True
    max_discrepancy: float = Field(default=0.01, gt=0)
    check_correlation: bool = True
    max_pairwise_correlation: float = Field(default=0.05, gt=0, lt=1)


class SamplingConfig(BaseModel):
    """Complete sampling configuration."""

    strategy: Literal["sobol_stratified"] = "sobol_stratified"
    sobol: SobolConfig = Field(default_factory=SobolConfig)
    stratification: StratificationConfig = Field(default_factory=StratificationConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    total_samples: int = Field(ge=100)
    batch_size: int = Field(default=1000, ge=1)

    @model_validator(mode='after')
    def validate_batch_size(self) -> 'SamplingConfig':
        if self.total_samples % self.batch_size != 0:
            raise ValueError(
                f"total_samples ({self.total_samples}) must be divisible by "
                f"batch_size ({self.batch_size})"
            )
        return self


# =============================================================================
# Simulation Configuration
# =============================================================================


class ParallelismConfig(BaseModel):
    """GPU parallelization configuration."""

    strategy: Literal["data_parallel", "ddp"] = "data_parallel"
    devices: Union[Literal["auto"], list[int]] = "auto"


class InputGenerationConfig(BaseModel):
    """Input field generation configuration."""

    method: Literal["random", "gaussian_random_field", "structured", "mixed", "sampled"] = "random"
    distribution: Literal["gaussian", "uniform"] = "gaussian"
    num_samples_per_operator: int = Field(default=100, ge=1)
    spatial_size: Union[Literal["from_operator"], int] = "from_operator"
    # Optional parameters for GRF generation
    length_scale: float = Field(default=0.1, gt=0, le=1)
    variance: float = Field(default=1.0, gt=0)
    # Optional parameters for sampled IC type method
    ic_type_weights: Dict[str, float] = Field(default_factory=dict)
    multiscale_grf: Dict[str, Any] = Field(default_factory=dict)
    localized: Dict[str, Any] = Field(default_factory=dict)
    composite: Dict[str, Any] = Field(default_factory=dict)
    heavy_tailed: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    compile: bool = False  # torch.compile (PyTorch 2.0+)
    benchmark_cudnn: bool = True
    deterministic: bool = False


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    device: Literal["cuda", "cpu"] = "cuda"
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    input_generation: InputGenerationConfig = Field(default_factory=InputGenerationConfig)
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    num_realizations: int = Field(default=10, ge=1, le=100)


# =============================================================================
# Dataset Configuration
# =============================================================================


class StorageConfig(BaseModel):
    """Storage backend configuration."""

    backend: Literal["hdf5", "zarr"] = "hdf5"
    compression: Literal["gzip", "lzf", "none"] = "gzip"
    compression_level: int = Field(default=4, ge=0, le=9)
    chunk_size: int = Field(default=1000, ge=1)


class DatasetConfig(BaseModel):
    """Complete dataset configuration."""

    output_path: Path
    storage: StorageConfig = Field(default_factory=StorageConfig)
    data_to_store: list[str] = Field(
        default_factory=lambda: [
            "operator_parameters",
            "operator_outputs",
            "input_data"
        ]
    )
    metadata: Dict[str, bool] = Field(
        default_factory=lambda: {
            "include_config": True,
            "include_sampling_metrics": True,
            "include_timestamps": True
        }
    )


# =============================================================================
# Logging Configuration
# =============================================================================


class MetricsConfig(BaseModel):
    """Performance metrics tracking configuration."""

    track_gpu_memory: bool = True
    track_sampling_time: bool = True
    track_simulation_time: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["structured", "plain"] = "structured"
    output: Union[Literal["stdout"], Path] = "stdout"
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


# =============================================================================
# Root Configuration
# =============================================================================


class MetadataConfig(BaseModel):
    """Experiment metadata."""

    name: str
    description: str = ""
    author: str = ""
    created: str = ""


class SpinlockConfig(BaseModel):
    """
    Root configuration model for Spinlock.

    Validates and provides type-safe access to all system configuration:
    - Parameter space definition
    - Sampling strategy
    - Simulation settings
    - Dataset storage
    - Logging and metrics

    Example:
        ```python
        config = SpinlockConfig.from_yaml("config.yaml")
        print(f"Total dimensions: {config.parameter_space.total_dimensions}")
        ```
    """

    version: str = "1.0"
    metadata: MetadataConfig
    parameter_space: ParameterSpace
    sampling: SamplingConfig
    simulation: SimulationConfig
    dataset: DatasetConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode='after')
    def validate_output_path(self) -> 'SpinlockConfig':
        """Ensure output directory exists or can be created."""
        output_path = self.dataset.output_path
        if not output_path.parent.exists():
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Cannot create output directory {output_path.parent}: {e}"
                )
        return self
