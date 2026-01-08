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

from typing import Literal, Union, Any, Iterator, Dict, Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
from spinlock.config.cloud import CloudConfig


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
    """Categorical parameter with discrete choices.

    Supports optional weighted sampling via the weights field.
    If weights are provided, they must be non-negative and sum to 1.0.
    If weights are omitted, uniform distribution is used (backward compatible).
    """

    type: Literal["choice"] = Field(default="choice", frozen=True)  # type: ignore[assignment]
    choices: list[Union[str, int, float]]
    weights: Optional[list[float]] = None

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v: list[Any]) -> list[Any]:
        if len(v) < 1:
            raise ValueError("Choice parameter must have at least 1 option")
        return v

    @model_validator(mode='after')
    def validate_weights(self) -> 'ChoiceParameter':
        """Validate weights if provided."""
        if self.weights is not None:
            if len(self.weights) != len(self.choices):
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match "
                    f"choices length ({len(self.choices)})"
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("All weights must be non-negative")

            total = sum(self.weights)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {total}. "
                    f"Normalize: {[w/total for w in self.weights]}"
                )
        return self


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
# U-AFNO Parameter Space
# =============================================================================


class UAFNOParameterSpace(BaseModel):
    """
    Parameter space configuration for U-AFNO operators.

    Defines the Sobol-sampable hyperparameters for U-AFNO architecture:
    - modes: Number of Fourier modes for spectral mixing
    - hidden_dim: Hidden dimension in AFNO MLP
    - encoder_levels: Depth of U-Net encoder
    - afno_blocks: Number of stacked AFNO blocks in bottleneck
    - blocks_per_level: Residual blocks per U-Net level
    """

    modes: IntegerParameter = Field(
        default_factory=lambda: IntegerParameter(bounds=(8, 64)),
        description="Number of Fourier modes to keep in AFNO spectral mixing"
    )
    hidden_dim: IntegerParameter = Field(
        default_factory=lambda: IntegerParameter(bounds=(32, 256)),
        description="Hidden dimension for AFNO MLP (default: 2x bottleneck channels)"
    )
    encoder_levels: IntegerParameter = Field(
        default_factory=lambda: IntegerParameter(bounds=(2, 5)),
        description="Number of U-Net encoder levels (each halves spatial resolution)"
    )
    afno_blocks: IntegerParameter = Field(
        default_factory=lambda: IntegerParameter(bounds=(2, 8)),
        description="Number of stacked AFNO blocks in bottleneck"
    )
    blocks_per_level: IntegerParameter = Field(
        default_factory=lambda: IntegerParameter(bounds=(1, 4)),
        description="Number of residual blocks per U-Net level"
    )


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

    # U-AFNO specific parameter space (optional, only when operator_type="u_afno")
    u_afno: Optional[UAFNOParameterSpace] = None

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
        yield from self.evolution.values()
        # Include U-AFNO params if specified
        if self.u_afno is not None:
            yield self.u_afno.modes
            yield self.u_afno.hidden_dim
            yield self.u_afno.encoder_levels
            yield self.u_afno.afno_blocks
            yield self.u_afno.blocks_per_level

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
    total_samples: int = Field(ge=1, description="Total number of parameter sets to sample")
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

    method: Literal[
        "random", "gaussian_random_field", "structured", "mixed", "sampled",
        # Tier 1 domain-specific ICs
        "quantum_wave_packet", "turing_pattern", "thermal_gradient",
        "morphogen_gradient", "reaction_front",
        # Tier 2 domain-specific ICs
        "light_cone", "critical_fluctuation", "phase_boundary",
        "bz_reaction", "shannon_entropy",
        # Tier 3 domain-specific ICs
        "interference_pattern", "cell_population", "chromatin_domain",
        "shock_front", "gene_expression",
        # Tier 4 research frontiers ICs
        "coherent_state", "relativistic_wave_packet", "mutual_information",
        "regulatory_network", "dla_cluster", "error_correcting_code"
    ] = "random"
    distribution: Literal["gaussian", "uniform"] = "gaussian"
    num_samples_per_operator: int = Field(default=100, ge=1)
    spatial_size: Union[Literal["from_operator"], int] = "from_operator"
    # Optional parameters for GRF generation
    length_scale: float = Field(default=0.1, gt=0, le=1)
    variance: float = Field(default=1.0, gt=0)
    # Optional parameters for sampled IC type method
    ic_type_weights: Dict[str, float] = Field(default_factory=dict)
    # Existing IC type parameter dicts
    multiscale_grf: Dict[str, Any] = Field(default_factory=dict)
    localized: Dict[str, Any] = Field(default_factory=dict)
    composite: Dict[str, Any] = Field(default_factory=dict)
    heavy_tailed: Dict[str, Any] = Field(default_factory=dict)
    # Tier 1 domain-specific IC parameter dicts
    quantum_wave_packet: Dict[str, Any] = Field(default_factory=dict)
    turing_pattern: Dict[str, Any] = Field(default_factory=dict)
    thermal_gradient: Dict[str, Any] = Field(default_factory=dict)
    morphogen_gradient: Dict[str, Any] = Field(default_factory=dict)
    reaction_front: Dict[str, Any] = Field(default_factory=dict)
    # Tier 2 domain-specific IC parameter dicts
    light_cone: Dict[str, Any] = Field(default_factory=dict)
    critical_fluctuation: Dict[str, Any] = Field(default_factory=dict)
    phase_boundary: Dict[str, Any] = Field(default_factory=dict)
    bz_reaction: Dict[str, Any] = Field(default_factory=dict)
    shannon_entropy: Dict[str, Any] = Field(default_factory=dict)
    # Tier 3 domain-specific IC parameter dicts
    interference_pattern: Dict[str, Any] = Field(default_factory=dict)
    cell_population: Dict[str, Any] = Field(default_factory=dict)
    chromatin_domain: Dict[str, Any] = Field(default_factory=dict)
    shock_front: Dict[str, Any] = Field(default_factory=dict)
    gene_expression: Dict[str, Any] = Field(default_factory=dict)
    # Tier 4 research frontiers IC parameter dicts
    coherent_state: Dict[str, Any] = Field(default_factory=dict)
    relativistic_wave_packet: Dict[str, Any] = Field(default_factory=dict)
    mutual_information: Dict[str, Any] = Field(default_factory=dict)
    regulatory_network: Dict[str, Any] = Field(default_factory=dict)
    dla_cluster: Dict[str, Any] = Field(default_factory=dict)
    error_correcting_code: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    compile: bool = False  # torch.compile (PyTorch 2.0+) - legacy single-operator mode
    benchmark_cudnn: bool = True
    deterministic: bool = False

    # Phase 1: Architecture partitioning + torch.compile per partition
    # Groups operators by (num_layers, channels_bucket, kernel_size) and compiles ONE
    # kernel per partition, reusing it across all operators with same architecture.
    partition_by_architecture: bool = True  # Default: enabled
    warmup_templates: bool = False  # Pre-compile all templates (can OOM with 100+ architectures)
    channel_bucket_size: int = 16  # Bucket channels in groups (16, 32, 48, 64)
    compile_mode: Literal["reduce-overhead", "max-autotune", "default"] = "reduce-overhead"

    # Phase 2: Batched operator execution with vmap (future)
    batched_execution: bool = False  # Not yet implemented
    max_batch_size: int = 32  # Max operators per vmap batch


class SimulationConfig(BaseModel):
    """Complete simulation configuration."""

    device: Literal["cuda", "cpu"] = "cuda"
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    input_generation: InputGenerationConfig = Field(default_factory=InputGenerationConfig)
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    num_realizations: int = Field(default=10, ge=1, le=100)
    num_timesteps: int = Field(default=1, ge=1, le=10000)
    extract_operator_features: bool = Field(default=False)

    # Operator architecture type
    operator_type: Literal["cnn", "u_afno"] = Field(
        default="cnn",
        description="Neural operator architecture: 'cnn' for convolutional, 'u_afno' for U-Net+AFNO"
    )


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


class TemporalFeaturesConfig(BaseModel):
    """TEMPORAL feature family configuration (per-timestep time series)."""

    enabled: bool = Field(
        default=True,
        description="Extract per-timestep time series features. Disable for SUMMARY-only mode."
    )


class LearnedFeaturesConfig(BaseModel):
    """Learned feature configuration (neural operator latent extraction).

    Supports both U-AFNO and CNN operators:
    - U-AFNO: bottleneck, skips, or all
    - CNN: early, mid, pre_output, or all
    """

    enabled: bool = Field(
        default=False,
        description="Enable learned feature extraction from neural operator latent representations."
    )
    extract_from: Literal["bottleneck", "skips", "all", "early", "mid", "pre_output"] = Field(
        default="all",
        description=(
            "Which latents to extract. "
            "U-AFNO: 'bottleneck', 'skips', or 'all'. "
            "CNN: 'early', 'mid', 'pre_output', or 'all'."
        )
    )
    # U-AFNO specific
    skip_levels: List[int] = Field(
        default_factory=lambda: [0, 1, 2],
        description="(U-AFNO) Which encoder levels to extract (0 = shallowest, higher = deeper)."
    )
    # CNN specific
    layer_indices: Optional[List[int]] = Field(
        default=None,
        description="(CNN) Which mid layer indices to extract. None = all mid layers."
    )
    temporal_agg: Literal["mean", "max", "mean_max", "std"] = Field(
        default="mean_max",
        description="Temporal aggregation method across timesteps."
    )
    spatial_agg: Literal["gap", "flatten"] = Field(
        default="gap",
        description="Spatial aggregation: gap (global average pooling) or flatten."
    )
    projection_dim: Optional[int] = Field(
        default=None,
        ge=8,
        le=512,
        description="Optional fixed output dimension via MLP projection."
    )

    # Training config for learned features
    training_epochs: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Number of epochs to train each operator on next-step prediction."
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0,
        description="Learning rate for operator training (Adam optimizer)."
    )
    lr_scheduler: Literal["constant", "cosine"] = Field(
        default="cosine",
        description="Learning rate schedule: constant or cosine annealing."
    )
    early_stopping_patience: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Stop training if no improvement for this many epochs."
    )


class SummaryFeaturesConfig(BaseModel):
    """SUMMARY feature family configuration (aggregated scalars)."""

    enabled: bool = Field(
        default=True,
        description="Extract aggregated summary features."
    )
    summary_mode: Literal["manual", "learned", "hybrid"] = Field(
        default="manual",
        description=(
            "Feature mode: 'manual' (hand-crafted features only), "
            "'learned' (neural operator latent features only), "
            "'hybrid' (both concatenated)."
        )
    )
    learned: LearnedFeaturesConfig = Field(
        default_factory=LearnedFeaturesConfig,
        description="Configuration for learned feature extraction from neural operator latents (U-AFNO or CNN)."
    )


class FeaturesConfig(BaseModel):
    """Feature extraction configuration.

    Two feature families:
    - TEMPORAL: Per-timestep time series features [N, T, D]
    - SUMMARY: Aggregated scalar features [N, D]
    """

    temporal: TemporalFeaturesConfig = Field(default_factory=TemporalFeaturesConfig)
    summary: SummaryFeaturesConfig = Field(default_factory=SummaryFeaturesConfig)


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
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cloud: CloudConfig = Field(
        default_factory=CloudConfig,
        description="Cloud provider configuration (Lambda Labs, RunPod, S3)"
    )

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
