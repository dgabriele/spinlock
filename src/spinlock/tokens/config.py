"""Configuration models for VQ tokenizer."""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal, Union
from pathlib import Path

from spinlock.features.grouping.models import GroupingConfig


class QuantizerConfig(BaseModel):
    """Vector quantization configuration.

    Note: num_embeddings is computed adaptively per quantizer using v1's formula.
    Do not specify in config - it will be ignored if provided.
    """
    num_embeddings: Optional[int] = Field(
        default=None,
        gt=0,
        description="Codebook size (auto-computed adaptively, do not set in config)"
    )
    embedding_dim: int = Field(default=64, gt=0, description="Embedding dimension")
    commitment_cost: float = Field(default=0.25, ge=0.0)
    use_ema: bool = Field(default=True)
    ema_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    epsilon: float = Field(default=1e-5, gt=0.0)


class InitialEncoderConfig(BaseModel):
    """Initial condition encoder configuration.

    Dimensions (manual_dim, in_channels) are automatically detected from dataset
    if not specified. Manual specification only needed for overrides.
    """
    variant: Literal["cnn", "hybrid"] = "hybrid"
    manual_dim: Optional[int] = Field(
        default=None,
        ge=0,
        description="Initial feature dimension (auto-detected if None)"
    )
    cnn_embedding_dim: int = Field(default=256, gt=0)
    in_channels: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of input channels (auto-detected if None)"
    )
    pretrained_cnn_path: Optional[Path] = None
    use_final_batchnorm: bool = False
    encode_manual: bool = False


class VariableLengthConfig(BaseModel):
    """Variable-length sequence configuration for pyramid encoder."""
    enabled: bool = True
    min_timesteps: int = Field(default=16, ge=1)
    max_timesteps: int = Field(default=256, ge=1)
    length_bins: Optional[List[int]] = None
    sampling_strategy: Literal["fixed_bins", "uniform"] = "fixed_bins"
    adaptive_pyramid: bool = True
    min_pyramid_length: int = Field(default=1, ge=1)
    mask_downsample_method: Literal["ceil", "floor"] = "ceil"


class LearnedTemporalConfig(BaseModel):
    """Config for learned CNN temporal feature extraction.

    Used when feature_source="learned". The CNN replaces the entire
    hand-crafted TemporalFeatureOrchestrator pipeline.

    Two architectures:
      - ``"per_frame"``: Per-frame CNN → sequential group slicing → per-group
        pyramid. Legacy mode with known variance imbalance issues.
      - ``"pyramid_first"``: Raw trajectory → SpatioTemporalPyramid → shared CNN →
        per-level temporal aggregation → learned group projection. Fixes the
        temporal context and group balance problems.
    """
    architecture: Literal["per_frame", "pyramid_first"] = Field(
        default="per_frame",
        description="Encoder architecture: 'per_frame' (legacy) or 'pyramid_first' (recommended)",
    )
    in_channels: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of input channels per frame (auto-detected from data)"
    )
    # per_frame mode only:
    embedding_dim: int = Field(
        default=240,
        gt=0,
        description="Total CNN output dim per frame (num_groups * group_dim). Only used in per_frame mode."
    )
    # Shared:
    num_groups: int = Field(
        default=30,
        gt=0,
        description="Number of temporal groups"
    )
    # pyramid_first mode:
    d_cnn: int = Field(
        default=256,
        gt=0,
        description="Per-frame CNN output dim (pyramid_first mode)"
    )
    d_agg: int = Field(
        default=128,
        gt=0,
        description="Per-level temporal aggregation dim (pyramid_first mode)"
    )
    # Gated groups (pyramid_first mode): learnable per-group gates for
    # downstream-driven group selection during joint fine-tuning.
    gated_groups: bool = Field(
        default=False,
        description="Enable per-group learnable gates (pyramid_first mode)"
    )
    gate_init_bias: float = Field(
        default=5.0,
        description="Initial gate logit bias (sigmoid(5)≈0.993, gates start open)"
    )

    @field_validator('embedding_dim')
    @classmethod
    def validate_divisible(cls, v, info):
        """Ensure embedding_dim is divisible by num_groups (per_frame mode only)."""
        architecture = info.data.get('architecture', 'per_frame')
        if architecture != "per_frame":
            return v  # Skip validation for pyramid_first mode
        num_groups = info.data.get('num_groups', 30)
        if v % num_groups != 0:
            raise ValueError(
                f"embedding_dim ({v}) must be divisible by num_groups ({num_groups})"
            )
        return v


class TemporalEncoderConfig(BaseModel):
    """Temporal sequence encoder configuration."""
    variant: Literal["mean", "cnn", "pyramid"] = "pyramid"
    level_dims: List[int] = Field(default=[32, 64, 96, 128])
    downsample_factors: List[int] = Field(default=[1, 2, 4, 8])
    variable_length: Union[bool, VariableLengthConfig] = True
    min_timesteps: int = Field(default=16, ge=1)
    max_timesteps: int = Field(default=256, ge=1)
    adaptive_pyramid: bool = True
    learned: Optional[LearnedTemporalConfig] = Field(
        default=None,
        description="Config for learned CNN temporal features (used when feature_source='learned')"
    )


class ThetaEncoderConfig(BaseModel):
    """Configuration for theta (parameter) encoder.

    Parameter dimension (param_dim) is automatically detected from dataset
    if not specified. Manual specification only needed for overrides.

    Variants:
        - "mlp": Two-layer MLP mapping all params to shared embedding space.
        - "direct": No encoder — each parameter gets its own independent VQ group.
          This gives ~10 bits/param vs ~0.8 bits/param with the MLP bottleneck.
    """

    variant: Literal["mlp", "direct"] = Field(
        default="mlp",
        description="Encoder variant: 'mlp' (shared embedding) or 'direct' (per-param VQ)"
    )
    param_dim: Optional[int] = Field(
        default=None,
        ge=1,
        description="Dimensionality of input parameters (auto-detected if None)"
    )
    hidden_dim: int = Field(
        default=64,
        gt=0,
        description="Hidden layer size"
    )
    output_dim: int = Field(
        default=32,
        gt=0,
        description="Output embedding dimensionality"
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout probability"
    )
    use_layer_norm: bool = Field(
        default=True,
        description="Whether to apply LayerNorm"
    )


class EncoderConfig(BaseModel):
    """Multi-family encoder configuration."""
    initial: InitialEncoderConfig = Field(default_factory=InitialEncoderConfig)
    temporal: TemporalEncoderConfig = Field(default_factory=TemporalEncoderConfig)
    theta: Optional[ThetaEncoderConfig] = None
    embedding_dim: int = Field(default=64, gt=0)
    hidden_dim: int = Field(default=128, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    use_layer_norm: bool = True


class HierarchyConfig(BaseModel):
    """Hierarchical quantization configuration."""
    num_levels: int = Field(default=3, ge=1, le=5)
    compression_ratios: str = "0.5:1.0:1.5"
    min_latent_dim: int = Field(default=4, ge=2)
    max_latent_dim: int = Field(default=64, ge=2)

    @field_validator('compression_ratios')
    @classmethod
    def validate_ratios(cls, v: str) -> str:
        ratios = [float(r) for r in v.split(':')]
        if any(r <= 0 for r in ratios):
            raise ValueError("All ratios must be positive")
        return v


class AuxHeadConfig(BaseModel):
    """Config for auxiliary decoder heads in temporal-only mode.

    When temporal_only=True, theta and IC are NOT separate token families.
    Instead, they are DECODED from the concatenated quantized temporal latents
    via lightweight auxiliary MLP/CNN heads. This eliminates the cross-family
    consistency problem: every valid temporal token sequence automatically
    produces valid (theta, IC) by construction.

    Note: theta_param_dim and initial channels/spatial_size are auto-detected
    from encoder config and dataset at runtime.
    """
    theta_enabled: bool = Field(default=True, description="Enable theta auxiliary head")
    theta_hidden_dim: int = Field(default=128, gt=0, description="Hidden dim for theta MLP")
    theta_weight: float = Field(default=1.0, ge=0.0, description="Weight for theta aux loss")
    initial_enabled: bool = Field(default=True, description="Enable IC auxiliary head")
    initial_base_channels: int = Field(default=256, gt=0, description="Base channels for IC CNN decoder")
    initial_weight: float = Field(default=0.5, ge=0.0, description="Weight for IC aux loss")

    # Pre-VQ theta probe: predicts theta from encoded temporal features BEFORE
    # quantization, giving the CNN a direct gradient signal to encode
    # param-dependent dynamics without passing through the VQ bottleneck.
    theta_probe_enabled: bool = Field(default=True, description="Enable pre-VQ theta probe")
    theta_probe_hidden_dim: int = Field(default=256, gt=0, description="Hidden dim for probe MLP")
    theta_probe_weight: float = Field(default=5.0, ge=0.0, description="Weight for probe loss")

    # Pre-VQ IC probe: predicts ICs from encoded temporal features BEFORE
    # quantization, bypassing the VQ bottleneck for a direct gradient signal.
    initial_probe_enabled: bool = Field(default=False, description="Enable pre-VQ IC probe")
    initial_probe_weight: float = Field(default=1.0, ge=0.0, description="Weight for IC probe loss")


class InverseHeadConfig(BaseModel):
    """Configuration for inverse decoder heads.

    Note: Dimensions like theta_param_dim, initial_channels, initial_spatial_size
    are inferred from the dataset at runtime. Only specify architectural hyperparameters here.
    """

    # Theta inverse (dimensions inferred from encoder config and data)
    theta_hidden_dim: int = Field(default=64, description="Hidden dimension for theta MLP")
    theta_dropout: float = Field(default=0.1, description="Dropout rate for theta MLP")

    # Initial inverse (dimensions inferred from encoder config and data)
    initial_base_channels: int = Field(default=256, description="Base channels for CNN decoder")

    # Temporal inverse (only used in learned feature mode)
    temporal_hidden_dim: int = Field(default=1536, description="Hidden dimension for temporal inverse MLP")
    temporal_dropout: float = Field(default=0.1, description="Dropout rate for temporal inverse MLP")
    temporal_roundtrip_timesteps: int = Field(
        default=8,
        description="Number of synthetic timesteps for temporal roundtrip re-encoding",
    )


class RoundtripLossConfig(BaseModel):
    """Configuration for roundtrip consistency loss."""

    enabled: bool = Field(default=True, description="Enable roundtrip loss")
    weight: float = Field(default=1.0, description="Weight for roundtrip loss in total loss")
    theta_weight: float = Field(default=1.0, description="Weight for theta roundtrip")
    initial_weight: float = Field(default=1.0, description="Weight for initial roundtrip")
    temporal_weight: float = Field(default=1.0, description="Weight for temporal roundtrip")


class LossConfig(BaseModel):
    """Loss function configuration."""
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    orthogonality_weight: float = Field(default=0.1, ge=0.0)
    informativeness_weight: float = Field(default=0.1, ge=0.0)
    informativeness_mode: str = Field(
        default="log_barrier",
        description=(
            "Informativeness loss formulation: 'floor' (original, activates only "
            "on near-collapse) or 'log_barrier' (continuously active, penalizes "
            "any group with variance < 1.0 via -log(var))"
        ),
    )
    topographic_weight: float = Field(default=0.0, ge=0.0)
    topographic_n_samples: int = Field(
        default=64,
        ge=4,
        description=(
            "Number of samples drawn from the batch for pairwise distance computation "
            "in the topographic loss. Scale proportionally with batch_size to keep "
            "the same fraction of the batch covered."
        ),
    )
    normalize_reconstruction: bool = True
    normalize_loss_scales: bool = Field(
        default=False,
        description=(
            "EMA loss-scale normalization: each loss L_i is replaced by "
            "L_i / EMA(L_i), so weights reflect actual gradient ratios "
            "regardless of raw loss magnitudes. Eliminates manual tuning."
        ),
    )
    loss_scale_ema_momentum: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="EMA momentum for loss-scale normalization (higher = slower adaptation)",
    )

    # Group balance loss (learned mode): penalizes per-group variance imbalance
    # in CNN output features. CV = std(group_vars)/mean(group_vars).
    group_balance_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for group balance loss (0 = disabled, try 0.5-2.0 for learned mode)",
    )

    # Gate sparsity loss (pyramid_first gated mode): L1 penalty on group gates
    # to encourage the model to concentrate info into fewer groups.
    # Set to 0.0 during tokenizer pretraining; increase during joint fine-tuning.
    gate_sparsity_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for L1 gate sparsity loss (0 = disabled)",
    )

    # Roundtrip loss configuration
    roundtrip: Optional[RoundtripLossConfig] = Field(
        default=None,
        description="Configuration for roundtrip consistency loss"
    )


class TrainingConfig(BaseModel):
    """Training loop configuration."""
    num_epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    val_split: float = Field(default=0.2, ge=0.0, le=0.5)
    val_every_n_epochs: int = Field(default=5, ge=1)

    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_clip_norm: Optional[float] = None

    use_scheduler: bool = False
    scheduler_type: Literal["cosine", "step", "exponential"] = "cosine"
    warmup_epochs: int = Field(default=0, ge=0)

    early_stopping_patience: int = Field(default=20, ge=1)
    early_stopping_min_delta: float = Field(default=1e-4, ge=0.0)

    dead_code_reset_interval: int = Field(default=10, ge=0, description="0 = disabled")
    dead_code_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    device: Literal["cuda", "cpu", "auto"] = "auto"
    compile_model: bool = False

    # OPQ (Optimized Product Quantization) rotation calibration
    opq_enabled: bool = Field(default=False, description="Enable OPQ rotation before training")
    opq_warmup_epochs: int = Field(
        default=0, ge=0,
        description="Epochs to train before OPQ calibration (0 = calibrate immediately)",
    )
    opq_n_calibration_batches: int = Field(
        default=50, ge=1, description="Batches to collect CNN features from",
    )
    opq_max_samples: int = Field(
        default=50000, ge=100, description="Max flattened frames for OPQ fitting",
    )
    opq_n_iter: int = Field(
        default=20, ge=1, description="Alternating optimization iterations",
    )
    opq_n_codes: int = Field(
        default=32, ge=2, description="Proxy codebook size per subspace",
    )


class NormalizationConfig(BaseModel):
    """Feature normalization configuration."""
    method: Literal["per_category", "global", "none"] = "per_category"
    clip_std_multiplier: Optional[float] = None


class FeatureCleaningConfig(BaseModel):
    """Feature cleaning configuration (pre-training)."""
    enabled: bool = Field(default=False, description="Enable feature cleaning")
    pre_categorization: bool = Field(
        default=True,
        description="Clean features before categorization (recommended)"
    )
    variance_threshold: float = Field(
        default=1e-8,
        ge=0.0,
        description="Remove features with std below this threshold"
    )
    deduplicate_threshold: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Remove features with |corr| above this threshold"
    )
    use_intelligent_dedup: bool = Field(
        default=True,
        description="Keep more informative feature from correlated pairs"
    )
    outlier_method: Literal["percentile", "iqr", "mad", "none"] = Field(
        default="percentile",
        description="Method for outlier capping"
    )
    percentile_range: tuple[float, float] = Field(
        default=(0.5, 99.5),
        description="Percentile range for clipping (e.g., (0.5, 99.5))"
    )


class TokenizerConfig(BaseModel):
    """Complete VQ tokenizer configuration."""
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    quantizer: QuantizerConfig = Field(default_factory=QuantizerConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    grouping: Optional[GroupingConfig] = None
    feature_cleaning: Optional[FeatureCleaningConfig] = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    random_seed: Optional[int] = None
    verbose: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Inverse heads configuration
    inverse_heads: Optional[InverseHeadConfig] = Field(
        default=None,
        description="Configuration for inverse decoder heads (theta/initial reconstruction)"
    )

    # Temporal-only mode: only temporal tokens, theta/IC decoded from aux heads
    temporal_only: bool = Field(
        default=False,
        description=(
            "When True, only temporal features go through VQ encoding/quantization. "
            "Theta and IC are decoded from quantized temporal latents via auxiliary heads, "
            "eliminating the cross-family consistency problem for D3PM."
        ),
    )
    aux_heads: Optional[AuxHeadConfig] = Field(
        default=None,
        description="Auxiliary head config for temporal-only mode (theta/IC prediction from temporal tokens)"
    )

    # Learned temporal feature mode
    feature_source: Literal["manual", "learned"] = Field(
        default="manual",
        description=(
            "'manual': hand-crafted TemporalFeatureOrchestrator (default). "
            "'learned': per-frame CNN replaces entire feature pipeline."
        ),
    )
    generation_config_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to dataset generation config YAML for on-the-fly trajectory replay. "
            "Auto-detects operator type (CNO, Lenia, etc.) and creates the appropriate replayer."
        ),
    )
    cno_config_path: Optional[str] = Field(
        default=None,
        description="Deprecated: use generation_config_path instead. Kept for backward compatibility."
    )
    replayer_cache_size: int = Field(
        default=8,
        ge=0,
        description="Operator cache size for CNOReplayer (ignored by Lenia)"
    )
    generation_timesteps: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of timesteps for on-the-fly trajectory generation (auto-detected if None)"
    )


class PretrainingConfig(BaseModel):
    """CNN encoder pretraining configuration."""
    embedding_dim: int = Field(default=256, gt=0)
    in_channels: int = Field(default=3, ge=1)
    use_final_batchnorm: bool = False
    num_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    val_split: float = Field(default=0.1, ge=0.0, le=0.5)
    optimizer: Literal["adam", "adamw"] = "adam"
    use_scheduler: bool = True
    scheduler_type: Literal["cosine", "step", "exponential"] = "cosine"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    log_every_n_epochs: int = Field(default=1, ge=1)
    verbose: bool = True
