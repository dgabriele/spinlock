"""Configuration models for VQ tokenizer."""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal
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
    """Initial condition encoder configuration."""
    variant: Literal["cnn", "hybrid"] = "hybrid"
    manual_dim: int = Field(default=42, ge=0)
    cnn_embedding_dim: int = Field(default=256, gt=0)
    in_channels: int = Field(default=3, ge=1)
    pretrained_cnn_path: Optional[Path] = None
    use_final_batchnorm: bool = False
    encode_manual: bool = False


class TemporalEncoderConfig(BaseModel):
    """Temporal sequence encoder configuration."""
    variant: Literal["mean", "cnn", "pyramid"] = "pyramid"
    level_dims: List[int] = Field(default=[32, 64, 96, 128])
    downsample_factors: List[int] = Field(default=[1, 2, 4, 8])
    variable_length: bool = True
    min_timesteps: int = Field(default=16, ge=1)
    max_timesteps: int = Field(default=256, ge=1)
    adaptive_pyramid: bool = True


class ThetaEncoderConfig(BaseModel):
    """Configuration for theta (parameter) encoder."""

    variant: Literal["mlp"] = Field(
        default="mlp",
        description="Encoder variant (currently only MLP supported)"
    )
    param_dim: int = Field(
        default=14,
        ge=1,
        description="Dimensionality of input parameters"
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


class RoundtripLossConfig(BaseModel):
    """Configuration for roundtrip consistency loss."""

    enabled: bool = Field(default=True, description="Enable roundtrip loss")
    weight: float = Field(default=1.0, description="Weight for roundtrip loss in total loss")
    theta_weight: float = Field(default=1.0, description="Weight for theta roundtrip")
    initial_weight: float = Field(default=1.0, description="Weight for initial roundtrip")


class LossConfig(BaseModel):
    """Loss function configuration."""
    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    orthogonality_weight: float = Field(default=0.1, ge=0.0)
    informativeness_weight: float = Field(default=0.1, ge=0.0)
    topographic_weight: float = Field(default=0.0, ge=0.0)
    normalize_reconstruction: bool = True

    # NEW: Roundtrip loss configuration
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

    dead_code_reset_interval: int = Field(default=10, ge=1)
    dead_code_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    device: Literal["cuda", "cpu", "auto"] = "auto"
    compile_model: bool = False


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

    # NEW: Inverse heads configuration
    inverse_heads: Optional[InverseHeadConfig] = Field(
        default=None,
        description="Configuration for inverse decoder heads (theta/initial reconstruction)"
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
