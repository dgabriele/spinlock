"""Configuration for Token-Conditioned CVAE.

This module defines Pydantic configuration classes for the TokenConditionedCVAE,
following spinlock's config pattern with type-safe validation.

The CVAE models P(theta, IC | temporal_tokens) — given what the dynamics
look like (temporal tokens), generate plausible physical parameters and
initial conditions. Unlike the standard TokenToRolloutVAE, the CVAE encoder
sees BOTH the target (theta, IC) AND the condition (tokens) during training,
learning the posterior q(z | theta, IC, tokens).
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class CVAEEncoderConfig(BaseModel):
    """CVAE recognition network (encoder) configuration.

    Input dim = target_features_dim + condition_dim (resolved at runtime).
    """

    hidden_dims: List[int] = Field(
        default=[512, 256],
        description="Hidden layer dimensions for recognition network MLP",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class ConditionConfig(BaseModel):
    """Token conditioning network configuration.

    Converts frozen codebook embeddings into a fixed-size condition vector
    via per-group MLPs (weight-tied across groups) and mean pooling.
    """

    group_mlp_hidden_dim: int = Field(
        default=128,
        gt=0,
        description="Hidden dim for per-group MLP projection",
    )
    group_mlp_output_dim: int = Field(
        default=64,
        gt=0,
        description="Output dim for per-group MLP (= condition vector dim)",
    )
    pooling: Literal["mean"] = Field(
        default="mean",
        description="Pooling strategy over group embeddings",
    )


class ParameterDecoderConfig(BaseModel):
    """Parameter decoder configuration (shared with rollout_vae)."""

    hidden_dims: List[int] = Field(
        default=[256, 128],
        description="Hidden layer dimensions for parameter decoder MLP",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class GridDecoderConfig(BaseModel):
    """Grid decoder configuration (shared with rollout_vae)."""

    hidden_channels: List[int] = Field(
        default=[512, 256, 128, 64, 32],
        description="ConvTranspose2d channel progression (spatial size resolved at runtime)",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class TargetEncoderConfig(BaseModel):
    """Target encoder configuration.

    Encodes GT theta + IC into a feature vector for the recognition network.
    Only used during training.
    """

    theta_hidden_dim: int = Field(
        default=256, gt=0, description="Hidden dim for theta MLP branch"
    )
    ic_hidden_dim: int = Field(
        default=256, gt=0, description="Hidden dim for IC CNN branch"
    )
    ic_channels: List[int] = Field(
        default=[32, 64, 128],
        description="Conv2d channel progression for IC encoder",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class CVAEModelConfig(BaseModel):
    """TokenConditionedCVAE model configuration.

    Note: theta_dim, grid_shape, and embedding dimensions are resolved
    at runtime from the VQTokenizer checkpoint and dataset.
    """

    latent_dim: int = Field(
        default=256, gt=0, description="Latent space dimensionality"
    )
    encoder: CVAEEncoderConfig = Field(default_factory=CVAEEncoderConfig)
    condition: ConditionConfig = Field(default_factory=ConditionConfig)
    target_encoder: TargetEncoderConfig = Field(
        default_factory=TargetEncoderConfig
    )
    param_decoder: ParameterDecoderConfig = Field(
        default_factory=ParameterDecoderConfig
    )
    grid_decoder: GridDecoderConfig = Field(default_factory=GridDecoderConfig)


class CVAETrainingConfig(BaseModel):
    """CVAE training hyperparameters."""

    num_epochs: int = Field(default=200, gt=0, description="Number of training epochs")
    batch_size: int = Field(default=256, gt=0, description="Training batch size")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Learning rate")
    beta_schedule: Literal["linear", "constant"] = Field(
        default="linear", description="KL annealing schedule type"
    )
    beta_max: float = Field(
        default=1.0, ge=0.0, description="Maximum KL divergence weight"
    )
    beta_warmup_epochs: int = Field(
        default=100, ge=0, description="Number of epochs to ramp beta from 0 to beta_max"
    )
    optimizer: Literal["adam", "adamw"] = Field(default="adam", description="Optimizer type")
    weight_decay: float = Field(default=0.0, ge=0.0, description="Weight decay (L2 penalty)")
    scheduler_type: Optional[Literal["cosine", "step", "none"]] = Field(
        default="cosine", description="Learning rate scheduler"
    )
    grad_clip_norm: Optional[float] = Field(
        default=1.0, ge=0.0, description="Gradient clipping norm (None to disable)"
    )


class CVAEDataConfig(BaseModel):
    """Dataset paths and split configuration.

    Uses generic field names (not operator-specific like 'cno_dataset').
    """

    vq_checkpoint: Path = Field(
        ..., description="Path to frozen VQTokenizer checkpoint"
    )
    dataset: Path = Field(
        ..., description="Path to dataset HDF5 (e.g., 50k_baseline.h5)"
    )
    tokenized_dataset: Path = Field(
        ..., description="Path to pre-tokenized dataset HDF5"
    )
    temporal_keys_only: bool = Field(
        default=True,
        description="Filter to temporal-family token keys only (removes initial/theta)",
    )
    train_split: float = Field(default=0.9, ge=0.0, le=1.0, description="Training split ratio")
    val_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation split ratio")

    def model_post_init(self, __context) -> None:
        """Validate that splits sum to 1.0."""
        if not abs(self.train_split + self.val_split - 1.0) < 1e-6:
            raise ValueError(
                f"train_split ({self.train_split}) + val_split ({self.val_split}) must equal 1.0"
            )


class CVAEValidationConfig(BaseModel):
    """Validation settings."""

    freq_epochs: int = Field(
        default=5, gt=0, description="Run validation every N epochs"
    )


class TokenConditionedCVAEConfig(BaseModel):
    """Complete configuration for TokenConditionedCVAE training.

    All input/output dimensions (theta_dim, grid_shape, embedding dims) are
    resolved at runtime by inspecting the VQTokenizer checkpoint and dataset.
    """

    model: CVAEModelConfig = Field(default_factory=CVAEModelConfig)
    training: CVAETrainingConfig = Field(default_factory=CVAETrainingConfig)
    data: CVAEDataConfig
    validation: CVAEValidationConfig = Field(default_factory=CVAEValidationConfig)
    output_dir: Path = Field(..., description="Output directory for checkpoints and logs")
    device: str = Field(default="cuda", description="Device for training (cuda or cpu)")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: Path) -> "TokenConditionedCVAEConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Validated configuration object
        """
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML config
        """
        import yaml

        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_dict = convert_paths(self.model_dump())

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
