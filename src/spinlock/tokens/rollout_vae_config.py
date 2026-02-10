"""Configuration for Token-to-Rollout VAE.

This module defines Pydantic configuration classes for the TokenToRolloutVAE,
following spinlock's config pattern with type-safe validation.
"""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    """VAE encoder configuration."""

    hidden_dims: List[int] = Field(
        default=[2048, 1024], description="Hidden layer dimensions for MLP encoder"
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class ParameterDecoderConfig(BaseModel):
    """Parameter decoder configuration."""

    hidden_dims: List[int] = Field(
        default=[256, 128],
        description="Hidden layer dimensions for parameter decoder MLP",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class GridDecoderConfig(BaseModel):
    """Grid decoder configuration."""

    hidden_channels: List[int] = Field(
        default=[512, 256, 128, 64, 32],
        description="ConvTranspose2d channel progression (spatial size resolved at runtime)",
    )
    dropout: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Dropout probability"
    )


class ModelConfig(BaseModel):
    """TokenToRolloutVAE model configuration.

    Note: input_dim, theta_dim, and grid_shape are resolved at runtime from
    the VQTokenizer checkpoint and CNO dataset.
    """

    latent_dim: int = Field(
        default=512, gt=0, description="Latent space dimensionality"
    )
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    param_decoder: ParameterDecoderConfig = Field(
        default_factory=ParameterDecoderConfig
    )
    grid_decoder: GridDecoderConfig = Field(default_factory=GridDecoderConfig)


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

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


class DataConfig(BaseModel):
    """Dataset paths and split configuration."""

    vq_checkpoint: Path = Field(
        ..., description="Path to frozen VQTokenizer checkpoint"
    )
    cno_dataset: Path = Field(
        ..., description="Path to CNO dataset (e.g., 50k_baseline.h5)"
    )
    tokenized_dataset: Path = Field(
        ..., description="Path to pre-tokenized dataset (e.g., 50k_baseline_tokenized.h5)"
    )
    train_split: float = Field(default=0.9, ge=0.0, le=1.0, description="Training split ratio")
    val_split: float = Field(default=0.1, ge=0.0, le=1.0, description="Validation split ratio")

    def model_post_init(self, __context) -> None:
        """Validate that splits sum to 1.0."""
        if not abs(self.train_split + self.val_split - 1.0) < 1e-6:
            raise ValueError(
                f"train_split ({self.train_split}) + val_split ({self.val_split}) must equal 1.0"
            )


class ValidationConfig(BaseModel):
    """Validation settings."""

    freq_epochs: int = Field(
        default=5, gt=0, description="Run validation every N epochs"
    )
    n_samples: int = Field(
        default=100, gt=0, description="Number of samples for validation metrics"
    )
    sobol_seed: int = Field(default=42, description="Seed for Sobol sampling validation")


class TokenToRolloutVAEConfig(BaseModel):
    """Complete configuration for TokenToRolloutVAE training.

    All input/output dimensions (input_dim, theta_dim, grid_shape) are resolved
    at runtime by inspecting the VQTokenizer checkpoint and CNO dataset.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output_dir: Path = Field(..., description="Output directory for checkpoints and logs")
    device: str = Field(default="cuda", description="Device for training (cuda or cpu)")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    class Config:
        extra = "forbid"  # Raise error on unknown fields

    @classmethod
    def from_yaml(cls, path: Path) -> "TokenToRolloutVAEConfig":
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

        # Convert model_dump to dict with Path objects as strings
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
