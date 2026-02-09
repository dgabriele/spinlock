"""Pydantic configuration models for discrete diffusion experiments."""

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
from enum import Enum


class MaskingStrategy(str, Enum):
    """Masking strategies for hierarchical tokens."""
    RANDOM = "random"
    COARSE_ONLY = "coarse_only"
    HIERARCHICAL = "hierarchical"
    FAMILY_SELECTIVE = "family_selective"


class DatasetConfig(BaseModel):
    """Dataset configuration for diffusion training."""
    # Pre-tokenized mode (fast)
    use_pretokenized: bool = False
    tokenized_path: Optional[Path] = None

    # On-the-fly tokenization mode (slow, flexible)
    path: Optional[Path] = None
    tokenizer_checkpoint: Optional[Path] = None
    cache_tokens: bool = False
    max_cache_size: Optional[int] = None
    device: str = "cpu"

    @field_validator("tokenized_path")
    @classmethod
    def validate_pretokenized_path(cls, v, info):
        """Validate pre-tokenized dataset exists if use_pretokenized=True."""
        if info.data.get("use_pretokenized") and v is not None:
            if not v.exists():
                raise ValueError(f"Pre-tokenized dataset not found: {v}")
        return v

    @field_validator("path")
    @classmethod
    def validate_dataset_path(cls, v, info):
        """Validate dataset exists if not using pre-tokenized."""
        if not info.data.get("use_pretokenized") and v is not None:
            if not v.exists():
                raise ValueError(f"Dataset not found: {v}")
        return v

    @field_validator("tokenizer_checkpoint")
    @classmethod
    def validate_tokenizer(cls, v, info):
        """Validate tokenizer exists if not using pre-tokenized."""
        if not info.data.get("use_pretokenized") and v is not None:
            if not v.exists():
                raise ValueError(f"Tokenizer checkpoint not found: {v}")
        return v

    def model_post_init(self, __context):
        """Validate configuration consistency after initialization."""
        if self.use_pretokenized and self.tokenized_path is None:
            raise ValueError("tokenized_path required when use_pretokenized=True")
        if not self.use_pretokenized and (self.path is None or self.tokenizer_checkpoint is None):
            raise ValueError("path and tokenizer_checkpoint required when use_pretokenized=False")


class MaskingConfig(BaseModel):
    """Masking configuration for diffusion training."""
    strategy: MaskingStrategy = MaskingStrategy.RANDOM
    mask_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int = 42


class DiffusionConfig(BaseModel):
    """Diffusion process configuration."""
    num_timesteps: int = Field(default=50, ge=1)
    beta_start: float = Field(default=0.0001, gt=0.0)
    beta_end: float = Field(default=0.02, gt=0.0)
    schedule_type: str = Field(default="cosine", pattern="^(linear|cosine|sqrt)$")

    @field_validator("beta_end")
    @classmethod
    def validate_beta_range(cls, v, info):
        """Ensure beta_end > beta_start."""
        if "beta_start" in info.data and v <= info.data["beta_start"]:
            raise ValueError(f"beta_end ({v}) must be > beta_start ({info.data['beta_start']})")
        return v


class ModelConfig(BaseModel):
    """Denoising network model configuration."""
    hidden_dim: int = Field(default=256, ge=64)
    num_layers: int = Field(default=6, ge=1)
    num_heads: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    use_hierarchical_guidance: bool = True
    hierarchical_guidance_weight: float = Field(default=0.1, ge=0.0)


class LRSchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: str = Field(default="cosine", pattern="^(cosine|linear|step)$")
    warmup_epochs: int = Field(default=2, ge=0)
    min_lr: float = Field(default=1e-6, gt=0.0)


class TrainingConfig(BaseModel):
    """Training configuration."""
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=30, ge=1)
    num_workers: int = Field(default=0, ge=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-5, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)

    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)

    val_batch_size: int = Field(default=128, ge=1)
    val_split: float = Field(default=0.1, gt=0.0, lt=1.0)
    val_frequency: int = Field(default=1, ge=1)
    val_samples: int = Field(default=5, ge=1)

    checkpoint_frequency: int = Field(default=5, ge=1)
    save_best: bool = True

    log_frequency: int = Field(default=50, ge=1)
    use_wandb: bool = False


class OutputConfig(BaseModel):
    """Output configuration."""
    dir: Path
    checkpoint_prefix: str = "diffusion"

    @field_validator("dir")
    @classmethod
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class DiffusionExperimentConfig(BaseModel):
    """Complete configuration for discrete diffusion experiment."""
    dataset: DatasetConfig
    masking: MaskingConfig
    diffusion: DiffusionConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig

    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    seed: int = 42

    class Config:
        """Pydantic config."""
        use_enum_values = True
