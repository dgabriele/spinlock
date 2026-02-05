"""Base configuration schemas for experiments."""

from pydantic import BaseModel, Field, field_validator
from pathlib import Path
from typing import Optional
from datetime import datetime


class ExperimentMetadata(BaseModel):
    """Metadata for all experiments."""
    name: str
    description: str
    author: str = "Daniel"
    created: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class CheckpointConfig(BaseModel):
    """Checkpoint loading configuration."""
    vqvae_path: Path = Path("checkpoints/vqvae/50k_baseline/best_model.pt")
    mno_path: Optional[Path] = Path("checkpoints/mno/50k_baseline/meta_operator_best.pt")

    @field_validator("vqvae_path", "mno_path")
    @classmethod
    def validate_checkpoint_exists(cls, v):
        if v and not v.exists():
            raise ValueError(f"Checkpoint not found: {v}")
        return v


class DataConfig(BaseModel):
    """Data configuration."""
    dataset_path: Path
    batch_size: int = 16
    val_split: float = 0.2
    num_workers: int = 4
    shuffle: bool = True

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_exists(cls, v):
        if not v.exists():
            raise ValueError(f"Dataset not found: {v}")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    seed: int = 42
    save_every: int = 5


class BaseExperimentConfig(BaseModel):
    """Base configuration for all experiments."""
    metadata: ExperimentMetadata
    checkpoints: CheckpointConfig
    data: DataConfig
    training: TrainingConfig
    output_dir: Path = Path("experiments/{experiment_name}/results")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
