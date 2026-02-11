"""
Common utilities and base classes for experiments.

Provides shared infrastructure following DRY principles:
- Config system (Pydantic schemas, YAML loading)
- Model wrappers (TrainedVQVAE, TrainedMNO)
- Training infrastructure (BaseExperimentTrainer)
- Data utilities (TrajectoryDataLoader)
"""

from spinlock.experimental.common.config.base import (
    ExperimentMetadata,
    CheckpointConfig,
    DataConfig,
    TrainingConfig,
    BaseExperimentConfig,
)
from spinlock.experimental.common.config.loader import (
    load_experiment_config,
    save_experiment_config,
)
from spinlock.experimental.common.models.trained_vqvae import TrainedVQVAE
from spinlock.experimental.common.models.trained_mno import TrainedMNO
from spinlock.experimental.common.training.trainer import BaseExperimentTrainer
from spinlock.experimental.common.data.trajectory_loader import TrajectoryDataLoader

__all__ = [
    "ExperimentMetadata",
    "CheckpointConfig",
    "DataConfig",
    "TrainingConfig",
    "BaseExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    "TrainedVQVAE",
    "TrainedMNO",
    "BaseExperimentTrainer",
    "TrajectoryDataLoader",
]
