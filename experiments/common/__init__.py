"""
Common utilities and base classes for experiments.

Provides shared infrastructure following DRY principles:
- Config system (Pydantic schemas, YAML loading)
- Model wrappers (TrainedVQVAE, TrainedMNO)
- Training infrastructure (BaseExperimentTrainer)
- Data utilities (TrajectoryDataLoader)
"""

from experiments.common.config.base import (
    ExperimentMetadata,
    CheckpointConfig,
    DataConfig,
    TrainingConfig,
    BaseExperimentConfig,
)
from experiments.common.config.loader import (
    load_experiment_config,
    save_experiment_config,
)
from experiments.common.models.trained_vqvae import TrainedVQVAE
from experiments.common.models.trained_mno import TrainedMNO
from experiments.common.training.trainer import BaseExperimentTrainer
from experiments.common.data.trajectory_loader import TrajectoryDataLoader

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
