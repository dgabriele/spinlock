"""Configuration system for experiments."""

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
    substitute_env_vars,
)

__all__ = [
    "ExperimentMetadata",
    "CheckpointConfig",
    "DataConfig",
    "TrainingConfig",
    "BaseExperimentConfig",
    "load_experiment_config",
    "save_experiment_config",
    "substitute_env_vars",
]
