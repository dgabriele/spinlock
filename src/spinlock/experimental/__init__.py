"""
Experimental components for Spinlock.

This package contains experimental features and research components:
- Common experiment infrastructure (config, training, data loading)
- Diffusion-based trajectory completion models
- Trajectory completion experiments
- Token coverage analysis utilities
- Clustering comparison tools

These components are production-quality but represent cutting-edge
research features that may evolve rapidly.
"""

__version__ = "1.0.0"

# Re-export common utilities for convenience
from spinlock.experimental.common import (
    ExperimentMetadata,
    CheckpointConfig,
    DataConfig,
    TrainingConfig,
    BaseExperimentConfig,
    load_experiment_config,
    save_experiment_config,
    TrainedVQVAE,
    TrainedMNO,
    BaseExperimentTrainer,
    TrajectoryDataLoader,
)

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
