"""NOA (Neural Operator Agent) - Phase 1 Implementation.

This module implements the NOA Phase 1 architecture:
- U-AFNO backbone for generating operator rollouts
- Feature extraction using proper SUMMARY/TEMPORAL extractors
- Training pipeline with real dataset support

All dimensions are resolved dynamically at runtime.
"""

from .backbone import NOABackbone
from .losses import VQVAEPerceptualLoss, FeatureProjector, NOALoss
from .feature_extraction import NOAFeatureExtractor
from .dataset import NOARealDataset, NOARealDatasetStreaming
from .training import (
    NOAPhase1Trainer,
    NOADataset,
    NOADatasetWithFeatures,
    NOARealDataTrainer,
    extract_trajectory_features,
    generate_synthetic_data,
)
from .cno_replay import CNOReplayer

__all__ = [
    # Backbone
    "NOABackbone",
    # Feature extraction (real data)
    "NOAFeatureExtractor",
    # Datasets
    "NOARealDataset",
    "NOARealDatasetStreaming",
    "NOADataset",
    "NOADatasetWithFeatures",
    # Training
    "NOARealDataTrainer",
    "NOAPhase1Trainer",
    # Losses
    "VQVAEPerceptualLoss",
    "FeatureProjector",
    "NOALoss",
    # Utilities (legacy/synthetic)
    "extract_trajectory_features",
    "generate_synthetic_data",
    # CNO replay for state supervision
    "CNOReplayer",
]
