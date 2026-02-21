"""MNO (Meta Neural Operator) - Neural Operator Implementation.

V2 uses trajectory-first training: MSE + contrastive loss, then separate
VQTokenizer training on MNO outputs.

For loading real HDF5 datasets, use spinlock.data.SpinlockDataset — it is the
single authoritative interface to GT field data in this codebase.

All dimensions are resolved dynamically at runtime.
"""

# Abstract base classes
from .base_backbone import BaseMNOBackbone
from .base_loss import BaseNOALoss, LossOutput

# Concrete backbone
from .backbone import MNOBackbone

# Training losses
from .losses import ContrastiveLoss

# Perceptual losses (VQ-VAE feature-based)
from .perceptual_losses import VQVAEPerceptualLoss, FeatureProjector, NOALoss

# Feature extraction
from .feature_extraction import MNOFeatureExtractor

# In-memory datasets (for synthetic / unit-test use only)
from .training import (
    MNOTrajectoryTrainer,
    MNOInMemoryDataset,
    MNOInMemoryDatasetWithFeatures,
    extract_trajectory_features,
    generate_synthetic_data,
)

# CNO replay and VQ alignment
from .cno_replay import CNOReplayer
from .vqvae_alignment import VQVAEAlignmentLoss, TrajectoryFeatureExtractor, AlignedFeatureExtractor
from .latent_projector import LatentProjector

# Truncated BPTT for long-horizon training
from .truncated_bptt import TruncatedBPTT

__all__ = [
    # Abstract base classes
    "BaseMNOBackbone",
    "BaseNOALoss",
    "LossOutput",
    # Backbone
    "MNOBackbone",
    # Training losses
    "ContrastiveLoss",
    # Feature extraction
    "MNOFeatureExtractor",
    # In-memory datasets (synthetic / unit-test only; use SpinlockDataset for real data)
    "MNOInMemoryDataset",
    "MNOInMemoryDatasetWithFeatures",
    # Training
    "MNOTrajectoryTrainer",
    # Perceptual losses (VQ-VAE feature-based)
    "VQVAEPerceptualLoss",
    "FeatureProjector",
    "NOALoss",
    # Utilities
    "extract_trajectory_features",
    "generate_synthetic_data",
    # CNO replay for state supervision
    "CNOReplayer",
    # VQ-VAE alignment for token-aligned training
    "VQVAEAlignmentLoss",
    "TrajectoryFeatureExtractor",
    "AlignedFeatureExtractor",
    "LatentProjector",
    # Truncated BPTT
    "TruncatedBPTT",
]
