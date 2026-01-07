"""NOA (Neural Operator Agent) - Neural Operator Agent Implementation.

This module implements the NOA architecture with two training paradigms:

MSE-led (Physics First):
    - Trajectory matching (L_traj) as primary loss
    - VQ alignment losses as auxiliary regularizers
    - Use when exact physics fidelity is required

VQ-led (Creative Observer):
    - VQ reconstruction (L_recon) as primary loss
    - Trajectory matching as auxiliary regularizer
    - Enables "creative" exploration where meaningful deviation is allowed

Core Components:
    - BaseNOABackbone: Abstract interface for autoregressive neural operators
    - BaseNOALoss: Abstract interface for training losses with LossOutput format
    - NOABackbone: U-AFNO implementation with gradient checkpointing
    - MSELedLoss: Physics-first training objective
    - VQLedLoss: Symbolic coherence training objective

All dimensions are resolved dynamically at runtime.
"""

# Abstract base classes
from .base_backbone import BaseNOABackbone
from .base_loss import BaseNOALoss, LossOutput

# Concrete backbone
from .backbone import NOABackbone

# Training losses (two paradigms)
from .losses import MSELedLoss, VQLedLoss

# Perceptual losses (VQ-VAE feature-based)
from .perceptual_losses import VQVAEPerceptualLoss, FeatureProjector, NOALoss

# Feature extraction
from .feature_extraction import NOAFeatureExtractor

# Datasets
from .dataset import NOARealDataset, NOARealDatasetStreaming
from .training import (
    NOAPhase1Trainer,
    NOADataset,
    NOADatasetWithFeatures,
    NOARealDataTrainer,
    extract_trajectory_features,
    generate_synthetic_data,
)

# CNO replay and VQ alignment
from .cno_replay import CNOReplayer
from .vqvae_alignment import VQVAEAlignmentLoss, TrajectoryFeatureExtractor, AlignedFeatureExtractor
from .latent_projector import LatentProjector

__all__ = [
    # Abstract base classes
    "BaseNOABackbone",
    "BaseNOALoss",
    "LossOutput",
    # Backbone
    "NOABackbone",
    # Training losses (two paradigms)
    "MSELedLoss",
    "VQLedLoss",
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
    # Perceptual losses (VQ-VAE feature-based)
    "VQVAEPerceptualLoss",
    "FeatureProjector",
    "NOALoss",
    # Utilities (legacy/synthetic)
    "extract_trajectory_features",
    "generate_synthetic_data",
    # CNO replay for state supervision
    "CNOReplayer",
    # VQ-VAE alignment for token-aligned training
    "VQVAEAlignmentLoss",
    "TrajectoryFeatureExtractor",
    "AlignedFeatureExtractor",
    "LatentProjector",
]
