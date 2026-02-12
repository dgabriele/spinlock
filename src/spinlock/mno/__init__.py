"""MNO (Meta Neural Operator) - Neural Operator Implementation.

This module implements the MNO architecture with two training paradigms:

MSE-led (Physics First):
    - Trajectory matching (L_traj) as primary loss
    - VQ alignment losses as auxiliary regularizers
    - Use when exact physics fidelity is required

VQ-led (Creative Observer):
    - VQ reconstruction (L_recon) as primary loss
    - Trajectory matching as auxiliary regularizer
    - Enables "creative" exploration where meaningful deviation is allowed

Core Components:
    - BaseMNOBackbone: Abstract interface for autoregressive neural operators
    - BaseNOALoss: Abstract interface for training losses with LossOutput format
    - MNOBackbone: U-AFNO implementation with gradient checkpointing
    - MSELedLoss: Physics-first training objective
    - VQLedLoss: Symbolic coherence training objective

All dimensions are resolved dynamically at runtime.
"""

# Abstract base classes
from .base_backbone import BaseMNOBackbone
from .base_loss import BaseNOALoss, LossOutput

# Concrete backbone
from .backbone import MNOBackbone

# Training losses (two paradigms)
from .losses import MSELedLoss, VQLedLoss

# Perceptual losses (VQ-VAE feature-based)
from .perceptual_losses import VQVAEPerceptualLoss, FeatureProjector, NOALoss

# Feature extraction
from .feature_extraction import MNOFeatureExtractor

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

# Truncated BPTT for long-horizon training
from .truncated_bptt import TruncatedBPTT

__all__ = [
    # Abstract base classes
    "BaseMNOBackbone",
    "BaseNOALoss",
    "LossOutput",
    # Backbone
    "MNOBackbone",
    # Training losses (two paradigms)
    "MSELedLoss",
    "VQLedLoss",
    # Feature extraction (real data)
    "MNOFeatureExtractor",
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
    # Truncated BPTT
    "TruncatedBPTT",
]
