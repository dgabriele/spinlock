"""Encoding package for VQ-VAE tokenization.

This package provides categorical hierarchical VQ-VAE implementation
for operator behavioral feature tokenization. Features include:

- Data-driven category discovery via hierarchical clustering
- Per-category MLP encoders/decoders
- 3-level hierarchical quantization (coarse → medium → fine)
- Intelligent formula-based defaults for latent dims and token counts
- Normalization utilities with save/load support
- Complete training pipeline with callbacks and loss functions

Ported from unisim.system (100% generic, adapted for spinlock).

Main components:
- CategoricalHierarchicalVQVAE: Main model
- CategoricalVQVAEConfig: Configuration dataclass
- DynamicCategoryAssignment: Auto-discovery via clustering
- VectorQuantizer: Core VQ layer
- VQVAETrainer: Training loop with callbacks

Usage:
    >>> from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
    >>> from spinlock.encoding import DynamicCategoryAssignment, VQVAETrainer
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Auto-discover categories from features
    >>> assigner = DynamicCategoryAssignment(num_categories=None)  # Auto-determine
    >>> group_indices = assigner.assign_categories(feature_names, features)
    >>>
    >>> # Create VQ-VAE config
    >>> config = CategoricalVQVAEConfig(
    ...     input_dim=features.shape[1],
    ...     group_indices=group_indices,
    ...     group_embedding_dim=64,
    ... )
    >>>
    >>> # Create model
    >>> model = CategoricalHierarchicalVQVAE(config)
    >>>
    >>> # Train model
    >>> trainer = VQVAETrainer(model, train_loader, val_loader)
    >>> history = trainer.train(epochs=500)
"""

from .categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from .category_assignment import CategoryAssignment, DynamicCategoryAssignment
from .clustering_assignment import (
    hierarchical_clustering_assignment,
    auto_determine_num_clusters,
    validate_cluster_orthogonality,
    get_cluster_statistics,
)
from .vector_quantizer import VectorQuantizer, compute_codebook_metrics
from .normalization import (
    NormalizationStats,
    RobustNormalizationStats,
    standard_normalize,
    robust_normalize,
    l2_normalize,
    compute_normalization_stats,
    compute_robust_normalization_stats,
    apply_standard_normalization,
    apply_robust_normalization,
    save_normalization_stats,
    load_normalization_stats,
)
from .feature_processor import FeatureProcessor
from .latent_dim_defaults import (
    compute_default_latent_dims,
    compute_default_num_tokens,
    fill_missing_latent_dims,
    fill_missing_num_tokens,
    parse_compression_ratios,
)
from .grouped_feature_extractor import GroupedFeatureExtractor, GroupMLP
from .categorical_projector import CategoricalProjector
from .training import VQVAETrainer, EarlyStopping, DeadCodeReset, Checkpointer

__all__ = [
    # Main model
    "CategoricalHierarchicalVQVAE",
    "CategoricalVQVAEConfig",
    # Category assignment
    "CategoryAssignment",
    "DynamicCategoryAssignment",
    # Clustering
    "hierarchical_clustering_assignment",
    "auto_determine_num_clusters",
    "validate_cluster_orthogonality",
    "get_cluster_statistics",
    # Vector quantization
    "VectorQuantizer",
    "compute_codebook_metrics",
    # Normalization
    "NormalizationStats",
    "RobustNormalizationStats",
    "standard_normalize",
    "robust_normalize",
    "l2_normalize",
    "compute_normalization_stats",
    "compute_robust_normalization_stats",
    "apply_standard_normalization",
    "apply_robust_normalization",
    "save_normalization_stats",
    "load_normalization_stats",
    # Feature preprocessing
    "FeatureProcessor",
    # Latent dimension defaults
    "compute_default_latent_dims",
    "compute_default_num_tokens",
    "fill_missing_latent_dims",
    "fill_missing_num_tokens",
    "parse_compression_ratios",
    # Encoders/decoders
    "GroupedFeatureExtractor",
    "GroupMLP",
    "CategoricalProjector",
    # Training
    "VQVAETrainer",
    "EarlyStopping",
    "DeadCodeReset",
    "Checkpointer",
]
