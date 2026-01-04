"""VQ-VAE visualization module.

Provides comprehensive visualizations for trained Categorical Hierarchical VQ-VAE models:

1. Engineering Dashboard - Model structure, training metrics, codebook health
2. Semantic Dashboard - Feature-to-category mappings, token meanings
3. Topological Dashboard - t-SNE codebook embeddings, usage heatmap, similarity matrix

Usage:
    from spinlock.visualization.vqvae import (
        create_engineering_dashboard,
        create_semantic_dashboard,
        create_topological_dashboard,
    )

    create_topological_dashboard(
        checkpoint_path="checkpoints/production/100k_full_features/",
        output_path="visualizations/vqvae_topology.png"
    )
"""

from .engineering_dashboard import create_engineering_dashboard
from .semantic_dashboard import create_semantic_dashboard
from .topological_dashboard import create_topological_dashboard
from .utils import load_vqvae_checkpoint, VQVAECheckpointData

__all__ = [
    "create_engineering_dashboard",
    "create_semantic_dashboard",
    "create_topological_dashboard",
    "load_vqvae_checkpoint",
    "VQVAECheckpointData",
]
