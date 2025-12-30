"""Training package for VQ-VAE tokenization.

Provides training infrastructure for categorical hierarchical VQ-VAE:
- VQVAETrainer: Main training loop with callbacks
- Loss functions: 5-component loss (reconstruction, VQ, orthogonality, informativeness, topographic)
- Callbacks: Early stopping, dead code reset, checkpointing

Ported from unisim.system.training (simplified, removed multimodal support).

Usage:
    >>> from spinlock.encoding.training import VQVAETrainer
    >>> from spinlock.encoding import CategoricalHierarchicalVQVAE
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # Create model and data loaders
    >>> model = CategoricalHierarchicalVQVAE(config)
    >>> train_loader = DataLoader(train_dataset, batch_size=512)
    >>> val_loader = DataLoader(val_dataset, batch_size=512)
    >>>
    >>> # Create trainer
    >>> trainer = VQVAETrainer(
    ...     model=model,
    ...     train_loader=train_loader,
    ...     val_loader=val_loader,
    ...     learning_rate=1e-3,
    ...     checkpoint_dir="checkpoints/vqvae",
    ... )
    >>>
    >>> # Train
    >>> history = trainer.train(epochs=500)
"""

from .trainer import VQVAETrainer
from .callbacks import EarlyStopping, DeadCodeReset, Checkpointer
from .losses import (
    reconstruction_loss,
    orthogonality_loss,
    informativeness_loss,
    topographic_similarity_loss,
    compute_total_loss,
)

__all__ = [
    # Trainer
    "VQVAETrainer",
    # Callbacks
    "EarlyStopping",
    "DeadCodeReset",
    "Checkpointer",
    # Loss functions
    "reconstruction_loss",
    "orthogonality_loss",
    "informativeness_loss",
    "topographic_similarity_loss",
    "compute_total_loss",
]
