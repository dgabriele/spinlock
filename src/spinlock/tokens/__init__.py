"""VQ Tokenizer for trajectory encoding.

This package provides a clean, modular implementation of hierarchical VQ-VAE
for multi-family trajectory tokenization with:
- Joint temporal + initial feature training
- Variable-length temporal sequence support
- End-to-end CNN training for initial conditions
- Integration with feature grouping system
- Clean V2-only checkpoint format

Example:
    >>> from spinlock.tokens import VQTokenizer, TokenizerConfig
    >>>
    >>> # Training
    >>> config = TokenizerConfig(...)
    >>> tokenizer = VQTokenizer(config)
    >>> tokenizer.train(dataset, output_dir="checkpoints/")
    >>>
    >>> # Inference
    >>> tokenizer = VQTokenizer.from_checkpoint("checkpoints/best.pt")
    >>> tokens = tokenizer.tokenize(trajectories)
"""

from .config import (
    TokenizerConfig,
    EncoderConfig,
    InitialEncoderConfig,
    TemporalEncoderConfig,
    QuantizerConfig,
    HierarchyConfig,
    LossConfig,
    TrainingConfig,
    NormalizationConfig,
    PretrainingConfig,
)

from .tokenizer import VQTokenizer
from .model import JointHierarchicalVQVAE
from .trainer import VQTokenizerTrainer
from .losses import VQVAELoss

from .checkpoint import save_checkpoint, load_checkpoint, verify_pretrained_cnn

from .pretraining import CNNPretrainer

__all__ = [
    # Main interface
    "VQTokenizer",

    # Configuration
    "TokenizerConfig",
    "EncoderConfig",
    "InitialEncoderConfig",
    "TemporalEncoderConfig",
    "QuantizerConfig",
    "HierarchyConfig",
    "LossConfig",
    "TrainingConfig",
    "NormalizationConfig",
    "PretrainingConfig",

    # Core components
    "JointHierarchicalVQVAE",
    "VQTokenizerTrainer",
    "VQVAELoss",

    # Utilities
    "save_checkpoint",
    "load_checkpoint",
    "verify_pretrained_cnn",

    # Pretraining
    "CNNPretrainer",
]
