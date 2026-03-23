"""Tokenizer package — VQ (discrete) and NL (continuous VAE + LFM).

This package provides modular tokenizer implementations for encoding
Lenia dynamics into latent representations:

1. **VQTokenizer**: Discrete codebook tokens via hierarchical VQ-VAE.
   Used with D3PM for inverse generation.
2. **NLTokenizer**: Continuous VAE latents projected into LFM's frozen
   autoregressive decoder for natural language expression generation.

Both share the family encoder pattern (temporal, IC, theta) via base classes.

Example (VQ):
    >>> from spinlock.tokens import VQTokenizer, TokenizerConfig
    >>> tokenizer = VQTokenizer(config)
    >>> tokenizer.train(dataset, output_dir="checkpoints/")

Example (NL):
    >>> from spinlock.tokens import NLTokenizer, NLTokenizerConfig
    >>> tokenizer = NLTokenizer(config)
    >>> tokenizer.train(dataset, output_dir="checkpoints/nl/")
    >>> text = tokenizer.generate_text(z)
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

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    verify_pretrained_cnn,
    TokenizerCheckpoint,
)

from .pretraining import CNNPretrainer
from .schema import TokenSchema, CategoryLevelInfo

# Base classes
from .base_model import BaseTokenizerModel
from .base_tokenizer import BaseTokenizer
from .base_trainer import BaseTokenizerTrainer

# NL tokenizer
from .nl_config import NLTokenizerConfig
from .nl_tokenizer import NLTokenizer
from .nl_model import NLTokenizerModel
from .nl_lfm_adapter import LFMAdapter, NLListener, NLTokenBridge

__all__ = [
    # Main interfaces
    "VQTokenizer",
    "NLTokenizer",

    # Schema
    "TokenSchema",
    "CategoryLevelInfo",

    # Base classes
    "BaseTokenizerModel",
    "BaseTokenizer",
    "BaseTokenizerTrainer",

    # VQ configuration
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

    # NL configuration
    "NLTokenizerConfig",

    # VQ core components
    "JointHierarchicalVQVAE",
    "VQTokenizerTrainer",
    "VQVAELoss",

    # NL core components
    "NLTokenizerModel",
    "LFMAdapter",
    "NLListener",
    "NLTokenBridge",

    # Utilities
    "save_checkpoint",
    "load_checkpoint",
    "verify_pretrained_cnn",
    "TokenizerCheckpoint",

    # Pretraining
    "CNNPretrainer",
]
