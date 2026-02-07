"""V2 Tokens package - DEPRECATED.

The V2 VQTokenizer system has been promoted to the primary tokens package.

DEPRECATED: Import from spinlock.tokens instead of spinlock.v2.tokens
"""

import warnings

warnings.warn(
    "spinlock.v2.tokens is deprecated. "
    "Import from spinlock.tokens instead:\n"
    "  Old: from spinlock.v2.tokens import VQTokenizer\n"
    "  New: from spinlock.tokens import VQTokenizer\n"
    "\n"
    "This deprecation shim will be removed in 6 months (August 2026).",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from spinlock.tokens import (
    VQTokenizer,
    TokenizerConfig,
    JointHierarchicalVQVAE,
    VQTokenizerTrainer,
    VQVAELoss,
    EncoderConfig,
    InitialEncoderConfig,
    TemporalEncoderConfig,
    QuantizerConfig,
    HierarchyConfig,
    LossConfig,
    TrainingConfig,
    NormalizationConfig,
    PretrainingConfig,
    save_checkpoint,
    load_checkpoint,
    verify_pretrained_cnn,
    CNNPretrainer,
)

__all__ = [
    "VQTokenizer",
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
    "JointHierarchicalVQVAE",
    "VQTokenizerTrainer",
    "VQVAELoss",
    "save_checkpoint",
    "load_checkpoint",
    "verify_pretrained_cnn",
    "CNNPretrainer",
]
