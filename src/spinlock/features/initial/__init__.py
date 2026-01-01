"""
IC (Initial Condition) feature family.

Hybrid architecture combining hand-crafted and learned features:
- Manual: 14D interpretable features (spatial, spectral, information, morphological)
- CNN: 28D learned embeddings (ResNet-3 encoder)
- Total: 42D per IC

Designed for bidirectional use:
1. Analysis: Extract features from existing ICs
2. Generation: Construct ICs from embeddings (VAE mode)

This enables the NOA to both understand and construct IC+NO pairs
that embody its "thoughts".
"""

from .config import (
    InitialConfig,
    InitialManualConfig,
    InitialCNNConfig,
)
from .manual_extractors import ICManualExtractor
from .cnn_encoder import (
    InitialCNNEncoder,
    InitialCNNDecoder,
    InitialVAE,
    ResidualBlock,
)

__all__ = [
    # Config
    'InitialConfig',
    'InitialManualConfig',
    'InitialCNNConfig',
    # Extractors
    'ICManualExtractor',
    # CNN components
    'InitialCNNEncoder',
    'InitialCNNDecoder',
    'InitialVAE',
    'ResidualBlock',
]
