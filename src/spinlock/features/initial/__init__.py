"""
INITIAL (Initial Condition) feature family.

Hybrid architecture combining hand-crafted and learned features:
- Manual: 14D interpretable features (spatial, spectral, information, morphological)
- CNN: 28D learned embeddings (ResNet-3 encoder)
- Total: 42D per initial condition

Designed for bidirectional use:
1. Analysis: Extract features from existing initial conditions
2. Generation: Construct initial conditions from embeddings (VAE mode)

This enables the NOA to both understand and construct INITIAL+ARCHITECTURE pairs
that embody its "thoughts".
"""

from .config import (
    InitialConfig,
    InitialManualConfig,
    InitialCNNConfig,
)
from .extractors import InitialExtractor
from .manual_extractors import InitialManualExtractor
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
    'InitialExtractor',
    'InitialManualExtractor',
    # CNN components
    'InitialCNNEncoder',
    'InitialCNNDecoder',
    'InitialVAE',
    'ResidualBlock',
]
