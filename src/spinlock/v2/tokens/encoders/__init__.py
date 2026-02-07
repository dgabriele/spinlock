"""V2 encoder re-exports from V1.

V2 reuses the well-designed and tested V1 encoders for temporal and initial features.
This module provides a clean namespace for V2 imports while maintaining compatibility
with the existing V1 encoder implementations.

Example:
    >>> from spinlock.v2.tokens.encoders import PyramidTemporalEncoder, InitialHybridEncoder
    >>> temporal_encoder = PyramidTemporalEncoder(input_dim=345, level_dims=[32, 64, 96, 128])
    >>> initial_encoder = InitialHybridEncoder(manual_dim=42, cnn_embedding_dim=256)
"""

from .temporal import (
    TemporalMeanEncoder,
    TemporalCNNEncoder,
    PyramidTemporalEncoder,
)

from .initial import (
    InitialCNNEncoder,
    InitialHybridEncoder,
)

__all__ = [
    # Temporal encoders
    "TemporalMeanEncoder",
    "TemporalCNNEncoder",
    "PyramidTemporalEncoder",
    # Initial encoders
    "InitialCNNEncoder",
    "InitialHybridEncoder",
]
