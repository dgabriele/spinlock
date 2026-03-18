"""VQ Tokenizer encoder modules.

Provides temporal and initial feature encoders for the VQ-VAE system.
These encoders handle multi-family feature encoding with support for:
- Variable-length temporal sequences (pyramid encoder)
- Hybrid manual + CNN initial conditions
- End-to-end CNN training

Example:
    >>> from spinlock.tokens.encoders import PyramidTemporalEncoder, InitialHybridEncoder
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
    FrameCNNEncoder,
)
from .initial_spectral import SpectralICEncoder
from .initial_spatial import SpatialICEncoder

from .temporal_cnn_feature import TemporalCNNFeatureEncoder
from .pyramid_first import PyramidFirstEncoder

__all__ = [
    # Temporal encoders
    "TemporalMeanEncoder",
    "TemporalCNNEncoder",
    "PyramidTemporalEncoder",
    "TemporalCNNFeatureEncoder",
    "PyramidFirstEncoder",
    # Initial encoders
    "InitialCNNEncoder",
    "InitialHybridEncoder",
    "FrameCNNEncoder",
    "SpectralICEncoder",
    "SpatialICEncoder",
]
