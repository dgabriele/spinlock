"""Temporal feature encoders for tokenizer.

Re-exports V1 temporal encoders for use in tokens package.
These encoders are well-designed and tested, no need to reimplement.

Available encoders:
- TemporalMeanEncoder: Simple mean pooling over time dimension
- TemporalCNNEncoder: ResNet-1D CNN encoder for temporal sequences
- PyramidTemporalEncoder: Multi-resolution pyramid encoder with shared backbone
"""

from spinlock.encoding.encoders.temporal_mean import TemporalMeanEncoder
from spinlock.encoding.encoders.temporal_cnn import TemporalCNNEncoder
from spinlock.encoding.encoders.pyramid_temporal import PyramidTemporalEncoder

__all__ = [
    "TemporalMeanEncoder",
    "TemporalCNNEncoder",
    "PyramidTemporalEncoder",
]
