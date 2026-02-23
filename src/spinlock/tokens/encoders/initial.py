"""Initial condition encoders for tokenizer.

Re-exports V1 initial encoders for use in tokens package.
These encoders are well-designed and tested, no need to reimplement.

Available encoders:
- InitialCNNEncoder: ResNet-3 CNN encoder for spatial IC processing
- InitialHybridEncoder: Hybrid encoder combining manual features + end-to-end CNN
- FrameCNNEncoder: ResNet-3 encoder for per-frame trajectory feature extraction
"""

from spinlock.encoding.encoders.initial_cnn import InitialCNNEncoder
from spinlock.encoding.encoders.initial_hybrid import InitialHybridEncoder
from spinlock.features.initial.cnn_encoder import FrameCNNEncoder

__all__ = [
    "InitialCNNEncoder",
    "InitialHybridEncoder",
    "FrameCNNEncoder",
]
