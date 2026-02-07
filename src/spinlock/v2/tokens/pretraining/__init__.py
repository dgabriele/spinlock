"""Encoder pretraining module.

Pretrain encoders via autoencoder reconstruction tasks before using them in the
VQ-VAE. This helps initialize encoders with meaningful features rather than
random initialization.

Available pretrainers:
- CNNPretrainer: Pretrain InitialCNNEncoder for spatial features
- TemporalPretrainer: Pretrain PyramidTemporalEncoder for temporal sequences

Example:
    >>> from spinlock.v2.tokens.pretraining import CNNPretrainer, TemporalPretrainer
    >>> from spinlock.v2.tokens.config import PretrainingConfig, TemporalPretrainingConfig
    >>>
    >>> # CNN pretraining
    >>> config = PretrainingConfig(embedding_dim=256, num_epochs=50)
    >>> pretrainer = CNNPretrainer(config)
    >>> pretrainer.train(dataset, output_path="cnn_pretrained.pt")
    >>>
    >>> # Temporal pretraining
    >>> config = TemporalPretrainingConfig(num_epochs=100)
    >>> pretrainer = TemporalPretrainer(config)
    >>> pretrainer.train(temporal_features, output_path="temporal_pretrained.pt")
"""

from .cnn_pretrainer import CNNPretrainer
from .temporal_pretrainer import TemporalPretrainer

__all__ = ["CNNPretrainer", "TemporalPretrainer"]
