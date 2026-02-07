"""CNN encoder pretraining module.

Pretrain the InitialCNNEncoder via autoencoder reconstruction task before
using it in the VQ-VAE. This helps initialize the CNN with meaningful spatial
features rather than random initialization.

Example:
    >>> from spinlock.tokens.pretraining import CNNPretrainer
    >>> from spinlock.tokens import PretrainingConfig
    >>>
    >>> config = PretrainingConfig(embedding_dim=256, num_epochs=50)
    >>> pretrainer = CNNPretrainer(config)
    >>> pretrainer.train(dataset, output_path="cnn_pretrained.pt")
"""

from .cnn_pretrainer import CNNPretrainer

__all__ = ["CNNPretrainer"]
