"""
Modular encoder registry for VQ-VAE feature family processing.

Encoders transform raw features or data into embeddings suitable for VQ-VAE tokenization.
Each feature family (SDF, NOP, IC) can specify its own encoder in config.

Available Encoders:
- IdentityEncoder: Pass-through (no transformation)
- MLPEncoder: MLP with configurable hidden layers
- ICCNNEncoder: ResNet-3 CNN for initial condition spatial processing
- TDCNNEncoder: ResNet-1D CNN for temporal dynamics encoding

Example Config:
    families:
      sdf:
        encoder: MLPEncoder
        encoder_params:
          hidden_dims: [128, 64]

      ic:
        encoder: ICCNNEncoder
        encoder_params:
          embedding_dim: 28
"""

import torch.nn as nn
from typing import Dict, Type, Any

from .base import BaseEncoder
from .identity import IdentityEncoder
from .mlp import MLPEncoder
from .ic_cnn import ICCNNEncoder
from .td_cnn import TDCNNEncoder


# Encoder Registry
_ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {
    "identity": IdentityEncoder,
    "IdentityEncoder": IdentityEncoder,
    "mlp": MLPEncoder,
    "MLPEncoder": MLPEncoder,
    "ic_cnn": ICCNNEncoder,
    "ICCNNEncoder": ICCNNEncoder,
    "td_cnn": TDCNNEncoder,
    "TDCNNEncoder": TDCNNEncoder,
}


def register_encoder(name: str, encoder_class: Type[BaseEncoder]) -> None:
    """Register a custom encoder class.

    Args:
        name: Encoder name for config reference
        encoder_class: Encoder class (must inherit from BaseEncoder)
    """
    if not issubclass(encoder_class, BaseEncoder):
        raise ValueError(f"Encoder {encoder_class} must inherit from BaseEncoder")
    _ENCODER_REGISTRY[name] = encoder_class


def get_encoder(name: str, **params: Any) -> BaseEncoder:
    """Get encoder instance by name.

    Args:
        name: Encoder name from config
        **params: Encoder-specific parameters

    Returns:
        Initialized encoder instance

    Raises:
        ValueError: If encoder name not found
    """
    if name not in _ENCODER_REGISTRY:
        available = ', '.join(sorted(_ENCODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown encoder: '{name}'. Available encoders: {available}"
        )

    encoder_class = _ENCODER_REGISTRY[name]
    return encoder_class(**params)


__all__ = [
    "BaseEncoder",
    "IdentityEncoder",
    "MLPEncoder",
    "ICCNNEncoder",
    "TDCNNEncoder",
    "register_encoder",
    "get_encoder",
]
