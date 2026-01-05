"""Hybrid INITIAL encoder combining manual features + end-to-end CNN.

Takes both pre-extracted manual features (14D) and raw ICs, then concatenates:
- Manual features: passed through directly (or MLP encoded)
- CNN features: InitialCNNEncoder trained end-to-end

Total output: manual_dim + cnn_embedding_dim (typically 14 + 28 = 42)
"""

import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseEncoder
from .initial_cnn import InitialCNNEncoder


class InitialHybridEncoder(BaseEncoder):
    """Hybrid encoder for INITIAL features.

    Combines:
    - Pre-extracted manual features (14D): spatial, spectral, information, morphological
    - End-to-end CNN embeddings (28D): InitialCNNEncoder trained jointly with VQ-VAE

    The manual features provide interpretable, domain-driven representations while
    the CNN learns complementary spatial patterns directly from raw ICs.

    Args:
        manual_dim: Dimension of pre-extracted manual features (default: 14)
        cnn_embedding_dim: Output dimension of CNN encoder (default: 28)
        encode_manual: If True, apply MLP to manual features (default: False)
        manual_hidden_dims: Hidden layer sizes for manual MLP (if encode_manual=True)
        manual_output_dim: Output dimension for manual MLP (if encode_manual=True)

    Example:
        >>> encoder = InitialHybridEncoder(manual_dim=14, cnn_embedding_dim=28)
        >>> manual_features = torch.randn(32, 14)  # Pre-extracted
        >>> raw_ics = torch.randn(32, 1, 128, 128)  # Raw initial conditions
        >>> embeddings = encoder(manual_features, raw_ics)  # [32, 42]
    """

    def __init__(
        self,
        manual_dim: int = 14,
        cnn_embedding_dim: int = 28,
        encode_manual: bool = False,
        manual_hidden_dims: Tuple[int, ...] = (64,),
        manual_output_dim: int = 14,
        in_channels: int = 1,
    ):
        super().__init__()

        self.manual_dim = manual_dim
        self.cnn_embedding_dim = cnn_embedding_dim
        self.encode_manual = encode_manual
        self.in_channels = in_channels

        # Manual feature encoder (optional)
        if encode_manual:
            layers = []
            prev_dim = manual_dim
            for hidden_dim in manual_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, manual_output_dim))
            layers.append(nn.BatchNorm1d(manual_output_dim))
            self.manual_encoder = nn.Sequential(*layers)
            self._manual_output_dim = manual_output_dim
        else:
            self.manual_encoder = nn.Identity()
            self._manual_output_dim = manual_dim

        # CNN encoder for raw ICs (trained end-to-end)
        self.cnn_encoder = InitialCNNEncoder(
            embedding_dim=cnn_embedding_dim,
            in_channels=in_channels,
        )

        self._output_dim = self._manual_output_dim + cnn_embedding_dim

    def forward(
        self,
        manual_features: torch.Tensor,
        raw_ics: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass combining manual features and CNN embeddings.

        Args:
            manual_features: Pre-extracted manual features [batch_size, manual_dim]
            raw_ics: Raw initial conditions [batch_size, in_channels, H, W]

        Returns:
            Combined embeddings [batch_size, output_dim]
        """
        # Encode manual features (identity or MLP)
        manual_encoded = self.manual_encoder(manual_features)

        # Encode raw ICs with CNN
        cnn_encoded = self.cnn_encoder(raw_ics)

        # Concatenate
        combined = torch.cat([manual_encoded, cnn_encoded], dim=-1)

        return combined

    @property
    def output_dim(self) -> int:
        """Total output embedding dimension (manual + CNN)."""
        return self._output_dim

    @property
    def cnn_output_dim(self) -> int:
        """CNN embedding dimension only."""
        return self.cnn_embedding_dim

    @property
    def manual_output_dim(self) -> int:
        """Manual features output dimension."""
        return self._manual_output_dim
