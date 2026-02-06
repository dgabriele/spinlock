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
        in_channels: Input channels for CNN encoder (default: 1)
        use_final_batchnorm: Apply BatchNorm1d to final CNN embedding (default: False).
            Setting to False allows natural variance preservation.

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
        use_final_batchnorm: bool = False,
        pretrained_cnn_path: str = None,
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
            use_final_batchnorm=use_final_batchnorm,
        )

        # Load pre-trained weights if provided
        if pretrained_cnn_path:
            self._load_pretrained_cnn(pretrained_cnn_path)

        self._output_dim = self._manual_output_dim + cnn_embedding_dim

    def _load_pretrained_cnn(self, checkpoint_path: str):
        """Load pre-trained CNN encoder weights from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file containing encoder weights
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Pre-trained checkpoint not found: {checkpoint_path}")

        print(f"Loading pre-trained CNN encoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # The checkpoint contains 'encoder_state_dict' key from pre-training
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
            embedding_dim = checkpoint.get('embedding_dim', None)

            # Verify embedding dimension matches
            if embedding_dim and embedding_dim != self.cnn_embedding_dim:
                raise ValueError(
                    f"Pre-trained embedding_dim ({embedding_dim}) does not match "
                    f"config embedding_dim ({self.cnn_embedding_dim})"
                )

            # Load weights
            self.cnn_encoder.load_state_dict(state_dict)
            print(f"✓ Loaded pre-trained CNN encoder (embedding_dim={self.cnn_embedding_dim})")

            if 'val_loss' in checkpoint:
                print(f"  Pre-training val_loss: {checkpoint['val_loss']:.6f}")
            if 'epoch' in checkpoint:
                print(f"  Pre-training epoch: {checkpoint['epoch']}")
        else:
            raise KeyError(f"Checkpoint does not contain 'encoder_state_dict' key")

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
