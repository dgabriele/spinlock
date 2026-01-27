"""Residual decoder blocks for improved VQ-VAE reconstruction.

Multi-layer residual decoder with skip connections prevents gradient vanishing
and allows deeper networks for better reconstruction quality.
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualDecoderBlock(nn.Module):
    """Residual block with skip connection for decoder.

    Architecture:
        x -> Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> LayerNorm -> + residual

    Args:
        dim_in: Input dimension
        dim_hidden: Hidden dimension (typically larger than dim_in/dim_out)
        dim_out: Output dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
            nn.LayerNorm(dim_out),
        )

        # Projection for residual connection if dimensions don't match
        self.proj = (
            nn.Linear(dim_in, dim_out)
            if dim_in != dim_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch, dim_in]

        Returns:
            Output tensor [batch, dim_out]
        """
        return self.net(x) + self.proj(x)


class MultiLayerResidualDecoder(nn.Module):
    """Multi-layer residual decoder for VQ-VAE reconstruction.

    4-layer architecture: latent → 512 → 768 → 512 → input_dim
    Each layer has residual connections and layer normalization.

    Args:
        latent_dim: Dimension of concatenated quantized latent vectors
        input_dim: Dimension of reconstructed input features
        hidden_dims: List of hidden dimensions for each layer
        dropout: Dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Default architecture: latent → 512 → 768 → 512 → input
        if hidden_dims is None:
            hidden_dims = [512, 768, 512]

        layers = []

        # Input layer: latent_dim → hidden_dims[0]
        layers.append(
            ResidualDecoderBlock(
                latent_dim,
                hidden_dims[0],
                hidden_dims[0],
                dropout
            )
        )

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(
                ResidualDecoderBlock(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    hidden_dims[i+1],
                    dropout
                )
            )

        # Output layer: hidden_dims[-1] → input_dim (no residual)
        layers.append(nn.Linear(hidden_dims[-1], input_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual decoder.

        Args:
            x: Concatenated quantized latent vectors [batch, latent_dim]

        Returns:
            Reconstructed features [batch, input_dim]
        """
        return self.decoder(x)


class AttentionDecoder(nn.Module):
    """Decoder with multi-head attention for selective latent code focus.

    Uses cross-attention to allow intermediate decoder layers to selectively
    attend to relevant parts of the quantized latent representation.

    Args:
        latent_dim: Dimension of concatenated quantized latent vectors
        input_dim: Dimension of reconstructed input features
        hidden_dim: Hidden dimension for decoder
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        input_dim: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Initial projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Decoder with residual blocks
        self.decoder = nn.Sequential(
            ResidualDecoderBlock(hidden_dim, hidden_dim * 2, hidden_dim, dropout),
            ResidualDecoderBlock(hidden_dim, hidden_dim, hidden_dim // 2, dropout),
            nn.Linear(hidden_dim // 2, input_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention decoder.

        Args:
            x: Concatenated quantized latent vectors [batch, latent_dim]

        Returns:
            Reconstructed features [batch, input_dim]
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # [batch, hidden_dim]

        # Add sequence dimension for attention (treat as single-token sequence)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # [batch, 1, hidden_dim]

        # Residual connection and layer norm
        x = self.layer_norm(x + attn_out)

        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, hidden_dim]

        # Decode to input dimension
        return self.decoder(x)
