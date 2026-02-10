"""Token-to-Rollout VAE: Decode tokens to generative parameters.

This module implements a standalone VAE that learns the inverse mapping:
    tokens → (theta, initial_grids)

The VAE enables Sobol sampling in latent space to generate diverse rollout
variations around a given token embedding.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbeddingExtractor(nn.Module):
    """Extract embeddings from frozen VQTokenizer.

    This module loads a pre-trained VQTokenizer and extracts embeddings for
    given token indices. All dimensions are resolved at runtime from the
    checkpoint.

    Args:
        vq_checkpoint_path: Path to VQTokenizer checkpoint
    """

    def __init__(self, vq_checkpoint_path: Path):
        super().__init__()
        self.vq_checkpoint_path = Path(vq_checkpoint_path)

        # Load frozen tokenizer
        from spinlock.tokens.tokenizer import VQTokenizer

        self.tokenizer = VQTokenizer.from_checkpoint(vq_checkpoint_path)
        self.tokenizer.model.eval()

        # Freeze all parameters
        for param in self.tokenizer.model.parameters():
            param.requires_grad = False

        # Runtime-resolve dimensions
        self.num_token_groups = len(self.tokenizer.model.quantizers)

        # Compute total embedding dim (sum across all quantizers, may vary)
        self.total_embedding_dim = sum(
            q.embedding_dim for q in self.tokenizer.model.quantizers.values()
        )

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings for token indices.

        Args:
            tokens: Dict mapping token keys to indices [B] (96 keys)

        Returns:
            Flattened embeddings [B, total_embedding_dim]
        """
        embeddings = []

        # Extract embedding for each token group
        quantizer_keys = sorted(self.tokenizer.model.quantizers.keys())
        sorted_token_keys = sorted(tokens.keys())

        # Get device from first token tensor
        device = next(iter(tokens.values())).device

        for quantizer_key, token_key in zip(quantizer_keys, sorted_token_keys):
            # Get quantizer for this group
            quantizer = self.tokenizer.model.quantizers[quantizer_key]
            token_indices = tokens[token_key]

            # Move token indices to same device as quantizer embeddings (CPU)
            token_indices_cpu = token_indices.cpu()

            # Lookup embeddings from quantizer's embedding layer
            emb = quantizer.embedding(token_indices_cpu)  # [B, embedding_dim]

            # Move embeddings to target device
            emb = emb.to(device)

            embeddings.append(emb)

        # Concatenate all embeddings
        return torch.cat(embeddings, dim=-1)  # [B, num_token_groups * embedding_dim]


class TokenVAEEncoder(nn.Module):
    """Encode token embeddings to latent distribution.

    Args:
        input_dim: Dimensionality of input embeddings
        latent_dim: Dimensionality of latent space
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Build MLP encoder
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Split to mu and logvar
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution.

        Args:
            x: Input embeddings [B, input_dim]

        Returns:
            Tuple of (mu [B, latent_dim], logvar [B, latent_dim])
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ParameterDecoder(nn.Module):
    """Decode latent to theta parameters.

    Args:
        latent_dim: Dimensionality of latent space
        theta_dim: Dimensionality of theta parameters
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        theta_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.theta_dim = theta_dim

        # Build MLP decoder
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer (no activation, as theta can be any real value)
        layers.append(nn.Linear(prev_dim, theta_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to theta.

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            Theta parameters [B, theta_dim]
        """
        return self.decoder(z)


class GridDecoder(nn.Module):
    """Decode latent to initial condition grids.

    Uses ConvTranspose2d layers to upsample from a small spatial size
    to the target grid resolution.

    Args:
        latent_dim: Dimensionality of latent space
        grid_shape: Target grid shape (C, H, W)
        hidden_channels: List of channel dimensions for upsampling layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        grid_shape: Tuple[int, int, int],
        hidden_channels: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_shape = grid_shape
        C, H, W = grid_shape

        # Start at H//16 × W//16 spatial size
        self.start_h = max(H // 16, 4)
        self.start_w = max(W // 16, 4)
        self.start_channels = hidden_channels[0]

        # Project latent to initial spatial feature map
        self.fc = nn.Linear(latent_dim, self.start_channels * self.start_h * self.start_w)

        # Build upsampling layers (double spatial size each time)
        layers = []
        in_channels = self.start_channels
        for out_channels in hidden_channels[1:]:
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        # Final layer to target channels (no activation)
        layers.append(
            nn.ConvTranspose2d(
                in_channels,
                C,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        self.decoder = nn.Sequential(*layers)

        # Calculate expected output size
        self.expected_h = self.start_h * (2 ** len(hidden_channels))
        self.expected_w = self.start_w * (2 ** len(hidden_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to grids.

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            Initial grids [B, C, H, W]
        """
        # Project to spatial features
        h = self.fc(z)  # [B, start_channels * start_h * start_w]
        h = h.view(-1, self.start_channels, self.start_h, self.start_w)

        # Upsample
        out = self.decoder(h)  # [B, C, expected_h, expected_w]

        # Resize to target if needed (handles rounding issues)
        C, H, W = self.grid_shape
        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out


class TokenToRolloutVAE(nn.Module):
    """Complete VAE: tokens → (theta, initial_grids).

    This VAE enables Sobol sampling in latent space to generate diverse
    rollout variations around a given token embedding.

    Args:
        vq_checkpoint: Path to frozen VQTokenizer checkpoint
        theta_dim: Dimensionality of theta parameters
        grid_shape: Shape of initial grids (C, H, W)
        latent_dim: Dimensionality of latent space
        encoder_hidden_dims: Hidden dimensions for encoder
        param_decoder_hidden_dims: Hidden dimensions for parameter decoder
        grid_decoder_hidden_channels: Channels for grid decoder
        dropout: Dropout probability
    """

    def __init__(
        self,
        vq_checkpoint: Path,
        theta_dim: int,
        grid_shape: Tuple[int, int, int],
        latent_dim: int = 512,
        encoder_hidden_dims: list[int] = [2048, 1024],
        param_decoder_hidden_dims: list[int] = [256, 128],
        grid_decoder_hidden_channels: list[int] = [512, 256, 128, 64, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.grid_shape = grid_shape
        self.latent_dim = latent_dim

        # Frozen embedding extractor
        self.embedding_extractor = TokenEmbeddingExtractor(vq_checkpoint)
        input_dim = self.embedding_extractor.total_embedding_dim

        # VAE components
        self.encoder = TokenVAEEncoder(input_dim, latent_dim, encoder_hidden_dims, dropout)
        self.param_decoder = ParameterDecoder(
            latent_dim, theta_dim, param_decoder_hidden_dims, dropout
        )
        self.grid_decoder = GridDecoder(
            latent_dim, grid_shape, grid_decoder_hidden_channels, dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * exp(0.5 * logvar).

        Args:
            mu: Mean [B, latent_dim]
            logvar: Log variance [B, latent_dim]

        Returns:
            Sampled latent codes [B, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            tokens: Dict mapping token keys to indices [B]

        Returns:
            Dictionary containing:
            - theta: [B, theta_dim] decoded parameters
            - grids: [B, C, H, W] decoded initial grids
            - mu: [B, latent_dim] latent mean
            - logvar: [B, latent_dim] latent log variance
            - z: [B, latent_dim] sampled latent codes
        """
        # Extract embeddings
        embeddings = self.embedding_extractor(tokens)  # [B, total_embedding_dim]

        # Encode to latent distribution
        mu, logvar = self.encoder(embeddings)  # [B, latent_dim]

        # Sample latent codes
        z = self.reparameterize(mu, logvar)  # [B, latent_dim]

        # Decode to theta and grids
        theta = self.param_decoder(z)  # [B, theta_dim]
        grids = self.grid_decoder(z)  # [B, C, H, W]

        return {
            "theta": theta,
            "grids": grids,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def encode(self, tokens: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode tokens to latent distribution.

        Args:
            tokens: Dict mapping token keys to indices

        Returns:
            Tuple of (mu, logvar)
        """
        embeddings = self.embedding_extractor(tokens)
        return self.encoder(embeddings)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent codes to (theta, grids).

        Args:
            z: Latent codes [B, latent_dim]

        Returns:
            Tuple of (theta [B, theta_dim], grids [B, C, H, W])
        """
        theta = self.param_decoder(z)
        grids = self.grid_decoder(z)
        return theta, grids

    @torch.no_grad()
    def sample_from_tokens(
        self,
        tokens: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Sample variations from token latent distribution.

        Args:
            tokens: Dict mapping token keys to indices [1]
            n_samples: Number of samples to generate

        Returns:
            Dictionary containing:
            - theta: [n_samples, theta_dim]
            - grids: [n_samples, C, H, W]
            - z: [n_samples, latent_dim]
        """
        self.eval()

        # Get latent distribution
        mu, logvar = self.encode(tokens)  # [1, latent_dim]

        # Sample multiple z values
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(n_samples, self.latent_dim, device=mu.device)
        z = mu + eps * std  # [n_samples, latent_dim]

        # Decode
        theta, grids = self.decode(z)

        return {
            "theta": theta,
            "grids": grids,
            "z": z,
        }
