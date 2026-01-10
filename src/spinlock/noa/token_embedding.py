"""Token embedding for conditioning MNO on VQ-VAE discrete codes."""

import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """Embed VQ token indices into continuous vectors.

    Each of N×L tokens has its own embedding table with vocabulary size K_i.
    Embeddings are concatenated and projected to a fixed dimension.

    Args:
        num_tokens: Total number of tokens (N×L, e.g., 21 for 7 categories × 3 levels)
        codebook_sizes: List of vocabulary sizes [K_0, K_1, ..., K_{N×L-1}]
        embed_dim: Embedding dimension per token (default: 32)
        projection_dim: Final projected dimension after concatenation (default: 64)

    Input:
        tokens: [B, num_tokens] integer indices in range [0, K_i-1] per token

    Output:
        [B, projection_dim] continuous embedding vector
    """

    def __init__(
        self,
        num_tokens: int,
        codebook_sizes: list[int],
        embed_dim: int = 32,
        projection_dim: int = 64,
    ):
        super().__init__()
        assert len(codebook_sizes) == num_tokens, \
            f"codebook_sizes must have length {num_tokens}, got {len(codebook_sizes)}"

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        # Separate embedding table per token
        self.embeddings = nn.ModuleList([
            nn.Embedding(K, embed_dim) for K in codebook_sizes
        ])

        # Project concatenated embeddings to fixed dimension
        self.projection = nn.Linear(num_tokens * embed_dim, projection_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token indices.

        Args:
            tokens: [B, num_tokens] integer indices

        Returns:
            [B, projection_dim] embedded tokens
        """
        B = tokens.shape[0]
        assert tokens.shape == (B, self.num_tokens), \
            f"Expected tokens shape [{B}, {self.num_tokens}], got {tokens.shape}"

        # Embed each token independently
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            token_idx = tokens[:, i]  # [B]
            embedded.append(emb_layer(token_idx))  # [B, embed_dim]

        # Concatenate all embeddings
        concat = torch.cat(embedded, dim=-1)  # [B, num_tokens * embed_dim]

        # Project to final dimension
        return self.projection(concat)  # [B, projection_dim]

    def initialize_from_vqvae(self, vqvae_checkpoint_path: str):
        """Initialize embeddings from VQ-VAE codebooks (optional).

        Args:
            vqvae_checkpoint_path: Path to VQ-VAE checkpoint with codebooks
        """
        # Load VQ-VAE
        checkpoint = torch.load(vqvae_checkpoint_path, map_location='cpu')
        vqvae_state = checkpoint['model_state_dict']

        # Extract codebook embeddings (one per VQ layer)
        for i, emb_layer in enumerate(self.embeddings):
            codebook_key = f"vq_layers.{i}.codebook"
            if codebook_key in vqvae_state:
                codebook = vqvae_state[codebook_key]  # [K_i, D_i]
                K_i, D_i = codebook.shape

                # Project codebook to embed_dim if dimensions differ
                if D_i == self.embed_dim:
                    emb_layer.weight.data.copy_(codebook)
                else:
                    # Use random projection
                    proj_matrix = torch.randn(D_i, self.embed_dim) / (D_i ** 0.5)
                    projected = codebook @ proj_matrix  # [K_i, embed_dim]
                    emb_layer.weight.data.copy_(projected)

                print(f"  Initialized token {i} embedding from VQ codebook (size {K_i})")
