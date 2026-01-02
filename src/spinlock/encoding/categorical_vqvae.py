"""Categorical Hierarchical VQ-VAE for semantic tokenization.

Produces N×L independent tokens (N categories × L hierarchical levels)
for operator behavioral feature tokenization.

Architecture:
1. GroupedFeatureExtractor: input → N category embeddings
2. CategoricalProjector: N embeddings → N×L latent vectors
3. N×L VectorQuantizers: N×L latent vectors → N×L discrete tokens
4. Shared decoder: concat(N×L quantized vectors) → reconstructed input

Ported from unisim.system.models.categorical_vqvae (100% generic, simplified).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass

from .vector_quantizer import VectorQuantizer
from .grouped_feature_extractor import GroupedFeatureExtractor
from .categorical_projector import CategoricalProjector
from .latent_dim_defaults import (
    fill_missing_latent_dims,
    compute_default_latent_dims,
    fill_missing_num_tokens,
    compute_default_num_tokens,
)


@dataclass
class CategoricalVQVAEConfig:
    """Configuration for categorical hierarchical VQ-VAE.

    Attributes:
        input_dim: Input feature dimension
        group_indices: Dict mapping category -> feature indices
        group_embedding_dim: Embedding dimension for each category
        group_hidden_dim: Hidden dimension for category MLPs
        levels: List of level configs (uniform across categories)
        category_levels: Dict mapping category -> list of levels (overrides 'levels')
        commitment_cost: VQ-VAE commitment cost
        orthogonality_weight: Weight for orthogonality loss
        informativeness_weight: Weight for informativeness loss
        use_ema: Whether to use EMA for codebook updates
        decay: EMA decay rate
        dropout: Dropout rate
    """

    input_dim: int
    group_indices: Dict[str, List[int]]
    group_embedding_dim: int = 64
    group_hidden_dim: int = 128
    levels: Optional[List[Dict]] = None  # Uniform levels across categories
    category_levels: Optional[
        Dict[str, List[Dict]]
    ] = None  # Per-category levels (overrides 'levels')
    commitment_cost: float = 0.25
    orthogonality_weight: float = 0.1
    informativeness_weight: float = 0.1
    use_ema: bool = True
    decay: float = 0.99
    dropout: float = 0.1
    compression_ratios: Optional[List[float]] = None  # Latent_dim ratios for autoscaling

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Derive categories from provided config
        if self.category_levels is not None:
            self.categories = list(self.category_levels.keys())
        elif self.group_indices is not None:
            self.categories = list(self.group_indices.keys())
        else:
            raise ValueError("Must provide either category_levels or group_indices")

        # Handle category_levels if provided
        if self.category_levels is not None:
            # Validate all categories are specified
            for cat in self.categories:
                if cat not in self.category_levels:
                    raise ValueError(f"Missing category in category_levels: {cat}")

            # Fill missing latent_dims and num_tokens for each category
            for cat in self.categories:
                cat_levels = self.category_levels[cat]
                if not cat_levels or len(cat_levels) == 0:
                    raise ValueError(f"Category '{cat}' must have at least one level")

                # Compute per-category embedding dim from actual feature count
                category_feature_dim = len(self.group_indices[cat])

                # Fill missing num_tokens FIRST
                self.category_levels[cat] = fill_missing_num_tokens(
                    levels=cat_levels,
                    group_embedding_dim=category_feature_dim,
                    n_samples=10000,  # Fallback
                    category_name=cat,
                )

                # Fill missing latent_dims SECOND
                self.category_levels[cat] = fill_missing_latent_dims(
                    levels=self.category_levels[cat],
                    group_embedding_dim=category_feature_dim,
                    n_samples=10000,  # Fallback
                    category_name=cat,
                )

                # Validate all levels have required fields
                for level in self.category_levels[cat]:
                    if "latent_dim" not in level or "num_tokens" not in level:
                        raise ValueError(
                            f"Each level must have 'latent_dim' and 'num_tokens'"
                        )
        elif self.levels is None:
            # Default 3-level hierarchy (uniform across categories)
            default_num_tokens = compute_default_num_tokens(
                num_levels=3,
                group_embedding_dim=self.group_embedding_dim,
                n_samples=10000,
            )
            default_latent_dims = compute_default_latent_dims(
                num_levels=3,
                group_embedding_dim=self.group_embedding_dim,
                num_tokens_per_level=default_num_tokens,
                category_name=None,
                compression_ratios=self.compression_ratios,
            )
            self.levels = [
                {
                    "latent_dim": default_latent_dims[0],
                    "num_tokens": default_num_tokens[0],
                },  # L0
                {
                    "latent_dim": default_latent_dims[1],
                    "num_tokens": default_num_tokens[1],
                },  # L1
                {
                    "latent_dim": default_latent_dims[2],
                    "num_tokens": default_num_tokens[2],
                },  # L2
            ]

        # Validate uniform levels if used
        if self.levels is not None:
            if len(self.levels) == 0:
                raise ValueError("Must specify at least one level")

            # Fill missing num_tokens FIRST
            self.levels = fill_missing_num_tokens(
                levels=self.levels,
                group_embedding_dim=self.group_embedding_dim,
                n_samples=10000,
                category_name=None,
            )

            # Fill missing latent_dims SECOND
            self.levels = fill_missing_latent_dims(
                levels=self.levels,
                group_embedding_dim=self.group_embedding_dim,
                n_samples=10000,
                category_name=None,
            )

            # Validate all levels have required fields
            for level in self.levels:
                if "latent_dim" not in level or "num_tokens" not in level:
                    raise ValueError("Each level must have 'latent_dim' and 'num_tokens'")

    def get_category_levels(self, category: str) -> List[Dict]:
        """Get levels for a specific category."""
        if self.category_levels is not None:
            return self.category_levels[category]
        else:
            return self.levels

    @property
    def num_levels(self) -> int:
        """Number of hierarchical levels (same across all categories)."""
        if self.category_levels is not None:
            level_counts = [len(levels) for levels in self.category_levels.values()]
            if len(set(level_counts)) > 1:
                raise ValueError(
                    f"All categories must have same number of levels. Got: {level_counts}"
                )
            return level_counts[0]
        else:
            return len(self.levels)

    @property
    def num_categories(self) -> int:
        """Number of categories."""
        return len(self.categories)

    @property
    def total_tokens(self) -> int:
        """Total number of independent tokens (categories × levels)."""
        return self.num_categories * self.num_levels

    @property
    def total_latent_dim(self) -> int:
        """Total latent dimension (sum of all latent vectors)."""
        if self.category_levels is not None:
            total = 0
            for category in self.categories:
                cat_levels = self.category_levels[category]
                total += sum(level["latent_dim"] for level in cat_levels)
            return total
        else:
            return sum(level["latent_dim"] for level in self.levels) * self.num_categories


class CategoricalHierarchicalVQVAE(nn.Module):
    """Categorical Hierarchical VQ-VAE with N×L independent token streams.

    Architecture:
        1. GroupedFeatureExtractor: input → N category embeddings
        2. CategoricalProjector: N embeddings → N×L latent vectors
        3. N×L VectorQuantizers: N×L latent vectors → N×L discrete tokens
        4. Shared decoder: concat(N×L quantized vectors) → reconstructed input

    Output tokens organized as [batch, N×L]:
        [category_1_L0, category_1_L1, category_1_L2,
         category_2_L0, category_2_L1, category_2_L2,
         ...]
    """

    def __init__(self, config: CategoricalVQVAEConfig):
        """Initialize categorical hierarchical VQ-VAE."""
        super().__init__()

        self.config = config

        # Feature extractor (input → N category embeddings)
        self.feature_extractor = GroupedFeatureExtractor(
            input_dim=config.input_dim,
            group_indices=config.group_indices,
            group_embedding_dim=config.group_embedding_dim,
            group_hidden_dim=config.group_hidden_dim,
            dropout=config.dropout,
        )

        # Categorical projector (N embeddings → N×L latent vectors)
        if config.category_levels is not None:
            self.projector = CategoricalProjector(
                group_embedding_dim=config.group_embedding_dim,
                category_levels=config.category_levels,
                dropout=config.dropout,
            )
        else:
            self.projector = CategoricalProjector(
                group_embedding_dim=config.group_embedding_dim,
                levels=config.levels,
                categories=config.categories,
                dropout=config.dropout,
            )

        # N×L VectorQuantizers (one per category-level pair)
        self.vq_layers = nn.ModuleList()
        for category in config.categories:
            cat_levels = config.get_category_levels(category)
            for level_config in cat_levels:
                self.vq_layers.append(
                    VectorQuantizer(
                        num_embeddings=level_config["num_tokens"],
                        embedding_dim=level_config["latent_dim"],
                        commitment_cost=config.commitment_cost,
                        use_ema=config.use_ema,
                        decay=config.decay,
                    )
                )

        # Shared decoder (concat all quantized vectors → input reconstruction)
        self.shared_decoder = nn.Sequential(
            nn.Linear(config.total_latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.input_dim),
        )

        # Partial decoders (each latent vector → input reconstruction)
        # For informativeness loss
        self.partial_decoders = nn.ModuleList()
        for category in config.categories:
            cat_levels = config.get_category_levels(category)
            for level_config in cat_levels:
                self.partial_decoders.append(
                    nn.Sequential(
                        nn.Linear(level_config["latent_dim"], 128),
                        nn.ReLU(),
                        nn.Linear(128, config.input_dim),
                    )
                )

    @property
    def quantizers(self):
        """Alias for vq_layers for compatibility."""
        return self.vq_layers

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode input to N×L latent vectors.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            List of N×L latent vectors [batch, latent_dim]
        """
        # Extract category embeddings
        group_embeddings = self.feature_extractor(x)

        # Project to N×L hierarchical latent vectors
        latent_vectors = self.projector(group_embeddings)

        return latent_vectors

    def quantize(
        self, z_list: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """Quantize all N×L latent vectors.

        Args:
            z_list: List of N×L latent vectors

        Returns:
            Tuple of:
                - z_q_list: List of N×L quantized vectors
                - tokens: Token indices [batch, N×L]
                - losses: Dictionary of VQ losses
        """
        z_q_list = []
        token_list = []
        vq_loss_total = 0.0

        for z, vq_layer in zip(z_list, self.vq_layers):
            # Quantize
            z_q, encodings, losses = vq_layer(z)
            z_q_list.append(z_q)

            # Extract token indices
            tokens = torch.argmax(encodings, dim=-1)  # [batch]
            token_list.append(tokens)

            # Accumulate losses
            vq_loss_total += losses["loss"]

        # Stack tokens to [batch, N×L]
        tokens = torch.stack(token_list, dim=1)

        losses = {"vq_loss": vq_loss_total}

        return z_q_list, tokens, losses

    def decode_shared(self, z_q_list: List[torch.Tensor]) -> torch.Tensor:
        """Decode from all quantized vectors using shared decoder.

        Args:
            z_q_list: List of N×L quantized vectors

        Returns:
            Reconstructed input [batch, input_dim]
        """
        # Concatenate all quantized vectors
        z_q_concat = torch.cat(z_q_list, dim=1)

        # Decode
        x_recon = self.shared_decoder(z_q_concat)

        return x_recon

    def decode_partial(self, z_q_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Decode from each quantized vector independently.

        Args:
            z_q_list: List of N×L quantized vectors

        Returns:
            List of N×L reconstructions [batch, input_dim]
        """
        return [decoder(z_q) for decoder, z_q in zip(self.partial_decoders, z_q_list)]

    def compute_orthogonality_loss(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute orthogonality loss across all latent vectors.

        Encourages different category-level pairs to capture orthogonal information.

        Args:
            z_list: List of N×L latent vectors

        Returns:
            Orthogonality loss (scalar)
        """
        if len(z_list) <= 1:
            return torch.tensor(0.0, device=z_list[0].device)

        # Normalize representations
        z_norm_list = [F.normalize(z, dim=1) for z in z_list]

        # Pad to same dimension
        max_dim = max(z.size(1) for z in z_norm_list)
        z_pad_list = [
            F.pad(z_norm, (0, max_dim - z_norm.size(1))) for z_norm in z_norm_list
        ]

        # Compute pairwise correlations
        total_corr = 0.0
        num_pairs = 0

        for i in range(len(z_pad_list)):
            for j in range(i + 1, len(z_pad_list)):
                corr = torch.abs(torch.sum(z_pad_list[i] * z_pad_list[j], dim=1).mean())
                total_corr += corr
                num_pairs += 1

        ortho_loss = (
            total_corr / num_pairs
            if num_pairs > 0
            else torch.tensor(0.0, device=z_list[0].device)
        )

        return ortho_loss

    def compute_informativeness_loss(
        self, x: torch.Tensor, partial_recons: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute informativeness loss for all partial decoders.

        Each latent vector should be independently useful for reconstruction.

        Args:
            x: Original input [batch, input_dim]
            partial_recons: List of N×L reconstructions

        Returns:
            Informativeness loss (scalar)
        """
        total_info_loss = 0.0

        for recon in partial_recons:
            info_loss = F.mse_loss(recon, x)
            total_info_loss += info_loss

        info_loss = total_info_loss / len(partial_recons)

        return info_loss

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Full forward pass: encode → quantize → decode.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Dictionary with keys:
                - reconstruction: Dict with "features" key (reconstructed input)
                - vq_losses: List of VQ loss values
                - partial_reconstructions: List of partial reconstructions
                - latents: List of latent vectors (for topographic loss)
                - tokens: Token indices [batch, N×L]
        """
        # Encode to N×L latent vectors
        z_list = self.encode(x)

        # Quantize
        z_q_list, tokens, vq_losses = self.quantize(z_list)

        # Decode (shared)
        x_recon = self.decode_shared(z_q_list)

        # Decode (partial - for informativeness loss)
        partial_recons = self.decode_partial(z_q_list)

        # Return in format expected by loss functions
        return {
            "reconstruction": {"features": x_recon},
            "vq_losses": [vq_losses["vq_loss"]],
            "partial_reconstructions": partial_recons,
            "latents": z_list,
            "tokens": tokens,
        }

    def get_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Extract N×L token indices from input.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Token indices [batch, N×L]
        """
        with torch.no_grad():
            z_list = self.encode(x)
            _, tokens, _ = self.quantize(z_list)

        return tokens

    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from N×L token indices.

        Args:
            tokens: Token indices [batch, N×L]

        Returns:
            Reconstructed input [batch, input_dim]
        """
        # Look up embeddings for each token
        z_q_list = []

        for i, vq_layer in enumerate(self.vq_layers):
            token_idx = tokens[:, i]  # [batch]
            z_q = vq_layer.embedding.weight[token_idx]  # [batch, latent_dim]
            z_q_list.append(z_q)

        # Decode
        x_recon = self.decode_shared(z_q_list)

        return x_recon

    def get_category_tokens(self, tokens: torch.Tensor, category: str) -> torch.Tensor:
        """Extract tokens for a specific category.

        Args:
            tokens: All tokens [batch, N×L]
            category: Category name (e.g., 'cluster_1')

        Returns:
            Category tokens [batch, num_levels]
        """
        category_idx = self.projector.categories.index(category)
        start_idx = category_idx * self.config.num_levels
        end_idx = start_idx + self.config.num_levels

        return tokens[:, start_idx:end_idx]

    def reset_dead_codes(
        self, training_batch: torch.Tensor, percentile_threshold: float = 10.0
    ) -> int:
        """Reset unused codebook entries across all quantizers.

        Args:
            training_batch: Recent training batch [batch, input_dim]
            percentile_threshold: Percentile threshold (default 10.0)

        Returns:
            Total number of codes reset
        """
        # Encode to get latent vectors
        with torch.no_grad():
            z_list = self.encode(training_batch)

        # Reset dead codes in each quantizer
        total_reset = 0
        for vq_layer, z in zip(self.vq_layers, z_list):
            if hasattr(vq_layer, "reset_dead_codes"):
                n_reset = vq_layer.reset_dead_codes(z, percentile_threshold)
                total_reset += n_reset

        return total_reset
