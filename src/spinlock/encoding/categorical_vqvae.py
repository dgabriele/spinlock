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

import logging
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
    parse_compression_ratios,
)

logger = logging.getLogger(__name__)


@dataclass
class CategoricalVQVAEConfig:
    """Configuration for categorical hierarchical VQ-VAE.

    Attributes:
        input_dim: Input feature dimension
        group_indices: Dict mapping category -> feature indices
        group_embedding_dim: Embedding dimension for each category
        group_hidden_dim: Hidden dimension for category MLPs
        levels: Per-category level configs (category_name -> list of level dicts)
                Each level can specify num_tokens and/or latent_dim
                If omitted, they are auto-computed using compression_ratios
        commitment_cost: VQ-VAE commitment cost
        orthogonality_weight: Weight for orthogonality loss
        informativeness_weight: Weight for informativeness loss
        use_ema: Whether to use EMA for codebook updates
        decay: EMA decay rate
        dropout: Dropout rate
        compression_ratios: Latent_dim scaling ratios [L0:L1:L2] relative to feature count
                           E.g. "0.5:1:1.5" means L0=feats×0.5, L1=feats×1.0, L2=feats×1.5
        uniform_codebook_init: If True, all levels start with L0's token count.
                           Dead code resets will naturally prune to appropriate sizes.
    """

    input_dim: int
    group_indices: Dict[str, List[int]]
    group_embedding_dim: int = 64
    group_hidden_dim: int = 128
    levels: Optional[Dict[str, List[Dict]]] = None  # Per-category levels
    commitment_cost: float = 0.25
    orthogonality_weight: float = 0.1
    informativeness_weight: float = 0.1
    use_ema: bool = True
    decay: float = 0.99
    dropout: float = 0.1
    compression_ratios: Optional[List[float]] = None  # Latent_dim ratios for autoscaling
    uniform_codebook_init: bool = False  # If True, all levels start with L0's token count

    def __post_init__(self):
        """Validate configuration and set defaults."""
        # Parse compression_ratios: can be string, list, or dict (per-category)
        parsed_compression_ratios = None
        per_category_ratios = None
        auto_strategy = "balanced"  # Default strategy for auto mode

        if self.compression_ratios is not None:
            if isinstance(self.compression_ratios, str):
                if self.compression_ratios.lower() == "auto":
                    # Auto mode: will compute adaptive ratios per category
                    parsed_compression_ratios = "auto"
                    logger.info(
                        "Compression ratios set to 'auto' mode. "
                        "Adaptive ratios will be computed per category based on feature characteristics."
                    )
                    logger.warning(
                        "AUTO mode requires category_features to be provided during training. "
                        "If using this config for training, ensure features are available."
                    )
                else:
                    # Assume 3 levels if parsing from string (e.g., "0.5:1:1.5")
                    parsed_compression_ratios = parse_compression_ratios(
                        self.compression_ratios, num_levels=3
                    )
            elif isinstance(self.compression_ratios, list):
                # Uniform ratios for all categories
                parsed_compression_ratios = self.compression_ratios
            elif isinstance(self.compression_ratios, dict):
                # Per-category ratios (from adaptive computation)
                per_category_ratios = self.compression_ratios
                logger.info(
                    f"Using adaptive per-category compression ratios for {len(per_category_ratios)} categories"
                )

        # Derive categories from group_indices
        if self.group_indices is None:
            raise ValueError("Must provide group_indices")
        self.categories = list(self.group_indices.keys())

        # Initialize per-category levels if not provided
        if self.levels is None:
            # Auto-create empty levels for each category (will be filled by auto-scaling)
            self.levels = {cat: [] for cat in self.categories}
        else:
            # Validate all categories are specified
            for cat in self.categories:
                if cat not in self.levels:
                    raise ValueError(f"Missing category '{cat}' in levels config")

        # Fill missing latent_dims and num_tokens for each category
        for cat in self.categories:
            cat_levels = self.levels[cat]

            # If empty, create default 3-level structure
            if not cat_levels or len(cat_levels) == 0:
                cat_levels = [
                    {'num_tokens': None, 'latent_dim': None},
                    {'num_tokens': None, 'latent_dim': None},
                    {'num_tokens': None, 'latent_dim': None},
                ]

            # CORRECT DESIGN: Use feature count for adaptive coarse-to-fine hierarchy
            # This implements proper hierarchical VQ-VAE with compression at coarse levels:
            # - Small categories (7 features): L0=4D (compression), L2=11D (expansion)
            # - Large categories (71 features): L0=35D (compression), L2=107D (expansion)
            # The information bottleneck at L0 forces meaningful clustering.
            # V11 tried "fixing" this to use group_embedding_dim (256), which broke
            # coarse-to-fine compression and caused 11% quality regression.
            category_feature_dim = len(self.group_indices[cat])

            # Fill missing num_tokens FIRST
            self.levels[cat] = fill_missing_num_tokens(
                levels=cat_levels,
                group_embedding_dim=category_feature_dim,
                n_samples=10000,  # Fallback
                category_name=cat,
                uniform_codebook_init=self.uniform_codebook_init,
            )

            # Fill missing latent_dims SECOND
            # Use per-category ratios if available, otherwise use uniform ratios
            cat_compression_ratios = parsed_compression_ratios
            if per_category_ratios is not None and cat in per_category_ratios:
                cat_compression_ratios = per_category_ratios[cat]
                logger.info(
                    f"Category '{cat}': using adaptive compression ratios {cat_compression_ratios}"
                )

            self.levels[cat] = fill_missing_latent_dims(
                levels=self.levels[cat],
                group_embedding_dim=category_feature_dim,
                n_samples=10000,  # Fallback
                category_name=cat,
                compression_ratios=cat_compression_ratios,
                category_features=None,  # Not needed - ratios pre-computed
                auto_strategy=auto_strategy,
            )

            # Validate all levels have required fields
            for level in self.levels[cat]:
                if "latent_dim" not in level or "num_tokens" not in level:
                    raise ValueError(
                        f"Each level must have 'latent_dim' and 'num_tokens'"
                    )

    def get_category_levels(self, category: str) -> List[Dict]:
        """Get levels for a specific category."""
        return self.levels[category]

    @property
    def num_levels(self) -> int:
        """Number of hierarchical levels (same across all categories)."""
        level_counts = [len(levels) for levels in self.levels.values()]
        if len(set(level_counts)) > 1:
            raise ValueError(
                f"All categories must have same number of levels. Got: {level_counts}"
            )
        return level_counts[0]

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
        total = 0
        for category in self.categories:
            cat_levels = self.levels[category]
            total += sum(level["latent_dim"] for level in cat_levels)
        return total


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
        self.projector = CategoricalProjector(
            group_embedding_dim=config.group_embedding_dim,
            category_levels=config.levels,  # Per-category levels dict
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

    def decode(self, z_q_list: List[torch.Tensor]) -> torch.Tensor:
        """Decode quantized vectors to feature space.

        Unified decode interface for VQ-VAE reconstruction.
        This is the reconstruction step: z_q → x_recon

        Used in tokenizable-led NOA training as:
            x_recon = vqvae.decode(z_q_list)
            L_recon = MSE(x_recon, x_original)

        Philosophy: This enables NOA to be evaluated on whether its outputs
        are "expressible" in the VQ vocabulary, not just accurate to physics.
        A high-quality decode means the NOA trajectory is symbolically coherent.

        Args:
            z_q_list: List of N×L quantized vectors (from quantize())

        Returns:
            Reconstructed features [batch, input_dim]
        """
        return self.decode_shared(z_q_list)

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
                - latents: List of PRE-quantization latent vectors
                - quantized: List of POST-quantization code embeddings
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
            "latents": z_list,  # PRE-quantization (continuous)
            "quantized": z_q_list,  # POST-quantization (discrete code embeddings)
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
        self, training_batch: torch.Tensor, percentile_threshold: float = 10.0, raw_ics=None
    ) -> int:
        """Reset unused codebook entries across all quantizers.

        Args:
            training_batch: Recent training batch [batch, input_dim]
            percentile_threshold: Percentile threshold (default 10.0)
            raw_ics: Raw initial conditions (unused for non-hybrid models, for API consistency)

        Returns:
            Total number of codes reset
        """
        # Note: raw_ics is ignored for non-hybrid models (VQVAEWithInitial uses it)
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
