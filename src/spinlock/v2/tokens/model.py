"""Joint Hierarchical VQ-VAE for multi-family trajectory tokenization.

Supports joint training of temporal and initial features with:
- Variable-length temporal sequences
- End-to-end CNN training for initial conditions
- Hierarchical VQ with N×L tokens per family
- Feature grouping integration
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from spinlock.encoding.vector_quantizer import VectorQuantizer
from spinlock.encoding.residual_decoder import MultiLayerResidualDecoder

from .encoders import (
    PyramidTemporalEncoder,
    TemporalMeanEncoder,
    TemporalCNNEncoder,
    InitialCNNEncoder,
    InitialHybridEncoder,
)
from .config import TokenizerConfig, HierarchyConfig
from .projector import HierarchicalProjector

logger = logging.getLogger(__name__)


class JointHierarchicalVQVAE(nn.Module):
    """Joint Hierarchical VQ-VAE for multi-family trajectory tokenization.

    Supports independent encoding and hierarchical quantization of:
    - Temporal features: Variable-length sequences encoded with pyramid encoder
    - Initial features: Hybrid manual + CNN encoding with optional pretraining

    Architecture:
        1. Family encoders: temporal_encoder(x_t) → z_t, initial_encoder(x_i, ic) → z_i
        2. Categorical projectors: z_f → {z_f^{c,l}} for each category c and level l
        3. Vector quantizers: z_f^{c,l} → quantized codes
        4. Shared decoder: concat(all quantized) → reconstructed features

    Args:
        config: Complete tokenizer configuration
        group_indices: Dict mapping "family_category" → feature indices
            E.g., {"temporal_group_1": [0,1,2], "initial_group_1": [345,346]}

    Example:
        >>> config = TokenizerConfig(...)
        >>> model = JointHierarchicalVQVAE(config, group_indices)
        >>> outputs = model(
        ...     temporal_features=[B, T, D_t],
        ...     initial_manual=[B, D_i_manual],
        ...     initial_raw=[B, C, H, W],
        ...     temporal_mask=[B, T],
        ...     temporal_lengths=[B]
        ... )
    """

    def __init__(
        self,
        config: TokenizerConfig,
        group_indices: Dict[str, List[int]],
    ):
        super().__init__()

        self.config = config
        self.group_indices = group_indices

        # Parse families from group_indices keys
        self.families = self._parse_families(group_indices)
        logger.info(f"Families detected: {self.families}")

        # Total input dimension (after encoding)
        self.temporal_dim = 0
        self.initial_dim = 0

        # Create family encoders
        self._create_encoders()

        # Compute total encoded dimension
        total_encoded_dim = self.temporal_dim + self.initial_dim

        # Create categorical projectors (one per family-category)
        self.projectors = nn.ModuleDict()
        for family_cat, indices in group_indices.items():
            family, _ = family_cat.split('_', 1)

            # Determine source dimension for this category
            if family == "temporal":
                cat_dim = len(indices)  # Feature count in this category
            elif family == "initial":
                cat_dim = len(indices)
            else:
                raise ValueError(f"Unknown family: {family}")

            # Create hierarchical projector
            self.projectors[family_cat] = self._create_projector(
                family_cat, cat_dim
            )

        # Create vector quantizers (N×L total, one per family-category-level)
        self.quantizers = nn.ModuleDict()
        for family_cat in group_indices.keys():
            family, _ = family_cat.split('_', 1)
            num_levels = config.hierarchy.num_levels

            for level_idx in range(num_levels):
                quantizer_key = f"{family_cat}_L{level_idx}"
                latent_dim = self._get_latent_dim(family_cat, level_idx)
                num_embeddings = config.quantizer.num_embeddings

                self.quantizers[quantizer_key] = VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=latent_dim,
                    commitment_cost=config.quantizer.commitment_cost,
                    use_ema=config.quantizer.use_ema,
                    decay=config.quantizer.ema_decay,
                    epsilon=config.quantizer.epsilon,
                )

        # Shared decoder
        total_latent_dim = sum(
            self._get_latent_dim(fc, l)
            for fc in group_indices.keys()
            for l in range(config.hierarchy.num_levels)
        )

        self.decoder = MultiLayerResidualDecoder(
            latent_dim=total_latent_dim,
            input_dim=total_encoded_dim,
            dropout=config.encoder.dropout,
        )

    def _parse_families(self, group_indices: Dict[str, List[int]]) -> List[str]:
        """Parse unique families from group_indices keys.

        Args:
            group_indices: Dict with keys like "temporal_group_1", "initial_group_2"

        Returns:
            List of unique families, e.g., ["temporal", "initial"]
        """
        families = set()
        for key in group_indices.keys():
            family = key.split('_', 1)[0]
            families.add(family)
        return sorted(families)

    def _create_encoders(self):
        """Create family-specific encoders based on config."""
        config = self.config

        # Temporal encoder
        if "temporal" in self.families:
            if config.encoder.temporal.variant == "pyramid":
                vl_config = None
                if config.encoder.temporal.variable_length:
                    vl_config = {
                        "enabled": True,
                        "adaptive_pyramid": config.encoder.temporal.adaptive_pyramid,
                        "min_pyramid_length": config.encoder.temporal.min_timesteps,
                    }

                # Input dim is determined by dataset - placeholder for now
                # Will be set during first forward pass or explicitly configured
                self.temporal_encoder = PyramidTemporalEncoder(
                    input_dim=1,  # Placeholder - will be overridden
                    level_dims=config.encoder.temporal.level_dims,
                    downsample_factors=config.encoder.temporal.downsample_factors,
                    variable_length_config=vl_config,
                )
                self.temporal_dim = sum(config.encoder.temporal.level_dims)

            elif config.encoder.temporal.variant == "mean":
                self.temporal_encoder = TemporalMeanEncoder(input_dim=1)
                self.temporal_dim = 1  # Will be overridden

            elif config.encoder.temporal.variant == "cnn":
                self.temporal_encoder = TemporalCNNEncoder(
                    input_dim=1,
                    embedding_dim=config.encoder.embedding_dim,
                )
                self.temporal_dim = config.encoder.embedding_dim
            else:
                raise ValueError(f"Unknown temporal variant: {config.encoder.temporal.variant}")

        # Initial encoder
        if "initial" in self.families:
            if config.encoder.initial.variant == "hybrid":
                self.initial_encoder = InitialHybridEncoder(
                    manual_dim=config.encoder.initial.manual_dim,
                    cnn_embedding_dim=config.encoder.initial.cnn_embedding_dim,
                    encode_manual=config.encoder.initial.encode_manual,
                    in_channels=config.encoder.initial.in_channels,
                    use_final_batchnorm=config.encoder.initial.use_final_batchnorm,
                    pretrained_cnn_path=str(config.encoder.initial.pretrained_cnn_path)
                    if config.encoder.initial.pretrained_cnn_path
                    else None,
                )
                self.initial_dim = self.initial_encoder.output_dim

            elif config.encoder.initial.variant == "cnn":
                self.initial_encoder = InitialCNNEncoder(
                    embedding_dim=config.encoder.initial.cnn_embedding_dim,
                    in_channels=config.encoder.initial.in_channels,
                    use_final_batchnorm=config.encoder.initial.use_final_batchnorm,
                )
                self.initial_dim = config.encoder.initial.cnn_embedding_dim
            else:
                raise ValueError(f"Unknown initial variant: {config.encoder.initial.variant}")

    def _create_projector(
        self, family_cat: str, category_dim: int
    ) -> HierarchicalProjector:
        """Create hierarchical projector for a family-category.

        Args:
            family_cat: Family-category key (e.g., "temporal_group_1")
            category_dim: Number of features in this category

        Returns:
            HierarchicalProjector instance
        """
        config = self.config
        num_levels = config.hierarchy.num_levels

        # Build level configs
        levels = []
        for level_idx in range(num_levels):
            latent_dim = self._get_latent_dim(family_cat, level_idx)
            levels.append({"latent_dim": latent_dim})

        return HierarchicalProjector(
            input_dim=category_dim,
            levels=levels,
            dropout=config.encoder.dropout,
        )

    def _get_latent_dim(self, family_cat: str, level_idx: int) -> int:
        """Compute latent dimension for a family-category-level.

        Uses compression ratios to scale based on feature count.

        Args:
            family_cat: Family-category key
            level_idx: Hierarchy level index

        Returns:
            Latent dimension for this level
        """
        config = self.config.hierarchy
        category_feature_count = len(self.group_indices[family_cat])

        # Parse compression ratios
        ratios = [float(r) for r in config.compression_ratios.split(':')]
        if level_idx >= len(ratios):
            raise ValueError(
                f"Level {level_idx} exceeds compression_ratios length {len(ratios)}"
            )

        # Compute latent dim
        latent_dim = int(category_feature_count * ratios[level_idx])

        # Clamp to min/max
        latent_dim = max(config.min_latent_dim, latent_dim)
        latent_dim = min(config.max_latent_dim, latent_dim)

        return latent_dim

    def forward(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through joint VQ-VAE.

        Args:
            temporal_features: Temporal sequences [B, T, D_t] (required if temporal family exists)
            initial_manual: Manual initial features [B, D_i_manual] (required if initial_hybrid)
            initial_raw: Raw initial conditions [B, C, H, W] (required if initial_hybrid)
            temporal_mask: Validity mask for temporal [B, T] (optional)
            temporal_lengths: Actual sequence lengths [B] (optional)

        Returns:
            Dict with keys:
                - reconstructed: Reconstructed features [B, total_dim]
                - quantized: Quantized latent vectors [B, total_latent_dim]
                - vq_loss: Vector quantization loss
                - perplexity: Average codebook perplexity
                - encodings: Dict of per-family-category-level encodings
        """
        batch_size = (
            temporal_features.shape[0] if temporal_features is not None
            else initial_manual.shape[0]
        )

        # Encode families
        encoded = {}

        if "temporal" in self.families:
            if temporal_features is None:
                raise ValueError("temporal_features required for temporal family")

            # Encode temporal
            if isinstance(self.temporal_encoder, PyramidTemporalEncoder):
                if temporal_mask is not None:
                    temp_encoded, mask_info = self.temporal_encoder(
                        temporal_features, mask=temporal_mask, lengths=temporal_lengths
                    )
                else:
                    temp_encoded = self.temporal_encoder(temporal_features)
            else:
                temp_encoded = self.temporal_encoder(temporal_features)

            encoded["temporal"] = temp_encoded

        if "initial" in self.families:
            if isinstance(self.initial_encoder, InitialHybridEncoder):
                if initial_manual is None or initial_raw is None:
                    raise ValueError(
                        "initial_manual and initial_raw required for initial_hybrid"
                    )
                init_encoded = self.initial_encoder(initial_manual, initial_raw)
            else:
                if initial_raw is None:
                    raise ValueError("initial_raw required for initial_cnn")
                init_encoded = self.initial_encoder(initial_raw)

            encoded["initial"] = init_encoded

        # Concatenate all encoded features
        all_encoded = []
        for family in sorted(self.families):
            all_encoded.append(encoded[family])
        all_encoded = torch.cat(all_encoded, dim=1)  # [B, total_encoded_dim]

        # Project to categorical latent spaces and quantize
        all_quantized = []
        vq_losses = []
        perplexities = []
        encodings_dict = {}

        for family_cat, indices in self.group_indices.items():
            family, cat_name = family_cat.split('_', 1)

            # Extract features for this category
            cat_features = all_encoded[:, indices]  # [B, cat_dim]

            # Project to hierarchical latents
            projector = self.projectors[family_cat]
            latents = projector(cat_features)  # List of [B, latent_dim_l]

            # Quantize each level
            for level_idx, latent in enumerate(latents):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = self.quantizers[quantizer_key]

                quantized, vq_loss, perplexity, _ = quantizer(latent)

                all_quantized.append(quantized)
                vq_losses.append(vq_loss)
                perplexities.append(perplexity)

                encodings_dict[quantizer_key] = quantized

        # Concatenate all quantized vectors
        all_quantized_cat = torch.cat(all_quantized, dim=1)  # [B, total_latent_dim]

        # Decode to reconstruct
        reconstructed = self.decoder(all_quantized_cat)  # [B, total_encoded_dim]

        # Aggregate VQ losses
        total_vq_loss = torch.stack(vq_losses).mean()
        avg_perplexity = torch.stack(perplexities).mean()

        return {
            "reconstructed": reconstructed,
            "quantized": all_quantized_cat,
            "vq_loss": total_vq_loss,
            "perplexity": avg_perplexity,
            "encodings": encodings_dict,
            "original_encoded": all_encoded,
        }

    def encode(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode inputs to discrete token indices.

        Args:
            Same as forward()

        Returns:
            Dict mapping "family_category_Ll" → token indices [B]
        """
        # Run forward to get quantized encodings
        outputs = self.forward(
            temporal_features=temporal_features,
            initial_manual=initial_manual,
            initial_raw=initial_raw,
            temporal_mask=temporal_mask,
            temporal_lengths=temporal_lengths,
        )

        # Extract token indices from quantizers
        tokens = {}
        for quantizer_key, quantizer in self.quantizers.items():
            # Get the quantized vector for this key
            quantized = outputs["encodings"][quantizer_key]

            # Find nearest codebook entry
            distances = torch.cdist(
                quantized,
                quantizer.embeddings.weight,
                p=2.0
            )
            token_indices = distances.argmin(dim=1)  # [B]

            tokens[quantizer_key] = token_indices

        return tokens
