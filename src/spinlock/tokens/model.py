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
from .encoders.theta import ThetaMLPEncoder
from .config import TokenizerConfig, HierarchyConfig
from .projector import HierarchicalProjector
from .inverse_models import ThetaInverseMLP, InitialInverseCNN

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
        temporal_input_dim: Optional[int] = None,
        initial_input_dim: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.group_indices = group_indices
        self.temporal_input_dim = temporal_input_dim
        self.initial_input_dim = initial_input_dim

        # Parse families from group_indices keys
        self.families = self._parse_families(group_indices)
        logger.info(f"Families detected: {self.families}")

        # Total input dimension (after encoding)
        self.temporal_dim = 0
        self.initial_dim = 0
        self.theta_dim = 0

        # Per-group encoder mode (active when grouping.method = pca_striped | opq)
        # Each entry encodes one group's raw features [B, G_k] → [B, embedding_dim]
        self.per_group_temporal_encoders: Optional[nn.ModuleDict] = None
        # Rotation buffers (registered so they move with .to(device))
        # Set via set_temporal_rotation() after model creation
        self.register_buffer("temporal_rotation_mean", None)
        self.register_buffer("temporal_rotation_matrix", None)

        # Create family encoders first (sets dims)
        self._create_encoders()

        # Inverse decoders (created if config provided, otherwise None)
        self.theta_inverse: Optional[ThetaInverseMLP] = None
        self.initial_inverse: Optional[InitialInverseCNN] = None

        if config.inverse_heads is not None:
            # Create theta inverse if theta family exists
            if "theta" in self.families and self.theta_dim > 0:
                # Infer param_dim from encoder config (adaptive to dataset)
                theta_param_dim = config.encoder.theta.param_dim if config.encoder.theta else 14
                self.theta_inverse = ThetaInverseMLP(
                    encoded_dim=self.theta_dim,
                    param_dim=theta_param_dim,
                    hidden_dim=config.inverse_heads.theta_hidden_dim,
                    dropout=config.inverse_heads.theta_dropout,
                )
                logger.info(f"Created ThetaInverseMLP: {self.theta_dim} → {theta_param_dim}")

            # Create initial inverse if initial family exists
            if "initial" in self.families and self.initial_dim > 0:
                # Infer channels and spatial_size from encoder config (adaptive to dataset)
                initial_channels = config.encoder.initial.in_channels
                # Spatial size is typically 64 for CNO, but could be inferred from data
                initial_spatial_size = 64  # Default, could be made configurable
                self.initial_inverse = InitialInverseCNN(
                    encoded_dim=self.initial_dim,
                    channels=initial_channels,
                    spatial_size=initial_spatial_size,
                )
                logger.info(f"Created InitialInverseCNN: {self.initial_dim} → [{initial_channels}, {initial_spatial_size}, {initial_spatial_size}]")

        # Compute total encoded dimension
        total_encoded_dim = self.temporal_dim + self.initial_dim + self.theta_dim

        # Create categorical projectors (one per family-category)
        self.projectors = nn.ModuleDict()
        for family_cat, indices in group_indices.items():
            family, _ = family_cat.split('_', 1)

            # Determine source dimension for this category.
            # In per-group encoder mode, each temporal group's projector receives
            # embedding_dim input (output of the per-group MLP) rather than the
            # raw feature count.  All other families still use raw feature count.
            if family == "temporal" and self.per_group_temporal_encoders is not None:
                cat_dim = config.encoder.embedding_dim
            else:
                cat_dim = len(indices)

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
                num_embeddings = self._get_num_embeddings(family_cat, level_idx)  # Adaptive!

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

    def _use_per_group_encoders(self) -> bool:
        """Return True when PCA/OPQ grouping is configured (per-group encoder path)."""
        cfg = self.config
        return (
            cfg.grouping is not None
            and getattr(cfg.grouping, "method", "correlation") in ("pca_striped", "opq")
        )

    def set_temporal_rotation(self, transform: Any) -> None:
        """Register a PCA/OPQ rotation transform as model buffers.

        Must be called after model creation when grouping.method is pca_striped
        or opq.  The transform is applied to time-averaged temporal features before
        they are split into per-group sub-spaces.

        Args:
            transform: LinearTransform with .mean [D] and .components [D, D].
        """
        import torch
        mean_t = torch.from_numpy(transform.mean).float()
        comps_t = torch.from_numpy(transform.components).float()
        # Re-register as persistent buffers (replaces the None registered in __init__)
        self.register_buffer("temporal_rotation_mean", mean_t)
        self.register_buffer("temporal_rotation_matrix", comps_t)
        logger.info(
            f"Temporal rotation set: mean {mean_t.shape}, components {comps_t.shape}"
        )

    def _create_encoders(self):
        """Create family-specific encoders based on config."""
        config = self.config

        # Temporal encoder
        if "temporal" in self.families:
            if config.encoder.temporal.variant == "pyramid":
                vl_config = None
                if config.encoder.temporal.variable_length:
                    # Handle both bool and VariableLengthConfig
                    if isinstance(config.encoder.temporal.variable_length, bool):
                        # Legacy: just a boolean
                        vl_config = {
                            "enabled": True,
                            "adaptive_pyramid": config.encoder.temporal.adaptive_pyramid,
                            "min_pyramid_length": config.encoder.temporal.min_timesteps,
                        }
                    else:
                        # New: VariableLengthConfig object
                        vl_cfg = config.encoder.temporal.variable_length
                        vl_config = {
                            "enabled": vl_cfg.enabled,
                            "adaptive_pyramid": vl_cfg.adaptive_pyramid,
                            "min_pyramid_length": vl_cfg.min_pyramid_length,
                            "mask_downsample_method": vl_cfg.mask_downsample_method,
                        }
                        # Note: length_bins and sampling_strategy are used by trainer, not encoder

                # Determine input dimension
                if self.temporal_input_dim is None:
                    raise ValueError(
                        "temporal_input_dim must be provided for pyramid encoder"
                    )

                self.temporal_encoder = PyramidTemporalEncoder(
                    input_dim=self.temporal_input_dim,
                    level_dims=config.encoder.temporal.level_dims,
                    downsample_factors=config.encoder.temporal.downsample_factors,
                    variable_length_config=vl_config,
                )
                self.temporal_dim = sum(config.encoder.temporal.level_dims)

            elif config.encoder.temporal.variant == "mean":
                if self.temporal_input_dim is None:
                    raise ValueError(
                        "temporal_input_dim must be provided for mean encoder"
                    )

                if self._use_per_group_encoders():
                    # Per-group encoder path (PCA/OPQ grouping)
                    # Each temporal group gets its own 2-layer MLP that takes the
                    # raw (rotated) group features [B, G_k] → [B, embedding_dim].
                    # No shared temporal encoder is needed.
                    embedding_dim = config.encoder.embedding_dim
                    temporal_groups = {
                        k: v for k, v in self.group_indices.items()
                        if k.startswith("temporal_")
                    }
                    n_temporal_groups = len(temporal_groups)

                    self.per_group_temporal_encoders = nn.ModuleDict({
                        family_cat: nn.Sequential(
                            nn.Linear(len(indices), embedding_dim),
                            nn.LayerNorm(embedding_dim),
                            nn.GELU(),
                            nn.Linear(embedding_dim, embedding_dim),
                        )
                        for family_cat, indices in temporal_groups.items()
                    })

                    # temporal_dim = total encoded temporal space for the decoder
                    self.temporal_dim = n_temporal_groups * embedding_dim
                    logger.info(
                        f"Per-group temporal encoders: {n_temporal_groups} groups × "
                        f"{embedding_dim}D = {self.temporal_dim}D total temporal dim"
                    )
                else:
                    # Shared mean encoder (legacy path)
                    self.temporal_encoder = TemporalMeanEncoder(input_dim=self.temporal_input_dim)
                    self.temporal_dim = self.temporal_input_dim

            elif config.encoder.temporal.variant == "cnn":
                if self.temporal_input_dim is None:
                    raise ValueError(
                        "temporal_input_dim must be provided for CNN encoder"
                    )
                self.temporal_encoder = TemporalCNNEncoder(
                    input_dim=self.temporal_input_dim,
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

        # Theta encoder
        if "theta" in self.families:
            theta_cfg = config.encoder.theta
            if theta_cfg is None:
                raise ValueError("Theta family enabled but theta encoder config missing")

            if theta_cfg.variant == "mlp":
                self.theta_encoder = ThetaMLPEncoder(
                    param_dim=theta_cfg.param_dim,
                    hidden_dim=theta_cfg.hidden_dim,
                    output_dim=theta_cfg.output_dim,
                    dropout=theta_cfg.dropout,
                    use_layer_norm=theta_cfg.use_layer_norm,
                )
                self.theta_dim = theta_cfg.output_dim
            else:
                raise ValueError(f"Unknown theta encoder variant: {theta_cfg.variant}")

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

    def _get_latent_dim(self, family_cat: str, level_idx: int, n_samples: int = 50000) -> int:
        """Compute latent dimension using v1's adaptive formula.

        Formula: latent_dim = category_size × base_expansion × level_multiplier × token_factor

        Args:
            family_cat: Family-category key
            level_idx: Hierarchy level index
            n_samples: Number of samples in dataset

        Returns:
            Latent dimension for this level
        """
        import numpy as np

        category_feature_count = len(self.group_indices[family_cat])

        # Compute num_tokens for this level (needed for token_factor)
        num_tokens = self._get_num_embeddings(family_cat, level_idx, n_samples)

        # Adaptive expansion based on category size
        # V1 formula: 1.0 + 0.8 * ((dim / 100.0) ** 0.7)
        base_expansion = 1.0 + 0.8 * ((category_feature_count / 100.0) ** 0.7)

        # Level progression: geometric decay (L0 → L1 → L2)
        level_multiplier = 0.5**level_idx

        # Token scaling (gentle log scaling)
        token_factor = max(1.0, np.log2(num_tokens) / 20.0)

        # Compute base value
        latent_dim_float = (
            category_feature_count * base_expansion * level_multiplier * token_factor
        )

        # Round to multiple of 4 (GPU alignment)
        latent_dim = int(np.ceil(latent_dim_float / 4.0)) * 4

        # Dataset-aware minimum capacity (L0 only)
        if level_idx == 0 and n_samples > 1000:
            # Minimum scales with dataset size
            min_latent_dim = int(np.ceil(np.log10(n_samples) * 12 / 4.0)) * 4
            # Cap at reasonable maximum
            min_latent_dim = min(64, max(8, min_latent_dim))
            latent_dim = max(min_latent_dim, latent_dim)
        else:
            # L1 and L2: preserve standard minimum (8D)
            latent_dim = max(8, latent_dim)

        # Enforce monotonicity: ensure L0 >= L1 >= L2 (will be done after all dims computed)
        # For now, just clamp to config limits
        latent_dim = max(self.config.hierarchy.min_latent_dim, latent_dim)
        latent_dim = min(self.config.hierarchy.max_latent_dim, latent_dim)

        return latent_dim

    def _get_num_embeddings(self, family_cat: str, level_idx: int, n_samples: int = 50000) -> int:
        """Compute adaptive codebook size using v1's formula.

        Scales codebook size with category dimension and dataset size.
        Implements hierarchical progression: L0 > L1 > L2

        Args:
            family_cat: Family-category key (e.g., "temporal_group_1")
            level_idx: Hierarchy level index (0, 1, 2)
            n_samples: Number of samples in dataset (for dataset-aware minimum)

        Returns:
            Number of codebook embeddings (codebook size)
        """
        import numpy as np

        # Get category feature count (NOT latent_dim!)
        category_feature_count = len(self.group_indices[family_cat])

        # Base token count scales with category capacity
        # V1 formula: base_tokens = log2(group_embedding_dim) * 5
        base_tokens = int(np.log2(max(category_feature_count, 2)) * 5)

        # L0: Apply dataset-aware minimum
        if level_idx == 0:
            l0_tokens_float = base_tokens * 1.0
            l0_tokens = (int(l0_tokens_float) // 4) * 4

            # Dataset-aware minimum (v1 formula)
            if n_samples > 1000:
                min_tokens = min(28, max(5, int(np.sqrt(n_samples / 1000.0) * 4.8)))
                l0_tokens = max(min_tokens, l0_tokens)

            return l0_tokens

        # L1 and L2: Geometric halving with minimum
        level_multiplier = 0.5**level_idx
        num_tokens_float = base_tokens * level_multiplier
        num_tokens = int(num_tokens_float)

        # L1 and L2: preserve standard minimum (6)
        num_tokens = max(6, num_tokens)

        return num_tokens

    def forward(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through joint VQ-VAE.

        Args:
            temporal_features: Temporal sequences [B, T, D_t] (required if temporal family exists)
            initial_manual: Manual initial features [B, D_i_manual] (required if initial_hybrid)
            initial_raw: Raw initial conditions [B, C, H, W] (required if initial_hybrid)
            theta_features: Operator parameters [B, param_dim] (required if theta family exists)
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
        # Determine batch size from available inputs
        if temporal_features is not None:
            batch_size = temporal_features.shape[0]
        elif initial_manual is not None:
            batch_size = initial_manual.shape[0]
        elif theta_features is not None:
            batch_size = theta_features.shape[0]
        else:
            raise ValueError("At least one feature input must be provided")

        # Encode families
        encoded = {}

        if "temporal" in self.families:
            if temporal_features is None:
                raise ValueError("temporal_features required for temporal family")

            if self.per_group_temporal_encoders is not None:
                # ── Per-group encoder path (PCA/OPQ grouping) ────────────────
                # Temporal summary: cat([mean, std]) over the time dimension.
                # mean(dim=1): trajectory-averaged level (correlated with steady state)
                # std(dim=1):  dynamical amplitude (correlated with parameter-driven dynamics)
                # Together: [B, 2*D_t] — avoids the rank-1 collapse that mean alone produces.
                temporal_mean = temporal_features.mean(dim=1)  # [B, D_t]
                temporal_std  = temporal_features.std(dim=1)   # [B, D_t]
                temporal_summary = torch.cat([temporal_mean, temporal_std], dim=1)  # [B, 2*D_t]

                # Apply PCA/OPQ rotation if set: [B, 2*D_t] → [B, 2*D_t]
                if self.temporal_rotation_matrix is not None:
                    temporal_summary = (
                        (temporal_summary - self.temporal_rotation_mean)
                        @ self.temporal_rotation_matrix.T
                    )

                # Encode each group independently; accumulate for the decoder
                group_encoded_parts = []
                for family_cat, indices in self.group_indices.items():
                    if family_cat.startswith("temporal_"):
                        group_feats = temporal_summary[:, indices]  # [B, G_k]
                        enc = self.per_group_temporal_encoders[family_cat](group_feats)
                        encoded[family_cat] = enc        # used in quantization loop
                        group_encoded_parts.append(enc)

                # Concatenate all group encodings: [B, n_groups * embedding_dim]
                encoded["temporal"] = torch.cat(group_encoded_parts, dim=1)

            else:
                # ── Shared encoder path (legacy: correlation grouping) ────────
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

        if "theta" in self.families:
            if theta_features is None:
                raise ValueError("theta_features required for theta family")

            # Encode theta parameters
            theta_encoded = self.theta_encoder(theta_features)
            encoded["theta"] = theta_encoded

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
        latents_dict = {}  # Pre-quantization latent vectors

        for family_cat, indices in self.group_indices.items():
            family, cat_name = family_cat.split('_', 1)

            # Extract features for this category.
            # Per-group path: temporal groups are already encoded in encoded[family_cat].
            # Legacy path: slice from the concatenated all_encoded.
            if family == "temporal" and self.per_group_temporal_encoders is not None:
                cat_features = encoded[family_cat]  # [B, embedding_dim]
            else:
                cat_features = all_encoded[:, indices]  # [B, cat_dim]

            # Project to hierarchical latents
            projector = self.projectors[family_cat]
            latents = projector(cat_features)  # List of [B, latent_dim_l]

            # Quantize each level
            for level_idx, latent in enumerate(latents):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = self.quantizers[quantizer_key]

                # Save pre-quantization latent
                latents_dict[quantizer_key] = latent

                quantized, encodings, losses = quantizer(latent)

                all_quantized.append(quantized)
                vq_losses.append(losses['loss'])
                # Compute perplexity from encodings
                avg_probs = encodings.mean(dim=0)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                perplexities.append(perplexity)

                encodings_dict[quantizer_key] = quantized

        # Concatenate all quantized vectors
        all_quantized_cat = torch.cat(all_quantized, dim=1)  # [B, total_latent_dim]

        # Decode to reconstruct
        reconstructed = self.decoder(all_quantized_cat)  # [B, total_encoded_dim]

        # Split reconstructed back into family components
        reconstructed_split = self._split_reconstructed(reconstructed, all_encoded)

        # Apply inverse heads if they exist (NEW!)
        decoded = {}
        if self.theta_inverse is not None and "theta" in reconstructed_split:
            decoded["theta"] = self.theta_inverse(reconstructed_split["theta"])
        if self.initial_inverse is not None and "initial" in reconstructed_split:
            decoded["initial"] = self.initial_inverse(reconstructed_split["initial"])

        # Aggregate VQ losses
        total_vq_loss = torch.stack(vq_losses).mean()
        avg_perplexity = torch.stack(perplexities).mean()

        # Extract token indices for topographic loss
        token_indices = self._get_token_indices(encodings_dict)

        return {
            "reconstructed": reconstructed,
            "reconstructed_split": reconstructed_split,  # NEW: split by family
            "decoded": decoded if decoded else None,  # NEW: decoded (theta, ICs)
            "quantized": all_quantized_cat,
            "vq_loss": total_vq_loss,
            "perplexity": avg_perplexity,
            "encodings": encodings_dict,
            "latents": latents_dict,  # Pre-quantization latent vectors
            "token_indices": token_indices,
            "original_encoded": all_encoded,
        }

    def encode(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
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
            theta_features=theta_features,
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
                quantizer.embedding.weight,
                p=2.0
            )
            token_indices = distances.argmin(dim=1)  # [B]

            tokens[quantizer_key] = token_indices

        return tokens

    def _split_reconstructed(
        self,
        reconstructed: torch.Tensor,
        original_encoded: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Split reconstructed tensor back into family components.

        Args:
            reconstructed: Reconstructed features [B, total_encoded_dim]
            original_encoded: Original encoded features [B, total_encoded_dim] (for shape reference)

        Returns:
            Dict mapping family → reconstructed features [B, family_dim]
        """
        split = {}
        offset = 0

        for family in sorted(self.families):
            if family == "temporal":
                family_dim = self.temporal_dim
            elif family == "initial":
                family_dim = self.initial_dim
            elif family == "theta":
                family_dim = self.theta_dim
            else:
                raise ValueError(f"Unknown family: {family}")

            split[family] = reconstructed[:, offset:offset + family_dim]
            offset += family_dim

        return split

    def _get_token_indices(self, encodings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract token indices from quantized encodings.

        Args:
            encodings: Dict mapping quantizer_key → quantized vectors [B, latent_dim]

        Returns:
            Dict mapping quantizer_key → token indices [B]
        """
        token_indices = {}
        for quantizer_key, quantizer in self.quantizers.items():
            quantized = encodings[quantizer_key]
            # Find nearest codebook entry
            distances = torch.cdist(
                quantized,
                quantizer.embedding.weight,
                p=2.0
            )
            indices = distances.argmin(dim=1)  # [B]
            token_indices[quantizer_key] = indices

        return token_indices
