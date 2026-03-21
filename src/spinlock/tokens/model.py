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
    SpectralICEncoder,
    SpatialICEncoder,
)
from .fsq import FiniteScalarQuantizer
from .encoders.theta import ThetaMLPEncoder
from .encoders.temporal_cnn_feature import TemporalCNNFeatureEncoder
from .encoders.pyramid_first import PyramidFirstEncoder
from .config import TokenizerConfig, HierarchyConfig, AuxHeadConfig
from .projector import HierarchicalProjector
from .inverse_models import (
    ThetaInverseMLP, InitialInverseCNN, InitialSpectralInverse,
    TemporalInverseMLP, TrajectoryPrototypeHead,
)
from .decoders import SpatialICDecoder, SpatialICLatentDecoder

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
        self.temporal_input_dim = temporal_input_dim
        self.initial_input_dim = initial_input_dim

        # Accept all groups present in group_indices — family presence is
        # determined by which groups exist, not by a boolean flag.
        self.group_indices = group_indices

        # Parse families from group_indices keys
        self.families = self._parse_families(self.group_indices)
        logger.info(f"Families detected: {self.families}")

        # Total input dimension (after encoding)
        self.temporal_dim = 0
        self.initial_dim = 0
        self.theta_dim = 0

        # Learned temporal CNN encoder (active when feature_source = "learned", architecture = "per_frame")
        self.temporal_cnn_encoder: Optional[TemporalCNNFeatureEncoder] = None
        # Pyramid-first encoder (active when feature_source = "learned", architecture = "pyramid_first")
        self.pyramid_first_encoder: Optional[PyramidFirstEncoder] = None
        # Per-group encoder mode (active when grouping.method = pca_striped | opq)
        # Each entry encodes one group's raw features [B, G_k] → [B, embedding_dim]
        self.per_group_temporal_encoders: Optional[nn.ModuleDict] = None
        # Per-group pyramid encoder mode (active when method = pca_raw + variant = pyramid)
        # OR when feature_source = "learned" (CNN output → per-group pyramid).
        # Each entry encodes [B, T, G_k] directly, preserving temporal dynamics.
        self.per_group_pyramid_encoders: Optional[nn.ModuleDict] = None
        # Rotation buffers (registered so they move with .to(device))
        # Set via set_temporal_rotation() after model creation
        self.register_buffer("temporal_rotation_mean", None)
        self.register_buffer("temporal_rotation_matrix", None)

        # Create family encoders first (sets dims)
        self._create_encoders()

        # Inverse decoders (created if config provided, otherwise None)
        self.theta_inverse: Optional[ThetaInverseMLP] = None
        self.initial_inverse: Optional[InitialInverseCNN] = None
        self.temporal_inverse: Optional[TemporalInverseMLP] = None

        # Dedicated bypass decoders for theta and IC.
        # The shared decoder's output space is dominated by temporal features,
        # so theta/IC gradients get diluted. Bypass decoders give these families
        # short, high-bandwidth gradient paths: quantized latents → MLP → output.
        self.theta_bypass_decoder: Optional[nn.Module] = None
        self.initial_bypass_inverse: Optional[nn.Module] = None
        # Latent-space IC decoder for roundtrip (skip CNN pixel path)
        self.initial_latent_decoder: Optional[SpatialICLatentDecoder] = None
        # Latent-space temporal decoder for roundtrip (skip shared decoder)
        self.temporal_latent_decoder: Optional[nn.Module] = None

        if config.inverse_heads is not None:
            # Create theta inverse if theta family exists
            if "theta" in self.families and self.theta_dim > 0:
                # Infer param_dim from encoder config (adaptive to dataset)
                theta_param_dim = config.encoder.theta.param_dim if config.encoder.theta else 14

                if getattr(self, '_theta_direct', False):
                    # Direct mode: bypass decoder maps quantized theta latents → params
                    theta_latent_dim = sum(
                        self._get_latent_dim(fc, l)
                        for fc in group_indices if fc.startswith("theta_")
                        for l in range(config.hierarchy.num_levels)
                    )
                    self.theta_bypass_decoder = nn.Sequential(
                        nn.Linear(theta_latent_dim, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, theta_param_dim),
                    )
                    logger.info(
                        f"Created theta bypass decoder: {theta_latent_dim}D → {theta_param_dim}D"
                    )
                    # No ThetaInverseMLP needed — bypass decoder does it directly
                else:
                    self.theta_inverse = ThetaInverseMLP(
                        encoded_dim=self.theta_dim,
                        param_dim=theta_param_dim,
                        hidden_dim=config.inverse_heads.theta_hidden_dim,
                        dropout=config.inverse_heads.theta_dropout,
                    )
                    logger.info(f"Created ThetaInverseMLP: {self.theta_dim} → {theta_param_dim}")

            # Create initial inverse if initial family exists
            if "initial" in self.families and self.initial_dim > 0:
                initial_channels = config.encoder.initial.in_channels
                initial_spatial_size = config.encoder.initial.spatial_size or 64

                if config.inverse_heads.initial_bypass_decoder:
                    # Bypass mode: concat IC quantized latents → inverse directly.
                    # Skips shared decoder, giving IC a high-bandwidth gradient path.
                    # Spatial IC groups have 1 level (FSQ), others have num_levels (VQ).
                    use_fsq = (
                        config.encoder.initial.spatial_quantizer == "fsq"
                    )
                    initial_latent_dim = 0
                    for fc in group_indices:
                        if not fc.startswith("initial_"):
                            continue
                        if fc.startswith("initial_spatial_") and use_fsq:
                            # FSQ: 1 level, dim = len(fsq_levels)
                            initial_latent_dim += len(config.encoder.initial.fsq_levels)
                        else:
                            for l in range(config.hierarchy.num_levels):
                                initial_latent_dim += self._get_latent_dim(fc, l)

                    if config.inverse_heads.initial_variant == "spatial":
                        # CNN decoder symmetric with SpatialICEncoder
                        self.initial_bypass_inverse = SpatialICDecoder(
                            encoded_dim=initial_latent_dim,
                            channels=initial_channels,
                            spatial_size=initial_spatial_size,
                            spatial_token_grid=config.encoder.initial.spatial_token_grid,
                            base_channels=config.encoder.initial.spatial_base_channels,
                        )
                        logger.info(
                            f"Created IC bypass decoder (spatial CNN): {initial_latent_dim}D → "
                            f"[{initial_channels}, {initial_spatial_size}, {initial_spatial_size}] "
                            f"(grid={config.encoder.initial.spatial_token_grid})"
                        )
                    else:
                        self.initial_bypass_inverse = InitialSpectralInverse(
                            encoded_dim=initial_latent_dim,
                            channels=initial_channels,
                            spatial_size=initial_spatial_size,
                            num_modes=config.inverse_heads.initial_spectral_num_modes,
                            hidden_dims=config.inverse_heads.initial_spectral_hidden_dims,
                        )
                        logger.info(
                            f"Created IC bypass decoder (spectral): {initial_latent_dim}D → "
                            f"[{initial_channels}, {initial_spatial_size}, {initial_spatial_size}] "
                            f"(K={config.inverse_heads.initial_spectral_num_modes})"
                        )
                elif config.inverse_heads.initial_variant == "spectral":
                    self.initial_inverse = InitialSpectralInverse(
                        encoded_dim=self.initial_dim,
                        channels=initial_channels,
                        spatial_size=initial_spatial_size,
                        num_modes=config.inverse_heads.initial_spectral_num_modes,
                        hidden_dims=config.inverse_heads.initial_spectral_hidden_dims,
                    )
                    logger.info(
                        f"Created InitialSpectralInverse: {self.initial_dim} → "
                        f"[{initial_channels}, {initial_spatial_size}, {initial_spatial_size}] "
                        f"(K={config.inverse_heads.initial_spectral_num_modes})"
                    )
                else:
                    self.initial_inverse = InitialInverseCNN(
                        encoded_dim=self.initial_dim,
                        channels=initial_channels,
                        spatial_size=initial_spatial_size,
                    )
                    logger.info(f"Created InitialInverseCNN: {self.initial_dim} → [{initial_channels}, {initial_spatial_size}, {initial_spatial_size}]")

                # Latent-space IC decoder for roundtrip (skip CNN pixel path)
                if config.inverse_heads.initial_latent_roundtrip:
                    # Compute total IC quantized latent dim (same as bypass decoder)
                    use_fsq_rt = (
                        config.encoder.initial.spatial_quantizer == "fsq"
                    )
                    initial_latent_dim = 0
                    for fc in group_indices:
                        if not fc.startswith("initial_"):
                            continue
                        if fc.startswith("initial_spatial_") and use_fsq_rt:
                            initial_latent_dim += len(config.encoder.initial.fsq_levels)
                        else:
                            for l in range(config.hierarchy.num_levels):
                                initial_latent_dim += self._get_latent_dim(fc, l)

                    # Output dim = SpatialICEncoder output (grid² × token_dim)
                    grid = config.encoder.initial.spatial_token_grid
                    spatial_token_dim = config.encoder.initial.spatial_token_dim
                    encoder_output_dim = (grid ** 2) * spatial_token_dim

                    self.initial_latent_decoder = SpatialICLatentDecoder(
                        encoded_dim=initial_latent_dim,
                        output_dim=encoder_output_dim,
                    )
                    logger.info(
                        f"Created IC latent decoder (roundtrip): "
                        f"{initial_latent_dim}D → {encoder_output_dim}D "
                        f"({grid}²×{spatial_token_dim})"
                    )

            # Create temporal inverse if learned mode (CNN features or pyramid_first)
            has_temporal_encoder = (
                self.temporal_cnn_encoder is not None
                or self.pyramid_first_encoder is not None
            )
            if (
                "temporal" in self.families
                and self.temporal_dim > 0
                and has_temporal_encoder
                and config.encoder.temporal.learned is not None
            ):
                learned_cfg = config.encoder.temporal.learned
                # In pyramid_first mode, cnn_dim = num_groups × D_group (embedding_dim);
                # in per_frame mode, cnn_dim = learned_cfg.embedding_dim.
                if self.pyramid_first_encoder is not None:
                    cnn_dim = learned_cfg.num_groups * config.encoder.embedding_dim
                else:
                    cnn_dim = learned_cfg.embedding_dim
                self.temporal_inverse = TemporalInverseMLP(
                    encoded_dim=self.temporal_dim,
                    cnn_dim=cnn_dim,
                    roundtrip_timesteps=config.inverse_heads.temporal_roundtrip_timesteps,
                    hidden_dim=config.inverse_heads.temporal_hidden_dim,
                    dropout=config.inverse_heads.temporal_dropout,
                )
                logger.info(
                    f"Created TemporalInverseMLP: {self.temporal_dim} → "
                    f"[{config.inverse_heads.temporal_roundtrip_timesteps}, {cnn_dim}]"
                )

                # Temporal latent bypass: MLP from quantized temporal latents directly
                # to per-group embeddings, skipping shared decoder + TemporalInverseMLP
                # + per_group_pyramid_encoders (6 layers → 2 layers in gradient chain).
                if config.inverse_heads.temporal_latent_roundtrip:
                    temporal_latent_dim = 0
                    for fc in group_indices:
                        if not fc.startswith("temporal_"):
                            continue
                        for l in range(config.hierarchy.num_levels):
                            temporal_latent_dim += self._get_latent_dim(fc, l)

                    temporal_output_dim = (
                        learned_cfg.num_groups * config.encoder.embedding_dim
                    )  # 30 × D_group
                    hidden = config.inverse_heads.temporal_hidden_dim

                    self.temporal_latent_decoder = nn.Sequential(
                        nn.Linear(temporal_latent_dim, hidden),
                        nn.LayerNorm(hidden),
                        nn.GELU(),
                        nn.Linear(hidden, hidden),
                        nn.LayerNorm(hidden),
                        nn.GELU(),
                        nn.Linear(hidden, temporal_output_dim),
                    )
                    logger.info(
                        f"Created temporal latent decoder (roundtrip bypass): "
                        f"{temporal_latent_dim}D → {temporal_output_dim}D "
                        f"({learned_cfg.num_groups} groups × "
                        f"{config.encoder.embedding_dim}D)"
                    )

        # Auxiliary heads for temporal-only mode (theta/IC decoded from temporal tokens)
        self.theta_aux_head: Optional[nn.Module] = None
        self.ic_aux_head: Optional[nn.Module] = None
        self.theta_probe: Optional[nn.Module] = None  # Pre-VQ theta probe
        self.ic_probe: Optional[nn.Module] = None     # Pre-VQ IC probe
        self.traj_prototype_head: Optional[TrajectoryPrototypeHead] = None

        if config.aux_heads is not None:
            # Compute total quantized latent dimension across all temporal groups
            temporal_groups = [fc for fc in group_indices if fc.startswith("temporal_")]
            total_quantized_dim = sum(
                self._get_latent_dim(fc, l)
                for fc in temporal_groups
                for l in range(config.hierarchy.num_levels)
            )

            if config.aux_heads.theta_enabled:
                theta_param_dim = (
                    config.encoder.theta.param_dim
                    if config.encoder.theta is not None and config.encoder.theta.param_dim is not None
                    else 14  # fallback, will be overridden at runtime
                )
                self.theta_aux_head = nn.Sequential(
                    nn.Linear(total_quantized_dim, config.aux_heads.theta_hidden_dim),
                    nn.LayerNorm(config.aux_heads.theta_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.aux_heads.theta_hidden_dim, theta_param_dim),
                    nn.Sigmoid(),  # params in [0,1]
                )
                logger.info(
                    f"Created theta aux head: {total_quantized_dim}D → {theta_param_dim}D"
                )

            if config.aux_heads.initial_enabled:
                initial_channels = (
                    config.encoder.initial.in_channels
                    if config.encoder.initial.in_channels is not None
                    else 3  # fallback, will be overridden at runtime
                )
                ic_spatial = config.encoder.initial.spatial_size or 64
                self.ic_aux_head = InitialInverseCNN(
                    encoded_dim=total_quantized_dim,
                    channels=initial_channels,
                    spatial_size=ic_spatial,
                )
                logger.info(
                    f"Created IC aux head: {total_quantized_dim}D → "
                    f"[{initial_channels}, {ic_spatial}, {ic_spatial}]"
                )

            # Pre-VQ theta probe: predicts theta from encoded temporal features
            # BEFORE quantization, giving the CNN a direct gradient path for
            # learning param-dependent dynamics (no VQ bottleneck in the way).
            if config.aux_heads.theta_probe_enabled:
                self.theta_probe = nn.Sequential(
                    nn.Linear(self.temporal_dim, config.aux_heads.theta_probe_hidden_dim),
                    nn.LayerNorm(config.aux_heads.theta_probe_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.aux_heads.theta_probe_hidden_dim, theta_param_dim),
                    nn.Sigmoid(),  # params in [0,1]
                )
                logger.info(
                    f"Created pre-VQ theta probe: {self.temporal_dim}D → {theta_param_dim}D"
                )

            # Pre-VQ IC probe: predicts ICs from encoded temporal features
            # BEFORE quantization, bypassing the VQ bottleneck for a direct
            # gradient signal to encode IC-relevant spatial patterns.
            if config.aux_heads.initial_probe_enabled:
                probe_channels = (
                    config.encoder.initial.in_channels
                    if config.encoder.initial.in_channels is not None
                    else 3
                )
                probe_spatial = config.encoder.initial.spatial_size or 64
                self.ic_probe = InitialInverseCNN(
                    encoded_dim=self.temporal_dim,
                    channels=probe_channels,
                    spatial_size=probe_spatial,
                )
                logger.info(
                    f"Created pre-VQ IC probe: {self.temporal_dim}D → "
                    f"[{probe_channels}, {probe_spatial}, {probe_spatial}]"
                )

            # Trajectory prototype head: decodes quantized temporal latents
            # → K keyframe images representing the behavioral class average.
            # Unlike the IC head (single frame at t=0), this produces a full
            # prototype trajectory that downstream systems can use for
            # behavioral adjacency sampling and navigation.
            if config.aux_heads.trajectory_enabled:
                traj_channels = (
                    config.encoder.initial.in_channels
                    if config.encoder.initial.in_channels is not None
                    else 3
                )
                self.traj_prototype_head = TrajectoryPrototypeHead(
                    encoded_dim=total_quantized_dim,
                    channels=traj_channels,
                    spatial_size=config.aux_heads.trajectory_spatial_size,
                    num_keyframes=config.aux_heads.trajectory_num_keyframes,
                    latent_dim=config.aux_heads.trajectory_latent_dim,
                    base_ch=config.aux_heads.trajectory_base_channels,
                )
                logger.info(
                    f"Created trajectory prototype head: {total_quantized_dim}D → "
                    f"[{config.aux_heads.trajectory_num_keyframes}, {traj_channels}, "
                    f"{config.aux_heads.trajectory_spatial_size}, {config.aux_heads.trajectory_spatial_size}]"
                )

        # Compute total encoded dimension
        total_encoded_dim = self.temporal_dim + self.initial_dim + self.theta_dim

        # Create categorical projectors (one per family-category)
        # Use self.group_indices (filtered in temporal-only mode)
        self.projectors = nn.ModuleDict()
        for family_cat, indices in self.group_indices.items():
            family, _ = family_cat.split('_', 1)

            # Determine source dimension for this category from the ENCODER
            # output, not from raw feature indices. Each projector receives the
            # output of its family's encoder, so the input dim must match.
            if family == "temporal" and self.pyramid_first_encoder is not None:
                cat_dim = config.encoder.embedding_dim  # D_group
            elif family == "temporal" and self.per_group_pyramid_encoders is not None:
                cat_dim = sum(config.encoder.temporal.level_dims)
            elif family == "temporal" and self.per_group_temporal_encoders is not None:
                cat_dim = config.encoder.embedding_dim
            elif family == "theta" and getattr(self, '_theta_direct', False):
                # Direct mode: each theta_param_i group receives 1D input
                cat_dim = 1
            elif family == "theta":
                cat_dim = self.theta_dim
            elif family == "initial" and family_cat.startswith("initial_spatial_"):
                # Spatial IC: each position gets spatial_token_dim as input
                cat_dim = config.encoder.initial.spatial_token_dim
            elif family == "initial":
                cat_dim = self.initial_dim
            else:
                cat_dim = len(indices)

            # Create hierarchical projector
            # Spatial IC groups with FSQ use 1 level (FSQ handles discretization directly)
            if (
                family_cat.startswith("initial_spatial_")
                and config.encoder.initial.spatial_quantizer == "fsq"
            ):
                self.projectors[family_cat] = self._create_projector(
                    family_cat, cat_dim, num_levels_override=1,
                    latent_dim_override=len(config.encoder.initial.fsq_levels),
                )
            else:
                self.projectors[family_cat] = self._create_projector(
                    family_cat, cat_dim
                )

        # Create vector quantizers (N×L total, one per family-category-level)
        # Spatial IC groups get FSQ (1 level) or VQ (3 levels) based on config.
        self.quantizers = nn.ModuleDict()
        for family_cat in self.group_indices.keys():
            family, _ = family_cat.split('_', 1)

            if (
                family_cat.startswith("initial_spatial_")
                and config.encoder.initial.spatial_quantizer == "fsq"
            ):
                # FSQ: single level, implicit codebook
                quantizer_key = f"{family_cat}_L0"
                self.quantizers[quantizer_key] = FiniteScalarQuantizer(
                    levels=config.encoder.initial.fsq_levels,
                )
                continue

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

        # Log hierarchy dimensions (once, for first group)
        if self.group_indices:
            first_fc = sorted(self.group_indices.keys())[0]
            num_levels = config.hierarchy.num_levels
            level_dims = [self._get_latent_dim(first_fc, l) for l in range(num_levels)]
            level_codes = [self._get_num_embeddings(first_fc, l) for l in range(num_levels)]
            # Show the projector's input dim, not the raw feature count
            family = first_fc.split("_", 1)[0]
            if family == "temporal" and self.pyramid_first_encoder is not None:
                input_dim = config.encoder.embedding_dim
            else:
                input_dim = len(self.group_indices[first_fc])
            logger.info(
                "Hierarchy (example %s): input_dim=%d → %s",
                first_fc,
                input_dim,
                ", ".join(
                    f"L{l}={level_dims[l]}D/{level_codes[l]}codes"
                    for l in range(num_levels)
                ),
            )
            total_per_group = sum(level_dims)
            total_all = total_per_group * len(self.group_indices)
            logger.info(
                "  Total quantized dim: %dD/group × %d groups = %dD",
                total_per_group,
                len(self.group_indices),
                total_all,
            )

        # Shared decoder: total quantized dim must match concatenated quantizer outputs.
        # Spatial IC groups with FSQ have 1 level; VQ spatial and everything else have num_levels.
        total_latent_dim = 0
        for fc in self.group_indices.keys():
            if (
                fc.startswith("initial_spatial_")
                and config.encoder.initial.spatial_quantizer == "fsq"
            ):
                # FSQ: 1 level, dim = len(fsq_levels)
                total_latent_dim += len(config.encoder.initial.fsq_levels)
            else:
                for l in range(config.hierarchy.num_levels):
                    total_latent_dim += self._get_latent_dim(fc, l)

        self.decoder = MultiLayerResidualDecoder(
            latent_dim=total_latent_dim,
            input_dim=total_encoded_dim,
            dropout=config.encoder.dropout,
        )

    def _parse_families(self, group_indices: Dict[str, List[int]]) -> List[str]:
        """Parse unique families from group_indices keys.

        Family presence is derived from group_indices: if there are keys
        starting with "theta_", the theta family exists. No boolean flag needed.

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

    def _use_per_group_pyramid(self) -> bool:
        """True when pca_raw grouping + pyramid variant: per-group pyramid encoders active.

        pca_raw assigns raw temporal features to groups by dominant PCA loading.
        Each group's slice [B, T, G_k] is passed directly to its own PyramidTemporalEncoder,
        preserving full temporal dynamics without any rotation at inference.
        """
        return (
            self.config.grouping is not None
            and getattr(self.config.grouping, "method", "correlation") == "pca_raw"
            and getattr(self.config.encoder.temporal, "variant", "mean") == "pyramid"
        )

    def _setup_per_group_pyramid_encoders(
        self,
        temporal_groups: dict,
    ) -> None:
        """Create one PyramidTemporalEncoder per temporal group (pca_raw path).

        Each encoder operates on the group's raw feature slice [B, T, G_k] and
        produces [B, sum(level_dims)] = [B, embedding_dim].  Group sizes vary;
        input_dim is set per-group from len(indices).

        Args:
            temporal_groups: Dict mapping group_name → list of raw feature indices.
        """
        pyramid_cfg = self.config.encoder.temporal

        # Build variable_length_config dict if configured (same logic as shared pyramid)
        vl_config = None
        if pyramid_cfg.variable_length:
            if isinstance(pyramid_cfg.variable_length, bool):
                vl_config = {
                    "enabled": True,
                    "adaptive_pyramid": pyramid_cfg.adaptive_pyramid,
                    "min_pyramid_length": pyramid_cfg.min_timesteps,
                }
            else:
                vl_cfg = pyramid_cfg.variable_length
                vl_config = {
                    "enabled": vl_cfg.enabled,
                    "adaptive_pyramid": vl_cfg.adaptive_pyramid,
                    "min_pyramid_length": vl_cfg.min_pyramid_length,
                    "mask_downsample_method": vl_cfg.mask_downsample_method,
                }

        self.per_group_pyramid_encoders = nn.ModuleDict({
            group_name: PyramidTemporalEncoder(
                input_dim=len(indices),
                level_dims=pyramid_cfg.level_dims,
                variable_length_config=vl_config,
            )
            for group_name, indices in temporal_groups.items()
            if group_name.startswith("temporal_")
        })

        embedding_dim = sum(pyramid_cfg.level_dims)
        logger.info(
            "Per-group pyramid encoders: %d groups × %dD = %dD total temporal dim",
            len(self.per_group_pyramid_encoders),
            embedding_dim,
            len(self.per_group_pyramid_encoders) * embedding_dim,
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
        # Detect model's current device so buffers land on the right device
        device = next(self.parameters()).device
        mean_t = torch.from_numpy(transform.mean).float().to(device)
        comps_t = torch.from_numpy(transform.components).float().to(device)
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
        if "temporal" in self.families and config.feature_source == "learned" and (
            config.encoder.temporal.learned is not None
            and config.encoder.temporal.learned.architecture == "pyramid_first"
        ):
            # ── Pyramid-first mode: raw trajectory → pyramid → CNN → learned groups ──
            learned_cfg = config.encoder.temporal.learned
            pyramid_cfg = config.encoder.temporal

            # Parse variable-length config for pyramid
            vl_kwargs = {}
            if pyramid_cfg.variable_length:
                if isinstance(pyramid_cfg.variable_length, bool):
                    vl_kwargs = {
                        "adaptive": pyramid_cfg.adaptive_pyramid,
                        "min_pyramid_length": pyramid_cfg.min_timesteps,
                    }
                else:
                    vl_cfg = pyramid_cfg.variable_length
                    vl_kwargs = {
                        "adaptive": vl_cfg.adaptive_pyramid,
                        "min_pyramid_length": vl_cfg.min_pyramid_length,
                        "mask_downsample_method": vl_cfg.mask_downsample_method,
                    }

            self.pyramid_first_encoder = PyramidFirstEncoder(
                in_channels=learned_cfg.in_channels,
                d_cnn=learned_cfg.d_cnn,
                d_agg=learned_cfg.d_agg,
                num_groups=learned_cfg.num_groups,
                d_group=config.encoder.embedding_dim,  # D_group = per-group VQ input dim
                downsample_factors=pyramid_cfg.downsample_factors,
                gated_groups=learned_cfg.gated_groups,
                gate_init_bias=learned_cfg.gate_init_bias,
                **vl_kwargs,
            )

            temporal_groups = {
                k: v for k, v in self.group_indices.items()
                if k.startswith("temporal_")
            }
            self.temporal_dim = len(temporal_groups) * config.encoder.embedding_dim
            # Store per-group dim for roundtrip re-encoding (used by losses.py)
            self._temporal_group_dim = config.encoder.embedding_dim

            # Roundtrip re-encoding path: lightweight per-group PyramidTemporalEncoders
            # that take decoded per-group features [B, T_rt, D_group] → [B, sum(level_dims)].
            # These are separate from PyramidFirstEncoder (which takes raw pixels).
            vl_config_rt = None
            if pyramid_cfg.variable_length:
                if isinstance(pyramid_cfg.variable_length, bool):
                    vl_config_rt = {
                        "enabled": True,
                        "adaptive_pyramid": pyramid_cfg.adaptive_pyramid,
                        "min_pyramid_length": pyramid_cfg.min_timesteps,
                    }
                else:
                    vl_cfg = pyramid_cfg.variable_length
                    vl_config_rt = {
                        "enabled": vl_cfg.enabled,
                        "adaptive_pyramid": vl_cfg.adaptive_pyramid,
                        "min_pyramid_length": vl_cfg.min_pyramid_length,
                        "mask_downsample_method": vl_cfg.mask_downsample_method,
                    }

            pyramid_out_dim = sum(pyramid_cfg.level_dims)
            self.per_group_pyramid_encoders = nn.ModuleDict({
                group_name: PyramidTemporalEncoder(
                    input_dim=config.encoder.embedding_dim,  # D_group
                    level_dims=pyramid_cfg.level_dims,
                    variable_length_config=vl_config_rt,
                )
                for group_name in temporal_groups.keys()
            })
            # Projection from pyramid output (sum(level_dims)) to D_group so that
            # the existing projectors (created with input_dim=D_group) accept the
            # roundtrip-encoded features. In per_frame mode this isn't needed
            # because the projector is created with input_dim=sum(level_dims).
            self._temporal_rt_projections = nn.ModuleDict({
                group_name: nn.Linear(pyramid_out_dim, config.encoder.embedding_dim)
                for group_name in temporal_groups.keys()
            })

            logger.info(
                "Pyramid-first mode: %d levels × %dD_agg → %d groups × %dD = %dD temporal "
                "(+ %d roundtrip pyramid encoders, %dD→%dD projection)",
                len(pyramid_cfg.downsample_factors),
                learned_cfg.d_agg,
                len(temporal_groups),
                config.encoder.embedding_dim,
                self.temporal_dim,
                len(self.per_group_pyramid_encoders),
                pyramid_out_dim,
                config.encoder.embedding_dim,
            )

        elif "temporal" in self.families and config.feature_source == "learned":
            # ── Legacy per-frame learned mode: per-frame CNN + per-group pyramid encoders ──
            learned_cfg = config.encoder.temporal.learned
            if learned_cfg is None:
                raise ValueError(
                    "feature_source='learned' requires encoder.temporal.learned config"
                )
            self.temporal_cnn_encoder = TemporalCNNFeatureEncoder(
                in_channels=learned_cfg.in_channels,
                embedding_dim=learned_cfg.embedding_dim,
            )
            group_dim = learned_cfg.embedding_dim // learned_cfg.num_groups

            # Set up per-group pyramid encoders on CNN output slices
            temporal_groups = {
                k: v for k, v in self.group_indices.items()
                if k.startswith("temporal_")
            }
            pyramid_cfg = config.encoder.temporal

            vl_config = None
            if pyramid_cfg.variable_length:
                if isinstance(pyramid_cfg.variable_length, bool):
                    vl_config = {
                        "enabled": True,
                        "adaptive_pyramid": pyramid_cfg.adaptive_pyramid,
                        "min_pyramid_length": pyramid_cfg.min_timesteps,
                    }
                else:
                    vl_cfg = pyramid_cfg.variable_length
                    vl_config = {
                        "enabled": vl_cfg.enabled,
                        "adaptive_pyramid": vl_cfg.adaptive_pyramid,
                        "min_pyramid_length": vl_cfg.min_pyramid_length,
                        "mask_downsample_method": vl_cfg.mask_downsample_method,
                    }

            self.per_group_pyramid_encoders = nn.ModuleDict({
                group_name: PyramidTemporalEncoder(
                    input_dim=group_dim,
                    level_dims=pyramid_cfg.level_dims,
                    variable_length_config=vl_config,
                )
                for group_name in temporal_groups.keys()
            })

            embedding_dim = sum(pyramid_cfg.level_dims)
            self.temporal_dim = len(temporal_groups) * embedding_dim
            logger.info(
                "Learned mode (per_frame): CNN(%dD) → %d groups × %dD pyramid = %dD temporal",
                learned_cfg.embedding_dim,
                len(temporal_groups),
                embedding_dim,
                self.temporal_dim,
            )

        elif "temporal" in self.families:
            if config.encoder.temporal.variant == "pyramid" and self._use_per_group_pyramid():
                # ── pca_raw + pyramid: per-group pyramid encoders ────────────
                # No shared temporal_encoder; each group slice goes to its own
                # PyramidTemporalEncoder([B, T, G_k]) → [B, embedding_dim].
                temporal_groups = {
                    k: v for k, v in self.group_indices.items()
                    if k.startswith("temporal_")
                }
                self._setup_per_group_pyramid_encoders(temporal_groups)
                embedding_dim = sum(config.encoder.temporal.level_dims)
                self.temporal_dim = len(temporal_groups) * embedding_dim

            elif config.encoder.temporal.variant == "pyramid":
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
                    manual_dim=self.initial_input_dim,   # auto-detected from dataset
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

            elif config.encoder.initial.variant == "spectral":
                # group_dim: compress spectral features per group for VQ.
                # Uses encoder.embedding_dim (same as temporal groups) so VQ
                # operates on manageable dimensionality.
                group_dim = config.encoder.embedding_dim  # e.g. 32
                self.initial_encoder = SpectralICEncoder(
                    in_channels=config.encoder.initial.in_channels,
                    spatial_size=config.encoder.initial.spatial_size or 128,
                    num_modes=config.encoder.initial.num_modes,
                    num_groups=config.encoder.initial.num_initial_groups,
                    group_dim=group_dim,
                )
                self.initial_dim = self.initial_encoder.output_dim
                logger.info(
                    f"Spectral IC encoder: {config.encoder.initial.in_channels}ch × "
                    f"{config.encoder.initial.num_modes}² modes → {self.initial_dim}D"
                )

            elif config.encoder.initial.variant == "spatial":
                ic_cfg = config.encoder.initial
                self.initial_encoder = SpatialICEncoder(
                    in_channels=ic_cfg.in_channels,
                    spatial_token_grid=ic_cfg.spatial_token_grid,
                    spatial_token_dim=ic_cfg.spatial_token_dim,
                    base_channels=ic_cfg.spatial_base_channels,
                )
                self.initial_dim = self.initial_encoder.output_dim
                logger.info(
                    f"Spatial IC encoder: {ic_cfg.in_channels}ch → "
                    f"{ic_cfg.spatial_token_grid}×{ic_cfg.spatial_token_grid} grid, "
                    f"{ic_cfg.spatial_token_dim}D/position, "
                    f"FSQ {ic_cfg.fsq_levels} → {self.initial_dim}D total"
                )
            else:
                raise ValueError(f"Unknown initial variant: {config.encoder.initial.variant}")

        # Theta encoder
        if "theta" in self.families:
            theta_cfg = config.encoder.theta
            if theta_cfg is None:
                raise ValueError("Theta family enabled but theta encoder config missing")

            if theta_cfg.variant == "direct":
                # Direct quantization: no encoder, raw params pass through.
                # Each parameter gets its own independent VQ group.
                self.theta_dim = theta_cfg.param_dim
                self._theta_direct = True
                logger.info(
                    f"Direct theta quantization: {theta_cfg.param_dim} params → "
                    f"{theta_cfg.param_dim} independent VQ groups"
                )
            elif theta_cfg.variant == "mlp":
                self.theta_encoder = ThetaMLPEncoder(
                    param_dim=theta_cfg.param_dim,
                    hidden_dim=theta_cfg.hidden_dim,
                    output_dim=theta_cfg.output_dim,
                    dropout=theta_cfg.dropout,
                    use_layer_norm=theta_cfg.use_layer_norm,
                )
                self.theta_dim = theta_cfg.output_dim
                self._theta_direct = False
            else:
                raise ValueError(f"Unknown theta encoder variant: {theta_cfg.variant}")

    def _create_projector(
        self,
        family_cat: str,
        category_dim: int,
        num_levels_override: int | None = None,
        latent_dim_override: int | None = None,
    ) -> HierarchicalProjector:
        """Create hierarchical projector for a family-category.

        Args:
            family_cat: Family-category key (e.g., "temporal_group_1")
            category_dim: Number of features in this category
            num_levels_override: Override number of hierarchy levels (e.g. 1 for FSQ)
            latent_dim_override: Override latent dim for all levels (e.g. len(fsq_levels))

        Returns:
            HierarchicalProjector instance
        """
        config = self.config
        num_levels = num_levels_override or config.hierarchy.num_levels

        # Build level configs
        levels = []
        for level_idx in range(num_levels):
            if latent_dim_override is not None:
                latent_dim = latent_dim_override
            else:
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

        When ``config.hierarchy.level_ratios`` is set, uses explicit ratios instead:
            latent_dim = ceil(input_dim × ratio / 4) × 4

        Args:
            family_cat: Family-category key
            level_idx: Hierarchy level index
            n_samples: Number of samples in dataset

        Returns:
            Latent dimension for this level
        """
        import numpy as np

        # Explicit override: level_ratios from config
        if self.config.hierarchy.level_ratios is not None:
            ratios = self.config.hierarchy.level_ratios
            if level_idx >= len(ratios):
                raise ValueError(
                    f"level_idx {level_idx} >= len(level_ratios) {len(ratios)}"
                )
            input_dim = len(self.group_indices[family_cat])
            latent_dim = int(np.ceil(input_dim * ratios[level_idx] / 4.0)) * 4
            latent_dim = max(self.config.hierarchy.min_latent_dim, latent_dim)
            latent_dim = min(self.config.hierarchy.max_latent_dim, latent_dim)
            return latent_dim

        category_feature_count = len(self.group_indices[family_cat])

        # For tiny categories (1-2 features, e.g., direct theta params),
        # use fixed small dims rather than the adaptive formula which
        # over-allocates due to dataset-aware minimums.
        if category_feature_count <= 2:
            tiny_dims = [8, 4, 4]  # L0, L1, L2
            return tiny_dims[min(level_idx, len(tiny_dims) - 1)]

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

        # L1: dataset-aware minimum (spare capacity for diverse data)
        if level_idx == 1 and n_samples > 1000:
            min_tokens_l1 = min(16, max(8, int(np.sqrt(n_samples / 1000.0) * 2.4)))
            num_tokens = max(min_tokens_l1, num_tokens)
        else:
            # L2+: preserve standard minimum (6)
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
        temporal_raw: Optional[torch.Tensor] = None,
        encode_only: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through joint VQ-VAE.

        Args:
            temporal_features: Temporal sequences [B, T, D_t] (required if temporal family exists in manual mode)
            initial_manual: Manual initial features [B, D_i_manual] (required if initial_hybrid)
            initial_raw: Raw initial conditions [B, C, H, W] (required if initial_hybrid)
            theta_features: Operator parameters [B, param_dim] (required if theta family exists)
            temporal_mask: Validity mask for temporal [B, T] (optional)
            temporal_lengths: Actual sequence lengths [B] (optional)
            temporal_raw: Raw trajectory [B, T, C, H, W] (required in learned feature mode)

        Returns:
            Dict with keys:
                - reconstructed: Reconstructed features [B, total_dim]
                - quantized: Quantized latent vectors [B, total_latent_dim]
                - vq_loss: Vector quantization loss
                - perplexity: Average codebook perplexity
                - encodings: Dict of per-family-category-level encodings
        """
        # Determine batch size from available inputs
        if temporal_raw is not None:
            batch_size = temporal_raw.shape[0]
        elif temporal_features is not None:
            batch_size = temporal_features.shape[0]
        elif initial_manual is not None:
            batch_size = initial_manual.shape[0]
        elif theta_features is not None:
            batch_size = theta_features.shape[0]
        else:
            raise ValueError("At least one feature input must be provided")

        # Encode families
        encoded = {}
        cnn_features = None  # Only set in learned mode

        if "temporal" in self.families and self.pyramid_first_encoder is not None:
            # ── Pyramid-first mode: raw trajectory → pyramid → CNN → learned groups ──
            if temporal_raw is None:
                raise ValueError(
                    "temporal_raw [B, T, C, H, W] required in pyramid_first mode"
                )
            per_group, multi_res_features = self.pyramid_first_encoder(
                temporal_raw, mask=temporal_mask, lengths=temporal_lengths
            )
            cnn_features = multi_res_features  # [B, num_levels*D_agg] for balance loss

            group_encoded_parts = []
            group_idx = 0
            for family_cat in sorted(self.group_indices.keys()):
                if not family_cat.startswith("temporal_"):
                    continue
                encoded[family_cat] = per_group[:, group_idx, :]  # [B, D_group]
                group_encoded_parts.append(encoded[family_cat])
                group_idx += 1
            encoded["temporal"] = torch.cat(group_encoded_parts, dim=1)

        elif "temporal" in self.families and self.temporal_cnn_encoder is not None:
            # ── Learned mode: raw trajectory → CNN → per-group pyramid ──
            if temporal_raw is None:
                raise ValueError(
                    "temporal_raw [B, T, C, H, W] required in learned feature mode"
                )
            # CNN: [B, T, C, H, W] → [B, T, D_per_frame]
            cnn_features = self.temporal_cnn_encoder(temporal_raw)

            # Apply OPQ rotation if calibrated (decorrelates features across groups)
            if self.temporal_rotation_matrix is not None:
                cnn_features = (
                    (cnn_features - self.temporal_rotation_mean)
                    @ self.temporal_rotation_matrix.T
                )

            learned_cfg = self.config.encoder.temporal.learned
            group_dim = learned_cfg.embedding_dim // learned_cfg.num_groups

            # Per-group pyramid encoding on sequential slices
            group_encoded_parts = []
            group_idx = 0
            for family_cat in sorted(self.group_indices.keys()):
                if not family_cat.startswith("temporal_"):
                    continue
                # Sequential slice: group i gets features [i*gd : (i+1)*gd]
                group_temporal = cnn_features[:, :, group_idx * group_dim:(group_idx + 1) * group_dim]
                encoder = self.per_group_pyramid_encoders[family_cat]
                if temporal_mask is not None:
                    enc, _ = encoder(group_temporal, mask=temporal_mask, lengths=temporal_lengths)
                else:
                    enc = encoder(group_temporal)
                encoded[family_cat] = enc
                group_encoded_parts.append(enc)
                group_idx += 1

            encoded["temporal"] = torch.cat(group_encoded_parts, dim=1)

        elif "temporal" in self.families:
            if temporal_features is None:
                raise ValueError("temporal_features required for temporal family")

            if self.per_group_pyramid_encoders is not None:
                # ── pca_raw path: per-group pyramid encoders on raw feature slices ──
                # No rotation. Each group's raw features are sliced from the temporal
                # tensor [B, T, D_t] → [B, T, G_k] and encoded by its own
                # PyramidTemporalEncoder → [B, embedding_dim].
                group_encoded_parts = []
                for family_cat, indices in self.group_indices.items():
                    if not family_cat.startswith("temporal_"):
                        continue
                    idx_t = torch.tensor(
                        indices, device=temporal_features.device, dtype=torch.long
                    )
                    group_temporal = temporal_features[:, :, idx_t]  # [B, T, G_k]
                    encoder = self.per_group_pyramid_encoders[family_cat]
                    if temporal_mask is not None:
                        enc, _ = encoder(
                            group_temporal, mask=temporal_mask, lengths=temporal_lengths
                        )
                    else:
                        enc = encoder(group_temporal)
                    encoded[family_cat] = enc
                    group_encoded_parts.append(enc)

                encoded["temporal"] = torch.cat(group_encoded_parts, dim=1)

            elif self.per_group_temporal_encoders is not None:
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
            elif isinstance(self.initial_encoder, SpatialICEncoder):
                if initial_raw is None:
                    raise ValueError("initial_raw required for spatial IC encoder")
                init_encoded = self.initial_encoder(initial_raw)  # [B, G²×token_dim]
                # Split into per-position entries for the quantizer loop
                for i in range(self.initial_encoder.num_positions):
                    start = i * self.initial_encoder.spatial_token_dim
                    end = start + self.initial_encoder.spatial_token_dim
                    encoded[f"initial_spatial_{i}"] = init_encoded[:, start:end]
            elif isinstance(self.initial_encoder, SpectralICEncoder):
                if initial_raw is None:
                    raise ValueError("initial_raw required for spectral IC encoder")
                init_encoded = self.initial_encoder(initial_raw)
            else:
                if initial_raw is None:
                    raise ValueError("initial_raw required for initial_cnn")
                init_encoded = self.initial_encoder(initial_raw)

            encoded["initial"] = init_encoded

        if "theta" in self.families:
            if theta_features is None:
                raise ValueError("theta_features required for theta family")

            if getattr(self, '_theta_direct', False):
                # Direct mode: raw parameters, each param → its own group
                encoded["theta"] = theta_features  # [B, P] for decoder
                theta_keys = sorted(
                    k for k in self.group_indices if k.startswith("theta_")
                )
                for i, key in enumerate(theta_keys):
                    encoded[key] = theta_features[:, i:i+1]  # [B, 1]
            else:
                # MLP mode: encode all params into shared embedding
                theta_encoded = self.theta_encoder(theta_features)
                encoded["theta"] = theta_encoded

        # Concatenate all encoded features
        all_encoded = []
        for family in sorted(self.families):
            all_encoded.append(encoded[family])
        all_encoded = torch.cat(all_encoded, dim=1)  # [B, total_encoded_dim]

        # Project to categorical latent spaces and quantize
        all_quantized = []
        temporal_quantized = []  # Temporal-only quantized vectors (for aux heads)
        vq_losses = []
        perplexities = []
        encodings_dict = {}
        latents_dict = {}  # Pre-quantization latent vectors

        for family_cat, indices in self.group_indices.items():
            family, cat_name = family_cat.split('_', 1)

            # Extract features for this category from the ENCODER output.
            # Each family's encoder produces the correct feature tensor;
            # we use it directly instead of slicing a concatenated tensor.
            if family == "temporal" and (
                self.per_group_temporal_encoders is not None
                or self.per_group_pyramid_encoders is not None
                or self.pyramid_first_encoder is not None
            ):
                cat_features = encoded[family_cat]  # [B, per_group_dim]
            elif family_cat.startswith("initial_spatial_") and family_cat in encoded:
                # Spatial IC: per-position features from SpatialICEncoder
                cat_features = encoded[family_cat]  # [B, spatial_token_dim]
            elif family == "theta" and family_cat in encoded:
                # Direct theta mode: per-param entries; MLP mode: falls through
                cat_features = encoded[family_cat]  # [B, 1] or [B, theta_dim]
            elif family in encoded:
                cat_features = encoded[family]  # [B, family_encoded_dim]
            else:
                cat_features = all_encoded[:, indices]  # fallback for legacy

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
                if family == "temporal":
                    temporal_quantized.append(quantized)
                vq_losses.append(losses['loss'])
                # Compute perplexity from encodings
                avg_probs = encodings.mean(dim=0)
                perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
                perplexities.append(perplexity)

                encodings_dict[quantizer_key] = quantized

        # Concatenate all quantized vectors
        all_quantized_cat = torch.cat(all_quantized, dim=1)  # [B, total_latent_dim]

        # Early return for encode-only path: skip decoder, inverse heads, aux heads.
        # Used by encode() to avoid wasted computation during inference.
        if encode_only:
            token_indices = self._get_token_indices(encodings_dict)
            return {"encodings": encodings_dict, "token_indices": token_indices}

        # Decode to reconstruct
        reconstructed = self.decoder(all_quantized_cat)  # [B, total_encoded_dim]

        # Split reconstructed back into family components
        reconstructed_split = self._split_reconstructed(reconstructed, all_encoded)

        # Apply decode heads: inverse heads (within-family) and aux heads
        # (cross-family) can both be active simultaneously.
        decoded = {}

        # ── Inverse heads (active when families have inverse configs) ──
        # Theta bypass: concatenated theta quantized latents → MLP → params
        if self.theta_bypass_decoder is not None:
            theta_q_parts = []
            for key in sorted(encodings_dict.keys()):
                if key.startswith("theta_"):
                    theta_q_parts.append(encodings_dict[key])
            if theta_q_parts:
                theta_q_cat = torch.cat(theta_q_parts, dim=1)
                decoded["theta"] = self.theta_bypass_decoder(theta_q_cat)
        elif self.theta_inverse is not None and "theta" in reconstructed_split:
            decoded["theta"] = self.theta_inverse(reconstructed_split["theta"])
        # IC: gather quantized latents (shared between bypass decoder and latent decoder)
        initial_q_cat = None
        if self.initial_bypass_inverse is not None or self.initial_latent_decoder is not None:
            initial_q_parts = []
            for key in sorted(encodings_dict.keys()):
                if key.startswith("initial_"):
                    initial_q_parts.append(encodings_dict[key])
            if initial_q_parts:
                initial_q_cat = torch.cat(initial_q_parts, dim=1)
        # IC bypass: concatenated IC quantized latents → spectral/spatial inverse → grid
        if self.initial_bypass_inverse is not None and initial_q_cat is not None:
            decoded["initial"] = self.initial_bypass_inverse(initial_q_cat)
        elif self.initial_inverse is not None and "initial" in reconstructed_split:
            decoded["initial"] = self.initial_inverse(reconstructed_split["initial"])
        # IC latent decoder for roundtrip (skip CNN re-encoding)
        if self.initial_latent_decoder is not None and initial_q_cat is not None:
            decoded["initial_latent"] = self.initial_latent_decoder(initial_q_cat)
        if self.temporal_inverse is not None and "temporal" in reconstructed_split:
            decoded["temporal"] = self.temporal_inverse(reconstructed_split["temporal"])
        # Temporal latent decoder for roundtrip (skip shared decoder path)
        if self.temporal_latent_decoder is not None:
            temporal_q_parts = []
            for key in sorted(encodings_dict.keys()):
                if key.startswith("temporal_"):
                    temporal_q_parts.append(encodings_dict[key])
            if temporal_q_parts:
                temporal_q_cat = torch.cat(temporal_q_parts, dim=1)
                decoded["temporal_latent"] = self.temporal_latent_decoder(
                    temporal_q_cat
                )

        # ── Aux heads (cross-family supervision — always active if configured) ──
        # Aux heads take TEMPORAL quantized features only (not all families),
        # testing whether temporal tokens alone encode enough to predict theta/IC.
        # Use _aux suffix when both inverse and aux heads are active to avoid
        # key collision (e.g. "theta" from inverse + "theta_aux" from aux).
        temporal_quantized_cat = (
            torch.cat(temporal_quantized, dim=1) if temporal_quantized
            else all_quantized_cat  # fallback: temporal-only models
        )
        if self.theta_aux_head is not None:
            key = "theta_aux" if "theta" in decoded else "theta"
            decoded[key] = self.theta_aux_head(temporal_quantized_cat)
        if self.ic_aux_head is not None:
            key = "initial_aux" if "initial" in decoded else "initial"
            decoded[key] = self.ic_aux_head(temporal_quantized_cat)
        if self.traj_prototype_head is not None:
            decoded["trajectory_prototype"] = self.traj_prototype_head(temporal_quantized_cat)
        # Pre-VQ probes: direct CNN → gradient (no VQ bottleneck)
        if self.theta_probe is not None and "temporal" in encoded:
            decoded["theta_probe"] = self.theta_probe(encoded["temporal"])
        if self.ic_probe is not None and "temporal" in encoded:
            decoded["initial_probe"] = self.ic_probe(encoded["temporal"])

        # Aggregate VQ losses
        total_vq_loss = torch.stack(vq_losses).mean()
        avg_perplexity = torch.stack(perplexities).mean()

        # Extract token indices for topographic loss
        token_indices = self._get_token_indices(encodings_dict)

        # Per-group encoded dict (for informativeness loss — uses correct dimensions)
        # Temporal groups have per-group keys in `encoded`; initial/theta use
        # the family-level key directly (each is a single group).
        per_group_encoded = {}
        for group_key in self.group_indices:
            if group_key in encoded:
                per_group_encoded[group_key] = encoded[group_key]
            else:
                family = group_key.split("_", 1)[0]
                if family in encoded:
                    per_group_encoded[group_key] = encoded[family]

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
            "per_group_encoded": per_group_encoded,
            "cnn_features": cnn_features if (self.temporal_cnn_encoder is not None or self.pyramid_first_encoder is not None) else None,
            "gate_values": self.pyramid_first_encoder.group_proj.get_gate_values() if self.pyramid_first_encoder is not None else None,
        }

    def encode(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        temporal_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode inputs to discrete token indices.

        Uses encode_only=True to skip decoder, inverse/aux heads, and avoids
        redundant cdist computation (token_indices are computed once in forward).

        Args:
            Same as forward()

        Returns:
            Dict mapping "family_category_Ll" → token indices [B]
        """
        outputs = self.forward(
            temporal_features=temporal_features,
            initial_manual=initial_manual,
            initial_raw=initial_raw,
            theta_features=theta_features,
            temporal_mask=temporal_mask,
            temporal_lengths=temporal_lengths,
            temporal_raw=temporal_raw,
            encode_only=True,
        )
        return outputs["token_indices"]

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
            if isinstance(quantizer, FiniteScalarQuantizer):
                # FSQ: mixed-radix index from quantized values
                indices = quantizer.values_to_indices(quantized)
            else:
                # VQ: nearest codebook entry
                distances = torch.cdist(
                    quantized,
                    quantizer.embedding.weight,
                    p=2.0
                )
                indices = distances.argmin(dim=1)  # [B]
            token_indices[quantizer_key] = indices

        return token_indices
