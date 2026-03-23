"""NLTokenizerModel — continuous VAE model for NL expression generation.

Supports two feature modes (matching VQTokenizer's dual paths):

Learned mode (production — default):
    temporal_raw [B, T, C, H, W] → PyramidFirstEncoder → per_group [B, G, D_group]
    → flatten + theta_encoded → h → VAE → z → LFM decoder → NL text

Manual mode (legacy):
    temporal_features [B, T, D] → PyramidTemporalEncoder → embedding
    + theta_encoded → h → VAE → z → LFM decoder → NL text
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base_model import BaseTokenizerModel
from .nl_config import NLTokenizerConfig

logger = logging.getLogger(__name__)


class NLTokenizerModel(BaseTokenizerModel):
    """Continuous VAE model for encoding dynamics into NL-compatible latents.

    The latent z is structured as (z_coarse ‖ z_fine) — a contiguous vector
    with semantic boundary at `coarse_dim`.

    Args:
        config: NLTokenizerConfig with encoder, VAE, and inverse settings
        group_indices: Dict mapping "family_category" → feature indices
        temporal_input_dim: Temporal feature dim (manual mode only)
        theta_param_dim: Auto-detected theta parameter dimension
        initial_input_dim: Auto-detected initial feature dimension
    """

    def __init__(
        self,
        config: NLTokenizerConfig,
        group_indices: Dict[str, List[int]],
        temporal_input_dim: Optional[int] = None,
        theta_param_dim: Optional[int] = None,
        initial_input_dim: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.group_indices = group_indices
        self.families = self.parse_families(group_indices)
        self._learned_mode = config.feature_source == "learned"
        logger.info(
            "NLTokenizerModel families=%s mode=%s",
            self.families, config.feature_source,
        )

        # Store auto-detected dims
        self._temporal_input_dim = temporal_input_dim
        self._theta_param_dim = theta_param_dim
        self._initial_input_dim = initial_input_dim

        # ── Family encoders ──
        self._create_encoders()

        # Total encoded dim = sum of all family encoder outputs
        total_encoded_dim = self._compute_total_encoded_dim()
        self._total_encoded_dim = total_encoded_dim
        logger.info(f"Total encoded dim: {total_encoded_dim}")

        # ── VAE encoder: h → (μ, logvar) ──
        vae_cfg = config.vae
        encoder_layers = []
        in_dim = total_encoded_dim
        for h_dim in vae_cfg.encoder_hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        self.vae_encoder = nn.Sequential(*encoder_layers)
        self.mu_head = nn.Linear(in_dim, vae_cfg.latent_dim)
        self.logvar_head = nn.Linear(in_dim, vae_cfg.latent_dim)

        # ── Feature decoder: z → ĥ ──
        decoder_layers = []
        in_dim = vae_cfg.latent_dim
        for h_dim in vae_cfg.decoder_hidden_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, total_encoded_dim))
        self.feature_decoder = nn.Sequential(*decoder_layers)

        # ── Inverse decoders ──
        inv_cfg = config.inverse

        self.theta_inverse = None
        if "theta" in self.families and theta_param_dim is not None:
            self.theta_inverse = nn.Sequential(
                nn.Linear(vae_cfg.latent_dim, inv_cfg.theta_hidden_dim),
                nn.LayerNorm(inv_cfg.theta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(inv_cfg.theta_dropout),
                nn.Linear(inv_cfg.theta_hidden_dim, theta_param_dim),
                nn.Sigmoid(),  # params in [0,1]
            )

        self.ic_inverse = None
        if "initial" in self.families and initial_input_dim is not None:
            self.ic_inverse = nn.Sequential(
                nn.Linear(vae_cfg.latent_dim, inv_cfg.ic_hidden_dim),
                nn.LayerNorm(inv_cfg.ic_hidden_dim),
                nn.ReLU(),
                nn.Dropout(inv_cfg.ic_dropout),
                nn.Linear(inv_cfg.ic_hidden_dim, initial_input_dim),
            )

    # ──────────────────────────────────────────────────────────────
    # Encoder creation
    # ──────────────────────────────────────────────────────────────

    def _create_encoders(self):
        """Create family-specific encoders.

        Two paths:
        - Learned (production): PyramidFirstEncoder on raw [B, T, C, H, W]
        - Manual (legacy): PyramidTemporalEncoder on extracted [B, T, D]
        """
        enc_cfg = self.config.encoder

        # ── Temporal encoder ──
        self.pyramid_first_encoder = None
        self.temporal_encoder = None
        self.temporal_dim = 0

        if "temporal" in self.families:
            if self._learned_mode:
                self._create_learned_temporal_encoder(enc_cfg)
            else:
                self._create_manual_temporal_encoder(enc_cfg)

        # ── Initial encoder (manual features via MLP) ──
        self.initial_encoder = None
        self.initial_dim = 0
        if "initial" in self.families and self._initial_input_dim is not None:
            self.initial_dim = enc_cfg.embedding_dim
            self.initial_encoder = nn.Sequential(
                nn.Linear(self._initial_input_dim, enc_cfg.hidden_dim),
                nn.LayerNorm(enc_cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(enc_cfg.dropout),
                nn.Linear(enc_cfg.hidden_dim, self.initial_dim),
            )
            logger.info(f"Initial encoder: {self._initial_input_dim}D → {self.initial_dim}D")

        # ── Theta encoder ──
        self.theta_encoder = None
        self.theta_dim = 0
        if "theta" in self.families:
            self._create_theta_encoder(enc_cfg)

    def _create_learned_temporal_encoder(self, enc_cfg):
        """Create PyramidFirstEncoder for raw trajectory [B, T, C, H, W] input.

        The encoder pipeline: SpatioTemporalPyramid → FrameCNN → per-level
        temporal aggregation → LearnedGroupProjection → [B, G, D_group].
        """
        from .encoders.pyramid_first import PyramidFirstEncoder

        learned_cfg = enc_cfg.temporal.learned
        if learned_cfg is None:
            raise ValueError(
                "feature_source='learned' requires encoder.temporal.learned config"
            )

        t_cfg = enc_cfg.temporal
        # Variable-length config for adaptive pyramid levels
        vl_kwargs = {}
        if t_cfg.variable_length:
            if isinstance(t_cfg.variable_length, bool):
                vl_kwargs = {
                    "adaptive": t_cfg.adaptive_pyramid,
                    "min_pyramid_length": t_cfg.min_timesteps,
                }
            else:
                vl_cfg = t_cfg.variable_length
                vl_kwargs = {
                    "adaptive": vl_cfg.adaptive_pyramid,
                    "min_pyramid_length": vl_cfg.min_pyramid_length,
                    "mask_downsample_method": vl_cfg.mask_downsample_method,
                }

        num_groups = learned_cfg.num_groups
        d_group = enc_cfg.embedding_dim

        self.pyramid_first_encoder = PyramidFirstEncoder(
            in_channels=learned_cfg.in_channels or 3,
            d_cnn=learned_cfg.d_cnn,
            d_agg=learned_cfg.d_agg,
            num_groups=num_groups,
            d_group=d_group,
            downsample_factors=t_cfg.downsample_factors,
            gated_groups=learned_cfg.gated_groups,
            gate_init_bias=learned_cfg.gate_init_bias,
            **vl_kwargs,
        )
        self.temporal_dim = num_groups * d_group
        self._num_temporal_groups = num_groups
        logger.info(
            "Learned temporal encoder: PyramidFirst %d groups × %dD = %dD "
            "(d_cnn=%d, d_agg=%d)",
            num_groups, d_group, self.temporal_dim,
            learned_cfg.d_cnn, learned_cfg.d_agg,
        )

    def _create_manual_temporal_encoder(self, enc_cfg):
        """Create PyramidTemporalEncoder for pre-extracted [B, T, D] features."""
        from .encoders import PyramidTemporalEncoder

        t_cfg = enc_cfg.temporal
        vl_config = None
        if t_cfg.variable_length:
            if isinstance(t_cfg.variable_length, bool):
                vl_config = {
                    "enabled": True,
                    "adaptive_pyramid": t_cfg.adaptive_pyramid,
                    "min_pyramid_length": t_cfg.min_timesteps,
                }
            else:
                vl_cfg = t_cfg.variable_length
                vl_config = {
                    "enabled": vl_cfg.enabled,
                    "adaptive_pyramid": vl_cfg.adaptive_pyramid,
                    "min_pyramid_length": vl_cfg.min_pyramid_length,
                    "mask_downsample_method": vl_cfg.mask_downsample_method,
                }

        input_dim = self._temporal_input_dim or sum(
            len(v) for k, v in self.group_indices.items()
            if k.startswith("temporal_")
        )
        self.temporal_encoder = PyramidTemporalEncoder(
            input_dim=input_dim,
            level_dims=t_cfg.level_dims,
            variable_length_config=vl_config,
        )
        self.temporal_dim = sum(t_cfg.level_dims)
        logger.info(f"Manual temporal encoder: {input_dim}D → {self.temporal_dim}D")

    def _create_theta_encoder(self, enc_cfg):
        """Create theta parameter encoder (MLP or direct pass-through)."""
        t_cfg = enc_cfg.theta
        if t_cfg is not None and t_cfg.variant == "mlp":
            from .encoders.theta import ThetaMLPEncoder
            param_dim = self._theta_param_dim or t_cfg.param_dim
            if param_dim is not None:
                self.theta_encoder = ThetaMLPEncoder(
                    param_dim=param_dim,
                    hidden_dim=t_cfg.hidden_dim,
                    output_dim=t_cfg.output_dim,
                    dropout=t_cfg.dropout,
                    use_layer_norm=t_cfg.use_layer_norm,
                )
                self.theta_dim = t_cfg.output_dim
                logger.info(f"Theta encoder: {param_dim}D → {self.theta_dim}D")
        elif t_cfg is None or t_cfg.variant == "direct":
            self.theta_dim = self._theta_param_dim or 0
            if self.theta_dim > 0:
                logger.info(f"Theta encoder: direct {self.theta_dim}D")

    def _compute_total_encoded_dim(self) -> int:
        """Sum of all family encoder output dimensions."""
        return self.temporal_dim + self.initial_dim + self.theta_dim

    # ──────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        temporal_raw: Optional[torch.Tensor] = None,
        encode_only: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Full forward pass: encode → VAE bottleneck → decode.

        Args:
            temporal_features: [B, T, D_t] manual temporal sequences
            initial_manual: [B, D_i] initial condition features
            theta_features: [B, param_dim] operator parameters
            temporal_mask: [B, T] validity mask
            temporal_lengths: [B] actual sequence lengths
            temporal_raw: [B, T, C, H, W] raw trajectories (learned mode)
            encode_only: If True, return after encoding (no decode)

        Returns:
            Dict with mu, logvar, z, h, h_hat, theta_hat, ic_hat
        """
        family_embs: Dict[str, torch.Tensor] = {}
        B = None

        # ── Temporal encoding ──
        if self.pyramid_first_encoder is not None and temporal_raw is not None:
            B = temporal_raw.shape[0]
            per_group, _multi_res = self.pyramid_first_encoder(
                temporal_raw, mask=temporal_mask, lengths=temporal_lengths,
            )
            family_embs["temporal"] = per_group.reshape(B, -1)

        elif self.temporal_encoder is not None and temporal_features is not None:
            B = temporal_features.shape[0]
            t_out = self.temporal_encoder(
                temporal_features, mask=temporal_mask, lengths=temporal_lengths,
            )
            if isinstance(t_out, tuple):
                t_out = t_out[0]
            family_embs["temporal"] = t_out

        # ── Initial encoding ──
        if self.initial_encoder is not None and initial_manual is not None:
            B = B or initial_manual.shape[0]
            family_embs["initial"] = self.initial_encoder(initial_manual)

        # ── Theta encoding ──
        if theta_features is not None:
            B = B or theta_features.shape[0]
            if self.theta_encoder is not None:
                family_embs["theta"] = self.theta_encoder(theta_features)
            elif self.theta_dim > 0:
                family_embs["theta"] = theta_features

        if not family_embs:
            raise ValueError("No input features provided")

        # Concatenate all family embeddings → h
        h = torch.cat(list(family_embs.values()), dim=-1)  # [B, total_encoded_dim]

        # ── VAE encoder ──
        h_enc = self.vae_encoder(h)
        mu = self.mu_head(h_enc)
        logvar = self.logvar_head(h_enc)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        result: Dict[str, Any] = {
            "mu": mu, "logvar": logvar, "z": z, "h": h,
            "family_embeddings": family_embs,
        }

        if encode_only:
            return result

        # ── Feature decoder ──
        result["h_hat"] = self.feature_decoder(z)

        # ── Inverse decoders (with behavioral re-encoding) ──
        if self.theta_inverse is not None:
            theta_hat = self.theta_inverse(z)
            result["theta_hat"] = theta_hat
            # Re-encode predicted params for behavioral equivalence loss:
            # ‖encoder(θ_hat) - encoder(θ_true)‖² instead of ‖θ_hat - θ_true‖²
            if self.theta_encoder is not None:
                result["theta_hat_encoded"] = self.theta_encoder(theta_hat)

        if self.ic_inverse is not None:
            result["ic_hat"] = self.ic_inverse(z)

        return result

    def encode(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode inputs to latent z. Accepts same args as forward()."""
        return self.forward(encode_only=True, **kwargs)
