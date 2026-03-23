"""NLTokenizerModel — per-group hierarchical VAE for NL expression generation.

Matches VQTokenizer's information capacity by preserving the per-group
structure through the bottleneck. Each of the 30 temporal groups gets
its own multi-level (μ, logvar) → z via a shared HierarchicalVAEHead,
producing ~1400+ total latent dims (vs VQ's ~1800).

Architecture:
    temporal_raw [B, T, C, H, W]
        → PyramidFirstEncoder → per_group [B, G, D_group]
        → shared HierarchicalVAEHead per group → z_g [B, sum(level_dims)]
        → concat all groups + theta_z → z_full [B, total_z_dim]
        → z_to_lfm projection → z_lfm [B, lfm_dim] → LFM decoder → NL
        → feature decoder → ĥ (reconstruction)
        → theta inverse → θ_hat → re-encode (behavioral equivalence)
        → listener(NL) → ẑ_full (roundtrip)
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .base_model import BaseTokenizerModel
from .nl_config import NLTokenizerConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Per-group VAE head (shared across groups)
# ──────────────────────────────────────────────────────────────────────

class HierarchicalVAEHead(nn.Module):
    """Multi-level VAE projection, shared across temporal groups.

    Mirrors VQ's HierarchicalProjector: per-level independent projections
    from the same encoded input, but with continuous (μ, logvar) outputs
    instead of discrete codebook lookups.

    Args:
        input_dim: Per-group embedding dim (D_group)
        level_dims: Latent dim per level, e.g. [32, 16]
        hidden_dim: Encoder hidden dim (default: input_dim × 2)
    """

    def __init__(
        self,
        input_dim: int,
        level_dims: List[int],
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 2
        self.level_dims = level_dims
        self.z_dim = sum(level_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mu_heads = nn.ModuleList([
            nn.Linear(hidden_dim, d) for d in level_dims
        ])
        self.logvar_heads = nn.ModuleList([
            nn.Linear(hidden_dim, d) for d in level_dims
        ])

    def forward(
        self, x: torch.Tensor, training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim] single group's embedding

        Returns:
            (z, mu, logvar) each [B, sum(level_dims)]
        """
        h = self.encoder(x)
        mus, logvars, zs = [], [], []
        for mu_head, logvar_head in zip(self.mu_heads, self.logvar_heads):
            mu = mu_head(h)
            logvar = logvar_head(h)
            if training:
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            else:
                z = mu
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z)
        return torch.cat(zs, -1), torch.cat(mus, -1), torch.cat(logvars, -1)


# ──────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────

class NLTokenizerModel(BaseTokenizerModel):
    """Per-group hierarchical VAE for NL-compatible dynamics encoding.

    Uses the same PyramidFirstEncoder + group structure as VQTokenizer,
    but replaces discrete codebooks with per-group continuous VAE heads.
    The HierarchicalVAEHead is shared across all temporal groups (same
    weights), matching VQ's per-group projector pattern with less params.

    Args:
        config: NLTokenizerConfig
        group_indices: Dict mapping "temporal_group_N" → feature indices
        theta_param_dim: Auto-detected operator parameter dimension
    """

    def __init__(
        self,
        config: NLTokenizerConfig,
        group_indices: Dict[str, List[int]],
        theta_param_dim: Optional[int] = None,
    ):
        super().__init__()

        self.config = config
        self.group_indices = group_indices
        self.families = self.parse_families(group_indices)
        self._theta_param_dim = theta_param_dim
        logger.info(f"NLTokenizerModel families={self.families}")

        # ── Encoders ──
        self._create_encoders()

        # ── Per-group temporal VAE head ──
        d_group = config.encoder.embedding_dim
        level_ratios = config.vae.level_ratios
        level_dims = [max(4, int(d_group * r)) for r in level_ratios]
        self._level_dims = level_dims
        self._num_temporal_groups = sum(
            1 for k in group_indices if k.startswith("temporal_")
        )

        self.temporal_vae_head = HierarchicalVAEHead(
            input_dim=d_group,
            level_dims=level_dims,
            hidden_dim=config.vae.group_encoder_hidden_dim,
        )
        temporal_z_dim = self._num_temporal_groups * self.temporal_vae_head.z_dim
        logger.info(
            "Temporal VAE: %d groups × %d levels (%s) = %dD z_temporal",
            self._num_temporal_groups, len(level_dims), level_dims, temporal_z_dim,
        )

        # ── Theta VAE head ──
        self.theta_vae_head = None
        theta_z_dim = 0
        if "theta" in self.families and self.theta_dim > 0:
            theta_z_dim = config.vae.theta_z_dim
            self.theta_vae_head = HierarchicalVAEHead(
                input_dim=self.theta_dim,
                level_dims=[theta_z_dim],  # single level for theta
            )
            logger.info(f"Theta VAE: {self.theta_dim}D → {theta_z_dim}D z_theta")

        # ── Totals ──
        self.z_full_dim = temporal_z_dim + theta_z_dim
        total_h_dim = (self._num_temporal_groups * d_group) + self.theta_dim
        self._total_h_dim = total_h_dim
        logger.info(f"z_full_dim={self.z_full_dim}, h_dim={total_h_dim}")

        # ── Feature decoder: z_full → ĥ ──
        dec_layers = []
        in_dim = self.z_full_dim
        for h_dim in config.vae.feature_decoder_hidden_dims:
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, total_h_dim))
        self.feature_decoder = nn.Sequential(*dec_layers)

        # ── z_full → z_lfm projection (for LFM adapter) ──
        lfm_dim = config.vae.lfm_projection_dim
        self.z_to_lfm = nn.Linear(self.z_full_dim, lfm_dim)

        # ── Theta inverse decoder (behavioral equivalence) ──
        self.theta_inverse = None
        if "theta" in self.families and theta_param_dim is not None:
            inv_cfg = config.inverse
            self.theta_inverse = nn.Sequential(
                nn.Linear(self.z_full_dim, inv_cfg.theta_hidden_dim),
                nn.LayerNorm(inv_cfg.theta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(inv_cfg.theta_dropout),
                nn.Linear(inv_cfg.theta_hidden_dim, theta_param_dim),
                nn.Sigmoid(),
            )

    # ──────────────────────────────────────────────────────────────
    # Encoder creation (reuses VQ's encoder classes)
    # ──────────────────────────────────────────────────────────────

    def _create_encoders(self):
        """Create PyramidFirstEncoder (temporal) and ThetaMLPEncoder (theta)."""
        enc_cfg = self.config.encoder

        # ── Temporal: PyramidFirstEncoder ──
        self.pyramid_first_encoder = None
        self.temporal_dim = 0  # Not used directly; per-group VAE handles dims

        if "temporal" in self.families:
            from .encoders.pyramid_first import PyramidFirstEncoder

            learned_cfg = enc_cfg.temporal.learned
            if learned_cfg is None:
                raise ValueError("NLTokenizer requires encoder.temporal.learned config")

            t_cfg = enc_cfg.temporal
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

            self.pyramid_first_encoder = PyramidFirstEncoder(
                in_channels=learned_cfg.in_channels or 3,
                d_cnn=learned_cfg.d_cnn,
                d_agg=learned_cfg.d_agg,
                num_groups=learned_cfg.num_groups,
                d_group=enc_cfg.embedding_dim,
                downsample_factors=t_cfg.downsample_factors,
                gated_groups=learned_cfg.gated_groups,
                gate_init_bias=learned_cfg.gate_init_bias,
                **vl_kwargs,
            )
            logger.info(
                "PyramidFirst: %d groups × %dD, d_cnn=%d, d_agg=%d",
                learned_cfg.num_groups, enc_cfg.embedding_dim,
                learned_cfg.d_cnn, learned_cfg.d_agg,
            )

        # ── Theta: MLP encoder ──
        self.theta_encoder = None
        self.theta_dim = 0

        if "theta" in self.families:
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
            elif t_cfg is None or t_cfg.variant == "direct":
                self.theta_dim = self._theta_param_dim or 0

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(
        self,
        temporal_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        encode_only: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass: encode → per-group VAE → decode.

        Args:
            temporal_raw: [B, T, C, H, W] raw trajectories
            theta_features: [B, param_dim] operator parameters
            temporal_mask: [B, T] validity mask
            temporal_lengths: [B] actual sequence lengths
            encode_only: Skip decoding if True

        Returns:
            Dict with z_full, z_lfm, mu, logvar, h, h_hat, theta_hat,
            theta_hat_encoded, family_embeddings
        """
        all_z, all_mu, all_logvar = [], [], []
        family_embs: Dict[str, torch.Tensor] = {}
        B = None

        # ── Temporal: PyramidFirst → per-group VAE ──
        if self.pyramid_first_encoder is not None and temporal_raw is not None:
            B = temporal_raw.shape[0]
            per_group, _multi_res = self.pyramid_first_encoder(
                temporal_raw, mask=temporal_mask, lengths=temporal_lengths,
            )
            # per_group: [B, G, D_group]

            # Apply shared VAE head independently to each group
            for g in range(self._num_temporal_groups):
                x_g = per_group[:, g, :]  # [B, D_group]
                z_g, mu_g, logvar_g = self.temporal_vae_head(
                    x_g, training=self.training,
                )
                all_z.append(z_g)
                all_mu.append(mu_g)
                all_logvar.append(logvar_g)

            # Store flattened temporal embedding for behavioral distances
            family_embs["temporal"] = per_group.reshape(B, -1)

        # ── Theta: encode → VAE ──
        if theta_features is not None:
            B = B or theta_features.shape[0]
            if self.theta_encoder is not None:
                theta_enc = self.theta_encoder(theta_features)
            else:
                theta_enc = theta_features
            family_embs["theta"] = theta_enc

            if self.theta_vae_head is not None:
                z_th, mu_th, logvar_th = self.theta_vae_head(
                    theta_enc, training=self.training,
                )
                all_z.append(z_th)
                all_mu.append(mu_th)
                all_logvar.append(logvar_th)

        if not all_z:
            raise ValueError("No input features provided")

        z_full = torch.cat(all_z, dim=-1)      # [B, z_full_dim]
        mu = torch.cat(all_mu, dim=-1)          # [B, z_full_dim]
        logvar = torch.cat(all_logvar, dim=-1)  # [B, z_full_dim]

        # Concatenated family embeddings (for reconstruction + topographic)
        h = torch.cat(list(family_embs.values()), dim=-1)

        # LFM projection
        z_lfm = self.z_to_lfm(z_full)  # [B, lfm_projection_dim]

        result: Dict[str, Any] = {
            "z_full": z_full,
            "z_lfm": z_lfm,
            "mu": mu,
            "logvar": logvar,
            "h": h,
            "family_embeddings": family_embs,
        }

        if encode_only:
            return result

        # ── Feature decoder ──
        result["h_hat"] = self.feature_decoder(z_full)

        # ── Theta inverse (behavioral equivalence) ──
        if self.theta_inverse is not None:
            theta_hat = self.theta_inverse(z_full)
            result["theta_hat"] = theta_hat
            if self.theta_encoder is not None:
                result["theta_hat_encoded"] = self.theta_encoder(theta_hat)

        return result

    def encode(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode inputs to latent z_full. Accepts same args as forward()."""
        return self.forward(encode_only=True, **kwargs)
