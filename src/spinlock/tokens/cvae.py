"""Token-Conditioned CVAE: P(theta, IC | temporal_tokens).

This module implements a Conditional VAE that generates plausible physical
parameters (theta) and initial conditions (IC) given temporal token observations.

Unlike the standard TokenToRolloutVAE where the encoder sees only tokens,
the CVAE encoder (recognition network) sees BOTH the target (theta, IC) AND
the condition (temporal tokens) during training, learning q(z | theta, IC, tokens).
At inference, z is sampled from the prior N(0,I) and the decoder conditions
on tokens alone.

Architecture:
    Temporal tokens (~90 keys)
        -> frozen codebook lookup
        -> per-group MLP (weight-tied)
        -> mean pool
        -> condition c [B, condition_dim]

    Target (theta[P] + IC[C,H,W])
        -> TargetEncoder
        -> target_features [B, target_dim]       (training only)

    concat(target_features, c)
        -> CVAEEncoder
        -> z [B, latent_dim]                      (training only)

    concat(z, c)
        -> ParameterDecoder -> theta_hat [B, P]
        -> GridDecoder -> IC_hat [B, C, H, W]

Reuses ParameterDecoder and GridDecoder from rollout_vae.py.
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from spinlock.tokens.rollout_vae import ParameterDecoder, GridDecoder, SpectralGridDecoder
from spinlock.tokens.schema import TokenSchema


class TemporalTokenConditioner(nn.Module):
    """Convert frozen codebook embeddings into a fixed-size condition vector.

    For each temporal token key:
        1. Look up frozen codebook embedding
        2. Project through per-group MLP (weight-tied across all groups)
        3. Pool across all groups (mean or attention)

    Pooling strategies:
        - "mean": uniform average across all groups (baseline)
        - "attention": learned per-group importance scores. A small scoring
          network produces scalar attention weights per group, softmax-
          normalized, then weighted sum. This lets the model upweight
          discriminative groups and suppress uninformative ones.

    Args:
        vq_checkpoint_path: Path to VQTokenizer checkpoint
        group_mlp_hidden_dim: Hidden dim for projection MLP
        group_mlp_output_dim: Output dim for projection MLP (= condition_dim)
        pooling: Pooling strategy ("mean" or "attention")
    """

    def __init__(
        self,
        vq_checkpoint_path: Path,
        group_mlp_hidden_dim: int = 128,
        group_mlp_output_dim: int = 64,
        pooling: str = "mean",
    ):
        super().__init__()
        self.group_mlp_output_dim = group_mlp_output_dim
        self.pooling = pooling

        # Load frozen tokenizer
        from spinlock.tokens.tokenizer import VQTokenizer

        self.tokenizer = VQTokenizer.from_checkpoint(vq_checkpoint_path)
        self.tokenizer.model.eval()

        # Freeze all tokenizer parameters
        for param in self.tokenizer.model.parameters():
            param.requires_grad = False

        # Discover temporal-only quantizer keys
        schema = TokenSchema.from_tokenizer(self.tokenizer)
        self.temporal_keys = schema.keys_for_family("temporal")
        self.num_temporal_keys = len(self.temporal_keys)

        # Build per-group MLP (weight-tied: single MLP shared across all groups)
        # Input dim = max embedding dim across temporal quantizers (pad smaller ones)
        self.embedding_dims = {}
        max_emb_dim = 0
        for key in self.temporal_keys:
            quantizer = self.tokenizer.model.quantizers[key]
            emb_dim = quantizer.embedding_dim
            self.embedding_dims[key] = emb_dim
            max_emb_dim = max(max_emb_dim, emb_dim)

        self.max_emb_dim = max_emb_dim

        self.group_mlp = nn.Sequential(
            nn.Linear(max_emb_dim, group_mlp_hidden_dim),
            nn.LayerNorm(group_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(group_mlp_hidden_dim, group_mlp_output_dim),
        )

        # Attention pooling scorer (Ilse et al. 2018, "Attention-based MIL")
        if pooling == "attention":
            self.attention_scorer = nn.Sequential(
                nn.Linear(group_mlp_output_dim, group_mlp_output_dim),
                nn.Tanh(),
                nn.Linear(group_mlp_output_dim, 1),
            )

    @property
    def condition_dim(self) -> int:
        """Dimensionality of the output condition vector."""
        return self.group_mlp_output_dim

    def forward(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute condition vector from temporal tokens.

        Args:
            tokens: Dict mapping token keys to indices [B].
                    Only temporal keys are used; others are ignored.

        Returns:
            Condition vector [B, condition_dim]
        """
        device = next(iter(tokens.values())).device
        projections = []

        for key in self.temporal_keys:
            if key not in tokens:
                continue

            quantizer = self.tokenizer.model.quantizers[key]
            token_indices = tokens[key].cpu()

            # Lookup frozen codebook embedding
            emb = quantizer.embedding(token_indices)  # [B, emb_dim]
            emb = emb.to(device)

            # Pad to max_emb_dim if needed
            if emb.shape[-1] < self.max_emb_dim:
                padding = torch.zeros(
                    emb.shape[0], self.max_emb_dim - emb.shape[-1],
                    device=device,
                )
                emb = torch.cat([emb, padding], dim=-1)

            # Project through shared MLP
            proj = self.group_mlp(emb)  # [B, condition_dim]
            projections.append(proj)

        # Pool across all temporal groups
        stacked = torch.stack(projections, dim=1)  # [B, num_groups, condition_dim]

        if self.pooling == "attention":
            # Learned attention weights per group
            scores = self.attention_scorer(stacked).squeeze(-1)  # [B, num_groups]
            weights = torch.softmax(scores, dim=-1)  # [B, num_groups]
            condition = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, condition_dim]
        else:
            condition = stacked.mean(dim=1)  # [B, condition_dim]

        return condition


class TargetEncoder(nn.Module):
    """Encode ground-truth theta + IC into features for the recognition network.

    Only used during training. At inference, the CVAE samples z from the prior.

    Args:
        theta_dim: Dimensionality of theta parameters
        grid_shape: Shape of initial condition grids (C, H, W)
        theta_hidden_dim: Hidden dim for theta MLP branch
        ic_hidden_dim: Hidden dim for IC CNN branch
        ic_channels: Conv2d channel progression for IC encoder
        dropout: Dropout probability
    """

    def __init__(
        self,
        theta_dim: int,
        grid_shape: Tuple[int, int, int],
        theta_hidden_dim: int = 256,
        ic_hidden_dim: int = 256,
        ic_channels: list[int] = [32, 64, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.grid_shape = grid_shape
        C, H, W = grid_shape

        # Theta branch: MLP
        self.theta_mlp = nn.Sequential(
            nn.Linear(theta_dim, theta_hidden_dim),
            nn.LayerNorm(theta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # IC branch: Conv2d layers + adaptive pool
        ic_layers = []
        in_ch = C
        for out_ch in ic_channels:
            ic_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ])
            in_ch = out_ch
        ic_layers.append(nn.AdaptiveAvgPool2d(1))
        self.ic_cnn = nn.Sequential(*ic_layers)

        self.ic_proj = nn.Sequential(
            nn.Linear(ic_channels[-1], ic_hidden_dim),
            nn.LayerNorm(ic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = theta_hidden_dim + ic_hidden_dim

    def forward(
        self, theta: torch.Tensor, grids: torch.Tensor
    ) -> torch.Tensor:
        """Encode targets to feature vector.

        Args:
            theta: Parameters [B, theta_dim]
            grids: Initial condition grids [B, C, H, W]

        Returns:
            Target features [B, output_dim]
        """
        theta_feat = self.theta_mlp(theta)  # [B, theta_hidden_dim]

        ic_feat = self.ic_cnn(grids)  # [B, ic_channels[-1], 1, 1]
        ic_feat = ic_feat.flatten(1)  # [B, ic_channels[-1]]
        ic_feat = self.ic_proj(ic_feat)  # [B, ic_hidden_dim]

        return torch.cat([theta_feat, ic_feat], dim=-1)  # [B, output_dim]


class CVAEEncoder(nn.Module):
    """Recognition network q(z | target, condition).

    Takes concatenated target features and condition vector, produces
    mean and log-variance of the approximate posterior.

    Args:
        input_dim: target_dim + condition_dim
        latent_dim: Dimensionality of latent space
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = [512, 256],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution.

        Args:
            x: Concatenated [target_features, condition] [B, input_dim]

        Returns:
            Tuple of (mu [B, latent_dim], logvar [B, latent_dim])
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class TokenConditionedCVAE(nn.Module):
    """Complete CVAE: P(theta, IC | temporal_tokens).

    Training: encoder sees both targets and condition
    Inference: z sampled from prior, decoder conditioned on tokens only

    Args:
        vq_checkpoint: Path to frozen VQTokenizer checkpoint
        theta_dim: Dimensionality of theta parameters
        grid_shape: Shape of initial grids (C, H, W)
        latent_dim: Dimensionality of latent space
        condition_config: Conditioning network parameters
        target_encoder_config: Target encoder parameters
        encoder_hidden_dims: Hidden dimensions for recognition network
        param_decoder_hidden_dims: Hidden dimensions for parameter decoder
        grid_decoder_hidden_channels: Channels for grid decoder
        dropout: Dropout probability
    """

    def __init__(
        self,
        vq_checkpoint: Path,
        theta_dim: int,
        grid_shape: Tuple[int, int, int],
        latent_dim: int = 256,
        group_mlp_hidden_dim: int = 128,
        group_mlp_output_dim: int = 64,
        pooling: str = "mean",
        theta_hidden_dim: int = 256,
        ic_hidden_dim: int = 256,
        ic_channels: list[int] = [32, 64, 128],
        encoder_hidden_dims: list[int] = [512, 256],
        param_decoder_hidden_dims: list[int] = [256, 128],
        grid_decoder_hidden_channels: list[int] = [512, 256, 128, 64, 32],
        grid_decoder_type: str = "conv",
        grid_decoder_num_modes: int = 16,
        grid_decoder_spectral_hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.grid_shape = grid_shape
        self.latent_dim = latent_dim
        self.grid_decoder_type = grid_decoder_type

        # Conditioning network (frozen codebook embeddings → condition vector)
        self.conditioner = TemporalTokenConditioner(
            vq_checkpoint,
            group_mlp_hidden_dim=group_mlp_hidden_dim,
            group_mlp_output_dim=group_mlp_output_dim,
            pooling=pooling,
        )
        condition_dim = self.conditioner.condition_dim

        # Target encoder (training only)
        self.target_encoder = TargetEncoder(
            theta_dim=theta_dim,
            grid_shape=grid_shape,
            theta_hidden_dim=theta_hidden_dim,
            ic_hidden_dim=ic_hidden_dim,
            ic_channels=ic_channels,
            dropout=dropout,
        )
        target_dim = self.target_encoder.output_dim

        # Recognition network q(z | target, condition)
        self.encoder = CVAEEncoder(
            input_dim=target_dim + condition_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
        )

        # Decoders: input is concat(z, condition) → latent_dim + condition_dim
        decoder_input_dim = latent_dim + condition_dim
        self.param_decoder = ParameterDecoder(
            latent_dim=decoder_input_dim,
            theta_dim=theta_dim,
            hidden_dims=param_decoder_hidden_dims,
            dropout=dropout,
        )

        # Select grid decoder based on type
        if grid_decoder_type == "spectral":
            self.grid_decoder = SpectralGridDecoder(
                latent_dim=decoder_input_dim,
                grid_shape=grid_shape,
                num_modes=grid_decoder_num_modes,
                hidden_dims=grid_decoder_spectral_hidden_dims,
                dropout=dropout,
            )
        else:
            self.grid_decoder = GridDecoder(
                latent_dim=decoder_input_dim,
                grid_shape=grid_shape,
                hidden_channels=grid_decoder_hidden_channels,
                dropout=dropout,
            )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        tokens: Dict[str, torch.Tensor],
        theta: torch.Tensor,
        grids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass — encoder sees targets.

        Args:
            tokens: Dict mapping token keys to indices [B]
            theta: Ground truth parameters [B, theta_dim]
            grids: Ground truth initial grids [B, C, H, W]

        Returns:
            Dictionary containing:
            - theta: [B, theta_dim] decoded parameters
            - grids: [B, C, H, W] decoded initial grids
            - mu: [B, latent_dim] latent mean
            - logvar: [B, latent_dim] latent log variance
            - z: [B, latent_dim] sampled latent codes
        """
        # Condition from temporal tokens
        condition = self.conditioner(tokens)  # [B, condition_dim]

        # Encode targets (training only)
        target_features = self.target_encoder(theta, grids)  # [B, target_dim]

        # Recognition network: q(z | target, condition)
        encoder_input = torch.cat([target_features, condition], dim=-1)
        mu, logvar = self.encoder(encoder_input)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode: concat z with condition
        decoder_input = torch.cat([z, condition], dim=-1)  # [B, latent_dim + condition_dim]
        theta_hat = self.param_decoder(decoder_input)
        grids_hat = self.grid_decoder(decoder_input)

        return {
            "theta": theta_hat,
            "grids": grids_hat,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def sample(
        self,
        tokens: Dict[str, torch.Tensor],
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Sample (theta, IC) conditioned on temporal tokens.

        At inference, z is sampled from the prior N(0,I) — no target encoder.

        Args:
            tokens: Dict mapping token keys to indices [B] (typically B=1)
            n_samples: Number of samples to generate per input

        Returns:
            Dictionary containing:
            - theta: [B*n_samples, theta_dim]
            - grids: [B*n_samples, C, H, W]
            - z: [B*n_samples, latent_dim]
        """
        self.eval()

        # Condition from temporal tokens
        condition = self.conditioner(tokens)  # [B, condition_dim]

        B = condition.shape[0]
        device = condition.device

        if n_samples > 1:
            # Repeat condition for multiple samples
            condition = condition.repeat_interleave(n_samples, dim=0)  # [B*n_samples, condition_dim]

        # Sample z from prior
        z = torch.randn(B * n_samples, self.latent_dim, device=device)

        # Decode
        decoder_input = torch.cat([z, condition], dim=-1)
        theta_hat = self.param_decoder(decoder_input)
        grids_hat = self.grid_decoder(decoder_input)

        return {
            "theta": theta_hat,
            "grids": grids_hat,
            "z": z,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: str = "cpu",
    ) -> "TokenConditionedCVAE":
        """Load trained CVAE from checkpoint.

        Args:
            checkpoint_path: Path to saved checkpoint
            device: Target device

        Returns:
            TokenConditionedCVAE instance with loaded weights
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Reconstruct model from saved config + dimensions
        config = checkpoint["config"]
        theta_dim = checkpoint["theta_dim"]
        grid_shape = tuple(checkpoint["grid_shape"])

        # Extract model config parameters
        model_cfg = config.get("model", {})
        condition_cfg = model_cfg.get("condition", {})
        target_cfg = model_cfg.get("target_encoder", {})
        encoder_cfg = model_cfg.get("encoder", {})
        param_cfg = model_cfg.get("param_decoder", {})
        grid_cfg = model_cfg.get("grid_decoder", {})
        data_cfg = config.get("data", {})

        model = cls(
            vq_checkpoint=Path(data_cfg["vq_checkpoint"]),
            theta_dim=theta_dim,
            grid_shape=grid_shape,
            latent_dim=model_cfg.get("latent_dim", 256),
            group_mlp_hidden_dim=condition_cfg.get("group_mlp_hidden_dim", 128),
            group_mlp_output_dim=condition_cfg.get("group_mlp_output_dim", 64),
            pooling=condition_cfg.get("pooling", "mean"),
            theta_hidden_dim=target_cfg.get("theta_hidden_dim", 256),
            ic_hidden_dim=target_cfg.get("ic_hidden_dim", 256),
            ic_channels=target_cfg.get("ic_channels", [32, 64, 128]),
            encoder_hidden_dims=encoder_cfg.get("hidden_dims", [512, 256]),
            param_decoder_hidden_dims=param_cfg.get("hidden_dims", [256, 128]),
            grid_decoder_hidden_channels=grid_cfg.get("hidden_channels", [512, 256, 128, 64, 32]),
            grid_decoder_type=grid_cfg.get("type", "conv"),
            grid_decoder_num_modes=grid_cfg.get("num_modes", 16),
            grid_decoder_spectral_hidden_dims=grid_cfg.get("spectral_hidden_dims", [256, 128]),
            dropout=encoder_cfg.get("dropout", 0.1),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return model
