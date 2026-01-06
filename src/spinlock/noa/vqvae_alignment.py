"""VQ-VAE Alignment Loss for NOA Training.

Implements the three-loss structure for token-aligned NOA training:
1. L_traj: MSE on trajectories (handled externally, not in this module)
2. L_latent: Pre-quantized latent alignment
3. L_commit: VQ commitment regularizer (manifold adherence)

The key insight is that we use PRE-quantization embeddings for smooth gradients,
and add a commitment loss to force NOA outputs onto the VQ manifold.

This module handles:
- Loading VQ-VAE from checkpoint
- Feature extraction from trajectories
- Per-category normalization
- Latent alignment and commitment loss computation

Usage:
    alignment = VQVAEAlignmentLoss.from_checkpoint(
        vqvae_path="checkpoints/production/100k_full_features/best_model.pt",
        device="cuda",
    )

    # In training loop
    losses = alignment.compute_losses(pred_trajectory, target_trajectory, ic)
    total_loss = state_loss + 0.1 * losses['latent'] + 0.5 * losses['commit']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class VQVAEAlignmentLoss(nn.Module):
    """VQ-VAE alignment loss for NOA training.

    Computes two loss components:
    1. L_latent: MSE between normalized pre-quantization latents
    2. L_commit: MSE between pre-quant latent and quantized (stop-grad)

    The VQ-VAE weights are FROZEN - it acts as a pre-trained feature extractor.
    """

    def __init__(
        self,
        vqvae: nn.Module,
        feature_extractor: nn.Module,
        normalization_stats: Dict[str, Any],
        group_indices: Dict[str, List[int]],
        device: str = "cuda",
    ):
        """Initialize alignment loss.

        Args:
            vqvae: Pre-trained CategoricalHierarchicalVQVAE model (frozen)
            feature_extractor: Module to extract features from trajectories
            normalization_stats: Per-category mean/std for normalization
            group_indices: Category → feature indices mapping
            device: Computation device
        """
        super().__init__()

        self.device = torch.device(device)
        self.vqvae = vqvae.to(self.device)
        self.feature_extractor = feature_extractor.to(self.device)
        self.normalization_stats = normalization_stats
        self.group_indices = group_indices

        # Freeze VQ-VAE weights
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval()

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature cleaning and normalization using existing infrastructure.

        Uses FeaturePreprocessor for NaN handling and standard normalization.

        Args:
            features: Raw features [batch, D]

        Returns:
            Cleaned and normalized features [batch, D]
        """
        from spinlock.encoding.normalization import standard_normalize

        # First: replace any NaN/Inf values with 0 (FeaturePreprocessor cleans known bad indices,
        # but we may have NaN from dynamic extraction)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Skip normalization if batch size is 1 (can't compute meaningful std)
        if features.shape[0] < 2:
            return features

        # Use global standard normalization
        normalized = standard_normalize(features)

        # Final safety: replace any NaN/Inf from normalization (e.g., zero-variance features)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized

    def compute_losses(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute alignment losses.

        Args:
            pred_trajectory: NOA predicted trajectory [B, T, C, H, W] or [B, M, T, C, H, W]
            target_trajectory: CNO target trajectory [B, T, C, H, W] or [B, M, T, C, H, W]
            ic: Initial condition [B, C, H, W] (optional, for initial features)

        Returns:
            Dictionary with:
                - 'latent': Pre-quantized latent alignment loss
                - 'commit': VQ commitment loss (manifold adherence)
                - 'z_pred': Pre-quantized latent for pred (for logging)
                - 'z_target': Pre-quantized latent for target (for logging)
        """
        # Extract features from trajectories
        pred_features = self.feature_extractor(pred_trajectory, ic=ic)
        target_features = self.feature_extractor(target_trajectory, ic=ic)

        # Normalize features
        pred_norm = self._normalize_features(pred_features)
        target_norm = self._normalize_features(target_features)

        # Encode to pre-quantization latents
        with torch.no_grad():
            z_target_list = self.vqvae.encode(target_norm)
            z_target = torch.cat(z_target_list, dim=1)  # [B, total_latent_dim]

        # Encode pred (gradients flow through feature_extractor)
        z_pred_list = self.vqvae.encode(pred_norm)
        z_pred = torch.cat(z_pred_list, dim=1)  # [B, total_latent_dim]

        # L_latent: Normalized MSE between pre-quant latents
        # Normalize per-channel for stable training
        z_pred_norm = F.normalize(z_pred, p=2, dim=-1)
        z_target_norm = F.normalize(z_target, p=2, dim=-1)
        latent_loss = F.mse_loss(z_pred_norm, z_target_norm)

        # L_commit: Force pred to be close to its quantized version
        # This ensures NOA outputs are expressible in VQ vocabulary
        z_q_pred_list, _, _ = self.vqvae.quantize(z_pred_list)
        z_q_pred = torch.cat(z_q_pred_list, dim=1)
        commit_loss = F.mse_loss(z_pred, z_q_pred.detach())

        return {
            'latent': latent_loss,
            'commit': commit_loss,
            'z_pred': z_pred.detach(),
            'z_target': z_target.detach(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        vqvae_path: str,
        device: str = "cuda",
        feature_extractor: Optional[nn.Module] = None,
    ) -> "VQVAEAlignmentLoss":
        """Load alignment loss from VQ-VAE checkpoint.

        Args:
            vqvae_path: Path to VQ-VAE checkpoint directory or .pt file
            device: Computation device
            feature_extractor: Optional custom feature extractor
                             If None, creates default from checkpoint config

        Returns:
            Configured VQVAEAlignmentLoss instance
        """
        path = Path(vqvae_path)

        # Determine checkpoint file
        if path.is_dir():
            checkpoint_path = path / "best_model.pt"
            stats_path = path / "normalization_stats.npz"
        else:
            checkpoint_path = path
            stats_path = path.parent / "normalization_stats.npz"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load normalization stats
        if stats_path.exists():
            normalization_stats = dict(np.load(stats_path, allow_pickle=True))
        else:
            normalization_stats = checkpoint.get('normalization_stats', {})

        # Get config
        config = checkpoint.get('config', {})
        group_indices = config.get('group_indices', checkpoint.get('group_indices', {}))

        # Create VQ-VAE model
        from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

        vqvae_config = CategoricalVQVAEConfig(
            input_dim=config.get('input_dim', 225),
            group_indices=group_indices,
            group_embedding_dim=config.get('group_embedding_dim', 256),
            group_hidden_dim=config.get('group_hidden_dim', 512),
            levels=config.get('levels'),
        )

        vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

        # Handle different checkpoint formats (compiled models, nested structures)
        state_dict = checkpoint['model_state_dict']

        # Check if this is a nested model (e.g., from torch.compile or wrapper)
        # The model might be saved as: _orig_mod.vqvae.* (compiled + nested)
        # or vqvae.* (nested) or just the raw weights
        has_orig_mod_vqvae = any(k.startswith('_orig_mod.vqvae.') for k in state_dict.keys())
        has_vqvae_prefix = any(k.startswith('vqvae.') for k in state_dict.keys())

        if has_orig_mod_vqvae:
            # Extract VQ-VAE weights from compiled nested structure
            vqvae_state = {}
            prefix = '_orig_mod.vqvae.'
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    vqvae_state[k[len(prefix):]] = v
            state_dict = vqvae_state
        elif has_vqvae_prefix:
            # Nested but not compiled
            vqvae_state = {}
            prefix = 'vqvae.'
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    vqvae_state[k[len(prefix):]] = v
            state_dict = vqvae_state
        else:
            # Check for _orig_mod. prefix (compiled model, flat structure)
            sample_key = next(iter(state_dict.keys()))
            if sample_key.startswith('_orig_mod.'):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        vqvae.load_state_dict(state_dict)

        # Create or use provided feature extractor
        if feature_extractor is None:
            feature_extractor = TrajectoryFeatureExtractor(
                input_dim=config.get('input_dim', 225),
                device=device,
            )

        return cls(
            vqvae=vqvae,
            feature_extractor=feature_extractor,
            normalization_stats=normalization_stats,
            group_indices=group_indices,
            device=device,
        )


class TrajectoryFeatureExtractor(nn.Module):
    """Extract features from trajectories matching VQ-VAE input format.

    This extractor produces a feature vector compatible with the VQ-VAE
    by combining:
    - Summary features (aggregated statistics from trajectory)
    - Temporal features (encoded temporal dynamics)

    Note: Architecture features are NOT included as they're from CNO parameters,
    not extractable from trajectories. Initial features can be added if IC is provided.
    """

    def __init__(
        self,
        input_dim: int = 225,
        device: str = "cuda",
    ):
        """Initialize feature extractor.

        Args:
            input_dim: Expected output dimension (must match VQ-VAE input_dim)
            device: Computation device
        """
        super().__init__()

        self.input_dim = input_dim
        self.device = torch.device(device)

        # Import extractors
        from spinlock.features.summary.config import SummaryConfig
        from spinlock.features.summary.extractors import SummaryExtractor

        # Create summary extractor with config that avoids NaN for M=1
        # Disable std/cv aggregations since we only have one realization
        config = SummaryConfig(
            # Only use mean for realization aggregation (std/cv undefined for M=1)
            realization_aggregation=["mean"],
            # Only use mean for temporal aggregation (std can produce NaN for constant sequences)
            temporal_aggregation=["mean"],
        )
        self.summary_extractor = SummaryExtractor(device=self.device, config=config)

        # Dimension tracking (will be set on first forward)
        self._output_dim = None

    def forward(
        self,
        trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from trajectory.

        Args:
            trajectory: Trajectory tensor [B, T, C, H, W] or [B, M, T, C, H, W]
            ic: Initial condition [B, C, H, W] (optional)

        Returns:
            Features [B, D] compatible with VQ-VAE input
        """
        # Handle multi-realization trajectories
        if trajectory.dim() == 6:
            # [B, M, T, C, H, W] - use M=1 dimension
            trajectory = trajectory[:, 0]  # [B, T, C, H, W]

        # Add realization dimension for extractor
        # [B, T, C, H, W] -> [B, 1, T, C, H, W]
        traj_with_m = trajectory.unsqueeze(1)

        # Extract summary features
        result = self.summary_extractor.extract_all(traj_with_m)

        # Get per_trajectory features [B, 1, D_summary] -> [B, D_summary]
        summary_features = result['per_trajectory'].squeeze(1)

        # Get temporal features [B, T, D_temporal]
        temporal_features = result['per_timestep']

        # Aggregate temporal to single vector (mean over time)
        temporal_agg = temporal_features.mean(dim=1)  # [B, D_temporal]

        # Concatenate
        features = torch.cat([summary_features, temporal_agg], dim=1)

        # Handle dimension mismatch
        if features.shape[1] != self.input_dim:
            # Pad or truncate to match expected dimension
            if features.shape[1] < self.input_dim:
                padding = torch.zeros(
                    features.shape[0],
                    self.input_dim - features.shape[1],
                    device=features.device,
                    dtype=features.dtype,
                )
                features = torch.cat([features, padding], dim=1)
            else:
                features = features[:, :self.input_dim]

        # Replace NaN/Inf with 0 (critical for gradient stability)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features
