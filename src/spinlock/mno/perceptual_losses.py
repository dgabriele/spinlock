"""NOA Phase 1 Losses - VQ-VAE perceptual loss.

This module provides the VQ-VAE perceptual loss for NOA Phase 1:
- Loads frozen VQ-VAE tokenizer
- Projects trajectory features to VQ-VAE input space
- Computes loss in VQ-VAE latent space (before quantization)

Architecture:
    trajectory_features → projection → VQ-VAE encoder → latent vectors → loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class VQVAEPerceptualLoss(nn.Module):
    """Frozen VQ-VAE as perceptual loss encoder.

    The VQ-VAE encodes features into a learned latent space. By computing
    loss in this space, we guide the NOA to generate trajectories that
    produce "meaningful" features (as learned by the VQ-VAE).

    Args:
        checkpoint_path: Path to VQ-VAE checkpoint directory or file
        device: Device to use
        freeze_vqvae: Whether to freeze VQ-VAE parameters (default: True)

    Example:
        >>> loss_fn = VQVAEPerceptualLoss("checkpoints/production/100k_3family_v1/")
        >>> pred_features = torch.randn(8, 187)  # Must match VQ-VAE input_dim
        >>> gt_features = torch.randn(8, 187)
        >>> loss = loss_fn(pred_features, gt_features)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        freeze_vqvae: bool = True,
    ):
        super().__init__()
        self.device = device
        self.freeze_vqvae = freeze_vqvae

        # Load VQ-VAE model
        self.vqvae, self.config = self._load_vqvae(checkpoint_path, device)

        if freeze_vqvae:
            self._freeze_vqvae()

        # Store input dimension for validation
        # Note: VQVAEWithInitial uses EXPANDED input dim (includes CNN features)
        self.input_dim = self._get_actual_input_dim()

    def _load_vqvae(
        self, checkpoint_path: str, device: str
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Load VQ-VAE from checkpoint.

        Handles both:
        - Raw CategoricalHierarchicalVQVAE checkpoints
        - VQVAEWithInitial wrapped checkpoints (with torch.compile prefix)

        Args:
            checkpoint_path: Path to checkpoint directory or file
            device: Device to load model on

        Returns:
            Tuple of (model, config)
        """
        from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
        from spinlock.encoding.vqvae_with_initial import VQVAEWithInitial

        path = Path(checkpoint_path)

        # Handle directory or file path
        if path.is_dir():
            ckpt_file = path / "best_model.pt"
        else:
            ckpt_file = path

        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

        # Check if checkpoint has torch.compile prefix
        has_orig_mod = any(k.startswith("_orig_mod.") for k in state_dict.keys())
        if has_orig_mod:
            # Remove _orig_mod. prefix from all keys
            state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in state_dict.items()
            }

        # Check if it's a VQVAEWithInitial checkpoint (has initial_encoder and vqvae)
        is_wrapped = any(k.startswith("initial_encoder.") for k in state_dict.keys())

        # Create VQ-VAE config
        vqvae_config = CategoricalVQVAEConfig(
            input_dim=config["input_dim"],
            group_indices=config["group_indices"],
            group_embedding_dim=config["group_embedding_dim"],
            group_hidden_dim=config["group_hidden_dim"],
            levels=config["levels"],
        )

        if is_wrapped:
            # Extract only the inner VQ-VAE weights (ignore initial_encoder)
            # This avoids dimension mismatch from VQVAEWithInitial wrapper
            vqvae_state_dict = {
                k.replace("vqvae.", ""): v
                for k, v in state_dict.items()
                if k.startswith("vqvae.")
            }

            model = CategoricalHierarchicalVQVAE(vqvae_config)
            model.load_state_dict(vqvae_state_dict)
        else:
            # Load raw CategoricalHierarchicalVQVAE
            model = CategoricalHierarchicalVQVAE(vqvae_config)
            model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        return model, config

    def _get_actual_input_dim(self) -> int:
        """Get the actual input dimension the VQ-VAE expects."""
        return self.vqvae.config.input_dim

    def _freeze_vqvae(self):
        """Freeze all VQ-VAE parameters."""
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Encode features through VQ-VAE encoder.

        Args:
            features: Input features [batch, input_dim]

        Returns:
            Concatenated latent vectors [batch, total_latent_dim]
        """
        # Get list of latent vectors from VQ-VAE encoder
        z_list = self.vqvae.encode(features)

        # Concatenate all latent vectors
        z_concat = torch.cat(z_list, dim=1)

        return z_concat

    def forward(
        self,
        pred_features: torch.Tensor,
        gt_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute perceptual loss in VQ-VAE latent space.

        Args:
            pred_features: Predicted features [batch, input_dim]
            gt_features: Ground truth features [batch, input_dim]

        Returns:
            Perceptual loss (scalar)
        """
        # Validate input dimensions
        if pred_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {pred_features.shape[-1]}"
            )

        # Encode through VQ-VAE
        if self.freeze_vqvae:
            with torch.no_grad():
                gt_latent = self.encode(gt_features)
        else:
            gt_latent = self.encode(gt_features)

        pred_latent = self.encode(pred_features)

        # MSE in latent space
        loss = F.mse_loss(pred_latent, gt_latent)

        return loss

    def get_latent_dim(self) -> int:
        """Get total latent dimension of VQ-VAE."""
        return self.vqvae.config.total_latent_dim


class FeatureProjector(nn.Module):
    """Projects trajectory features to VQ-VAE input dimension.

    Since trajectory features from extract_trajectory_features() have
    a different dimension than VQ-VAE expects, this module learns
    a projection to match dimensions.

    Args:
        input_dim: Input feature dimension (from trajectory extraction)
        output_dim: Output dimension (VQ-VAE input_dim)
        hidden_dim: Hidden dimension for MLP

    Example:
        >>> projector = FeatureProjector(input_dim=8, output_dim=187)
        >>> traj_features = torch.randn(8, 8)
        >>> vqvae_features = projector(traj_features)
        >>> vqvae_features.shape
        torch.Size([8, 187])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to VQ-VAE input dimension.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Projected features [batch, output_dim]
        """
        return self.projector(x)


class NOALoss(nn.Module):
    """Combined loss for NOA Phase 1 training.

    Combines:
    1. Grid MSE loss (predicted vs ground-truth trajectory)
    2. Feature MSE loss (trajectory statistics)
    3. VQ-VAE perceptual loss (optional)

    Args:
        vqvae_checkpoint: Path to VQ-VAE checkpoint (None to disable)
        feature_weight: Weight for feature MSE loss
        perceptual_weight: Weight for VQ-VAE perceptual loss
        trajectory_feature_dim: Dimension of trajectory features (for projector)
        device: Device to use

    Example:
        >>> loss_fn = NOALoss(
        ...     vqvae_checkpoint="checkpoints/production/100k_3family_v1/",
        ...     feature_weight=0.5,
        ...     perceptual_weight=0.1,
        ... )
        >>> pred_traj, gt_traj = ..., ...
        >>> pred_features, gt_features = ..., ...
        >>> losses = loss_fn(pred_traj, gt_traj, pred_features, gt_features)
    """

    def __init__(
        self,
        vqvae_checkpoint: Optional[str] = None,
        feature_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        trajectory_feature_dim: int = 8,  # From extract_trajectory_features
        device: str = "cuda",
    ):
        super().__init__()

        self.feature_weight = feature_weight
        self.perceptual_weight = perceptual_weight
        self.device = device

        # VQ-VAE perceptual loss (optional)
        self.vqvae_loss = None
        self.projector = None

        if vqvae_checkpoint is not None and perceptual_weight > 0:
            self.vqvae_loss = VQVAEPerceptualLoss(
                checkpoint_path=vqvae_checkpoint,
                device=device,
                freeze_vqvae=True,
            )

            # Create projector to map trajectory features to VQ-VAE input dim
            self.projector = FeatureProjector(
                input_dim=trajectory_feature_dim,
                output_dim=self.vqvae_loss.input_dim,
            ).to(device)

    def forward(
        self,
        pred_trajectory: torch.Tensor,
        gt_trajectory: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        gt_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined NOA loss.

        Args:
            pred_trajectory: Predicted trajectory [B, T, C, H, W]
            gt_trajectory: Ground truth trajectory [B, T, C, H, W]
            pred_features: Predicted trajectory features [B, D] (optional)
            gt_features: Ground truth features [B, D] (optional)

        Returns:
            Dictionary with loss components: {total, grid, feature, perceptual}
        """
        # 1. Grid MSE loss
        grid_loss = F.mse_loss(pred_trajectory, gt_trajectory)

        # 2. Feature MSE loss (if features provided)
        feature_loss = torch.tensor(0.0, device=self.device)
        if pred_features is not None and gt_features is not None and self.feature_weight > 0:
            feature_loss = F.mse_loss(pred_features, gt_features)

        # 3. VQ-VAE perceptual loss (if enabled)
        perceptual_loss = torch.tensor(0.0, device=self.device)
        if self.vqvae_loss is not None and self.perceptual_weight > 0:
            if pred_features is not None and gt_features is not None:
                # Project features to VQ-VAE input space
                pred_projected = self.projector(pred_features)
                gt_projected = self.projector(gt_features)

                perceptual_loss = self.vqvae_loss(pred_projected, gt_projected)

        # Combined loss
        total_loss = (
            grid_loss
            + self.feature_weight * feature_loss
            + self.perceptual_weight * perceptual_loss
        )

        return {
            "total": total_loss,
            "grid": grid_loss,
            "feature": feature_loss,
            "perceptual": perceptual_loss,
        }
