"""VQ-VAE Alignment Loss for NOA Training.

Implements two-loss structure for token-aligned NOA training:
1. L_traj: MSE on trajectories (handled externally, not in this module)
2. L_commit: VQ commitment regularizer (manifold adherence)

The key insight is that we use PRE-quantization embeddings for smooth gradients,
and add a commitment loss to force NOA outputs onto the VQ manifold.

Usage:
    alignment = VQVAEAlignmentLoss.from_checkpoint(
        vqvae_path="checkpoints/production/100k_full_features/best_model.pt",
        device="cuda",
    )

    # In training loop
    losses = alignment.compute_losses(pred_trajectory, target_trajectory, ic)
    total_loss = state_loss + lambda_commit * losses['commit']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


class VQVAEAlignmentLoss(nn.Module):
    """VQ-VAE alignment loss for NOA training.

    Computes commitment loss:
    L_commit: MSE between pre-quant latent and quantized (stop-grad)

    This ensures NOA outputs are expressible in the VQ-VAE vocabulary.
    The VQ-VAE weights are FROZEN - it acts as a pre-trained feature extractor.
    """

    def __init__(
        self,
        vqvae: nn.Module,
        feature_extractor: nn.Module,
        normalization_stats: Dict[str, Any],
        group_indices: Dict[str, List[int]],
        device: str = "cuda",
        is_hybrid_model: bool = False,
        noa: Optional[nn.Module] = None,
        enable_latent_loss: bool = False,
        latent_sample_steps: int = 3,
    ):
        """Initialize alignment loss.

        Args:
            vqvae: Pre-trained CategoricalHierarchicalVQVAE model (frozen)
            feature_extractor: Module to extract features from trajectories
            normalization_stats: Per-category mean/std for normalization
            group_indices: Category → feature indices mapping
            device: Computation device
            is_hybrid_model: Whether VQ-VAE is a VQVAEWithInitial (takes raw ICs)
            noa: NOA backbone for latent loss (required if enable_latent_loss=True)
            enable_latent_loss: Enable L_latent (NOA-VQ latent alignment)
            latent_sample_steps: Number of timesteps to sample for latent loss (3=default, -1=all)
        """
        super().__init__()

        self.device = torch.device(device)
        self.vqvae = vqvae.to(self.device)
        self.feature_extractor = feature_extractor.to(self.device)
        self.normalization_stats = normalization_stats
        self.group_indices = group_indices
        self._is_hybrid_model = is_hybrid_model

        # Freeze VQ-VAE weights
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.vqvae.eval()

        # Latent alignment components (optional)
        self.enable_latent_loss = enable_latent_loss
        self.latent_sample_steps = latent_sample_steps
        self.noa = noa

        if enable_latent_loss:
            if noa is None:
                raise ValueError("enable_latent_loss=True requires noa parameter")

            from spinlock.noa.latent_projector import LatentProjector

            # Infer dimensions dynamically
            vq_latent_dim = self._infer_vq_latent_dim(vqvae)
            noa_latent_dim = self._infer_noa_latent_dim(noa)

            # Create projector with inferred dimensions
            self.latent_projector = LatentProjector(
                noa_latent_dim=noa_latent_dim,
                vq_latent_dim=vq_latent_dim,
            ).to(self.device)
        else:
            self.latent_projector = None

    def _infer_vq_latent_dim(self, vqvae: nn.Module) -> int:
        """Infer VQ-VAE latent dimension from model architecture.

        Returns:
            Total latent dimension (sum across all category encoders)
        """
        # Get input dimension - try multiple approaches
        input_dim = None

        if hasattr(vqvae, 'vqvae'):  # HybridVQVAEWrapper
            if hasattr(vqvae.vqvae, 'input_dim'):
                input_dim = vqvae.vqvae.input_dim
        elif hasattr(vqvae, 'input_dim'):
            input_dim = vqvae.input_dim

        # Try to infer from encoder if not found
        if input_dim is None and hasattr(vqvae, 'group_encoders'):
            # Get first encoder's input dimension
            first_encoder = list(vqvae.group_encoders.values())[0]
            if hasattr(first_encoder, 'encoder') and hasattr(first_encoder.encoder[0], 'in_features'):
                # This is the total input across all groups
                input_dim = first_encoder.encoder[0].in_features

        # Fallback: try the feature extractor's output dimension
        if input_dim is None and hasattr(self, 'feature_extractor'):
            if hasattr(self.feature_extractor, 'input_dim'):
                input_dim = self.feature_extractor.input_dim

        # Last resort: conservative default
        if input_dim is None:
            input_dim = 187

        dummy_input = torch.zeros(1, input_dim, device=next(vqvae.parameters()).device)

        with torch.no_grad():
            # For VQVAEWithInitial (hybrid models), need dummy IC too
            if hasattr(vqvae, 'initial_encoder'):
                # Infer grid size from NOA (if available) or use conservative default
                grid_size = 64
                if self.noa is not None:
                    grid_size = self._infer_grid_size(self.noa)

                dummy_ic = torch.zeros(1, 1, grid_size, grid_size, device=dummy_input.device)
                z_list = vqvae.encode(dummy_input, raw_ics=dummy_ic)
            else:
                z_list = vqvae.encode(dummy_input)

            # Concatenate to get total dimension
            z_total = torch.cat(z_list, dim=1)
            return z_total.shape[1]

    def _infer_noa_latent_dim(self, noa: nn.Module) -> int:
        """Infer NOA bottleneck feature dimension.

        Returns:
            Channel dimension of bottleneck features
        """
        grid_size = self._infer_grid_size(noa)

        dummy_state = torch.zeros(1, noa.in_channels, grid_size, grid_size,
                                   device=next(noa.parameters()).device)

        with torch.no_grad():
            features = noa.get_intermediate_features(dummy_state, extract_from="bottleneck")
            bottleneck = features['bottleneck']  # [1, C, H, W]
            return bottleneck.shape[1]  # Return channel dimension

    def _infer_grid_size(self, noa: nn.Module) -> int:
        """Infer grid size from NOA operator architecture.

        Returns:
            Grid size (H = W assumed square)
        """
        # Try to infer from U-AFNO operator config if available
        if hasattr(noa, 'operator'):
            # U-AFNO typically works with power-of-2 grids
            # Test with common grid sizes
            test_sizes = [64, 128, 256]

            for size in test_sizes:
                try:
                    dummy = torch.zeros(1, noa.in_channels, size, size,
                                        device=next(noa.parameters()).device)
                    with torch.no_grad():
                        _ = noa.operator(dummy)
                    # If successful, this is a valid grid size
                    return size
                except:
                    continue

        # Fallback default
        return 64

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

    def _compute_latent_alignment(
        self,
        pred_trajectory: torch.Tensor,
        vq_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Compute latent alignment loss between NOA and VQ-VAE.

        Args:
            pred_trajectory: [B, T, C, H, W] predicted states
            vq_latents: [B, D_vq] VQ encoder pre-quantization latents

        Returns:
            Scalar latent alignment loss
        """
        B, T, C, H, W = pred_trajectory.shape

        # Determine timesteps to sample
        if self.latent_sample_steps <= 0 or self.latent_sample_steps >= T:
            # Sample all timesteps
            sample_indices = list(range(T))
        else:
            # Sample evenly spaced timesteps
            sample_indices = [int(i * (T - 1) / (self.latent_sample_steps - 1))
                            for i in range(self.latent_sample_steps)]

        noa_latents_sampled = []

        for t in sample_indices:
            state_t = pred_trajectory[:, t, :, :, :]  # [B, C, H, W]

            # Extract bottleneck features from NOA
            noa_features = self.noa.get_intermediate_features(
                state_t,
                extract_from="bottleneck"
            )
            bottleneck = noa_features['bottleneck']  # [B, C_noa, H', W']

            # Project to VQ space
            proj_latent = self.latent_projector(bottleneck)  # [B, D_vq]
            noa_latents_sampled.append(proj_latent)

        # Aggregate across sampled timesteps (mean pooling for stability)
        noa_latents_trajectory = torch.stack(noa_latents_sampled, dim=1)  # [B, n_samples, D_vq]
        noa_latents_aggregated = noa_latents_trajectory.mean(dim=1)  # [B, D_vq]

        # Compute MSE between NOA latents and VQ latents
        # No normalization - this is the key difference from previous failed attempt
        latent_loss = F.mse_loss(noa_latents_aggregated, vq_latents.detach())

        return latent_loss

    def compute_losses(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute alignment losses.

        Args:
            pred_trajectory: NOA predicted trajectory [B, T, C, H, W] or [B, M, T, C, H, W]
            target_trajectory: CNO target trajectory (unused, kept for API compat)
            ic: Initial condition [B, C, H, W] (optional, for initial features)

        Returns:
            Dictionary with:
                - 'commit': VQ commitment loss (manifold adherence)
                - 'z_pred': Pre-quantized latent for pred (for logging)
        """
        # Extract features from predicted trajectory
        pred_result = self.feature_extractor(pred_trajectory, ic=ic)

        if isinstance(pred_result, tuple):
            pred_features, pred_raw_ics = pred_result
        else:
            pred_features = pred_result
            pred_raw_ics = ic

        # Normalize features
        pred_norm = self._normalize_features(pred_features)

        # Encode pred to pre-quantization latents
        if self._is_hybrid_model and pred_raw_ics is not None:
            z_pred_list = self.vqvae.encode(pred_norm, raw_ics=pred_raw_ics)
        else:
            z_pred_list = self.vqvae.encode(pred_norm)
        z_pred = torch.cat(z_pred_list, dim=1)  # [B, total_latent_dim]

        # L_commit: Force pred to be close to its quantized version
        # This ensures NOA outputs are expressible in VQ vocabulary
        z_q_pred_list, _, _ = self.vqvae.quantize(z_pred_list)
        z_q_pred = torch.cat(z_q_pred_list, dim=1)
        commit_loss = F.mse_loss(z_pred, z_q_pred.detach())

        losses = {
            'commit': commit_loss,
            'z_pred': z_pred.detach(),
        }

        # L_latent: NOA-VQ latent alignment (optional)
        if self.enable_latent_loss and self.latent_projector is not None:
            latent_loss = self._compute_latent_alignment(
                pred_trajectory=pred_trajectory,
                vq_latents=z_pred,
            )
            losses['latent'] = latent_loss

        return losses

    @classmethod
    def from_checkpoint(
        cls,
        vqvae_path: str,
        device: str = "cuda",
        feature_extractor: Optional[nn.Module] = None,
        use_aligned_extractor: bool = True,
        noa: Optional[nn.Module] = None,
        enable_latent_loss: bool = False,
        latent_sample_steps: int = 3,
    ) -> "VQVAEAlignmentLoss":
        """Load alignment loss from VQ-VAE checkpoint.

        Args:
            vqvae_path: Path to VQ-VAE checkpoint directory or .pt file
            device: Computation device
            feature_extractor: Optional custom feature extractor
                             If None, creates from checkpoint config
            use_aligned_extractor: If True, use AlignedFeatureExtractor for
                                  3-family models (default). If False, use legacy
                                  TrajectoryFeatureExtractor.
            noa: NOA backbone for latent loss (required if enable_latent_loss=True)
            enable_latent_loss: Enable L_latent (NOA-VQ latent alignment)
            latent_sample_steps: Number of timesteps to sample for latent loss (3=default, -1=all)

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

        # Get config and families
        config = checkpoint.get('config', {})
        families = checkpoint.get('families', {})
        group_indices = config.get('group_indices', checkpoint.get('group_indices', {}))
        state_dict = checkpoint['model_state_dict']

        # Detect hybrid model (VQVAEWithInitial) by checking for initial_encoder
        is_hybrid_model = any('initial_encoder' in k for k in state_dict.keys())

        # Create appropriate VQ-VAE model
        if is_hybrid_model:
            from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
            from spinlock.encoding.encoders.initial_hybrid import InitialHybridEncoder

            # Handle compiled model prefix
            if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            # For hybrid models, the checkpoint was saved with ADJUSTED dimensions
            # (input_dim already accounts for CNN features added to INITIAL)
            # We need to load the inner vqvae and initial_encoder separately

            # Get initial encoder config
            initial_config = families.get('initial', {}).get('encoder_params', {})
            manual_dim = initial_config.get('manual_dim', 14)
            cnn_dim = initial_config.get('cnn_embedding_dim', 28)
            in_channels = initial_config.get('in_channels', 1)

            # The checkpoint's input_dim is already adjusted (187D)
            # The inner VQ-VAE expects this dimension
            vqvae_config = CategoricalVQVAEConfig(
                input_dim=config.get('input_dim', 187),
                group_indices=group_indices,
                group_embedding_dim=config.get('group_embedding_dim', 256),
                group_hidden_dim=config.get('group_hidden_dim', 512),
                levels=config.get('levels'),
            )

            # Create inner VQ-VAE
            inner_vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

            # Load inner VQ-VAE weights
            vqvae_state = {k.replace('vqvae.', ''): v for k, v in state_dict.items()
                          if k.startswith('vqvae.')}
            inner_vqvae.load_state_dict(vqvae_state)

            # Create initial encoder
            initial_encoder = InitialHybridEncoder(
                manual_dim=manual_dim,
                cnn_embedding_dim=cnn_dim,
                encode_manual=False,
                in_channels=in_channels,
            )

            # Load initial encoder weights
            encoder_state = {k.replace('initial_encoder.', ''): v for k, v in state_dict.items()
                            if k.startswith('initial_encoder.')}
            initial_encoder.load_state_dict(encoder_state)

            # Create a wrapper that matches VQVAEWithInitial interface but doesn't re-adjust
            class HybridVQVAEWrapper(nn.Module):
                """Wrapper for hybrid VQ-VAE that doesn't re-adjust dimensions."""

                def __init__(self, vqvae, initial_encoder, manual_dim, cnn_dim):
                    super().__init__()
                    self.vqvae = vqvae
                    self.initial_encoder = initial_encoder
                    self.initial_manual_dim = manual_dim
                    self.initial_cnn_dim = cnn_dim
                    self.initial_feature_offset = 0

                def encode(self, features, raw_ics=None):
                    """Encode to pre-quantization latents."""
                    if raw_ics is not None:
                        features = self._combine_features(features, raw_ics)
                    return self.vqvae.encode(features)

                def quantize(self, z_list):
                    """Quantize latents."""
                    return self.vqvae.quantize(z_list)

                def _combine_features(self, features, raw_ics):
                    """Combine manual features with CNN embeddings."""
                    # Extract manual INITIAL features
                    manual_features = features[:, :self.initial_manual_dim]

                    # Get CNN embeddings
                    initial_embeddings = self.initial_encoder(manual_features, raw_ics)

                    # Replace manual with hybrid
                    features_after = features[:, self.initial_manual_dim:]

                    return torch.cat([initial_embeddings, features_after], dim=1)

            vqvae = HybridVQVAEWrapper(inner_vqvae, initial_encoder, manual_dim, cnn_dim)
        else:
            from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

            vqvae_config = CategoricalVQVAEConfig(
                input_dim=config.get('input_dim', 225),
                group_indices=group_indices,
                group_embedding_dim=config.get('group_embedding_dim', 256),
                group_hidden_dim=config.get('group_hidden_dim', 512),
                levels=config.get('levels'),
            )

            vqvae = CategoricalHierarchicalVQVAE(vqvae_config)

            # Handle different checkpoint formats
            has_orig_mod_vqvae = any(k.startswith('_orig_mod.vqvae.') for k in state_dict.keys())
            has_vqvae_prefix = any(k.startswith('vqvae.') for k in state_dict.keys())

            if has_orig_mod_vqvae:
                vqvae_state = {}
                prefix = '_orig_mod.vqvae.'
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        vqvae_state[k[len(prefix):]] = v
                state_dict = vqvae_state
            elif has_vqvae_prefix:
                vqvae_state = {}
                prefix = 'vqvae.'
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        vqvae_state[k[len(prefix):]] = v
                state_dict = vqvae_state
            else:
                sample_key = next(iter(state_dict.keys()))
                if sample_key.startswith('_orig_mod.'):
                    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

            vqvae.load_state_dict(state_dict)

        # Create or use provided feature extractor
        if feature_extractor is None:
            if use_aligned_extractor and families:
                # Use AlignedFeatureExtractor for 3-family models
                feature_extractor = AlignedFeatureExtractor.from_checkpoint(
                    checkpoint_path=str(path),
                    device=device,
                )
            else:
                # Legacy extractor
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
            is_hybrid_model=is_hybrid_model,
            noa=noa,
            enable_latent_loss=enable_latent_loss,
            latent_sample_steps=latent_sample_steps,
        )


class TrajectoryFeatureExtractor(nn.Module):
    """Extract features from trajectories matching VQ-VAE input format.

    DEPRECATED: Use AlignedFeatureExtractor for proper 3-family alignment.

    This extractor produces a feature vector compatible with simple VQ-VAE
    by combining summary and temporal features. For 3-family VQ-VAE models
    (100k_3family_v1), use AlignedFeatureExtractor instead.
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
        config = SummaryConfig(
            realization_aggregation=["mean"],
            temporal_aggregation=["mean"],
        )
        self.summary_extractor = SummaryExtractor(device=self.device, config=config)

    def forward(
        self,
        trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract features from trajectory."""
        if trajectory.dim() == 6:
            trajectory = trajectory[:, 0]

        traj_with_m = trajectory.unsqueeze(1)
        result = self.summary_extractor.extract_all(traj_with_m)

        summary_features = result['per_trajectory'].squeeze(1)
        temporal_features = result['per_timestep']
        temporal_agg = temporal_features.mean(dim=1)

        features = torch.cat([summary_features, temporal_agg], dim=1)

        if features.shape[1] != self.input_dim:
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

        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features


class AlignedFeatureExtractor(nn.Module):
    """Extract features from trajectories matching 3-family VQ-VAE format.

    This extractor produces features compatible with production VQ-VAE checkpoints
    (e.g., 100k_3family_v1) by extracting and encoding features in the same way
    as dataset generation + VQ-VAE training:

    1. INITIAL (14D manual): Basic spatial/spectral stats from IC
    2. SUMMARY (128D encoded): Trajectory-level aggregated stats via MLPEncoder
    3. TEMPORAL (128D encoded): Per-timestep time series via TemporalCNNEncoder

    The encoders are loaded from the VQ-VAE checkpoint to ensure alignment.

    For VQVAEWithInitial models, this extractor also returns raw ICs for the
    CNN portion of InitialHybridEncoder (trained end-to-end in VQ-VAE).
    """

    def __init__(
        self,
        input_dim: int = 187,
        device: str = "cuda",
        summary_encoder: Optional[nn.Module] = None,
        temporal_encoder: Optional[nn.Module] = None,
    ):
        """Initialize aligned feature extractor.

        Args:
            input_dim: Expected output dimension (after concatenation + cleanup)
            device: Computation device
            summary_encoder: Pre-trained MLPEncoder for SUMMARY features
            temporal_encoder: Pre-trained TemporalCNNEncoder for TEMPORAL features
        """
        super().__init__()

        self.input_dim = input_dim
        self.device = torch.device(device)

        # Store encoders (frozen, from VQ-VAE checkpoint)
        self.summary_encoder = summary_encoder
        self.temporal_encoder = temporal_encoder

        if summary_encoder is not None:
            summary_encoder.to(self.device)
            summary_encoder.eval()
            for p in summary_encoder.parameters():
                p.requires_grad = False

        if temporal_encoder is not None:
            temporal_encoder.to(self.device)
            temporal_encoder.eval()
            for p in temporal_encoder.parameters():
                p.requires_grad = False

        # Import extractors
        from spinlock.features.summary.config import SummaryConfig
        from spinlock.features.summary.extractors import SummaryExtractor
        from spinlock.features.initial.manual_extractors import InitialManualExtractor

        # INITIAL extractor (manual 14D features from IC)
        self.initial_extractor = InitialManualExtractor(device=self.device)

        # SUMMARY extractor with config matching dataset generation
        summary_config = SummaryConfig(
            realization_aggregation=["mean"],
            temporal_aggregation=["mean"],
        )
        self.summary_extractor = SummaryExtractor(device=self.device, config=summary_config)

        # Feature dimensions (before encoding)
        self.initial_dim = 14  # Manual INITIAL features
        self.summary_raw_dim = 360  # Raw SUMMARY aggregated features
        self.temporal_raw_dim = 63  # Per-timestep TEMPORAL features

    def forward(
        self,
        trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from trajectory.

        Args:
            trajectory: Trajectory tensor [B, T, C, H, W] or [B, M, T, C, H, W]
            ic: Initial condition [B, C, H, W] (required for INITIAL features)

        Returns:
            Tuple of:
            - features: Flat feature vector [B, input_dim]
            - raw_ics: Raw ICs [B, C, H, W] for VQ-VAE's InitialHybridEncoder
        """
        B = trajectory.shape[0]

        # Handle multi-realization trajectories
        if trajectory.dim() == 6:
            # [B, M, T, C, H, W] - use first realization
            trajectory = trajectory[:, 0]  # [B, T, C, H, W]

        # Add realization dimension for extractor: [B, T, C, H, W] -> [B, 1, T, C, H, W]
        traj_with_m = trajectory.unsqueeze(1)

        # === INITIAL features (14D manual) ===
        if ic is not None:
            # Extract manual INITIAL features from IC
            # InitialManualExtractor expects [B, M, C, H, W], add M=1 dim
            ic_with_m = ic.unsqueeze(1)  # [B, 1, C, H, W]
            initial_features = self.initial_extractor.extract_all(ic_with_m)  # [B, 1, 14]
            initial_features = initial_features.squeeze(1)  # [B, 14]
        else:
            # Use trajectory's first frame as IC approximation
            ic_approx = trajectory[:, 0]  # [B, C, H, W]
            ic_with_m = ic_approx.unsqueeze(1)  # [B, 1, C, H, W]
            initial_features = self.initial_extractor.extract_all(ic_with_m)  # [B, 1, 14]
            initial_features = initial_features.squeeze(1)  # [B, 14]

        # === SUMMARY features ===
        summary_result = self.summary_extractor.extract_all(traj_with_m)

        # Get aggregated SUMMARY features
        if 'aggregated_mean' in summary_result:
            summary_raw = summary_result['aggregated_mean']  # [B, D_summary]
        else:
            summary_raw = summary_result['per_trajectory'].squeeze(1)  # [B, D_summary]

        # Encode SUMMARY if encoder provided
        if self.summary_encoder is not None:
            # Pad/truncate to expected input dim
            if summary_raw.shape[1] < self.summary_raw_dim:
                pad = torch.zeros(B, self.summary_raw_dim - summary_raw.shape[1],
                                  device=summary_raw.device, dtype=summary_raw.dtype)
                summary_raw = torch.cat([summary_raw, pad], dim=1)
            elif summary_raw.shape[1] > self.summary_raw_dim:
                summary_raw = summary_raw[:, :self.summary_raw_dim]

            summary_encoded = self.summary_encoder(summary_raw)  # [B, 128]
        else:
            # No encoder - use raw (will be truncated later)
            summary_encoded = summary_raw

        # === TEMPORAL features ===
        # Get per-timestep features [B, T, D_temporal]
        temporal_raw = summary_result['per_timestep']  # [B, T, D_temporal]

        # Encode TEMPORAL if encoder provided
        if self.temporal_encoder is not None:
            # Pad/truncate feature dimension to expected input
            if temporal_raw.shape[2] < self.temporal_raw_dim:
                pad = torch.zeros(B, temporal_raw.shape[1],
                                  self.temporal_raw_dim - temporal_raw.shape[2],
                                  device=temporal_raw.device, dtype=temporal_raw.dtype)
                temporal_raw = torch.cat([temporal_raw, pad], dim=2)
            elif temporal_raw.shape[2] > self.temporal_raw_dim:
                temporal_raw = temporal_raw[:, :, :self.temporal_raw_dim]

            temporal_encoded = self.temporal_encoder(temporal_raw)  # [B, 128]
        else:
            # No encoder - aggregate via mean
            temporal_encoded = temporal_raw.mean(dim=1)  # [B, D_temporal]

        # === Concatenate all features ===
        features = torch.cat([
            initial_features,  # [B, 14]
            summary_encoded,   # [B, 128] or [B, D_summary]
            temporal_encoded,  # [B, 128] or [B, D_temporal]
        ], dim=1)

        # Handle dimension mismatch (pad/truncate to input_dim)
        if features.shape[1] < self.input_dim:
            padding = torch.zeros(B, self.input_dim - features.shape[1],
                                  device=features.device, dtype=features.dtype)
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > self.input_dim:
            features = features[:, :self.input_dim]

        # Clean NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Return features and raw ICs (for VQ-VAE's InitialHybridEncoder)
        raw_ics = ic if ic is not None else trajectory[:, 0]

        return features, raw_ics

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "AlignedFeatureExtractor":
        """Create extractor with encoders loaded from VQ-VAE checkpoint.

        Args:
            checkpoint_path: Path to VQ-VAE checkpoint directory or .pt file
            device: Computation device

        Returns:
            Configured AlignedFeatureExtractor with encoders from checkpoint
        """
        from pathlib import Path
        from spinlock.encoding.encoders import get_encoder

        path = Path(checkpoint_path)
        if path.is_dir():
            checkpoint_file = path / "best_model.pt"
        else:
            checkpoint_file = path

        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        families = checkpoint.get('families', {})

        input_dim = config.get('input_dim', 187)

        # Create encoders from family configs
        summary_encoder = None
        temporal_encoder = None

        if 'summary' in families:
            summary_config = families['summary']
            encoder_name = summary_config.get('encoder')
            if encoder_name and encoder_name not in ['identity', 'IdentityEncoder', 'initial_hybrid']:
                params = summary_config.get('encoder_params', {})
                # Get input_dim from raw feature size
                summary_encoder = get_encoder(
                    encoder_name,
                    input_dim=360,  # Raw SUMMARY dimension
                    **params
                )

        if 'temporal' in families:
            temporal_config = families['temporal']
            encoder_name = temporal_config.get('encoder')
            if encoder_name and encoder_name not in ['identity', 'IdentityEncoder', 'initial_hybrid']:
                params = temporal_config.get('encoder_params', {})
                temporal_encoder = get_encoder(
                    encoder_name,
                    input_dim=63,  # Per-timestep TEMPORAL feature dimension
                    **params
                )

        return cls(
            input_dim=input_dim,
            device=device,
            summary_encoder=summary_encoder,
            temporal_encoder=temporal_encoder,
        )
