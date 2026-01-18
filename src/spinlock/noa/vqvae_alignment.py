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

Documentation:
    - Stage 2 VQ-led training: docs/two-stage-curriculum-architecture.md
    - VQ-VAE checkpoint format: docs/vqvae/checkpoint-format.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path


def _reconstruct_levels_from_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    group_indices: Dict[str, List[int]]
) -> Dict[str, List[Dict]]:
    """Reconstruct per-category levels dict from VQ layer dimensions in checkpoint.

    This provides backward compatibility for old checkpoints that have levels=[]
    instead of the new per-category levels dict.

    Args:
        state_dict: Model state dict with 'vqvae.vq_layers.X.embedding.weight' keys
        group_indices: Dict mapping category name -> list of feature indices

    Returns:
        Dict mapping category name -> list of level configs
        Example: {
            'cluster_1': [
                {'num_tokens': 16, 'latent_dim': 4},
                {'num_tokens': 16, 'latent_dim': 8},
                {'num_tokens': 16, 'latent_dim': 12}
            ],
            ...
        }
    """
    # Extract VQ layer dimensions from embeddings
    vq_dims = []
    layer_idx = 0
    while True:
        key = f'vqvae.vq_layers.{layer_idx}.embedding.weight'
        if key not in state_dict:
            break
        embedding = state_dict[key]
        num_tokens, latent_dim = embedding.shape
        vq_dims.append({'num_tokens': int(num_tokens), 'latent_dim': int(latent_dim)})
        layer_idx += 1

    # Determine number of categories and levels per category
    num_categories = len(group_indices)
    if len(vq_dims) % num_categories != 0:
        raise ValueError(
            f"Cannot reconstruct levels: {len(vq_dims)} VQ layers is not "
            f"divisible by {num_categories} categories"
        )

    num_levels = len(vq_dims) // num_categories

    # Distribute layers to categories
    # Assumption: VQ layers are stored in category-major order
    # (cat0_L0, cat0_L1, cat0_L2, cat1_L0, cat1_L1, cat1_L2, ...)
    levels_dict = {}
    category_names = list(group_indices.keys())

    for cat_idx, cat_name in enumerate(category_names):
        cat_levels = []
        for level_idx in range(num_levels):
            vq_layer_idx = cat_idx * num_levels + level_idx
            cat_levels.append(vq_dims[vq_layer_idx])
        levels_dict[cat_name] = cat_levels

    return levels_dict


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
        feature_mask: Optional[np.ndarray] = None,
        feature_cleaning_params: Optional[Dict[str, Any]] = None,
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
            feature_mask: Boolean mask for feature cleaning (from VQ-VAE training)
            feature_cleaning_params: Feature cleaning parameters (from VQ-VAE training)
        """
        super().__init__()

        self.device = torch.device(device)
        self.vqvae = vqvae.to(self.device)
        self.feature_extractor = feature_extractor.to(self.device)
        self.normalization_stats = normalization_stats
        self.group_indices = group_indices
        self._is_hybrid_model = is_hybrid_model

        # Store feature cleaning info (needed for VQ-led loss)
        self.feature_mask = feature_mask
        self.feature_cleaning_params = feature_cleaning_params

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

    # NOTE: _normalize_features() method removed - normalization now handled by UnifiedFeaturePipeline

    def _apply_feature_cleaning(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature selection to match VQ-VAE input dimensions.

        This is necessary for VQ-led loss where VQ-VAE was trained on a subset of features.
        The feature extractor produces all features (e.g., 187), but VQ-VAE expects only
        the subset defined by group_indices (e.g., 171).

        Args:
            features: [B, D_extractor] features from feature extractor

        Returns:
            features_selected: [B, D_vqvae] features after selecting VQ-VAE subset
                              where D_vqvae = sum of lengths of all group_indices
        """
        # Collect all feature indices used by VQ-VAE from group_indices
        all_indices = []
        for group_name in sorted(self.group_indices.keys()):  # Sort for consistency
            all_indices.extend(self.group_indices[group_name])

        # Sort indices to maintain order
        all_indices = sorted(all_indices)

        # Convert to tensor
        indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=features.device)

        # Select features
        # features has shape [B, D_extractor]
        # indices_tensor has shape [D_vqvae]
        features_selected = features[:, indices_tensor]

        return features_selected

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
        normalization_stats_file: Optional[str] = None,
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
            normalization_stats_file: Optional path to external normalization stats file.
                                     If provided and checkpoint lacks normalization_stats,
                                     loads stats from this file instead. Required for
                                     Stage 2 VQ-led training if VQ-VAE was trained without
                                     normalization.

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

        # Load normalization stats from checkpoint
        # They're stored at top-level in the checkpoint dict (not in config)
        normalization_stats = checkpoint.get('normalization_stats')

        # If not in checkpoint, try loading from external file
        if normalization_stats is None and normalization_stats_file is not None:
            print(f"  Loading normalization stats from: {normalization_stats_file}")
            stats_data = torch.load(normalization_stats_file, map_location='cpu', weights_only=False)
            # Remove metadata key if present
            normalization_stats = {k: v for k, v in stats_data.items() if k != '_metadata'}
            print(f"  ✓ Loaded {len([k for k in normalization_stats if 'mean' in k])} feature groups")

        # Convert NormalizationStats objects to flat dict format
        # VQ-VAE checkpoint stores stats as {cluster_N: NormalizationStats(mean, std)}
        # but _normalize_features() expects {cluster_N_mean: [...], cluster_N_std: [...]}
        if normalization_stats is not None:
            from spinlock.encoding.normalization import NormalizationStats

            # Check if conversion is needed (first value is NormalizationStats object)
            first_key = list(normalization_stats.keys())[0] if normalization_stats else None
            if first_key and isinstance(normalization_stats[first_key], NormalizationStats):
                # Convert to flat dict format
                flat_stats = {}
                for group_name, stats_obj in normalization_stats.items():
                    # Convert numpy arrays to lists for JSON serialization
                    flat_stats[f"{group_name}_mean"] = stats_obj.mean.tolist() if hasattr(stats_obj.mean, 'tolist') else list(stats_obj.mean)
                    flat_stats[f"{group_name}_std"] = stats_obj.std.tolist() if hasattr(stats_obj.std, 'tolist') else list(stats_obj.std)
                normalization_stats = flat_stats
                print(f"  ✓ Converted {len(normalization_stats) // 2} normalization groups to flat format")

        # Get config and families
        # Try model_config first (has actual VQ-VAE params), fall back to training config
        model_config = checkpoint.get('model_config', {})
        config = checkpoint.get('config', {})
        families = config.get('families', checkpoint.get('families', {}))
        # Use model_config.group_indices (actual indices used during training)
        group_indices = model_config.get('group_indices', checkpoint.get('pre_model_group_indices', config.get('group_indices', {})))
        state_dict = checkpoint['model_state_dict']

        # Get input_dim from model_config if available, otherwise from feature_mask or config
        input_dim = model_config.get('input_dim')
        feature_mask = checkpoint.get('feature_mask', None)
        feature_cleaning_params = checkpoint.get('feature_cleaning_params', None)

        if input_dim is None and feature_mask is not None:
            # Count number of True values in feature_mask
            if hasattr(feature_mask, '__len__'):
                input_dim = int(np.sum(feature_mask))
        if input_dim is None:
            # Try config.input_dim before falling back to default
            input_dim = config.get('input_dim', 225)  # Default to 225 for production models

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

            # Reconstruct levels from checkpoint if needed (backward compatibility)
            # Old checkpoints have levels=[] which needs to be reconstructed from VQ layer dimensions
            # Try model_config.levels first (actual levels used during training)
            levels_from_config = model_config.get('levels') or config.get('levels')
            if levels_from_config is None or (isinstance(levels_from_config, list) and len(levels_from_config) == 0):
                # Get group_indices from checkpoint (try multiple keys)
                actual_group_indices = (
                    checkpoint.get('pre_model_group_indices') or
                    checkpoint.get('group_indices') or
                    config.get('group_indices') or
                    group_indices
                )
                # Extract VQ layer dimensions from state_dict (keys have 'vqvae.' prefix)
                vqvae_state = {k: v for k, v in state_dict.items() if k.startswith('vqvae.')}
                levels_from_config = _reconstruct_levels_from_checkpoint(
                    state_dict=vqvae_state,
                    group_indices=actual_group_indices
                )

            vqvae_config = CategoricalVQVAEConfig(
                input_dim=input_dim,
                group_indices=group_indices,
                group_embedding_dim=config.get('group_embedding_dim', 256),
                group_hidden_dim=config.get('group_hidden_dim', 512),
                levels=levels_from_config,
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

                def decode(self, z_q_list):
                    """Decode quantized latents to feature space."""
                    return self.vqvae.decode(z_q_list)

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
            feature_mask=feature_mask,
            feature_cleaning_params=feature_cleaning_params,
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
        from spinlock.features.temporal.config import SummaryConfig
        from spinlock.features.temporal.extractors import SummaryExtractor

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

    This is a thin wrapper around UnifiedFeaturePipeline to ensure VQ-VAE training
    and meta-operator training use IDENTICAL feature extraction and normalization.

    Features: 14D INITIAL + 128D SUMMARY (encoded) + 128D TEMPORAL (encoded) = 270D

    The pipeline handles:
    1. Feature extraction (per-family)
    2. Encoding (using frozen VQ-VAE encoders)
    3. Normalization (per-family mean/std from VQ-VAE checkpoint)

    This replaces the old 200+ line implementation with a DRY solution.
    """

    def __init__(
        self,
        pipeline: "UnifiedFeaturePipeline",
        input_dim: int = 270,
        device: str = "cuda",
    ):
        """Initialize aligned feature extractor.

        Args:
            pipeline: UnifiedFeaturePipeline with loaded encoders and normalization
            input_dim: Expected output dimension (270D for 3-family VQ-VAE)
            device: Computation device
        """
        super().__init__()
        self.pipeline = pipeline
        self.input_dim = input_dim
        self.device = torch.device(device)

    def forward(
        self,
        trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from trajectory using unified pipeline.

        Args:
            trajectory: Trajectory tensor [B, T, C, H, W] or [B, M, T, C, H, W]
            ic: Initial condition [B, C, H, W] (optional, extracted from trajectory[0] if None)

        Returns:
            Tuple of:
            - features: Normalized feature vector [B, 270D]
            - raw_ics: Raw ICs [B, C, H, W] for VQ-VAE's InitialHybridEncoder
        """
        # Handle multi-realization trajectories
        if trajectory.dim() == 6:
            # [B, M, T, C, H, W] - use first realization
            trajectory = trajectory[:, 0]  # [B, T, C, H, W]

        # Extract IC if not provided
        if ic is None:
            ic = trajectory[:, 0]  # [B, C, H, W]

        # Extract and normalize features using unified pipeline
        # Pipeline outputs: 14D + 128D + 128D = 270D (already normalized)
        features = self.pipeline(trajectory, ic, normalize=True)

        # Clean NaN/Inf (safety)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, ic

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
    ) -> "AlignedFeatureExtractor":
        """Create extractor with pipeline loaded from VQ-VAE checkpoint.

        Args:
            checkpoint_path: Path to VQ-VAE checkpoint directory or .pt file
            device: Computation device

        Returns:
            Configured AlignedFeatureExtractor with UnifiedFeaturePipeline
        """
        from pathlib import Path
        from spinlock.encoding import UnifiedFeaturePipeline

        path = Path(checkpoint_path)
        if path.is_dir():
            checkpoint_file = path / "best_model.pt"
        else:
            checkpoint_file = path

        # Load checkpoint to check if it has per-family normalization stats
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        norm_stats = checkpoint.get('normalization_stats', {})

        # Check if checkpoint has new per-family format
        has_new_format = (
            'initial' in norm_stats and
            'summary' in norm_stats and
            'temporal' in norm_stats
        )

        if has_new_format:
            # New checkpoint format - use UnifiedFeaturePipeline directly
            print("  ✓ Loading UnifiedFeaturePipeline from checkpoint (per-family normalization)")
            pipeline = UnifiedFeaturePipeline.from_checkpoint(str(checkpoint_file), device=device)
        else:
            # Old checkpoint format - create pipeline with temporary identity normalization
            # This avoids dimension mismatches while maintaining correct feature dimensions
            print("  ⚠ Old checkpoint format detected (cluster-based normalization)")
            print("    Creating UnifiedFeaturePipeline with identity normalization (mean=0, std=1)")
            print("    VQ-VAE should be retrained with new format for proper normalization")

            # Create pipeline from checkpoint (loads encoders)
            try:
                pipeline = UnifiedFeaturePipeline.from_checkpoint(str(checkpoint_file), device=device)
            except KeyError:
                # If checkpoint is missing some fields, build pipeline manually from config
                from spinlock.encoding import InitialFeatureExtractor, SummaryFeatureExtractor, TemporalFeatureExtractor
                from spinlock.encoding.encoders import get_encoder

                families = checkpoint.get('families', checkpoint.get('config', {}).get('families', {}))

                # Load encoders
                summary_encoder = None
                temporal_encoder = None

                if 'summary' in families:
                    summary_config = families['summary']
                    encoder_name = summary_config.get('encoder')
                    if encoder_name and encoder_name not in ['identity', 'IdentityEncoder']:
                        params = summary_config.get('encoder_params', {})
                        summary_encoder = get_encoder(encoder_name, input_dim=360, **params)

                if 'temporal' in families:
                    temporal_config = families['temporal']
                    encoder_name = temporal_config.get('encoder')
                    if encoder_name and encoder_name not in ['identity', 'IdentityEncoder']:
                        params = temporal_config.get('encoder_params', {})
                        temporal_encoder = get_encoder(encoder_name, input_dim=63, **params)

                # Create extractors
                initial = InitialFeatureExtractor(device=device)
                summary = SummaryFeatureExtractor(summary_encoder, device=device) if summary_encoder else None
                temporal = TemporalFeatureExtractor(temporal_encoder, device=device) if temporal_encoder else None

                pipeline = UnifiedFeaturePipeline(initial, summary, temporal)

            # Set identity normalization (mean=0, std=1) to avoid dimension mismatches
            # This doesn't normalize features but avoids the dimension mismatch that was causing huge losses
            pipeline.initial.normalization_stats = (
                torch.zeros(14, device=device),
                torch.ones(14, device=device)
            )
            pipeline.summary.normalization_stats = (
                torch.zeros(128, device=device),
                torch.ones(128, device=device)
            )
            pipeline.temporal.normalization_stats = (
                torch.zeros(128, device=device),
                torch.ones(128, device=device)
            )

        return cls(
            pipeline=pipeline,
            input_dim=270,
            device=device,
        )
