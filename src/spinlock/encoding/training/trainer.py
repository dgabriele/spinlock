"""VQ-VAE trainer for operator feature tokenization.

Simple training loop with:
- 5-component loss function
- Early stopping
- Dead code reset
- Checkpointing
- Validation every N epochs

Ported from unisim.system.training.trainer (simplified, removed multimodal/IC support).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import time
import logging

from ..categorical_vqvae import CategoricalHierarchicalVQVAE
from .losses import compute_total_loss
from .callbacks import EarlyStopping, DeadCodeReset, SmartDeadCodeReset, Checkpointer

logger = logging.getLogger(__name__)


class VQVAETrainer:
    """Trainer for categorical hierarchical VQ-VAE."""

    def __init__(
        self,
        model: CategoricalHierarchicalVQVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        device: str = "cuda",
        # Loss weights
        orthogonality_weight: float = 0.1,
        informativeness_weight: float = 0.1,
        category_reconstruction_weight: float = 0.0,
        topo_weight: float = 0.02,
        topo_samples: int = 64,
        reference_reg_weight: float = 0.0,
        entropy_weight: float = 0.0,
        normalize_mse: bool = True,
        # Callbacks
        early_stopping_patience: int = 100,
        early_stopping_min_delta: float = 0.01,
        dead_code_reset_interval: int = 100,
        dead_code_threshold: float = 10.0,
        dead_code_max_reset_fraction: float = 0.25,
        use_smart_reset: bool = False,
        checkpoint_dir: Optional[Path] = None,
        # Optimization
        use_torch_compile: bool = True,
        val_every_n_epochs: int = 5,
        gradient_clip_norm: Optional[float] = None,
        warmup_epochs: int = 0,
        scheduler_config: Optional[dict] = None,
        # Logging
        verbose: bool = True,
        # Metadata for checkpoint reproducibility
        config: Optional[dict] = None,
        group_indices: Optional[dict] = None,
        normalization_stats: Optional[dict] = None,
        per_family_normalization_stats: Optional[dict] = None,
        feature_names: Optional[list] = None,
        encoder_state_dicts: Optional[dict] = None,
        feature_mask: Optional[np.ndarray] = None,
        feature_cleaning_params: Optional[dict] = None,
        # Variable-length mode support
        temporal_encoder: Optional[nn.Module] = None,
        temporal_encoder_output_dim: Optional[int] = None,
        encoded_initial_features: Optional[np.ndarray] = None,
    ):
        """Initialize trainer.

        Args:
            model: CategoricalHierarchicalVQVAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to use
            orthogonality_weight: Weight for orthogonality loss
            informativeness_weight: Weight for informativeness loss
            category_reconstruction_weight: Weight for per-category reconstruction regularizer (0.0 = disabled)
            topo_weight: Weight for topographic loss
            topo_samples: Number of samples for topographic loss
            reference_reg_weight: Weight for reference feature regularization (0.0 = disabled)
            entropy_weight: Weight for entropy regularization to encourage uniform codebook usage (0.0 = disabled)
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Min delta for early stopping
            dead_code_reset_interval: Interval for dead code reset (0 to disable, only for legacy mode)
            dead_code_threshold: Percentile threshold for dead code detection
            dead_code_max_reset_fraction: Max fraction of codebook to reset at once
            use_smart_reset: Use intelligent SmartDeadCodeReset instead of fixed-interval resets
            checkpoint_dir: Directory for checkpoints (None to disable)
            use_torch_compile: Use torch.compile() for JIT compilation
            val_every_n_epochs: Validate every N epochs
            gradient_clip_norm: Max gradient norm for clipping (None to disable)
            warmup_epochs: Number of linear warmup epochs (0 to disable)
            verbose: Whether to print progress
            config: Full raw training config for reproducibility
            group_indices: Category to feature indices mapping
            normalization_stats: Normalization statistics per category
            feature_names: Feature names list
            encoder_state_dicts: State dicts for frozen input encoders (MLPEncoder, etc.)
            feature_mask: Boolean array indicating which features survived cleaning
            feature_cleaning_params: Feature cleaning parameters for reproducibility
        """
        # Configure logging if verbose
        if verbose and not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s',
                force=True
            )

        # Configure CUDA optimizations
        if device == "cuda":
            # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
            torch.set_float32_matmul_precision("high")
            if verbose:
                logger.info("Enabled TF32 matmul for faster training")

        self.model = model.to(device)

        # Apply torch.compile() for speedup (PyTorch 2.0+)
        if use_torch_compile and device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="default")
                if verbose:
                    logger.info(
                        "Applied torch.compile() - expect 30-40% speedup after warmup"
                    )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"torch.compile() failed: {e}, continuing without compilation"
                    )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.verbose = verbose
        self.val_every_n_epochs = val_every_n_epochs

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.base_learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Learning rate scheduler (optional)
        self.scheduler = None
        if scheduler_config is not None:
            scheduler_type = scheduler_config.get("type", "cosine")
            if scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get("T_max", 1000),
                    eta_min=scheduler_config.get("eta_min", 0.0001),
                )
            elif scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get("step_size", 200),
                    gamma=scheduler_config.get("gamma", 0.5),
                )
            elif scheduler_type == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=scheduler_config.get("gamma", 0.95),
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        # Loss weights
        self.orthogonality_weight = orthogonality_weight
        self.informativeness_weight = informativeness_weight
        self.category_reconstruction_weight = category_reconstruction_weight
        self.topo_weight = topo_weight
        self.topo_samples = topo_samples
        self.reference_reg_weight = reference_reg_weight
        self.entropy_weight = entropy_weight
        self.normalize_mse = normalize_mse

        # Feature weights (for A3 feature-weighted reconstruction)
        self.feature_weights = None  # Will be set externally if needed

        # Store normalization stats and group indices for normalized metrics
        self.normalization_stats = normalization_stats
        self.group_indices = group_indices

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=verbose,
        )

        # Dead code reset: use smart reset if requested, otherwise legacy
        if use_smart_reset:
            self.dead_code_reset = SmartDeadCodeReset(
                base_threshold=dead_code_threshold,
                utilization_threshold=0.25,
                min_interval=50,
                lookback_window=10,
                verbose=verbose,
            )
            if verbose:
                logger.info("Using SmartDeadCodeReset (intelligent, condition-based)")
        else:
            self.dead_code_reset = DeadCodeReset(
                interval=dead_code_reset_interval,
                threshold=dead_code_threshold,
                max_reset_fraction=dead_code_max_reset_fraction,
                verbose=verbose,
            )
            if verbose:
                if dead_code_reset_interval == 0:
                    logger.info("DeadCodeReset: disabled")
                else:
                    logger.info(f"Using DeadCodeReset (legacy, fixed interval={dead_code_reset_interval})")
        self.checkpointer = (
            Checkpointer(
                checkpoint_dir,
                verbose=verbose,
                config=config,
                group_indices=group_indices,
                normalization_stats=normalization_stats,
                per_family_normalization_stats=per_family_normalization_stats,
                feature_names=feature_names,
                encoder_state_dicts=encoder_state_dicts,
                feature_mask=feature_mask,
                feature_cleaning_params=feature_cleaning_params,
            )
            if checkpoint_dir is not None
            else None
        )

        # Variable-length mode support
        self.temporal_encoder = temporal_encoder
        self.temporal_encoder_output_dim = temporal_encoder_output_dim
        self.encoded_initial_features = encoded_initial_features

        # Detect if variable-length mode is enabled
        self.vl_enabled = False
        if temporal_encoder is not None and hasattr(temporal_encoder, 'vl_config'):
            self.vl_enabled = temporal_encoder.vl_config.get('enabled', False)

        if temporal_encoder is not None:
            self.temporal_encoder = temporal_encoder.to(device)
            self.temporal_encoder.eval()  # Keep in eval mode (not trained separately)
            # Convert encoded_initial_features to torch tensor if provided
            if encoded_initial_features is not None:
                self.encoded_initial_features_tensor = torch.from_numpy(encoded_initial_features).float().to(device)
            else:
                self.encoded_initial_features_tensor = None

        # Feature cleaning mask
        # Tracks which features survived cleaning during category discovery
        self.feature_mask = feature_mask
        self.temporal_feature_mask_tensor = None  # For variable-length mode

        if feature_mask is not None and temporal_encoder is not None:
            # Variable-length mode: extract temporal portion of feature_mask
            # feature_mask structure: [initial features | temporal_p0 | temporal_p1 | temporal_p2 | temporal_p3]
            # We need to create a mask for the concatenated temporal encoding [320D]
            # that matches the pyramid level structure

            # Get dimensions of each pyramid level from the temporal encoder
            if hasattr(temporal_encoder, 'output_dims_per_level'):
                level_dims = temporal_encoder.output_dims_per_level  # e.g., [32, 64, 96, 128]

                # Determine where initial features end in the original feature_mask
                # The initial features come first, followed by pyramid levels
                num_initial = len(encoded_initial_features[0]) if encoded_initial_features is not None else 0

                # Extract temporal portion of the mask (everything after initial features)
                temporal_mask_flat = feature_mask[num_initial:]

                # The temporal_mask_flat corresponds to split pyramid levels in order:
                # [p0_features... | p1_features... | p2_features... | p3_features...]
                # We need to map this to the concatenated encoding structure
                temporal_feature_mask = temporal_mask_flat

                self.temporal_feature_mask_tensor = torch.from_numpy(temporal_feature_mask).bool().to(device)
                if verbose:
                    logger.info(f"Temporal feature mask loaded: {temporal_feature_mask.sum()}/{len(temporal_feature_mask)} temporal features kept")

            self.feature_mask_tensor = None  # Don't use the full mask in variable-length mode

        elif feature_mask is not None and temporal_encoder is None:
            # Non-variable-length mode: use feature_mask as-is
            mask_tensor = torch.from_numpy(feature_mask).bool()
            # Ensure mask is 1D (flatten if needed)
            if mask_tensor.dim() > 1:
                mask_tensor = mask_tensor.flatten()
            self.feature_mask_tensor = mask_tensor.to(device)
            if verbose:
                logger.info(f"Feature mask loaded: {mask_tensor.sum()}/{len(mask_tensor)} features kept")
        else:
            self.feature_mask_tensor = None

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

    def train_epoch(self):
        """Train for one epoch.

        Returns:
            Tuple of (average training loss, loss components dict, last batch features, last batch raw_ics)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        last_batch = None
        last_raw_ics = None

        # Track loss components
        loss_components = {
            "reconstruction": 0.0,
            "reconstruction_global": 0.0,  # Global reconstruction (primary)
            "reconstruction_category": 0.0,  # Per-category reconstruction (regularizer)
            "reconstruction_raw": 0.0,  # Raw (unnormalized) MSE for logging
            "vq": 0.0,
            "orthogonality": 0.0,
            "informativeness": 0.0,
            "informativeness_raw": 0.0,  # Raw (unnormalized) MSE for logging
            "topographic": 0.0,
            "reference_regularization": 0.0,
            "reference_regularization_raw": 0.0,  # Raw (unnormalized) MSE for logging
            "entropy": 0.0,
        }

        for batch in self.train_loader:
            features = batch["features"].to(self.device)

            # VARIABLE-LENGTH MODE: Encode temporal features at runtime with sampled lengths
            # If we have a temporal encoder, features are raw [B, T, D] temporal data
            # Need to encode with mask, then concatenate with initial features
            if self.temporal_encoder is not None:
                # Extract mask and length from batch
                mask = batch.get("mask")
                length = batch.get("length")

                # Encode temporal features with mask
                with torch.no_grad():  # Encoder not trained separately
                    if mask is not None:
                        mask = mask.to(self.device)
                        length = length.to(self.device)
                        # Encode with variable lengths
                        encoded_temporal, mask_info = self.temporal_encoder(
                            features,  # [B, T, D] raw temporal
                            mask=mask,
                            lengths=length
                        )
                    else:
                        # No mask - encode normally
                        encoded_temporal = self.temporal_encoder(features)

                # Apply temporal feature cleaning mask if present
                if self.temporal_feature_mask_tensor is not None:
                    encoded_temporal = encoded_temporal[:, self.temporal_feature_mask_tensor]

                # Concatenate with encoded initial features if present
                # Get from batch (correctly shuffled with data)
                initial_features = batch.get("encoded_initial_features")
                if initial_features is not None:
                    initial_features = initial_features.to(self.device)
                    features = torch.cat([initial_features, encoded_temporal], dim=1)
                else:
                    features = encoded_temporal

                # Feature cleaning masks have been applied above
                # In variable-length mode:
                #   - Category discovery uses pyramid-level-split features [initial + p0 + p1 + p2 + p3]
                #   - Training uses concatenated features [initial + temporal_full]
                # These have different dimensions, so the feature_mask is incompatible
                # Feature cleaning was already applied during category discovery
                # (features are pre-cleaned before being stored as initial_features_only)

            # DEBUG: Print final features shape before model
            if self.current_epoch == 0 and n_batches == 0:
                print(f"DEBUG: Passing to model: features.shape = {features.shape}")

            # Forward pass WITHOUT raw_ics (features are pre-encoded during category discovery)
            # Model expects pre-encoded dimension (encoded_dim initial + temporal)
            outputs = self.model(features)

            # Compute loss
            # For hybrid INITIAL models, use the expanded input_features as target
            # (includes CNN embeddings that must be reconstructed)
            if "input_features" in outputs:
                targets = {"features": outputs["input_features"]}
            else:
                targets = {"features": features}

            # Add raw_summary to targets for physics consistency measurement
            if "raw_summary" in batch:
                targets["raw_summary"] = batch["raw_summary"].to(self.device)

            # Extract mask info for variable-length support
            from ..variable_length_utils import extract_mask_info_from_batch
            mask_info = extract_mask_info_from_batch(batch, self.device)

            losses = compute_total_loss(
                outputs,
                targets,
                self.model,
                orthogonality_weight=self.orthogonality_weight,
                informativeness_weight=self.informativeness_weight,
                category_reconstruction_weight=self.category_reconstruction_weight,
                topo_weight=self.topo_weight,
                topo_samples=self.topo_samples,
                reference_reg_weight=self.reference_reg_weight,
                reference_features=batch.get("reference_features").to(self.device) if batch.get("reference_features") is not None else None,
                is_interpolated=batch.get("is_interpolated").to(self.device) if batch.get("is_interpolated") is not None else None,
                normalize_mse=self.normalize_mse,
                feature_weights=self.feature_weights,
                entropy_weight=self.entropy_weight,
                mask_info=mask_info,
                normalization_stats=self.normalization_stats,
                group_indices=self.group_indices,
            )

            loss = losses["total"]

            # Track individual loss components
            for key in loss_components.keys():
                if key in losses:
                    loss_components[key] += losses[key].item()

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            loss.backward()

            # Gradient clipping (if enabled)
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Save last batch for dead code reset
            last_batch = features

        avg_loss = total_loss / n_batches

        # Average loss components
        avg_loss_components = {k: v / n_batches for k, v in loss_components.items()}

        # raw_ics not needed since features are pre-encoded
        return avg_loss, avg_loss_components, last_batch, None

    def validate(self):
        """Validate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)

                # VARIABLE-LENGTH MODE: Encode temporal features at runtime with sampled lengths
                if self.temporal_encoder is not None:
                    mask = batch.get("mask")
                    length = batch.get("length")

                    if mask is not None:
                        mask = mask.to(self.device)
                        length = length.to(self.device)
                        encoded_temporal, mask_info = self.temporal_encoder(
                            features,
                            mask=mask,
                            lengths=length
                        )
                    else:
                        encoded_temporal = self.temporal_encoder(features)

                    # Apply temporal feature cleaning mask if present
                    if self.temporal_feature_mask_tensor is not None:
                        encoded_temporal = encoded_temporal[:, self.temporal_feature_mask_tensor]

                    # Concatenate with encoded initial features if present
                    # Get from batch (correctly shuffled with data)
                    initial_features = batch.get("encoded_initial_features")
                    if initial_features is not None:
                        initial_features = initial_features.to(self.device)
                        features = torch.cat([initial_features, encoded_temporal], dim=1)
                    else:
                        features = encoded_temporal

                    # Apply feature cleaning mask to concatenated features (non-variable-length mode)
                    if self.feature_mask_tensor is not None:
                        features = features[:, self.feature_mask_tensor]

                # Handle raw_ics for hybrid INITIAL encoder
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(self.device)

                # Forward pass (pass raw_ics if model supports hybrid INITIAL)
                if raw_ics is not None and hasattr(self.model, 'initial_encoder'):
                    outputs = self.model(features, raw_ics=raw_ics)
                elif raw_ics is not None and hasattr(self.model, '_orig_mod') and hasattr(self.model._orig_mod, 'initial_encoder'):
                    # Handle torch.compile wrapped model
                    outputs = self.model(features, raw_ics=raw_ics)
                else:
                    outputs = self.model(features)

                # Compute loss
                # For hybrid INITIAL models, use the expanded input_features as target
                if "input_features" in outputs:
                    targets = {"features": outputs["input_features"]}
                else:
                    targets = {"features": features}

                # Extract mask info for variable-length support
                from ..variable_length_utils import extract_mask_info_from_batch
                mask_info = extract_mask_info_from_batch(batch, self.device)

                losses = compute_total_loss(
                    outputs,
                    targets,
                    self.model,
                    orthogonality_weight=self.orthogonality_weight,
                    informativeness_weight=self.informativeness_weight,
                    category_reconstruction_weight=self.category_reconstruction_weight,
                    topo_weight=self.topo_weight,
                    topo_samples=self.topo_samples,
                    reference_reg_weight=self.reference_reg_weight,
                    reference_features=batch.get("reference_features"),
                    is_interpolated=batch.get("is_interpolated"),
                    feature_weights=self.feature_weights,
                    entropy_weight=self.entropy_weight,
                    mask_info=mask_info,
                    normalize_mse=self.normalize_mse,
                    normalization_stats=self.normalization_stats,
                    group_indices=self.group_indices,
                )

                loss = losses["total"]
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute validation metrics.

        Returns:
            Dict with utilization, reconstruction error (raw and normalized), topographic similarity, and detailed per-category metrics
        """
        from .metrics import (
            compute_per_category_metrics,
            compute_reconstruction_error,
            compute_normalized_reconstruction_error_per_category,
        )
        from .losses import topographic_similarity_loss

        # Unwrap compiled model if using torch.compile
        model_for_metrics = self.model
        if hasattr(self.model, '_orig_mod'):
            model_for_metrics = self.model._orig_mod

        # Compute reconstruction error (raw MSE)
        reconstruction_error = compute_reconstruction_error(
            model_for_metrics,
            self.val_loader,
            device=self.device,
            temporal_encoder=self.temporal_encoder,
            temporal_feature_mask=self.temporal_feature_mask_tensor,
        )

        # Compute normalized reconstruction error per category
        normalized_errors = {}
        if self.normalization_stats is not None and self.group_indices is not None:
            normalized_errors = compute_normalized_reconstruction_error_per_category(
                model_for_metrics,
                self.val_loader,
                self.normalization_stats,
                self.group_indices,
                device=self.device,
                temporal_encoder=self.temporal_encoder,
                temporal_feature_mask=self.temporal_feature_mask_tensor,
            )

        # Compute overall normalized error (weighted by feature count)
        # NOTE: This is per-category average, not global reconstruction quality
        overall_normalized_error = None
        if normalized_errors and self.group_indices:
            weighted_sum = 0.0
            total_features = 0
            for cat_name, indices in self.group_indices.items():
                norm_key = f"{cat_name}/reconstruction_error_normalized"
                if norm_key in normalized_errors:
                    weighted_sum += normalized_errors[norm_key] * len(indices)
                    total_features += len(indices)

            if total_features > 0:
                overall_normalized_error = weighted_sum / total_features

        # Compute GLOBAL reconstruction quality (full 486D reconstruction)
        # This directly measures: MSE(full_recon, full_target) / variance(full_target)
        # Unlike per-category average, this shows true reconstruction fidelity
        global_normalized_error = None
        if reconstruction_error > 0 and self.normalization_stats:
            # Compute global variance from all features
            all_variances = []
            for cat_name, stats in self.normalization_stats.items():
                # stats.std is per-feature std for this category
                cat_variances = stats.std ** 2
                all_variances.append(cat_variances)

            if all_variances:
                # Concatenate all feature variances
                global_var_array = np.concatenate(all_variances)
                global_variance = global_var_array.mean() + 1e-8
                global_normalized_error = reconstruction_error / global_variance

        # Compute detailed metrics on validation set
        detailed_metrics = compute_per_category_metrics(
            model_for_metrics,
            self.val_loader,
            device=self.device,
            max_batches=None,  # Use full val set
            temporal_encoder=self.temporal_encoder,
            temporal_feature_mask=self.temporal_feature_mask_tensor,
        )

        # Extract average utilization across all category-levels
        utilization_metrics = [
            v for k, v in detailed_metrics.items()
            if "utilization" in k and "level" in k
        ]

        if utilization_metrics:
            avg_utilization = sum(utilization_metrics) / len(utilization_metrics)
        else:
            avg_utilization = 0.0

        # Compute topographic similarity (PRE and POST quantization)
        topo_pre_sum = 0.0
        topo_post_sum = 0.0
        n_batches = 0

        model_for_metrics.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(self.device)

                # VARIABLE-LENGTH MODE: Encode temporal features at runtime
                if self.temporal_encoder is not None:
                    # Extract mask and length from batch
                    mask = batch.get("mask")
                    length = batch.get("length")

                    # Encode temporal features with mask
                    if mask is not None:
                        mask = mask.to(self.device)
                        length = length.to(self.device)
                        # Encode with variable lengths
                        encoded_temporal, mask_info = self.temporal_encoder(
                            features,  # [B, T, D] raw temporal
                            mask=mask,
                            lengths=length
                        )
                    else:
                        # No mask - encode normally
                        encoded_temporal = self.temporal_encoder(features)

                    # Apply temporal feature cleaning mask if present
                    if self.temporal_feature_mask_tensor is not None:
                        encoded_temporal = encoded_temporal[:, self.temporal_feature_mask_tensor]

                    # Concatenate with encoded initial features if present
                    initial_features = batch.get("encoded_initial_features")
                    if initial_features is not None:
                        initial_features = initial_features.to(self.device)
                        features = torch.cat([initial_features, encoded_temporal], dim=1)
                    else:
                        features = encoded_temporal

                # Forward pass
                if raw_ics is not None and hasattr(model_for_metrics, 'initial_encoder'):
                    outputs = model_for_metrics(features, raw_ics=raw_ics)
                else:
                    outputs = model_for_metrics(features)

                # Compute topographic similarity
                if "input_features" in outputs:
                    targets = {"features": outputs["input_features"]}
                else:
                    targets = {"features": features}

                _, topo_metrics = topographic_similarity_loss(
                    outputs, targets, n_samples=min(256, features.size(0))
                )
                topo_pre_sum += topo_metrics["topo_pre"]
                topo_post_sum += topo_metrics["topo_post"]
                n_batches += 1

                # Only compute on first 10 batches for efficiency
                if n_batches >= 10:
                    break

        if n_batches > 0:
            avg_topo_pre = topo_pre_sum / n_batches
            avg_topo_post = topo_post_sum / n_batches
        else:
            avg_topo_pre = 0.0
            avg_topo_post = 0.0

        # Compute physics consistency metrics (MNO vs CNO SUMMARY features)
        physics_metrics = self.compute_physics_consistency()

        # Return both aggregate and detailed metrics
        result = {
            "utilization": avg_utilization,
            "reconstruction_error": reconstruction_error,  # Raw MSE
            "topo_pre": avg_topo_pre,  # Pre-quantization topographic similarity
            "topo_post": avg_topo_post,  # Post-quantization topographic similarity
        }

        # Add overall normalized error if computed (per-category average)
        if overall_normalized_error is not None:
            result["reconstruction_error_normalized"] = overall_normalized_error

        # Add global normalized error if computed (true global reconstruction quality)
        if global_normalized_error is not None:
            result["global_reconstruction_error_normalized"] = global_normalized_error

        result.update(detailed_metrics)  # Include all detailed metrics
        result.update(normalized_errors)  # Per-category normalized errors
        result.update(physics_metrics)  # Include physics consistency metrics

        return result

    def compute_physics_consistency(self) -> Dict[str, float]:
        """Compute physics consistency: MSE between MNO and CNO SUMMARY features.

        Returns:
            Dict with physics_mse_all, physics_mse_interpolated, physics_mse_exact
        """
        import torch.nn.functional as F

        mse_all = []
        mse_interpolated = []
        mse_exact = []

        n_batches_checked = 0
        n_batches_skipped = 0

        with torch.no_grad():
            for batch in self.val_loader:
                raw_summary = batch.get("raw_summary")
                reference_features = batch.get("reference_features")
                is_interpolated = batch.get("is_interpolated")

                if raw_summary is None or reference_features is None:
                    # No reference features available, skip physics consistency
                    continue

                raw_summary = raw_summary.to(self.device)
                reference_features = reference_features.to(self.device)

                # Verify dimensions match (reference should be pre-cleaned)
                if reference_features.shape[1] != raw_summary.shape[1]:
                    # Skip this batch rather than crash - log warning
                    print(f"Warning: Dimension mismatch in physics consistency check: "
                          f"raw_summary={raw_summary.shape[1]}D, "
                          f"reference={reference_features.shape[1]}D. Skipping batch.")
                    continue

                # Compute MSE for all samples
                batch_mse = F.mse_loss(raw_summary, reference_features, reduction='none').mean(dim=1)
                mse_all.extend(batch_mse.cpu().tolist())

                n_batches_checked += 1

                # Break down by interpolated vs exact if mask available
                if is_interpolated is not None:
                    is_interpolated = is_interpolated.to(self.device)
                    mse_interpolated.extend(batch_mse[is_interpolated].cpu().tolist())
                    mse_exact.extend(batch_mse[~is_interpolated].cpu().tolist())

        # Average MSE values
        result = {}
        if mse_all:
            result["physics_mse_all"] = sum(mse_all) / len(mse_all) if mse_all else 0.0
        else:
            result["physics_mse_all"] = 0.0

        if mse_interpolated:
            result["physics_mse_interpolated"] = sum(mse_interpolated) / len(mse_interpolated) if mse_interpolated else 0.0
        else:
            result["physics_mse_interpolated"] = 0.0

        if mse_exact:
            result["physics_mse_exact"] = sum(mse_exact) / len(mse_exact) if mse_exact else 0.0
        else:
            result["physics_mse_exact"] = 0.0

        return result

    def _compute_variable_length_metrics(self) -> Dict[str, Any]:
        """Compute variable-length specific metrics during validation.

        Tracks:
        - Length distribution (how often each bin sampled)
        - Per-length reconstruction quality
        - Active pyramid levels by length
        - Masking efficiency

        Returns:
            Dict with VL metrics
        """
        from collections import defaultdict
        import torch.nn.functional as F

        length_counts = defaultdict(int)
        quality_by_length = defaultdict(list)
        active_levels_by_length = defaultdict(list)
        valid_fractions = []

        # Unwrap compiled model if using torch.compile
        model_for_metrics = self.model
        if hasattr(self.model, '_orig_mod'):
            model_for_metrics = self.model._orig_mod

        model_for_metrics.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                # Check if batch has length information
                if "length" not in batch:
                    continue

                lengths = batch["length"].cpu().numpy()
                features = batch["features"].to(self.device)
                mask = batch.get("mask")

                # Track length distribution
                for length in lengths:
                    length_counts[int(length)] += 1

                # Encode temporal features with mask
                if self.temporal_encoder is not None:
                    if mask is not None:
                        mask = mask.to(self.device)
                        length_tensor = batch["length"].to(self.device)
                        # Encode with variable lengths
                        encoded_temporal, mask_info = self.temporal_encoder(
                            features,  # [B, T, D] raw temporal
                            mask=mask,
                            lengths=length_tensor
                        )

                        # Track active pyramid levels if available
                        if "num_active_levels" in mask_info:
                            num_active = mask_info["num_active_levels"]
                            # num_active_levels can be either a tensor/array (per-sample) or a single int (same for all samples)
                            if isinstance(num_active, (int, float)):
                                # Same number of active levels for all samples in batch
                                for length in lengths:
                                    active_levels_by_length[int(length)].append(float(num_active))
                            else:
                                # Per-sample active levels
                                for length, n_active in zip(lengths, num_active):
                                    active_levels_by_length[int(length)].append(n_active.item() if hasattr(n_active, 'item') else float(n_active))

                        # Track masking efficiency
                        valid_frac = mask.float().mean(dim=1)
                        valid_fractions.extend(valid_frac.cpu().numpy())
                    else:
                        # No mask - encode normally
                        encoded_temporal = self.temporal_encoder(features)

                    # Apply temporal feature cleaning mask if present
                    if self.temporal_feature_mask_tensor is not None:
                        encoded_temporal = encoded_temporal[:, self.temporal_feature_mask_tensor]

                    # Concatenate with encoded initial features if present
                    initial_features = batch.get("encoded_initial_features")
                    if initial_features is not None:
                        initial_features = initial_features.to(self.device)
                        features_encoded = torch.cat([initial_features, encoded_temporal], dim=1)
                    else:
                        features_encoded = encoded_temporal
                else:
                    features_encoded = features

                # Forward pass through VQ-VAE
                outputs = model_for_metrics(features_encoded)

                # Compute reconstruction quality per length
                recon = outputs["reconstruction"]["features"]
                mse = F.mse_loss(features_encoded, recon, reduction='none').mean(dim=1)

                for length, mse_val in zip(lengths, mse):
                    # Convert MSE to quality score (1 - normalized_mse)
                    quality_by_length[int(length)].append(1.0 - mse_val.item())

        # Average metrics
        vl_metrics = {}

        # Length distribution
        if length_counts:
            vl_metrics["vl_length_distribution"] = dict(length_counts)

        # Per-length quality
        if quality_by_length:
            vl_metrics["vl_per_length_quality"] = {
                k: float(np.mean(v)) for k, v in quality_by_length.items()
            }

        # Active levels by length
        if active_levels_by_length:
            vl_metrics["vl_active_levels_by_length"] = {
                k: float(np.mean(v)) for k, v in active_levels_by_length.items()
            }

        # Masking efficiency
        if valid_fractions:
            vl_metrics["vl_masking_efficiency"] = float(np.mean(valid_fractions))

        return vl_metrics

    def train(self, epochs: int, start_epoch: int = 1):
        """Train for specified number of epochs.

        Args:
            epochs: Total number of epochs to train to (not number of additional epochs)
            start_epoch: Starting epoch number (default: 1, for resume: loaded_epoch + 1)

        Returns:
            Training history dict
        """
        if self.verbose:
            logger.info("=" * 70)
            logger.info("VQ-VAE TRAINING")
            logger.info("=" * 70)
            if start_epoch > 1:
                logger.info(f"Resuming from epoch {start_epoch - 1}")
                logger.info(f"Training epochs {start_epoch} to {epochs} ({epochs - start_epoch + 1} more epochs)")
            else:
                logger.info(f"Epochs: {epochs}")
            logger.info(f"Training samples: {len(self.train_loader.dataset)}")
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Validation frequency: every {self.val_every_n_epochs} epochs")
            logger.info(f"\nLoss weights:")
            logger.info(f"  Orthogonality: {self.orthogonality_weight}")
            logger.info(f"  Informativeness: {self.informativeness_weight}")
            logger.info(f"  Topographic: {self.topo_weight}")
            logger.info(f"  Reference regularization: {self.reference_reg_weight}")
            logger.info(f"  Entropy regularization: {self.entropy_weight} {'(ACTIVE)' if self.entropy_weight > 0 else '(disabled)'}")

        start_time = time.time()
        last_val_loss = None

        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()
            self.current_epoch = epoch

            # Apply warmup scheduler (if enabled)
            if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                warmup_factor = epoch / self.warmup_epochs
                current_lr = self.base_learning_rate * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            # Train
            train_loss, loss_components, last_batch, last_raw_ics = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Store loss components in history
            if "loss_components" not in self.history:
                self.history["loss_components"] = []
            self.history["loss_components"].append(loss_components)

            # Validate (only every N epochs or last epoch)
            should_validate = (epoch % self.val_every_n_epochs == 0) or (epoch == epochs)
            if should_validate:
                val_loss = self.validate()
                last_val_loss = val_loss
                self.history["val_loss"].append(val_loss)
            else:
                # Skip validation but append last value for history continuity
                if last_val_loss is not None:
                    self.history["val_loss"].append(last_val_loss)
                else:
                    # First epoch - must validate
                    val_loss = self.validate()
                    last_val_loss = val_loss
                    self.history["val_loss"].append(val_loss)

            # Compute metrics
            metrics = self.compute_metrics()
            self.history["metrics"].append(metrics)

            epoch_time = time.time() - epoch_start

            # Logging
            if self.verbose:
                # Main summary line
                msg = f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s): "
                msg += f"train={train_loss:.6f}"
                if should_validate:
                    msg += f", val={val_loss:.6f}"
                else:
                    msg += f", val={last_val_loss:.6f} (cached)"

                util = metrics.get("utilization", 0.0)
                msg += f", util={util:.1%}"

                # Show current learning rate if scheduler is active
                if self.scheduler is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    msg += f", lr={current_lr:.6f}"

                logger.info(msg)

                # Training loss components (normalized - what the optimizer sees)
                components_msg = "  Train: "
                if self.normalize_mse:
                    # Show raw, global, and category reconstruction
                    components_msg += f"recon={loss_components.get('reconstruction_raw', 0.0):.3f}"
                    components_msg += f" (global={loss_components.get('reconstruction_global', 0.0):.3f}"

                    # Only show category if weight is non-zero
                    if self.category_reconstruction_weight > 0:
                        components_msg += f", cat={loss_components.get('reconstruction_category', 0.0):.3f}), "
                    else:
                        components_msg += "), "
                else:
                    components_msg += f"recon={loss_components.get('reconstruction', 0.0):.3f}, "

                components_msg += f"vq={loss_components.get('vq', 0.0):.3f}, "
                components_msg += f"ortho={loss_components.get('orthogonality', 0.0):.3f}, "

                if self.normalize_mse:
                    components_msg += f"info={loss_components.get('informativeness_raw', 0.0):.3f}"
                    components_msg += f" (norm={loss_components.get('informativeness', 0.0):.3f}), "
                else:
                    components_msg += f"info={loss_components.get('informativeness', 0.0):.3f}, "

                components_msg += f"topo={loss_components.get('topographic', 0.0):.3f}"

                entropy_loss = loss_components.get("entropy", 0.0)
                if entropy_loss != 0.0:
                    components_msg += f", entropy={entropy_loss:.3f}"

                ref_reg = loss_components.get("reference_regularization", 0.0)
                if ref_reg > 0:
                    if self.normalize_mse:
                        components_msg += f", ref={loss_components.get('reference_regularization_raw', 0.0):.3f}"
                        components_msg += f" (norm={ref_reg:.3f})"
                    else:
                        components_msg += f", ref={ref_reg:.3f}"

                logger.info(components_msg)

                # Validation metrics
                val_msg = "  Val: "
                recon_error = metrics.get("reconstruction_error", 0.0)

                if "reconstruction_error_normalized" in metrics:
                    recon_norm = metrics["reconstruction_error_normalized"]

                    # Show global first (consistent with training logs)
                    if "global_reconstruction_error_normalized" in metrics:
                        global_norm = metrics["global_reconstruction_error_normalized"]
                        val_msg += f"recon={recon_error:.3f} (global={global_norm:.3f}, per-cat={recon_norm:.3f})"
                    else:
                        val_msg += f"recon={recon_error:.3f} (per-cat={recon_norm:.3f})"
                else:
                    val_msg += f"recon={recon_error:.3f}"

                topo_pre = metrics.get("topo_pre", 0.0)
                topo_post = metrics.get("topo_post", 0.0)
                val_msg += f", topo={topo_pre:.3f}→{topo_post:.3f}"

                logger.info(val_msg)

                # Add physics consistency metrics if available (separate line for readability)
                # Only log if actually active (any value > 0)
                physics_mse_all = metrics.get("physics_mse_all", 0.0)
                physics_mse_interp = metrics.get("physics_mse_interpolated", 0.0)
                physics_mse_exact = metrics.get("physics_mse_exact", 0.0)
                if physics_mse_all > 0 or physics_mse_interp > 0 or physics_mse_exact > 0:
                    physics_msg = "  Physics consistency: "
                    if physics_mse_all > 0:
                        physics_msg += f"all={physics_mse_all:.6f}"
                    if physics_mse_interp > 0:
                        physics_msg += f", interp={physics_mse_interp:.6f}"
                    if physics_mse_exact > 0:
                        physics_msg += f", exact={physics_mse_exact:.6f}"
                    logger.info(physics_msg)

            # Callbacks
            # 1. Dead code reset
            if last_batch is not None:
                # Check if using SmartDeadCodeReset (needs additional params)
                if isinstance(self.dead_code_reset, SmartDeadCodeReset):
                    current_util = metrics.get("utilization", 0.0)

                    # Extract per-category utilization for smarter resets
                    per_category_utils = {}
                    for key, val in metrics.items():
                        if "/utilization" in key and "/level_" in key:
                            # Extract category name from "cluster_1/level_0/utilization"
                            category = key.split("/")[0]
                            if category not in per_category_utils:
                                per_category_utils[category] = []
                            per_category_utils[category].append(val)

                    # Average utilization across levels for each category
                    per_category_utils = {
                        cat: sum(utils) / len(utils)
                        for cat, utils in per_category_utils.items()
                    }

                    # Extract feature counts per category from model config
                    per_category_feature_counts = {}
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'group_indices'):
                        # Unwrap compiled model if needed
                        model_for_config = self.model
                        if hasattr(self.model, '_orig_mod'):
                            model_for_config = self.model._orig_mod

                        if hasattr(model_for_config.config, 'group_indices'):
                            for category, feature_indices in model_for_config.config.group_indices.items():
                                per_category_feature_counts[category] = len(feature_indices)

                    self.dead_code_reset(
                        self.model,
                        last_batch,
                        epoch,
                        current_util,
                        val_loss,
                        self.early_stopping.counter,
                        per_category_utils,
                        per_category_feature_counts,
                        raw_ics=last_raw_ics,
                    )
                else:
                    # Legacy DeadCodeReset (fixed interval)
                    self.dead_code_reset(self.model, last_batch, epoch, raw_ics=last_raw_ics)

            # 2. Checkpointing (only when we validated)
            if should_validate and self.checkpointer is not None:
                self.checkpointer(self.model, self.optimizer, val_loss, epoch, metrics, self.history)

            # 3. Early stopping (only when we validated)
            if should_validate and self.early_stopping(val_loss, epoch):
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # 4. Learning rate scheduler step (after warmup completes)
            if self.scheduler is not None and epoch > self.warmup_epochs:
                self.scheduler.step()

        # Final metrics
        final_metrics = self.compute_metrics()
        self.history["final_metrics"] = final_metrics

        elapsed = time.time() - start_time

        if self.verbose:
            logger.info("=" * 70)
            logger.info("TRAINING COMPLETE")
            logger.info(f"Total time: {elapsed:.1f}s")
            logger.info("Final metrics:")
            for key, val in final_metrics.items():
                if isinstance(val, float):
                    logger.info(f"  {key}: {val:.4f}")
            logger.info("=" * 70)

        return self.history
