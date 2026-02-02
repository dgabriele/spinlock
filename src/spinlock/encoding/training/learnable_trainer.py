"""Learnable VQ-VAE trainer with end-to-end assignment learning."""

import logging
import torch
from typing import Optional, Dict, Any

from .trainer import VQVAETrainer
from .annealing import TemperatureScheduler
from .losses import compute_total_loss
from ..categorical_vqvae import CategoricalHierarchicalVQVAE
from ..variable_length_utils import extract_mask_info_from_batch

logger = logging.getLogger(__name__)


class LearnableVQVAETrainer(VQVAETrainer):
    """Extends VQVAETrainer with learnable category assignment optimization."""

    def __init__(
        self,
        model: CategoricalHierarchicalVQVAE,
        temp_scheduler: TemperatureScheduler,
        assignment_lr: float = 0.001,
        assignment_orthogonality_weight: float = 0.1,
        assignment_balance_weight: float = 0.05,
        gradient_clip_norm: Optional[float] = 1.0,
        **kwargs
    ):
        """Initialize learnable trainer.

        Args:
            model: CategoricalHierarchicalVQVAE model with assignment_matrix
            temp_scheduler: Temperature scheduler for Gumbel-Softmax
            assignment_lr: Learning rate for assignment matrix
            assignment_orthogonality_weight: Weight for assignment orthogonality loss
            assignment_balance_weight: Weight for assignment balance loss
            gradient_clip_norm: Gradient clipping norm (default 1.0 for stability)
            **kwargs: Additional arguments for base VQVAETrainer (including orthogonality_weight for VQ)
        """
        # Validate model has assignment_matrix
        if model.assignment_matrix is None:
            raise ValueError("Model must have assignment_matrix for learnable training")

        # Force gradient clipping for stability
        if gradient_clip_norm is None:
            gradient_clip_norm = 1.0
        kwargs["gradient_clip_norm"] = gradient_clip_norm

        super().__init__(model, **kwargs)

        self.temp_scheduler = temp_scheduler
        self.assignment_lr = assignment_lr
        self.assign_orthogonality_weight = assignment_orthogonality_weight
        self.assign_balance_weight = assignment_balance_weight

        # Create separate optimizer for assignment matrix
        self.assignment_optimizer = torch.optim.Adam(
            model.assignment_matrix.parameters(),
            lr=assignment_lr
        )

        if self.verbose:
            logger.info(f"Learnable assignment training enabled:")
            logger.info(f"  Assignment LR: {assignment_lr}")
            logger.info(f"  Temperature schedule: {temp_scheduler.schedule}")
            logger.info(f"  Assignment orthogonality weight: {assignment_orthogonality_weight}")
            logger.info(f"  Assignment balance weight: {assignment_balance_weight}")
            logger.info(f"  Gradient clip norm: {gradient_clip_norm}")

    def train_epoch(self):
        """Override train_epoch to add temperature and assignment losses."""
        self.model.train()

        total_loss = 0.0
        total_components = {}
        last_batch = None
        last_raw_ics = None

        # Get temperature for this epoch
        temperature = self.temp_scheduler(self.current_epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            # Extract batch data
            features = batch["features"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)
            lengths = batch.get("length")
            if lengths is not None:
                lengths = lengths.to(self.device)
            encoded_initial = batch.get("encoded_initial_features")
            if encoded_initial is not None:
                encoded_initial = encoded_initial.to(self.device)

            last_raw_ics = batch.get("raw_ics")
            if last_raw_ics is not None:
                last_raw_ics = last_raw_ics.to(self.device)

            # Get optional data
            reference_features = batch.get("reference_features")
            if reference_features is not None:
                reference_features = reference_features.to(self.device)

            # Forward - wrapper handles encoding if needed
            from .compilation_wrapper import DynamicEncodingWrapper
            if isinstance(self.model, DynamicEncodingWrapper):
                outputs = self.model(features, mask, lengths, encoded_initial, temperature=temperature)
                # Get encoded features for assignment losses (with mask for training)
                encoded_features = self.model.encode_features(features, mask, lengths, encoded_initial, apply_mask=True)
                # For dead code reset, learnable models need UNMASKED features
                # (assignment matrix operates on full feature space)
                last_batch = self.model.encode_features(features, mask, lengths, encoded_initial, apply_mask=False)
            else:
                # Legacy path
                encoded_features = self._encode_variable_length_features(batch, self.device)
                outputs = self.model(encoded_features, temperature=temperature)
                last_batch = encoded_features

            # Ensure last_batch is on the correct device (fix device mismatch in dead code reset)
            if last_batch is not None and hasattr(last_batch, 'to'):
                last_batch = last_batch.to(self.device)

            # Prepare targets dict (required by compute_total_loss)
            # For hybrid INITIAL models, use the expanded input_features as target
            # (includes CNN embeddings that must be reconstructed)
            if "input_features" in outputs:
                targets = {"features": outputs["input_features"]}
            else:
                targets = {"features": features}

            if reference_features is not None:
                targets["reference_features"] = reference_features

            # Add raw_summary to targets for physics consistency measurement
            if "raw_summary" in batch:
                targets["raw_summary"] = batch["raw_summary"].to(self.device)

            # Extract mask info for variable-length support
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
                reference_features=reference_features,
                feature_weights=self.feature_weights,
                entropy_weight=self.entropy_weight,
                mask_info=mask_info,
                normalize_mse=self.normalize_mse,
                normalization_stats=self.normalization_stats,
                group_indices=self.group_indices,
            )

            # Compute assignment losses (use encoded features)
            # Get the assignment matrix (handle wrapper)
            from .compilation_wrapper import DynamicEncodingWrapper
            if isinstance(self.model, DynamicEncodingWrapper):
                assignment_matrix = self.model._original_model.assignment_matrix
            else:
                assignment_matrix = self.model.assignment_matrix

            soft_assign = assignment_matrix(temperature)
            assign_losses = self.compute_assignment_losses(encoded_features, soft_assign)

            # Total loss
            loss = losses["total"] + assign_losses["total"]

            # Backward and optimize both model and assignments
            self.optimizer.zero_grad()
            self.assignment_optimizer.zero_grad()

            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

            self.optimizer.step()
            self.assignment_optimizer.step()

            # Accumulate losses
            total_loss += loss.item()

            # Merge loss components
            for key, val in losses.items():
                if key != "total":
                    if key not in total_components:
                        total_components[key] = 0.0
                    if hasattr(val, 'item'):
                        total_components[key] += val.item()
                    else:
                        total_components[key] += val

            # Add assignment losses to components
            for key, val in assign_losses.items():
                loss_key = f"assign_{key}"
                if loss_key not in total_components:
                    total_components[loss_key] = 0.0
                if hasattr(val, 'item'):
                    total_components[loss_key] += val.item()
                else:
                    total_components[loss_key] += val

        # Average over batches
        avg_loss = total_loss / len(self.train_loader)
        for key in total_components:
            total_components[key] /= len(self.train_loader)

        # Add temperature to components for logging
        total_components["temperature"] = temperature

        return avg_loss, total_components, last_batch, last_raw_ics

    def compute_assignment_losses(self, features, soft_assign):
        """Compute soft assignment losses.

        Args:
            features: Input features [B, D]
            soft_assign: Soft assignment matrix [D, C] or dict of [D_family, C_family]

        Returns:
            Dictionary of assignment losses
        """
        from .assignment_losses import (
            soft_orthogonality_loss,
            soft_balance_loss
        )

        ortho = soft_orthogonality_loss(features, soft_assign)
        balance = soft_balance_loss(soft_assign)

        return {
            "orthogonality": ortho * self.assign_orthogonality_weight,
            "balance": balance * self.assign_balance_weight,
            "total": ortho * self.assign_orthogonality_weight + balance * self.assign_balance_weight
        }
