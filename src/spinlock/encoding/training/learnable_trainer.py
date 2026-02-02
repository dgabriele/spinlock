"""Learnable VQ-VAE trainer with end-to-end assignment learning."""

import logging
import torch
from typing import Optional

from .trainer import VQVAETrainer
from .annealing import TemperatureScheduler
from ..categorical_vqvae import CategoricalHierarchicalVQVAE

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
            features = batch["features"].to(self.device)
            last_batch = features
            last_raw_ics = batch.get("raw_ics")

            # Get optional data
            reference_features = batch.get("reference_features")
            if reference_features is not None:
                reference_features = reference_features.to(self.device)

            # Forward pass with temperature
            outputs = self.model(features, temperature=temperature)

            # Compute standard VQ-VAE losses
            from .losses import compute_total_loss

            # Prepare targets dict (required by compute_total_loss)
            targets = {"features": features}
            if reference_features is not None:
                targets["reference_features"] = reference_features

            losses = compute_total_loss(
                outputs,
                targets,
                self.model,
                orthogonality_weight=self.orthogonality_weight,
                informativeness_weight=self.informativeness_weight,
                topo_weight=self.topo_weight,
                topo_samples=self.topo_samples,
                reference_reg_weight=self.reference_reg_weight,
                reference_features=reference_features,
                feature_weights=self.feature_weights,
                normalize_mse=self.normalize_mse,
                normalization_stats=self.normalization_stats,
                group_indices=self.group_indices,
            )

            # Compute assignment losses
            soft_assign = self.model.assignment_matrix(temperature)
            assign_losses = self.compute_assignment_losses(features, soft_assign)

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
