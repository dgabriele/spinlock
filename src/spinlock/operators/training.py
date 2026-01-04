"""
Operator Training Module.

Trains U-AFNO operators on next-step prediction before extracting learned features.

The training objective is:
    loss = MSE(operator(x_t), x_{t+1})

This ensures the operator's internal representations (bottleneck latents)
become meaningful learned features that capture the dynamics of the system.

Author: Claude (Anthropic)
Date: January 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
from typing import Literal, Optional
import time


@dataclass
class TrainingStats:
    """Statistics from operator training."""

    final_loss: float
    initial_loss: float
    epochs_completed: int
    training_time_sec: float
    best_loss: float
    converged: bool  # True if loss stopped decreasing


class OperatorTrainer:
    """
    Train U-AFNO operators on next-step prediction.

    The operator learns to predict x_{t+1} from x_t using MSE loss.
    After training, the operator's bottleneck latents become meaningful
    learned features that capture the dynamics of the system.

    Example:
        >>> trainer = OperatorTrainer(epochs=100, lr=1e-3)
        >>> trajectories = torch.randn(5, 50, 3, 64, 64)  # [M, T, C, H, W]
        >>> stats = trainer.train(model, trajectories)
        >>> print(f"Final loss: {stats.final_loss:.4f}")
    """

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 1e-3,
        lr_scheduler: Literal["constant", "cosine"] = "cosine",
        device: torch.device = torch.device("cuda"),
        early_stopping_patience: int = 20,
        min_delta: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Initialize operator trainer.

        Args:
            epochs: Number of training epochs
            lr: Learning rate for Adam optimizer
            lr_scheduler: Learning rate schedule ("constant" or "cosine")
            device: Torch device for training
            early_stopping_patience: Stop if no improvement for this many epochs
            min_delta: Minimum loss improvement to count as progress
            verbose: Print training progress
        """
        self.epochs = epochs
        self.lr = lr
        self.lr_scheduler_type = lr_scheduler
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.verbose = verbose

    def train(
        self,
        model: nn.Module,
        trajectories: torch.Tensor,
    ) -> TrainingStats:
        """
        Train operator on next-step prediction.

        Args:
            model: U-AFNO model (will be trained in-place)
            trajectories: Rollout trajectories [M, T, C, H, W]
                M = number of realizations
                T = number of timesteps
                C, H, W = channel, height, width

        Returns:
            TrainingStats with loss history and timing
        """
        M, T, C, H, W = trajectories.shape

        # Create training pairs: (x_t, x_{t+1})
        # x_t: all frames except last
        # x_{t+1}: all frames except first
        x_t = trajectories[:, :-1].reshape(-1, C, H, W)      # [M*(T-1), C, H, W]
        x_next = trajectories[:, 1:].reshape(-1, C, H, W)    # [M*(T-1), C, H, W]

        # Move to device
        x_t = x_t.to(self.device)
        x_next = x_next.to(self.device)

        # Ensure model is in training mode and gradients are enabled
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        # Setup optimizer
        optimizer = Adam(model.parameters(), lr=self.lr)

        # Setup scheduler
        scheduler: Optional[CosineAnnealingLR] = None
        if self.lr_scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 0.01)

        # Loss function
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        start_time = time.time()

        best_loss = float("inf")
        patience_counter = 0
        initial_loss: Optional[float] = None
        final_loss = 0.0
        epochs_completed = 0

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Forward pass
            predictions = model(x_t)
            loss = criterion(predictions, x_next)

            # Backward pass
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Track loss
            current_loss = loss.item()
            if initial_loss is None:
                initial_loss = current_loss
            final_loss = current_loss
            epochs_completed = epoch + 1

            # Early stopping check
            if current_loss < best_loss - self.min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch + 1}, loss: {current_loss:.6f}")
                break

            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, loss: {current_loss:.6f}")

        training_time = time.time() - start_time

        # Set back to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return TrainingStats(
            final_loss=final_loss,
            initial_loss=initial_loss if initial_loss is not None else final_loss,
            epochs_completed=epochs_completed,
            training_time_sec=training_time,
            best_loss=best_loss,
            converged=patience_counter >= self.early_stopping_patience,
        )

    def train_batch(
        self,
        models: list[nn.Module],
        trajectories_batch: torch.Tensor,
    ) -> list[TrainingStats]:
        """
        Train multiple operators on their respective trajectories.

        Args:
            models: List of N U-AFNO models
            trajectories_batch: Trajectories [N, M, T, C, H, W]

        Returns:
            List of TrainingStats, one per model
        """
        N = len(models)
        if trajectories_batch.shape[0] != N:
            raise ValueError(
                f"Number of models ({N}) must match batch dimension "
                f"of trajectories ({trajectories_batch.shape[0]})"
            )

        stats_list = []
        for i, model in enumerate(models):
            traj_i = trajectories_batch[i]  # [M, T, C, H, W]
            stats = self.train(model, traj_i)
            stats_list.append(stats)

        return stats_list
