"""NOA Phase 1 Training - Trainer with grid and feature loss.

Stage 1: Grid-level training only
- NOA backbone generates trajectories
- Loss = MSE(predicted_trajectory, ground_truth_trajectory)
- Validates: NOA architecture, autoregressive rollout, gradient flow

Stage 2: Add feature MSE loss
- Extract simple trajectory statistics as features
- Loss = grid_MSE + feature_MSE
- Validates: Feature extraction pipeline

Stage 3: VQ-VAE perceptual loss (TODO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple
import time

from .backbone import NOABackbone


# =============================================================================
# Stage 2: Simple Feature Extraction
# =============================================================================


def extract_trajectory_features(trajectory: torch.Tensor) -> torch.Tensor:
    """Extract simple summary features from a trajectory.

    Computes basic statistics that characterize trajectory dynamics:
    - Spatial statistics per timestep (mean, std, min, max)
    - Temporal dynamics (velocity mean/std, acceleration mean/std)
    - Global aggregates (total energy, temporal variance)

    Args:
        trajectory: Trajectory tensor [B, T, C, H, W]

    Returns:
        Features tensor [B, D] where D is the number of features
    """
    B, T, C, H, W = trajectory.shape

    features = []

    # 1. Spatial statistics per channel (averaged over time)
    # Mean per channel over space and time
    spatial_mean = trajectory.mean(dim=(1, 3, 4))  # [B, C]
    features.append(spatial_mean)

    # Std per channel over space and time
    spatial_std = trajectory.std(dim=(1, 3, 4))  # [B, C]
    features.append(spatial_std)

    # 2. Temporal dynamics
    if T > 1:
        # Velocity (first difference)
        velocity = trajectory[:, 1:] - trajectory[:, :-1]  # [B, T-1, C, H, W]

        # Mean absolute velocity
        vel_mean = velocity.abs().mean(dim=(1, 2, 3, 4))  # [B]
        features.append(vel_mean.unsqueeze(1))

        # Velocity std
        vel_std = velocity.std(dim=(1, 2, 3, 4))  # [B]
        features.append(vel_std.unsqueeze(1))

        if T > 2:
            # Acceleration (second difference)
            accel = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, C, H, W]
            accel_mean = accel.abs().mean(dim=(1, 2, 3, 4))
            features.append(accel_mean.unsqueeze(1))

    # 3. Energy-like measures
    # Total squared magnitude (proxy for energy)
    energy = (trajectory ** 2).mean(dim=(1, 2, 3, 4))  # [B]
    features.append(energy.unsqueeze(1))

    # Temporal variance at each spatial location, averaged
    temporal_var = trajectory.var(dim=1).mean(dim=(1, 2, 3))  # [B]
    features.append(temporal_var.unsqueeze(1))

    # 4. Spatial gradients (averaged over time)
    # Use finite differences
    grad_x = trajectory[:, :, :, :, 1:] - trajectory[:, :, :, :, :-1]  # [B, T, C, H, W-1]
    grad_y = trajectory[:, :, :, 1:, :] - trajectory[:, :, :, :-1, :]  # [B, T, C, H-1, W]

    # Align shapes: trim to common [H-1, W-1] region
    grad_x_trimmed = grad_x[:, :, :, :-1, :]  # [B, T, C, H-1, W-1]
    grad_y_trimmed = grad_y[:, :, :, :, :-1]  # [B, T, C, H-1, W-1]

    grad_magnitude = (grad_x_trimmed.abs() + grad_y_trimmed.abs()).mean(dim=(1, 2, 3, 4))
    features.append(grad_magnitude.unsqueeze(1))

    # Concatenate all features
    return torch.cat(features, dim=1)  # [B, D]


class NOADatasetWithFeatures(Dataset):
    """Dataset for NOA training with pre-computed features.

    Args:
        initial_conditions: Initial states [N, C, H, W]
        trajectories: Ground-truth trajectories [N, T, C, H, W]
        features: Optional pre-computed features [N, D]
    """

    def __init__(
        self,
        initial_conditions: torch.Tensor,
        trajectories: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ):
        self.initial_conditions = initial_conditions
        self.trajectories = trajectories
        self.features = features

        assert len(initial_conditions) == len(trajectories)
        if features is not None:
            assert len(features) == len(initial_conditions)

    def __len__(self) -> int:
        return len(self.initial_conditions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Returns (initial_condition, trajectory, features)."""
        feat = self.features[idx] if self.features is not None else None
        return self.initial_conditions[idx], self.trajectories[idx], feat


class NOADataset(Dataset):
    """Simple dataset for NOA training.

    Loads initial conditions and ground-truth trajectories from numpy arrays.

    Args:
        initial_conditions: Initial states [N, C, H, W]
        trajectories: Ground-truth trajectories [N, T, C, H, W]
    """

    def __init__(
        self,
        initial_conditions: torch.Tensor,
        trajectories: torch.Tensor,
    ):
        self.initial_conditions = initial_conditions
        self.trajectories = trajectories

        assert len(initial_conditions) == len(trajectories), \
            f"Mismatch: {len(initial_conditions)} ICs vs {len(trajectories)} trajectories"

    def __len__(self) -> int:
        return len(self.initial_conditions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (initial_condition, trajectory)."""
        return self.initial_conditions[idx], self.trajectories[idx]


class NOAPhase1Trainer:
    """Trainer for NOA Phase 1 (grid MSE + optional feature MSE loss).

    Supports two training modes:
    - Stage 1: Grid-level MSE only (feature_weight=0)
    - Stage 2: Grid MSE + Feature MSE (feature_weight>0)

    Args:
        noa: NOA backbone model
        device: Device to train on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        feature_weight: Weight for feature loss (0 = Stage 1, >0 = Stage 2)

    Example:
        >>> noa = NOABackbone()
        >>> # Stage 1: Grid loss only
        >>> trainer = NOAPhase1Trainer(noa, device="cuda", feature_weight=0.0)
        >>> # Stage 2: Grid + Feature loss
        >>> trainer = NOAPhase1Trainer(noa, device="cuda", feature_weight=0.5)
    """

    def __init__(
        self,
        noa: NOABackbone,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        feature_weight: float = 0.0,  # Stage 2: set > 0 to enable feature loss
    ):
        self.noa = noa.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_weight = feature_weight

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.noa.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "grid_loss": [],
            "feature_loss": [],
            "val_loss": [],
            "epoch_time": [],
        }

    def train_epoch(
        self,
        dataloader: DataLoader,
        steps: int = 64,
        clip_grad: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            steps: Number of rollout steps (must match trajectory length - 1)
            clip_grad: Gradient clipping value (None to disable)

        Returns:
            Dictionary with loss components: {total, grid, feature}
        """
        self.noa.train()
        total_loss = 0.0
        total_grid_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Handle both 2-tuple (Stage 1) and 3-tuple (Stage 2) batches
            if len(batch) == 2:
                u0, gt_trajectory = batch
                gt_features = None
            else:
                u0, gt_trajectory, gt_features = batch

            # Move to device
            u0 = u0.to(self.device)
            gt_trajectory = gt_trajectory.to(self.device)
            if gt_features is not None:
                gt_features = gt_features.to(self.device)

            # Determine steps from ground truth (trajectory includes u0)
            actual_steps = gt_trajectory.shape[1] - 1

            # Generate rollout
            pred_trajectory = self.noa(u0, steps=actual_steps, return_all_steps=True)

            # Compute grid MSE loss
            grid_loss = F.mse_loss(pred_trajectory, gt_trajectory)

            # Compute feature loss if enabled (Stage 2)
            feature_loss = torch.tensor(0.0, device=self.device)
            if self.feature_weight > 0:
                # Extract features from predicted and ground-truth trajectories
                pred_features = extract_trajectory_features(pred_trajectory)

                if gt_features is not None:
                    # Use pre-computed ground-truth features
                    feature_loss = F.mse_loss(pred_features, gt_features)
                else:
                    # Compute features from ground-truth trajectory on the fly
                    gt_features_computed = extract_trajectory_features(gt_trajectory)
                    feature_loss = F.mse_loss(pred_features, gt_features_computed)

            # Combined loss
            loss = grid_loss + self.feature_weight * feature_loss

            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.noa.parameters(), clip_grad)

            self.optimizer.step()

            total_loss += loss.item()
            total_grid_loss += grid_loss.item()
            total_feature_loss += feature_loss.item()
            num_batches += 1

            # Progress logging every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                msg = f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={avg_loss:.4f}"
                if self.feature_weight > 0:
                    msg += f" (grid={total_grid_loss/num_batches:.4f}, feat={total_feature_loss/num_batches:.4f})"
                print(msg)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        avg_grid_loss = total_grid_loss / max(num_batches, 1)
        avg_feature_loss = total_feature_loss / max(num_batches, 1)

        # Record history
        self.history["train_loss"].append(avg_loss)
        self.history["grid_loss"].append(avg_grid_loss)
        self.history["feature_loss"].append(avg_feature_loss)
        self.history["epoch_time"].append(epoch_time)

        return {
            "total": avg_loss,
            "grid": avg_grid_loss,
            "feature": avg_feature_loss,
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> float:
        """Validate on a dataset.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.noa.eval()
        total_loss = 0.0
        num_batches = 0

        for u0, gt_trajectory in dataloader:
            u0 = u0.to(self.device)
            gt_trajectory = gt_trajectory.to(self.device)

            actual_steps = gt_trajectory.shape[1] - 1
            pred_trajectory = self.noa(u0, steps=actual_steps, return_all_steps=True)

            loss = F.mse_loss(pred_trajectory, gt_trajectory)

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        val_every: int = 1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            val_every: Validate every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_result = self.train_epoch(train_loader)
            train_loss = train_result["total"]

            # Validate
            val_loss = None
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_loss = self.validate(val_loader)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Logging
            if verbose:
                epoch_time = self.history["epoch_time"][-1]
                msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s): loss={train_loss:.4f}"
                if self.feature_weight > 0:
                    msg += f" (grid={train_result['grid']:.4f}, feat={train_result['feature']:.4f})"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
                print(msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history


def generate_synthetic_data(
    n_samples: int = 100,
    timesteps: int = 65,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for testing.

    Creates random initial conditions and trajectories for debugging.

    Args:
        n_samples: Number of samples
        timesteps: Number of timesteps in trajectory (including u0)
        channels: Number of channels
        height: Grid height
        width: Grid width
        device: Device to create tensors on

    Returns:
        (initial_conditions, trajectories) tuple
    """
    # Random initial conditions
    u0 = torch.randn(n_samples, channels, height, width, device=device)

    # Simple diffusion-like dynamics for synthetic trajectories
    trajectories = [u0]
    x = u0
    for t in range(timesteps - 1):
        # Simple smoothing (blurring) as synthetic dynamics
        kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
        x_padded = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = F.conv2d(x_padded, kernel.expand(channels, 1, 3, 3), groups=channels)
        # Add some noise
        x = x + 0.01 * torch.randn_like(x)
        trajectories.append(x)

    trajectories = torch.stack(trajectories, dim=1)  # [N, T, C, H, W]

    return u0, trajectories


# =============================================================================
# Phase 1 Real Data Training
# =============================================================================


class NOARealDataTrainer:
    """Trainer for NOA Phase 1 on real data with proper feature extraction.

    Uses:
    - NOAFeatureExtractor (SummaryExtractor internally) for feature extraction
    - NOARealDataset for ground-truth features from HDF5
    - SUMMARY + TEMPORAL MSE losses

    All dimensions are resolved dynamically at runtime.

    Args:
        noa: NOA backbone model
        feature_extractor: NOAFeatureExtractor instance (or created automatically)
        summary_weight: Weight for SUMMARY MSE loss
        temporal_weight: Weight for TEMPORAL MSE loss
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on

    Example:
        >>> from spinlock.noa import NOABackbone, NOARealDataset, NOAFeatureExtractor
        >>> noa = NOABackbone(in_channels=1, out_channels=1, base_channels=32)
        >>> dataset = NOARealDataset("datasets/100k_full_features.h5", n_samples=1000)
        >>> trainer = NOARealDataTrainer(noa, device="cuda")
        >>> trainer.train(DataLoader(dataset, batch_size=16), epochs=10)
    """

    def __init__(
        self,
        noa: NOABackbone,
        feature_extractor: Optional["NOAFeatureExtractor"] = None,
        summary_weight: float = 1.0,
        temporal_weight: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
    ):
        from .feature_extraction import NOAFeatureExtractor

        self.device = device
        self.noa = noa.to(device)

        # Create feature extractor if not provided
        self.feature_extractor = feature_extractor or NOAFeatureExtractor(device=device)

        self.summary_weight = summary_weight
        self.temporal_weight = temporal_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.noa.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "summary_loss": [],
            "temporal_loss": [],
            "val_loss": [],
            "epoch_time": [],
        }

        # Dimension info (discovered on first batch)
        self._dims_initialized = False
        self._timesteps: Optional[int] = None

    def _init_dimensions(self, batch: Dict[str, torch.Tensor]):
        """Initialize dimension info from first batch."""
        if self._dims_initialized:
            return

        # Discover temporal steps from dataset
        # batch['temporal'] has shape [B, T, D], so timesteps is at index 1
        self._timesteps = batch['temporal'].shape[1]
        self._dims_initialized = True

        print(f"NOARealDataTrainer initialized:")
        print(f"  Temporal steps: {self._timesteps}")
        print(f"  Summary GT dim: {batch['summary'].shape[-1]}")
        print(f"  Temporal GT dim: {batch['temporal'].shape[-1]}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        clip_grad: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader with NOARealDataset
            clip_grad: Gradient clipping value (None to disable)

        Returns:
            Dictionary with loss components: {total, summary, temporal}
        """
        self.noa.train()
        total_loss = 0.0
        total_summary_loss = 0.0
        total_temporal_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Initialize dimensions on first batch
            self._init_dimensions(batch)

            # Move to device
            ic = batch['ic'].to(self.device)
            summary_gt = batch['summary'].to(self.device)
            temporal_gt = batch['temporal'].to(self.device)

            # Generate rollout
            # Steps = timesteps (NOA should produce [B, T+1, C, H, W] but we want [B, T, C, H, W])
            # Actually, NOA with return_all_steps returns [B, steps+1, C, H, W] including u0
            # We need steps = temporal_steps - 1 if trajectory includes u0
            # Or steps = temporal_steps if trajectory is just the evolved states

            # For TEMPORAL features, we have T timesteps of features
            # NOA generates T-step rollout (excluding IC) or T+1 if including IC
            # Let's generate T steps (not including IC separately)
            pred_trajectory = self.noa(ic, steps=self._timesteps, return_all_steps=True)

            # pred_trajectory: [B, T+1, C, H, W] (includes u0)
            # TEMPORAL features are for T timesteps, so we skip u0
            pred_rollout = pred_trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]

            # Extract features from rollout
            pred_features = self.feature_extractor.extract(pred_rollout)

            # Compute losses
            # Note: pred_features['summary'] is per_trajectory (from M=1), not aggregated
            # Dataset summary may be aggregated (360D) or per_trajectory (120D)
            # We need to handle dimension mismatch

            pred_summary = pred_features['summary']  # [B, D_pred]
            pred_temporal = pred_features['temporal']  # [B, T, D]

            # If dimension mismatch for summary, we compare what we can
            if pred_summary.shape[-1] != summary_gt.shape[-1]:
                # Assume dataset has aggregated (mean/std/cv), we only have mean
                # Use first 1/3 of ground truth (the mean portion)
                gt_per_traj_dim = pred_summary.shape[-1]
                summary_gt_trimmed = summary_gt[:, :gt_per_traj_dim]
                summary_loss = F.mse_loss(pred_summary, summary_gt_trimmed)
            else:
                summary_loss = F.mse_loss(pred_summary, summary_gt)

            # TEMPORAL loss
            temporal_loss = F.mse_loss(pred_temporal, temporal_gt)

            # Combined loss
            loss = self.summary_weight * summary_loss + self.temporal_weight * temporal_loss

            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}")
                continue

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.noa.parameters(), clip_grad)

            self.optimizer.step()

            total_loss += loss.item()
            total_summary_loss += summary_loss.item()
            total_temporal_loss += temporal_loss.item()
            num_batches += 1

            # Progress logging every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                avg_summary = total_summary_loss / num_batches
                avg_temporal = total_temporal_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                      f"loss={avg_loss:.4f} (sum={avg_summary:.4f}, temp={avg_temporal:.4f})")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        avg_summary_loss = total_summary_loss / max(num_batches, 1)
        avg_temporal_loss = total_temporal_loss / max(num_batches, 1)

        # Record history
        self.history["train_loss"].append(avg_loss)
        self.history["summary_loss"].append(avg_summary_loss)
        self.history["temporal_loss"].append(avg_temporal_loss)
        self.history["epoch_time"].append(epoch_time)

        return {
            "total": avg_loss,
            "summary": avg_summary_loss,
            "temporal": avg_temporal_loss,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on a dataset.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.noa.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            ic = batch['ic'].to(self.device)
            summary_gt = batch['summary'].to(self.device)
            temporal_gt = batch['temporal'].to(self.device)

            # Generate rollout
            pred_trajectory = self.noa(ic, steps=self._timesteps, return_all_steps=True)
            pred_rollout = pred_trajectory[:, 1:, :, :, :]

            # Extract features
            pred_features = self.feature_extractor.extract(pred_rollout)

            pred_summary = pred_features['summary']
            pred_temporal = pred_features['temporal']

            # Handle dimension mismatch
            if pred_summary.shape[-1] != summary_gt.shape[-1]:
                gt_per_traj_dim = pred_summary.shape[-1]
                summary_gt_trimmed = summary_gt[:, :gt_per_traj_dim]
                summary_loss = F.mse_loss(pred_summary, summary_gt_trimmed)
            else:
                summary_loss = F.mse_loss(pred_summary, summary_gt)

            temporal_loss = F.mse_loss(pred_temporal, temporal_gt)
            loss = self.summary_weight * summary_loss + self.temporal_weight * temporal_loss

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history["val_loss"].append(avg_loss)

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        val_every: int = 1,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            val_every: Validate every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_result = self.train_epoch(train_loader)
            train_loss = train_result["total"]

            # Validate
            val_loss = None
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_loss = self.validate(val_loader)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Logging
            if verbose:
                epoch_time = self.history["epoch_time"][-1]
                msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s): loss={train_loss:.4f}"
                msg += f" (sum={train_result['summary']:.4f}, temp={train_result['temporal']:.4f})"
                if val_loss is not None:
                    msg += f", val={val_loss:.4f}"
                print(msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history
