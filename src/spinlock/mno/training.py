"""MNO Training — trajectory reconstruction trainer with optional multi-scale roundtrip loss.

MNOTrajectoryTrainer:
    Trains MNO backbone to reconstruct operator trajectories.
    Supports three training modes:
    - Grid-only: trajectory MSE loss (feature_weight=0)
    - Grid + feature: trajectory MSE + summary-feature MSE (feature_weight>0)
    - Grid + multi-scale roundtrip: trajectory MSE + VQ tokenisation loss at
      every pyramid temporal resolution (requires vq_adapter + loss config)

For real HDF5 datasets, use SpinlockDataset (spinlock.data) to load training data.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, List, Tuple

from .backbone import MNOBackbone

logger = logging.getLogger(__name__)


# =============================================================================
# Simple Feature Extraction (grid-level statistics)
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
    spatial_mean = trajectory.mean(dim=(1, 3, 4))  # [B, C]
    features.append(spatial_mean)

    spatial_std = trajectory.std(dim=(1, 3, 4))  # [B, C]
    features.append(spatial_std)

    # 2. Temporal dynamics
    if T > 1:
        velocity = trajectory[:, 1:] - trajectory[:, :-1]  # [B, T-1, C, H, W]

        vel_mean = velocity.abs().mean(dim=(1, 2, 3, 4))  # [B]
        features.append(vel_mean.unsqueeze(1))

        vel_std = velocity.std(dim=(1, 2, 3, 4))  # [B]
        features.append(vel_std.unsqueeze(1))

        if T > 2:
            accel = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, C, H, W]
            accel_mean = accel.abs().mean(dim=(1, 2, 3, 4))
            features.append(accel_mean.unsqueeze(1))

    # 3. Energy-like measures
    energy = (trajectory ** 2).mean(dim=(1, 2, 3, 4))  # [B]
    features.append(energy.unsqueeze(1))

    temporal_var = trajectory.var(dim=1).mean(dim=(1, 2, 3))  # [B]
    features.append(temporal_var.unsqueeze(1))

    # 4. Spatial gradients (averaged over time)
    grad_x = trajectory[:, :, :, :, 1:] - trajectory[:, :, :, :, :-1]
    grad_y = trajectory[:, :, :, 1:, :] - trajectory[:, :, :, :-1, :]

    grad_x_trimmed = grad_x[:, :, :, :-1, :]
    grad_y_trimmed = grad_y[:, :, :, :, :-1]

    grad_magnitude = (grad_x_trimmed.abs() + grad_y_trimmed.abs()).mean(dim=(1, 2, 3, 4))
    features.append(grad_magnitude.unsqueeze(1))

    return torch.cat(features, dim=1)  # [B, D]


# =============================================================================
# In-Memory Datasets
# =============================================================================


class MNOInMemoryDataset(Dataset):
    """In-memory dataset for MNO training.

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
        return self.initial_conditions[idx], self.trajectories[idx]


class MNOInMemoryDatasetWithFeatures(Dataset):
    """In-memory dataset for MNO training with pre-computed features.

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
        feat = self.features[idx] if self.features is not None else None
        return self.initial_conditions[idx], self.trajectories[idx], feat


# =============================================================================
# MNOTrajectoryTrainer
# =============================================================================


class MNOTrajectoryTrainer:
    """Trainer for MNO trajectory reconstruction (grid MSE + optional multi-scale roundtrip loss).

    Supports three training modes:
    - Grid-only:   trajectory MSE (feature_weight=0, no vq_adapter)
    - Grid+feat:   trajectory MSE + summary-feature MSE (feature_weight>0)
    - Grid+roundtrip: trajectory MSE + multi-scale VQ tokenisation loss
                   (requires vq_adapter and config.loss.multi_scale_roundtrip.enabled)

    Multi-scale roundtrip loss operates at every temporal resolution defined by the
    tokeniser's pyramid encoder (e.g. T=32, 64, 128, 256), ensuring MNO trajectories
    tokenise consistently at all scales.

    GT tokens:
        Fast path — batches include a "gt_tokens" dict (loaded from pretokenized HDF5):
                    no feature re-extraction at training time
        Slow path — GT trajectory is tokenised on-the-fly via frozen VQCoherenceAdapter

    Args:
        mno: MNO backbone model
        device: Device to train on
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        feature_weight: Weight for feature loss (0 = grid only, >0 = grid+feature)
        vq_adapter: Optional VQCoherenceAdapter for multi-scale roundtrip loss
        loss_config: Optional LossConfig (required if vq_adapter is provided)
    """

    def __init__(
        self,
        mno: MNOBackbone,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        feature_weight: float = 0.0,
        vq_adapter=None,
        loss_config=None,
    ):
        self.mno = mno.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.feature_weight = feature_weight

        # Multi-scale roundtrip loss (optional)
        self.multi_scale_loss = self._build_loss(vq_adapter, loss_config)

        self.optimizer = torch.optim.AdamW(
            self.mno.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "grid_loss": [],
            "feature_loss": [],
            "roundtrip_loss": [],
            "val_loss": [],
            "epoch_time": [],
        }

        if self.multi_scale_loss is not None:
            logger.info(
                f"MNOTrajectoryTrainer: multi-scale roundtrip loss enabled, "
                f"scales={self.multi_scale_loss._truncation_lengths}"
            )

    def _build_loss(self, vq_adapter, loss_config):
        """Build multi-scale roundtrip loss if configured, else return None."""
        if vq_adapter is None or loss_config is None:
            return None
        msr_cfg = getattr(loss_config, 'multi_scale_roundtrip', None)
        if msr_cfg is None or not msr_cfg.enabled:
            return None

        from spinlock.mno.losses.multi_scale_roundtrip import MultiScaleRoundtripLoss
        return MultiScaleRoundtripLoss(vq_adapter, msr_cfg)

    def train_epoch(
        self,
        dataloader: DataLoader,
        steps: int = 64,
        clip_grad: Optional[float] = 1.0,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.  Batches may be:
                        - 2-tuple (u0, gt_trajectory)
                        - 3-tuple (u0, gt_trajectory, gt_features)
                        - dict with keys "u0", "trajectory", and optionally "gt_tokens"
            steps: Number of rollout steps (used only for synthetic in-memory data)
            clip_grad: Gradient clipping value (None to disable)

        Returns:
            Dictionary with loss components: {total, grid, feature, roundtrip}
        """
        import time

        self.mno.train()
        total_loss = 0.0
        total_grid_loss = 0.0
        total_feature_loss = 0.0
        total_roundtrip_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch (dict or tuple)
            gt_tokens = None
            if isinstance(batch, dict):
                u0 = batch["u0"].to(self.device)
                gt_trajectory = batch["trajectory"].to(self.device)
                gt_features = None
                gt_tokens = batch.get("gt_tokens")  # {T: {key: [B]}} or None
            elif len(batch) == 2:
                u0, gt_trajectory = batch
                u0 = u0.to(self.device)
                gt_trajectory = gt_trajectory.to(self.device)
                gt_features = None
            else:
                u0, gt_trajectory, gt_features = batch
                u0 = u0.to(self.device)
                gt_trajectory = gt_trajectory.to(self.device)
                if gt_features is not None:
                    gt_features = gt_features.to(self.device)

            actual_steps = gt_trajectory.shape[1] - 1
            pred_trajectory = self.mno(u0, steps=actual_steps, return_all_steps=True)

            # Grid MSE loss
            grid_loss = F.mse_loss(pred_trajectory, gt_trajectory)

            # Feature loss (optional)
            feature_loss = torch.tensor(0.0, device=self.device)
            if self.feature_weight > 0:
                pred_features = extract_trajectory_features(pred_trajectory)
                if gt_features is not None:
                    feature_loss = F.mse_loss(pred_features, gt_features)
                else:
                    gt_features_computed = extract_trajectory_features(gt_trajectory)
                    feature_loss = F.mse_loss(pred_features, gt_features_computed)

            # Multi-scale roundtrip loss (optional)
            roundtrip_loss = torch.tensor(0.0, device=self.device)
            if self.multi_scale_loss is not None:
                msr_out = self.multi_scale_loss.compute(
                    pred_trajectory=pred_trajectory,
                    target_trajectory=gt_trajectory if gt_tokens is None else None,
                    gt_tokens=gt_tokens,
                )
                roundtrip_loss = msr_out.total

            # Combined loss
            loss = grid_loss + self.feature_weight * feature_loss + roundtrip_loss

            if torch.isnan(loss):
                logger.warning(f"NaN loss at batch {batch_idx}, skipping")
                continue

            self.optimizer.zero_grad()
            loss.backward()

            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.mno.parameters(), clip_grad)

            self.optimizer.step()

            total_loss += loss.item()
            total_grid_loss += grid_loss.item()
            total_feature_loss += feature_loss.item()
            total_roundtrip_loss += roundtrip_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                msg = f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={avg_loss:.4f}"
                if self.feature_weight > 0:
                    msg += f" (grid={total_grid_loss/num_batches:.4f}, feat={total_feature_loss/num_batches:.4f})"
                if self.multi_scale_loss is not None:
                    msg += f" rt={total_roundtrip_loss/num_batches:.4f}"
                logger.info(msg)

        import time as _time
        epoch_time = _time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        avg_grid_loss = total_grid_loss / max(num_batches, 1)
        avg_feature_loss = total_feature_loss / max(num_batches, 1)
        avg_roundtrip_loss = total_roundtrip_loss / max(num_batches, 1)

        self.history["train_loss"].append(avg_loss)
        self.history["grid_loss"].append(avg_grid_loss)
        self.history["feature_loss"].append(avg_feature_loss)
        self.history["roundtrip_loss"].append(avg_roundtrip_loss)
        self.history["epoch_time"].append(epoch_time)

        return {
            "total": avg_loss,
            "grid": avg_grid_loss,
            "feature": avg_feature_loss,
            "roundtrip": avg_roundtrip_loss,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate on a dataset.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss (grid MSE only)
        """
        self.mno.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                u0 = batch["u0"].to(self.device)
                gt_trajectory = batch["trajectory"].to(self.device)
            else:
                u0, gt_trajectory = batch[0], batch[1]
                u0 = u0.to(self.device)
                gt_trajectory = gt_trajectory.to(self.device)

            actual_steps = gt_trajectory.shape[1] - 1
            pred_trajectory = self.mno(u0, steps=actual_steps, return_all_steps=True)

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
            train_result = self.train_epoch(train_loader)
            train_loss = train_result["total"]

            val_loss = None
            if val_loader is not None and (epoch + 1) % val_every == 0:
                val_loss = self.validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

            if verbose:
                epoch_time = self.history["epoch_time"][-1]
                msg = f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s): loss={train_loss:.4f}"
                if self.feature_weight > 0:
                    msg += f" (grid={train_result['grid']:.4f}, feat={train_result['feature']:.4f})"
                if self.multi_scale_loss is not None:
                    msg += f" rt={train_result['roundtrip']:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
                logger.info(msg)

            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        return self.history


# =============================================================================
# Synthetic Data Generator
# =============================================================================


def generate_synthetic_data(
    n_samples: int = 100,
    timesteps: int = 65,
    channels: int = 1,
    height: int = 64,
    width: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic diffusion-like data for testing.

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
    u0 = torch.randn(n_samples, channels, height, width, device=device)

    trajectories = [u0]
    x = u0
    for t in range(timesteps - 1):
        kernel = torch.ones(1, 1, 3, 3, device=device) / 9.0
        x_padded = F.pad(x, (1, 1, 1, 1), mode="circular")
        x = F.conv2d(x_padded, kernel.expand(channels, 1, 3, 3), groups=channels)
        x = x + 0.01 * torch.randn_like(x)
        trajectories.append(x)

    trajectories = torch.stack(trajectories, dim=1)  # [N, T, C, H, W]
    return u0, trajectories


