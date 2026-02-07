"""Temporal encoder pretraining via autoencoder reconstruction."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..config import TemporalPretrainingConfig
from ..encoders.temporal import PyramidTemporalEncoder

logger = logging.getLogger(__name__)


class TemporalDecoder(nn.Module):
    """1D deconvolutional decoder for temporal sequences.

    Symmetric architecture to PyramidTemporalEncoder, using transposed convolutions
    to upsample from concatenated pyramid embeddings back to temporal sequences.

    Args:
        embedding_dim: Total input embedding dimension (sum of level_dims)
        output_dim: Output feature dimension per timestep
        max_seq_length: Maximum sequence length to reconstruct
    """

    def __init__(self, embedding_dim: int, output_dim: int, max_seq_length: int = 256):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length

        # Project embedding to initial temporal feature map
        # Start with 256 channels and 16 timesteps
        self.initial_length = 16
        self.initial_channels = 256
        self.fc = nn.Linear(embedding_dim, self.initial_channels * self.initial_length)

        # Build upsampling stages: 16 → 32 → 64 → 128 → 256
        # Channel progression: 256 → 128 → 128 → 64 → 64 → output_dim
        self.upsample = nn.Sequential(
            # Stage 1: 16 → 32, 256 → 128
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Stage 2: 32 → 64, 128 → 128
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Stage 3: 64 → 128, 128 → 64
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Stage 4: 128 → 256, 64 → 64
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # Final projection to output dimension (no activation)
            nn.Conv1d(64, output_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embedding to temporal sequence.

        Args:
            x: Embedding [B, embedding_dim]

        Returns:
            Reconstructed sequence [B, T, D_out]
        """
        # Project to initial feature map
        h = self.fc(x)  # [B, initial_channels * initial_length]
        h = h.view(-1, self.initial_channels, self.initial_length)  # [B, 256, 16]

        # Upsample to target sequence length
        h = self.upsample(h)  # [B, output_dim, 256]

        # Transpose to [B, T, D]
        h = h.transpose(1, 2)  # [B, 256, output_dim]

        # Trim to max_seq_length if needed
        if h.shape[1] > self.max_seq_length:
            h = h[:, :self.max_seq_length, :]

        return h


def temporal_collate_fn(batch):
    """Collate function for variable-length temporal sequences.

    Pads sequences to max length in batch and creates validity masks.

    Args:
        batch: List of tuples (sequence, mask, length)
            - sequence: [T_i, D] temporal features
            - mask: [T_i] validity mask
            - length: int, actual sequence length

    Returns:
        Dictionary with keys:
            - sequences: [B, max_T, D] padded sequences
            - masks: [B, max_T] validity masks
            - lengths: [B] actual lengths
    """
    sequences, masks, lengths = zip(*batch)

    # Get dimensions
    batch_size = len(sequences)
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]

    # Create padded tensors
    padded_sequences = torch.zeros(batch_size, max_len, feature_dim)
    padded_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, (seq, mask, length) in enumerate(zip(sequences, masks, lengths)):
        seq_len = seq.shape[0]
        padded_sequences[i, :seq_len] = seq
        padded_masks[i, :seq_len] = mask

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return {
        "sequences": padded_sequences,
        "masks": padded_masks,
        "lengths": lengths_tensor,
    }


class TemporalPretrainer:
    """Pretrain PyramidTemporalEncoder via autoencoder reconstruction.

    Trains an autoencoder (encoder + decoder) to reconstruct temporal sequences,
    then saves only the encoder weights for use in VQ-VAE training.

    Args:
        config: Temporal pretraining configuration
        input_dim: Optional runtime-detected input dimension
        max_seq_length: Maximum sequence length for reconstruction

    Example:
        >>> config = TemporalPretrainingConfig(num_epochs=100)
        >>> pretrainer = TemporalPretrainer(config)
        >>> pretrainer.train(temporal_features, output_path="temporal_pretrained.pt")
    """

    def __init__(
        self,
        config: TemporalPretrainingConfig,
        input_dim: Optional[int] = None,
        max_seq_length: int = 256,
    ):
        self.config = config
        self.max_seq_length = max_seq_length

        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Use runtime-detected input_dim if provided
        self.input_dim = input_dim if input_dim is not None else config.input_dim

        # Variable-length config for encoder
        vl_config = None
        if config.variable_length:
            vl_config = {
                "enabled": True,
                "adaptive_pyramid": config.adaptive_pyramid,
                "min_pyramid_length": config.min_pyramid_length,
                "mask_downsample_method": "ceil",
            }

        # Create encoder
        self.encoder = PyramidTemporalEncoder(
            input_dim=self.input_dim,
            level_dims=config.level_dims,
            downsample_factors=config.downsample_factors,
            variable_length_config=vl_config,
        ).to(self.device)

        # Total embedding dimension
        self.embedding_dim = sum(config.level_dims)

        # Create decoder
        self.decoder = TemporalDecoder(
            embedding_dim=self.embedding_dim,
            output_dim=self.input_dim,
            max_seq_length=max_seq_length,
        ).to(self.device)

        # Loss function (will be masked MSE)
        self.criterion = nn.MSELoss(reduction="none")

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        elif config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=config.learning_rate, weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Scheduler (optional)
        self.scheduler = None
        if config.use_scheduler:
            if config.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=config.num_epochs
                )
            elif config.scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=config.num_epochs // 3, gamma=0.1
                )
            elif config.scheduler_type == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.95
                )

    def train(
        self,
        temporal_features: torch.Tensor,
        output_path: Path,
        masks: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        val_temporal_features: Optional[torch.Tensor] = None,
        val_masks: Optional[torch.Tensor] = None,
        val_lengths: Optional[torch.Tensor] = None,
        normalization_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Train autoencoder and save encoder checkpoint.

        Args:
            temporal_features: Training sequences [N, T, D]
            output_path: Path to save encoder checkpoint
            masks: Optional validity masks [N, T]
            lengths: Optional actual lengths [N]
            val_temporal_features: Optional validation sequences
            val_masks: Optional validation masks
            val_lengths: Optional validation lengths
            normalization_stats: Optional dict with 'temporal_mean' and 'temporal_std'

        Returns:
            Training history dict with keys:
                - train_losses: List of training losses per epoch
                - val_losses: List of validation losses per epoch
                - best_epoch: Epoch with lowest validation loss
                - best_val_loss: Best validation loss achieved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create masks if not provided (assume all valid)
        if masks is None:
            masks = torch.ones(temporal_features.shape[:2], dtype=torch.bool)

        # Create lengths if not provided
        if lengths is None:
            lengths = torch.full((len(temporal_features),), temporal_features.shape[1], dtype=torch.long)

        # Create train/val split if needed
        if val_temporal_features is None:
            dataset = TensorDataset(temporal_features, masks, lengths)
            val_size = int(len(dataset) * self.config.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            train_dataset = TensorDataset(temporal_features, masks, lengths)
            if val_masks is None:
                val_masks = torch.ones(val_temporal_features.shape[:2], dtype=torch.bool)
            if val_lengths is None:
                val_lengths = torch.full((len(val_temporal_features),), val_temporal_features.shape[1], dtype=torch.long)
            val_dataset = TensorDataset(val_temporal_features, val_masks, val_lengths)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        logger.info(f"Temporal Encoder Pretraining on {self.device}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Level dims: {self.config.level_dims}")
        logger.info(f"  Epochs: {self.config.num_epochs}")

        # Training loop
        train_metrics_history = {'mse': [], 'rmse': [], 'rel_l2': []}
        val_metrics_history = {'mse': [], 'rmse': [], 'rel_l2': []}
        best_val_mse = float('inf')
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self._train_epoch(train_loader)
            train_metrics_history['mse'].append(train_metrics['mse'])
            train_metrics_history['rmse'].append(train_metrics['rmse'])
            train_metrics_history['rel_l2'].append(train_metrics['rel_l2'])

            # Validate
            val_metrics = self._validate_epoch(val_loader)
            val_metrics_history['mse'].append(val_metrics['mse'])
            val_metrics_history['rmse'].append(val_metrics['rmse'])
            val_metrics_history['rel_l2'].append(val_metrics['rel_l2'])

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Track best model (based on MSE)
            if val_metrics['mse'] < best_val_mse:
                best_val_mse = val_metrics['mse']
                best_epoch = epoch

            # Log progress
            if self.config.verbose and (epoch + 1) % self.config.log_every_n_epochs == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "
                    f"Train MSE: {train_metrics['mse']:.4f} "
                    f"RMSE: {train_metrics['rmse']:.4f} "
                    f"Rel-L2: {train_metrics['rel_l2']:.4f} | "
                    f"Val MSE: {val_metrics['mse']:.4f} "
                    f"RMSE: {val_metrics['rmse']:.4f} "
                    f"Rel-L2: {val_metrics['rel_l2']:.4f} | "
                    f"LR: {lr:.6f}"
                )

        # Save checkpoint
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'level_dims': self.config.level_dims,
                'downsample_factors': self.config.downsample_factors,
                'variable_length': self.config.variable_length,
                'adaptive_pyramid': self.config.adaptive_pyramid,
            },
            'epoch': best_epoch,
            'best_val_mse': best_val_mse,
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'normalization_stats': normalization_stats,
        }

        torch.save(checkpoint, output_path)
        logger.info(f"Saved encoder checkpoint to {output_path}")
        logger.info(f"  Best epoch: {best_epoch+1}")
        logger.info(f"  Best val MSE: {best_val_mse:.6f}")
        logger.info(f"  Best val RMSE: {val_metrics_history['rmse'][best_epoch]:.6f}")
        logger.info(f"  Best val Rel-L2: {val_metrics_history['rel_l2'][best_epoch]:.6f}")

        return {
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'best_epoch': best_epoch,
            'best_val_mse': best_val_mse,
        }

    def _augment(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temporal augmentation.

        Args:
            x: Input batch [B, T, D]
            mask: Validity mask [B, T]

        Returns:
            Augmented batch and updated mask
        """
        # Temporal dropout: randomly mask 10% of valid timesteps
        if self.config.use_temporal_dropout:
            dropout_mask = torch.rand_like(mask.float()) > self.config.temporal_dropout_prob
            mask = mask & dropout_mask

        # Gaussian noise injection (optional)
        if self.config.use_noise_injection:
            noise = torch.randn_like(x) * self.config.noise_std
            x = x + noise

        return x, mask

    def _compute_masked_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """Compute masked loss metrics.

        Only compute loss on valid timesteps (mask == True).

        Args:
            pred: Predicted sequence [B, T, D]
            target: Target sequence [B, T, D]
            mask: Validity mask [B, T]

        Returns:
            Dict with keys:
                - 'mse': Mean squared error
                - 'rmse': Root mean squared error
                - 'rel_l2': Relative L2 error (||pred - target||_2 / ||target||_2)
        """
        # Expand mask to match dimensions
        mask_expanded = mask.unsqueeze(-1).float()  # [B, T, 1]

        # Compute element-wise MSE
        mse_elementwise = self.criterion(pred, target)  # [B, T, D]
        masked_mse = mse_elementwise * mask_expanded  # [B, T, D]

        # Average MSE over valid elements
        num_valid = mask_expanded.sum()
        if num_valid > 0:
            mse = masked_mse.sum() / num_valid
        else:
            mse = masked_mse.sum()  # Shouldn't happen, but handle gracefully

        # Compute RMSE
        rmse = torch.sqrt(mse)

        # Compute Relative L2 error: ||pred - target||_2 / ||target||_2
        # Apply mask before computing norms
        pred_masked = pred * mask_expanded
        target_masked = target * mask_expanded

        diff_norm = torch.norm(pred_masked - target_masked)
        target_norm = torch.norm(target_masked)

        if target_norm > 0:
            rel_l2 = diff_norm / target_norm
        else:
            rel_l2 = torch.tensor(0.0, device=pred.device)

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'rel_l2': rel_l2.item(),
        }

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            loader: Training data loader

        Returns:
            Dict of average metrics for the epoch (mse, rmse, rel_l2)
        """
        self.encoder.train()
        self.decoder.train()

        total_mse = 0.0
        total_rmse = 0.0
        total_rel_l2 = 0.0
        num_batches = 0

        for batch in loader:
            x = batch[0].to(self.device)  # [B, T, D]
            mask = batch[1].to(self.device)  # [B, T]
            lengths = batch[2].to(self.device)  # [B]

            # Apply augmentation (only during training)
            x_aug, mask_aug = self._augment(x, mask)

            # Forward pass through encoder
            if self.config.variable_length:
                embeddings, _ = self.encoder(x_aug, mask=mask_aug, lengths=lengths)
            else:
                embeddings = self.encoder(x_aug)

            # Forward pass through decoder
            reconstructed = self.decoder(embeddings)  # [B, T, D]

            # Compute masked loss metrics
            metrics = self._compute_masked_loss(reconstructed, x, mask)

            # Backward pass (use MSE as the optimization objective)
            loss = torch.tensor(metrics['mse'], device=self.device, requires_grad=True)
            # Need to recompute for gradients
            mse_elementwise = self.criterion(reconstructed, x)
            mask_expanded = mask.unsqueeze(-1).float()
            masked_mse = mse_elementwise * mask_expanded
            num_valid = mask_expanded.sum()
            loss = masked_mse.sum() / num_valid if num_valid > 0 else masked_mse.sum()

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    self.config.gradient_clip_norm
                )

            self.optimizer.step()

            total_mse += metrics['mse']
            total_rmse += metrics['rmse']
            total_rel_l2 += metrics['rel_l2']
            num_batches += 1

        return {
            'mse': total_mse / num_batches,
            'rmse': total_rmse / num_batches,
            'rel_l2': total_rel_l2 / num_batches,
        }

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch.

        Args:
            loader: Validation data loader

        Returns:
            Dict of average metrics for the epoch (mse, rmse, rel_l2)
        """
        self.encoder.eval()
        self.decoder.eval()

        total_mse = 0.0
        total_rmse = 0.0
        total_rel_l2 = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)
                mask = batch[1].to(self.device)
                lengths = batch[2].to(self.device)

                # Forward pass (no augmentation during validation)
                if self.config.variable_length:
                    embeddings, _ = self.encoder(x, mask=mask, lengths=lengths)
                else:
                    embeddings = self.encoder(x)

                reconstructed = self.decoder(embeddings)

                # Compute masked loss metrics
                metrics = self._compute_masked_loss(reconstructed, x, mask)

                total_mse += metrics['mse']
                total_rmse += metrics['rmse']
                total_rel_l2 += metrics['rel_l2']
                num_batches += 1

        return {
            'mse': total_mse / num_batches,
            'rmse': total_rmse / num_batches,
            'rel_l2': total_rel_l2 / num_batches,
        }
