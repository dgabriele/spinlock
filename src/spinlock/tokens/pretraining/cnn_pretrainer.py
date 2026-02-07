"""CNN encoder pretraining via autoencoder reconstruction."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from ..config import PretrainingConfig
from ..encoders import InitialCNNEncoder

logger = logging.getLogger(__name__)


class CNNDecoder(nn.Module):
    """Adaptive decoder for CNN autoencoder.

    Dynamically builds upsampling layers to match the encoder's downsampling,
    mirroring the InitialCNNEncoder architecture in reverse.

    Args:
        embedding_dim: Input embedding dimension
        out_channels: Output channels
        spatial_size: Output spatial size (must be power of 2, e.g., 32, 64, 128, 256)
    """

    def __init__(self, embedding_dim: int, out_channels: int, spatial_size: int):
        super().__init__()

        # Validate spatial_size is power of 2 and reasonable
        import math
        if spatial_size & (spatial_size - 1) != 0 or spatial_size < 16:
            raise ValueError(f"spatial_size must be power of 2 and >= 16, got {spatial_size}")

        self.spatial_size = spatial_size
        self.out_channels = out_channels

        # Determine initial feature map channels (should scale with embedding_dim)
        # For 256-dim embedding → 256 channels, for 768-dim → 512 channels, etc.
        self.initial_channels = min(512, max(256, embedding_dim // 3))

        # Project embedding to initial spatial feature map
        self.fc = nn.Linear(embedding_dim, self.initial_channels * 4 * 4)

        # Calculate number of 4x upsampling stages needed to go from 4x4 to target size
        # E.g., for 64x64: 4x4 → 16x16 → 64x64 = 2 stages
        #       for 256x256: 4x4 → 16x16 → 64x64 → 256x256 = 3 stages
        current_size = 4
        num_stages = 0
        while current_size < spatial_size:
            current_size *= 4
            num_stages += 1

        if current_size != spatial_size:
            raise ValueError(
                f"Cannot build decoder: spatial_size {spatial_size} not reachable "
                f"from 4x4 with 4x upsampling stages"
            )

        # Build upsampling layers
        layers = []
        in_ch = self.initial_channels
        channel_progression = [self.initial_channels, self.initial_channels // 2, self.initial_channels // 4,
                              self.initial_channels // 8, max(32, self.initial_channels // 16)]

        for stage in range(num_stages):
            out_ch = channel_progression[min(stage + 1, len(channel_progression) - 1)]

            # Last stage outputs to target channels
            if stage == num_stages - 1:
                out_ch = out_channels
                layers.append(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=4, padding=0, bias=True)
                )
            else:
                layers.extend([
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=4, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ])

            in_ch = out_ch

        self.upsample = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode embedding to image.

        Args:
            x: Embedding [B, embedding_dim]

        Returns:
            Reconstructed image [B, out_channels, spatial_size, spatial_size]
        """
        h = self.fc(x)  # [B, initial_channels * 4 * 4]
        h = h.view(-1, self.initial_channels, 4, 4)  # [B, initial_channels, 4, 4]

        # Upsample to target size
        h = self.upsample(h)  # [B, out_channels, spatial_size, spatial_size]
        return h


class CNNPretrainer:
    """Pretrain CNN encoder via autoencoder reconstruction.

    Trains an autoencoder (encoder + decoder) to reconstruct initial conditions,
    then saves only the encoder weights for use in VQ-VAE training.

    Args:
        config: Pretraining configuration

    Example:
        >>> config = PretrainingConfig(embedding_dim=256, num_epochs=50)
        >>> pretrainer = CNNPretrainer(config)
        >>> pretrainer.train(ics, output_path="cnn_pretrained.pt")
    """

    def __init__(self, config: PretrainingConfig, in_channels: Optional[int] = None, spatial_size: int = 64):
        self.config = config
        self.use_augmentation = config.use_augmentation

        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Use runtime-detected in_channels if provided, otherwise fall back to config
        self.in_channels = in_channels if in_channels is not None else config.in_channels
        self.spatial_size = spatial_size

        # Create encoder and decoder
        self.encoder = InitialCNNEncoder(
            embedding_dim=config.embedding_dim,
            in_channels=self.in_channels,
            use_final_batchnorm=config.use_final_batchnorm,
        ).to(self.device)

        self.decoder = CNNDecoder(
            embedding_dim=config.embedding_dim,
            out_channels=self.in_channels,
            spatial_size=self.spatial_size,
        ).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

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
        initial_conditions: torch.Tensor,
        output_path: Path,
        val_initial_conditions: Optional[torch.Tensor] = None,
        normalization_stats: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Train autoencoder and save encoder checkpoint.

        Args:
            initial_conditions: Training ICs [N, C, H, W]
            output_path: Path to save encoder checkpoint
            val_initial_conditions: Optional validation ICs [N_val, C, H, W]
                If None, will use config.val_split to create validation set

        Returns:
            Training history dict with keys:
                - train_losses: List of training losses per epoch
                - val_losses: List of validation losses per epoch
                - best_epoch: Epoch with lowest validation loss
                - best_val_loss: Best validation loss achieved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create train/val split if needed
        if val_initial_conditions is None:
            dataset = TensorDataset(initial_conditions)
            val_size = int(len(dataset) * self.config.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        else:
            train_dataset = TensorDataset(initial_conditions)
            val_dataset = TensorDataset(val_initial_conditions)

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

        logger.info(f"CNN Pretraining on {self.device}")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Val samples: {len(val_dataset)}")
        logger.info(f"  Embedding dim: {self.config.embedding_dim}")
        logger.info(f"  Epochs: {self.config.num_epochs}")

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            train_losses.append(train_loss)

            # Validate
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            # Log progress
            if self.config.verbose and (epoch + 1) % self.config.log_every_n_epochs == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"LR: {lr:.6f}"
                )

        # Save checkpoint
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'embedding_dim': self.config.embedding_dim,
            'in_channels': self.in_channels,
            'use_final_batchnorm': self.config.use_final_batchnorm,
            'epoch': best_epoch,
            'val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'normalization_stats': normalization_stats,
        }

        torch.save(checkpoint, output_path)
        logger.info(f"Saved encoder checkpoint to {output_path}")
        logger.info(f"  Best epoch: {best_epoch+1}")
        logger.info(f"  Best val loss: {best_val_loss:.6f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random flips and rotations for data augmentation.

        Args:
            x: Input batch [B, C, H, W]

        Returns:
            Augmented batch [B, C, H, W]
        """
        # Random horizontal flip (50% probability)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])

        # Random vertical flip (50% probability)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-2])

        # Random 90° rotation (0°, 90°, 180°, or 270°)
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            x = torch.rot90(x, k, dims=[-2, -1])

        return x

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            loader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        self.encoder.train()
        self.decoder.train()

        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            x = batch[0].to(self.device)  # [B, C, H, W]

            # Apply augmentation (only during training)
            if self.use_augmentation:
                x = self._augment(x)

            # Forward pass
            embeddings = self.encoder(x)  # [B, embedding_dim]
            reconstructed = self.decoder(embeddings)  # [B, C, H, W]

            # Compute loss
            loss = self.criterion(reconstructed, x)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch.

        Args:
            loader: Validation data loader

        Returns:
            Average validation loss for the epoch
        """
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.device)

                # Forward pass
                embeddings = self.encoder(x)
                reconstructed = self.decoder(embeddings)

                # Compute loss
                loss = self.criterion(reconstructed, x)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches
