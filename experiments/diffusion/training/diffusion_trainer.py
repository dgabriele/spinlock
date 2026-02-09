"""Training loop for discrete diffusion on hierarchical dict tokens."""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from experiments.diffusion.models import DiscreteD3PM, DenoisingNetwork
from experiments.diffusion.config import DiffusionExperimentConfig

logger = logging.getLogger(__name__)


class DiffusionTrainer:
    """Trainer for discrete diffusion token completion.

    Implements training loop with:
    - Per-timestep loss computation
    - Per-category-level cross-entropy on target positions
    - Learning rate scheduling with warmup
    - Checkpointing and validation
    - Optional wandb logging

    Args:
        diffusion_model: DiscreteD3PM instance
        denoising_network: DenoisingNetwork instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Training configuration (DiffusionExperimentConfig)
        output_dir: Directory for checkpoints and logs
        device: Device for training

    Example:
        >>> trainer = DiffusionTrainer(diffusion, denoiser, train_loader, val_loader, config)
        >>> history = trainer.train(num_epochs=30)
    """

    def __init__(
        self,
        diffusion_model: DiscreteD3PM,
        denoising_network: DenoisingNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DiffusionExperimentConfig,
        output_dir: Path,
        device: str = "cuda",
    ):
        self.diffusion = diffusion_model
        self.denoiser = denoising_network
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device

        # Move models to device
        self.diffusion.to(device)
        self.denoiser.to(device)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        # Optional wandb logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.use_wandb = False

        logger.info(f"DiffusionTrainer initialized: device={device}, output_dir={output_dir}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer."""
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        optimizer = AdamW(
            self.denoiser.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        logger.info(f"Optimizer created: lr={lr}, weight_decay={weight_decay}")
        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler with warmup."""
        lr_config = self.config.training.lr_scheduler

        if lr_config.type != 'cosine':
            return None

        warmup_epochs = lr_config.warmup_epochs
        min_lr = lr_config.min_lr
        num_epochs = self.config.training.num_epochs

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs * len(self.train_loader),
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(num_epochs - warmup_epochs) * len(self.train_loader),
            eta_min=min_lr,
        )

        # Sequential scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * len(self.train_loader)],
        )

        logger.info(f"LR scheduler created: warmup={warmup_epochs}ep, min_lr={min_lr}")
        return scheduler

    def train(self, num_epochs: int) -> Dict[str, list]:
        """Run full training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training history dict
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate
            if epoch % self.config.training.val_frequency == 0:
                val_metrics = self.validate()

                # Log
                logger.info(
                    f"Epoch {self.current_epoch}/{num_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

                # Track history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )

                # Wandb logging
                if self.use_wandb:
                    self.wandb.log({
                        'epoch': self.current_epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                    })

                # Save best model
                if self.config.training.save_best and val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best model saved (val_loss={self.best_val_loss:.4f})")

            # Periodic checkpoint
            if epoch % self.config.training.checkpoint_frequency == 0:
                self.save_checkpoint(is_best=False)

        logger.info("Training complete")
        return self.history

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict with epoch metrics
        """
        self.denoiser.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            tokens = {k: v.to(self.device) for k, v in batch['tokens'].items()}
            observed = {k: v.to(self.device) for k, v in batch['observed'].items()}
            target = {k: v.to(self.device) for k, v in batch['target'].items()}

            # Sample random timesteps for each batch element
            batch_size = next(iter(tokens.values())).shape[0]
            t = torch.randint(
                0, self.diffusion.schedule.num_timesteps, (batch_size,), device=self.device
            )

            # Forward diffusion: add noise to tokens
            noisy_tokens, _ = self.diffusion.forward_process(tokens, t[0].item(), mask_dict=target)

            # Predict clean tokens
            predicted_logits = self.denoiser(noisy_tokens, t, observed_dict=observed)

            # Compute loss on target positions only
            loss = self._compute_loss(predicted_logits, tokens, target)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip_norm:
                nn.utils.clip_grad_norm_(
                    self.denoiser.parameters(),
                    self.config.training.gradient_clip_norm
                )

            self.optimizer.step()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log
            if batch_idx % self.config.training.log_frequency == 0:
                logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                    f"loss={loss.item():.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )

        avg_loss = epoch_loss / num_batches
        return {'loss': avg_loss}

    def _compute_loss(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        target_mask: Dict[str, torch.BoolTensor],
    ) -> torch.Tensor:
        """Compute per-category-level cross-entropy on target positions.

        Args:
            predicted_logits: Dict mapping key → logits [B, V]
            target_tokens: Dict mapping key → true tokens [B]
            target_mask: Dict mapping key → mask [B] (True = predict this position)

        Returns:
            Scalar loss tensor
        """
        total_loss = 0.0
        total_count = 0

        for key in predicted_logits.keys():
            logits = predicted_logits[key]  # [B, V]
            targets = target_tokens[key]  # [B]
            mask = target_mask[key]  # [B]

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, targets, reduction='none')  # [B]

            # Apply mask (only count target positions)
            masked_loss = loss * mask.float()

            # Accumulate
            total_loss += masked_loss.sum()
            total_count += mask.sum()

        # Average over all target positions
        if total_count > 0:
            avg_loss = total_loss / total_count
        else:
            avg_loss = total_loss

        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dict with validation metrics
        """
        self.denoiser.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Move to device
            tokens = {k: v.to(self.device) for k, v in batch['tokens'].items()}
            observed = {k: v.to(self.device) for k, v in batch['observed'].items()}
            target = {k: v.to(self.device) for k, v in batch['target'].items()}

            # Sample random timesteps
            batch_size = next(iter(tokens.values())).shape[0]
            t = torch.randint(
                0, self.diffusion.schedule.num_timesteps, (batch_size,), device=self.device
            )

            # Forward diffusion
            noisy_tokens, _ = self.diffusion.forward_process(tokens, t[0].item(), mask_dict=target)

            # Predict
            predicted_logits = self.denoiser(noisy_tokens, t, observed_dict=observed)

            # Compute loss
            loss = self._compute_loss(predicted_logits, tokens, target)
            val_loss += loss.item()

            # Compute accuracy on target positions
            accuracy = self._compute_accuracy(predicted_logits, tokens, target)
            val_accuracy += accuracy

            num_batches += 1

        avg_loss = val_loss / num_batches
        avg_accuracy = val_accuracy / num_batches

        return {'loss': avg_loss, 'accuracy': avg_accuracy}

    def _compute_accuracy(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        target_mask: Dict[str, torch.BoolTensor],
    ) -> float:
        """Compute token accuracy on target positions.

        Args:
            predicted_logits: Dict mapping key → logits [B, V]
            target_tokens: Dict mapping key → true tokens [B]
            target_mask: Dict mapping key → mask [B] (True = predict this position)

        Returns:
            Accuracy as float (0 to 1)
        """
        total_correct = 0
        total_count = 0

        for key in predicted_logits.keys():
            logits = predicted_logits[key]  # [B, V]
            targets = target_tokens[key]  # [B]
            mask = target_mask[key]  # [B]

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)  # [B]

            # Compute accuracy on target positions
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_count += mask.sum().item()

        accuracy = total_correct / total_count if total_count > 0 else 0.0
        return accuracy

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'denoiser_state_dict': self.denoiser.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'history': self.history,
        }

        prefix = self.config.output.checkpoint_prefix

        if is_best:
            path = self.output_dir / f"{prefix}_best.pt"
        else:
            path = self.output_dir / f"{prefix}_epoch{self.current_epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.denoiser.load_state_dict(checkpoint['denoiser_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        logger.info(f"Checkpoint loaded: {checkpoint_path}, epoch={self.current_epoch}")
