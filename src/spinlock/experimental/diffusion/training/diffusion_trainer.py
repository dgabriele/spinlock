"""Training loop for discrete diffusion on hierarchical dict tokens."""

import logging
import signal
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

from spinlock.experimental.diffusion.models import DiscreteD3PM, DenoisingNetwork
from spinlock.experimental.diffusion.config import DiffusionExperimentConfig
from spinlock.experimental.diffusion.training.physics_loss import (
    PhysicsDecodeHead,
    PhysicsAwareLoss,
)

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

        # Cache SNR weights if enabled
        self.snr_weights = None
        if config.training.use_snr_weighting:
            snr = self.diffusion.get_timestep_weights()  # [T], normalized mean=1
            self.snr_weights = snr.to(device)

        # Cache vocab-size loss weights if enabled
        self.vocab_loss_weights = {}
        if config.training.use_vocab_size_weighting:
            v_max = max(self.diffusion.vocab_sizes.values())
            log_v_max = math.log(v_max)
            for key, v in self.diffusion.vocab_sizes.items():
                self.vocab_loss_weights[key] = math.log(v) / log_v_max

        # Cache graded schedule flag
        self._graded_enabled = bool(self.diffusion._key_scale_factors)

        # Optional wandb logging
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, disabling logging")
                self.use_wandb = False

        # Physics-aware auxiliary loss (frozen tokenizer decode pipeline)
        self.physics_loss = None
        if config.training.physics_loss.enabled:
            tokenizer_ckpt = config.dataset.tokenizer_checkpoint
            if tokenizer_ckpt is None:
                raise ValueError(
                    "physics_loss.enabled=True requires dataset.tokenizer_checkpoint"
                )
            decode_head = PhysicsDecodeHead.from_tokenizer_checkpoint(
                tokenizer_ckpt,
                families=config.training.physics_loss.families,
                device=self.device,
            )
            self.physics_loss = PhysicsAwareLoss(
                decode_head, config.training.physics_loss
            )
            self.physics_loss.to(self.device)

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
        """Create learning rate scheduler with warmup and cosine annealing."""
        lr_config = self.config.training.lr_scheduler

        if lr_config.type != 'cosine':
            return None

        warmup_epochs = lr_config.warmup_epochs
        min_lr = lr_config.min_lr
        num_epochs = self.config.training.num_epochs
        base_lr = self.config.training.learning_rate

        warmup_steps = warmup_epochs * len(self.train_loader)
        total_steps = num_epochs * len(self.train_loader)
        cosine_steps = total_steps - warmup_steps

        def lr_lambda(current_step: int) -> float:
            """Compute learning rate multiplier for current step.

            Combines linear warmup with cosine annealing:
            - Steps 0 to warmup_steps: linear warmup from 0.1 to 1.0
            - Steps warmup_steps to total_steps: cosine annealing from 1.0 to min_lr
            """
            if current_step < warmup_steps:
                # Linear warmup: 0.1 → 1.0
                return 0.1 + 0.9 * (current_step / warmup_steps)
            else:
                # Cosine annealing: 1.0 → min_lr/base_lr
                progress = (current_step - warmup_steps) / cosine_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine_decay

        scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(f"LR scheduler created: warmup={warmup_epochs}ep, min_lr={min_lr}")
        return scheduler


    def train(self, num_epochs: int) -> Dict[str, list]:
        """Run full training loop.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Training history dict
        """
        start_epoch = self.current_epoch  # >0 when resuming from checkpoint
        logger.info(f"Starting training for {num_epochs} epochs (from epoch {start_epoch + 1})")

        # Ctrl+C graceful shutdown: prompt user for validate+save
        self._interrupt_requested = False
        original_sigint = signal.getsignal(signal.SIGINT)

        def _sigint_handler(signum, frame):
            if self._interrupt_requested:
                # Second Ctrl+C: hard exit
                logger.info("Second interrupt — exiting immediately")
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            self._interrupt_requested = True
            logger.info(
                "\nInterrupt received. Will stop after current batch. "
                "Press Ctrl+C again to force quit."
            )

        signal.signal(signal.SIGINT, _sigint_handler)

        try:
            for epoch in range(start_epoch, num_epochs):
                self.current_epoch = epoch + 1

                # Train epoch
                train_metrics = self.train_epoch()

                # Check for interrupt after epoch
                if self._interrupt_requested:
                    self._handle_interrupt(num_epochs)
                    break

                # Validate
                if epoch % self.config.training.val_frequency == 0:
                    val_metrics = self.validate()

                    # Log
                    val_log = (
                        f"Epoch {self.current_epoch}/{num_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.4f}"
                    )
                    if 'physics_loss' in val_metrics:
                        val_log += f", val_phys={val_metrics['physics_loss']:.6f}"
                    logger.info(val_log)

                    # Track history
                    self.history['train_loss'].append(train_metrics['loss'])
                    self.history['val_loss'].append(val_metrics['loss'])
                    self.history['learning_rate'].append(
                        self.optimizer.param_groups[0]['lr']
                    )

                    # Wandb logging
                    if self.use_wandb:
                        wandb_metrics = {
                            'epoch': self.current_epoch,
                            'train_loss': train_metrics['loss'],
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy'],
                            'learning_rate': self.optimizer.param_groups[0]['lr'],
                        }
                        if 'physics_loss' in val_metrics:
                            wandb_metrics['val_physics_loss'] = val_metrics['physics_loss']
                        self.wandb.log(wandb_metrics)

                    # Save best model
                    if self.config.training.save_best and val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint(is_best=True)
                        logger.info(f"New best model saved (val_loss={self.best_val_loss:.4f})")

                # Periodic checkpoint
                if epoch % self.config.training.checkpoint_frequency == 0:
                    self.save_checkpoint(is_best=False)

            logger.info("Training complete")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_sigint)

        return self.history

    def _handle_interrupt(self, num_epochs: int):
        """Handle graceful interruption: prompt user for validate+save."""
        logger.info(f"\nTraining interrupted at epoch {self.current_epoch}/{num_epochs}")
        try:
            response = input("Run validation and save checkpoint? [Y/n] ").strip().lower()
        except EOFError:
            response = "y"

        if response in ("", "y", "yes"):
            logger.info("Running validation...")
            val_metrics = self.validate()
            logger.info(
                f"Interrupt checkpoint — val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            self.save_checkpoint(is_best=False)
            if self.config.training.save_best and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=True)
                logger.info(f"New best model saved (val_loss={self.best_val_loss:.4f})")
            logger.info("Checkpoint saved. Exiting.")
        else:
            logger.info("Skipping validation. Exiting.")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dict with epoch metrics
        """
        self.denoiser.train()
        if self.physics_loss is not None:
            self.physics_loss.eval()
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

            # Compute per-key effective timesteps (graded schedule)
            eff_t_dict = (
                self.diffusion.compute_effective_timesteps(t, batch_size)
                if self._graded_enabled else None
            )

            # Forward diffusion: add noise to tokens (per-sample timesteps)
            noisy_tokens, _ = self.diffusion.forward_process(
                tokens, t, mask_dict=target,
                effective_timesteps_dict=eff_t_dict,
            )

            # Predict clean tokens
            predicted_logits = self.denoiser(
                noisy_tokens, t, observed_dict=observed,
                **({"effective_timesteps_dict": eff_t_dict} if eff_t_dict is not None else {}),
            )

            # Compute loss on target positions only (with optional SNR/vocab weighting)
            loss = self._compute_loss(
                predicted_logits, tokens, target, t=t,
                effective_timesteps_dict=eff_t_dict,
            )

            # Physics-aware auxiliary loss (after warmup)
            physics_loss_val = 0.0
            if (
                self.physics_loss is not None
                and self.current_epoch > self.config.training.physics_loss.warmup_epochs
            ):
                p_loss = self.physics_loss(
                    predicted_logits, tokens, target, t,
                    T=self.diffusion.schedule.num_timesteps,
                )
                loss = loss + self.config.training.physics_loss.weight * p_loss
                physics_loss_val = p_loss.item()

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
                log_msg = (
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                    f"loss={loss.item():.4f}, lr={self.optimizer.param_groups[0]['lr']:.6f}"
                )
                if physics_loss_val > 0:
                    log_msg += f", phys={physics_loss_val:.6f}"

                logger.info(log_msg)

            # Check for graceful interrupt
            if getattr(self, '_interrupt_requested', False):
                logger.info(f"Stopping epoch early at batch {batch_idx}/{len(self.train_loader)}")
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    def _compute_loss(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        target_mask: Dict[str, torch.BoolTensor],
        t: Optional[torch.Tensor] = None,
        effective_timesteps_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute per-category-level cross-entropy on target positions.

        Supports optional SNR timestep weighting and vocab-size normalization.
        When graded schedule is active, SNR weighting uses per-key effective
        timesteps so the loss reflects each key's actual noise level.

        Args:
            predicted_logits: Dict mapping key → logits [B, V]
            target_tokens: Dict mapping key → true tokens [B]
            target_mask: Dict mapping key → mask [B] (True = predict this position)
            t: Optional timestep tensor [B] for SNR weighting
            effective_timesteps_dict: Optional per-key effective timesteps
                from graded schedule. When provided and SNR weighting is
                enabled, each key's loss is weighted by 1/β_{eff_t(key)}
                instead of 1/β_{global_t}.

        Returns:
            Scalar loss tensor
        """
        B = next(iter(predicted_logits.values())).shape[0]
        device = next(iter(predicted_logits.values())).device
        per_sample_loss = torch.zeros(B, device=device)
        per_sample_count = torch.zeros(B, device=device)

        # Per-key SNR weighting when graded schedule is active
        use_per_key_snr = (
            self.snr_weights is not None
            and t is not None
            and effective_timesteps_dict is not None
        )

        for key in predicted_logits.keys():
            logits = predicted_logits[key]  # [B, V]
            targets = target_tokens[key]  # [B]
            mask = target_mask[key]  # [B]

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, targets, reduction='none')  # [B]

            # Apply vocab-size weighting
            w_v = self.vocab_loss_weights.get(key, 1.0)

            # Apply per-key SNR weighting (graded: each key at its effective t)
            if use_per_key_snr:
                key_t = effective_timesteps_dict[key]  # [B]
                loss = loss * self.snr_weights[key_t]  # [B]

            # Accumulate per-sample weighted loss
            per_sample_loss = per_sample_loss + loss * mask.float() * w_v
            per_sample_count = per_sample_count + mask.float() * w_v

        # Normalize per sample
        per_sample_loss = per_sample_loss / per_sample_count.clamp(min=1.0)

        # Apply global SNR timestep weighting (only when NOT using per-key SNR)
        if self.snr_weights is not None and t is not None and not use_per_key_snr:
            per_sample_loss = per_sample_loss * self.snr_weights[t]

        return per_sample_loss.mean()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dict with validation metrics (loss, accuracy, and optionally physics_loss).
            When using temporal_causal masking, also includes per-cutoff breakdowns.
        """
        self.denoiser.eval()
        val_loss = 0.0
        val_physics_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0

        physics_active = (
            self.physics_loss is not None
            and self.current_epoch > self.config.training.physics_loss.warmup_epochs
        )

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

            # Compute per-key effective timesteps (graded schedule)
            eff_t_dict = (
                self.diffusion.compute_effective_timesteps(t, batch_size)
                if self._graded_enabled else None
            )

            # Forward diffusion (per-sample timesteps)
            noisy_tokens, _ = self.diffusion.forward_process(
                tokens, t, mask_dict=target,
                effective_timesteps_dict=eff_t_dict,
            )

            # Predict
            predicted_logits = self.denoiser(
                noisy_tokens, t, observed_dict=observed,
                **({"effective_timesteps_dict": eff_t_dict} if eff_t_dict is not None else {}),
            )

            # Compute loss (with optional SNR/vocab weighting)
            loss = self._compute_loss(
                predicted_logits, tokens, target, t=t,
                effective_timesteps_dict=eff_t_dict,
            )
            val_loss += loss.item()

            # Physics loss (monitoring only, no gradient in validation)
            if physics_active:
                p_loss = self.physics_loss(
                    predicted_logits, tokens, target, t,
                    T=self.diffusion.schedule.num_timesteps,
                )
                val_physics_loss += p_loss.item()

            # Compute accuracy on target positions
            accuracy = self._compute_accuracy(predicted_logits, tokens, target)
            val_accuracy += accuracy

            num_batches += 1

        avg_loss = val_loss / num_batches
        avg_accuracy = val_accuracy / num_batches

        metrics = {'loss': avg_loss, 'accuracy': avg_accuracy}
        if physics_active:
            metrics['physics_loss'] = val_physics_loss / num_batches

        return metrics

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
