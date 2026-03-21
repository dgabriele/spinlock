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
from spinlock.experimental.diffusion.training.roundtrip_loss import (
    DenoisingRoundtripHead,
    DenoisingRoundtripLoss,
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
        self.global_epoch = 0  # never resets (curriculum-safe)
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

        # Trajectory probe counter (incremented each validate() call)
        self._validation_count = 0

        # Focal loss gamma (0 = standard CE)
        self._focal_gamma = config.training.focal_gamma
        if self._focal_gamma > 0:
            logger.info(f"Focal loss enabled: gamma={self._focal_gamma}")

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
        self._decode_head = None  # shared between physics_loss and roundtrip_loss
        if config.training.physics_loss.enabled:
            tokenizer_ckpt = config.dataset.tokenizer_checkpoint
            if tokenizer_ckpt is None:
                raise ValueError(
                    "physics_loss.enabled=True requires dataset.tokenizer_checkpoint"
                )
            self._decode_head = PhysicsDecodeHead.from_tokenizer_checkpoint(
                tokenizer_ckpt,
                families=config.training.physics_loss.families,
                device=self.device,
            )
            self.physics_loss = PhysicsAwareLoss(
                self._decode_head, config.training.physics_loss
            )
            self.physics_loss.to(self.device)

        # Denoising roundtrip consistency loss (frozen VQ re-encode pipeline)
        self.roundtrip_loss = None
        if config.training.roundtrip_loss.enabled:
            tokenizer_ckpt = config.dataset.tokenizer_checkpoint
            if tokenizer_ckpt is None:
                raise ValueError(
                    "roundtrip_loss.enabled=True requires dataset.tokenizer_checkpoint"
                )
            aux_trunc = config.dataset.aux_truncation_lengths
            if not aux_trunc:
                raise ValueError(
                    "roundtrip_loss.enabled=True requires "
                    "dataset.aux_truncation_lengths to be set"
                )
            # Reuse decode_head if already created for physics_loss
            if self._decode_head is None:
                self._decode_head = PhysicsDecodeHead.from_tokenizer_checkpoint(
                    tokenizer_ckpt, device=self.device,
                )
            rt_head = DenoisingRoundtripHead.from_tokenizer_checkpoint(
                tokenizer_ckpt,
                decode_head=self._decode_head,
                device=self.device,
            )
            # Truncation levels: aux + primary (if set)
            trunc_levels = sorted(set(aux_trunc))
            if config.dataset.truncation_length is not None:
                trunc_levels = sorted(set(trunc_levels) | {config.dataset.truncation_length})
            self.roundtrip_loss = DenoisingRoundtripLoss(
                rt_head, config.training.roundtrip_loss, trunc_levels,
            )
            self.roundtrip_loss.to(self.device)

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
                self.global_epoch += 1

                # Train epoch
                train_metrics = self.train_epoch()

                # Check for interrupt after epoch
                if self._interrupt_requested:
                    self._handle_interrupt(num_epochs)
                    break

                # Validate
                if epoch % self.config.training.val_frequency == 0:
                    val_metrics = self.validate()

                    # Log headline metrics
                    val_log = (
                        f"Epoch {self.current_epoch}/{num_epochs}: "
                        f"train_loss={train_metrics['loss']:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.4f}"
                    )
                    if 'physics_loss' in val_metrics:
                        val_log += f", val_phys={val_metrics['physics_loss']:.6f}"
                    if 'roundtrip_loss' in val_metrics:
                        val_log += f", val_rt={val_metrics['roundtrip_loss']:.6f}"
                    logger.info(val_log)

                    # Log diagnostic breakdowns
                    noise_parts = []
                    for band in ['low', 'mid', 'high']:
                        k = f'acc_{band}_noise'
                        if k in val_metrics:
                            noise_parts.append(f"{band}={val_metrics[k]:.4f}")
                    if noise_parts:
                        logger.info(f"  Noise bands: {', '.join(noise_parts)}")

                    family_parts = []
                    for k, v in sorted(val_metrics.items()):
                        if k.startswith('acc_') and not k.endswith('_noise') and k != 'accuracy':
                            family_parts.append(f"{k[4:]}={v:.4f}")
                    if family_parts:
                        logger.info(f"  Families: {', '.join(family_parts)}")

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
                        if 'roundtrip_loss' in val_metrics:
                            wandb_metrics['val_roundtrip_loss'] = val_metrics['roundtrip_loss']
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
        if self.roundtrip_loss is not None:
            self.roundtrip_loss.eval()
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

            # Denoising roundtrip consistency loss (after warmup)
            roundtrip_loss_val = 0.0
            if (
                self.roundtrip_loss is not None
                and self.current_epoch > self.config.training.roundtrip_loss.warmup_epochs
            ):
                aux_trunc = {
                    tl: {k: v.to(self.device) for k, v in trunc_dict.items()}
                    for tl, trunc_dict in batch.get('aux_trunc_tokens', {}).items()
                }
                rt_loss = self.roundtrip_loss(
                    predicted_logits, tokens, aux_trunc, target, t,
                    eff_t_dict, T=self.diffusion.schedule.num_timesteps,
                )
                loss = loss + self.config.training.roundtrip_loss.weight * rt_loss
                roundtrip_loss_val = rt_loss.item()

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
                if roundtrip_loss_val > 0:
                    log_msg += f", rt={roundtrip_loss_val:.6f}"

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

            # Focal loss: down-weight easy predictions by (1-p_t)^γ
            if self._focal_gamma > 0:
                with torch.no_grad():
                    p_t = F.softmax(logits, dim=-1)  # [B, V]
                    p_correct = p_t.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]
                    focal_weight = (1 - p_correct) ** self._focal_gamma  # [B]
                loss = loss * focal_weight

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
        """Run validation with detailed diagnostic breakdowns.

        Returns:
            Dict with metrics including:
            - loss, accuracy: overall
            - acc_high_noise: accuracy at t > 2T/3 (hardest regime)
            - acc_<family>: per-family accuracy (temporal, initial, theta)
            - physics_loss: optional
        """
        self.denoiser.eval()
        val_loss = 0.0
        val_physics_loss = 0.0
        val_roundtrip_loss = 0.0
        num_batches = 0

        # Accumulate detailed accuracy counters
        total_correct = 0
        total_count = 0

        # Per-noise-band: low (t < T/3), mid (T/3 <= t < 2T/3), high (t >= 2T/3)
        T = self.diffusion.schedule.num_timesteps
        band_correct = {'low': 0, 'mid': 0, 'high': 0}
        band_count = {'low': 0, 'mid': 0, 'high': 0}

        # Per-family
        cat_info = self.diffusion.category_level_info
        family_correct: Dict[str, int] = {}
        family_count: Dict[str, int] = {}

        physics_active = (
            self.physics_loss is not None
            and self.current_epoch > self.config.training.physics_loss.warmup_epochs
        )
        roundtrip_active = (
            self.roundtrip_loss is not None
            and self.current_epoch > self.config.training.roundtrip_loss.warmup_epochs
        )

        for batch in self.val_loader:
            tokens = {k: v.to(self.device) for k, v in batch['tokens'].items()}
            observed = {k: v.to(self.device) for k, v in batch['observed'].items()}
            target = {k: v.to(self.device) for k, v in batch['target'].items()}

            batch_size = next(iter(tokens.values())).shape[0]
            t = torch.randint(0, T, (batch_size,), device=self.device)

            eff_t_dict = (
                self.diffusion.compute_effective_timesteps(t, batch_size)
                if self._graded_enabled else None
            )

            noisy_tokens, _ = self.diffusion.forward_process(
                tokens, t, mask_dict=target,
                effective_timesteps_dict=eff_t_dict,
            )

            predicted_logits = self.denoiser(
                noisy_tokens, t, observed_dict=observed,
                **({"effective_timesteps_dict": eff_t_dict} if eff_t_dict is not None else {}),
            )

            loss = self._compute_loss(
                predicted_logits, tokens, target, t=t,
                effective_timesteps_dict=eff_t_dict,
            )
            val_loss += loss.item()

            if physics_active:
                p_loss = self.physics_loss(
                    predicted_logits, tokens, target, t,
                    T=T,
                )
                val_physics_loss += p_loss.item()

            if roundtrip_active:
                aux_trunc = {
                    tl: {k: v.to(self.device) for k, v in trunc_dict.items()}
                    for tl, trunc_dict in batch.get('aux_trunc_tokens', {}).items()
                }
                rt_loss = self.roundtrip_loss(
                    predicted_logits, tokens, aux_trunc, target, t,
                    eff_t_dict, T=T,
                )
                val_roundtrip_loss += rt_loss.item()

            # Noise band masks [B]
            band_low = t < (T // 3)
            band_mid = (t >= T // 3) & (t < 2 * T // 3)
            band_high = t >= (2 * T // 3)

            # Accumulate per-key accuracy with band and family breakdowns
            for key in predicted_logits.keys():
                logits = predicted_logits[key]
                targets = tokens[key]
                mask = target[key]

                preds = torch.argmax(logits, dim=-1)
                correct = (preds == targets) & mask  # [B]
                n_masked = mask.sum().item()

                total_correct += correct.sum().item()
                total_count += n_masked

                # Per-band (intersect correct/mask with band membership)
                for band_name, band_mask in [('low', band_low), ('mid', band_mid), ('high', band_high)]:
                    bm = correct & band_mask  # correct AND in this band AND masked
                    bc = mask & band_mask       # masked AND in this band
                    band_correct[band_name] += bm.sum().item()
                    band_count[band_name] += bc.sum().item()

                # Per-family
                family = cat_info.get(key, {}).get('family', 'unknown')
                family_correct[family] = family_correct.get(family, 0) + correct.sum().item()
                family_count[family] = family_count.get(family, 0) + n_masked

            num_batches += 1

        avg_loss = val_loss / num_batches
        avg_accuracy = total_correct / total_count if total_count > 0 else 0.0

        metrics = {'loss': avg_loss, 'accuracy': avg_accuracy}

        # Noise band accuracies
        for band_name in ['low', 'mid', 'high']:
            bc = band_count[band_name]
            metrics[f'acc_{band_name}_noise'] = (
                band_correct[band_name] / bc if bc > 0 else 0.0
            )

        # Per-family accuracies
        for family in sorted(family_count.keys()):
            fc = family_count[family]
            metrics[f'acc_{family}'] = (
                family_correct[family] / fc if fc > 0 else 0.0
            )

        if physics_active:
            metrics['physics_loss'] = val_physics_loss / num_batches
        if roundtrip_active:
            metrics['roundtrip_loss'] = val_roundtrip_loss / num_batches

        # Trajectory probe: sample denoising trajectories and measure
        # per-step agreement against truncation-matched ground truth
        self._validation_count += 1
        rt_cfg = self.config.training.roundtrip_loss
        if (
            rt_cfg.trajectory_probe_frequency > 0
            and self._validation_count % rt_cfg.trajectory_probe_frequency == 0
            and self.roundtrip_loss is not None
        ):
            self._run_trajectory_probe(metrics)

        return metrics

    def _run_trajectory_probe(self, metrics: Dict[str, float]):
        """Sample denoising trajectories and log per-step agreement.

        Takes a small batch from val_loader, runs sample() with snapshot
        recording at representative denoising steps, and measures how well
        each snapshot's tokens agree with the nearest truncation-level GT.

        Args:
            metrics: Validation metrics dict (updated in-place with probe results).
        """
        rt_cfg = self.config.training.roundtrip_loss
        n_probe = rt_cfg.trajectory_probe_samples
        T = self.diffusion.schedule.num_timesteps

        # Representative probe steps: ~80%, 60%, 40%, 20% of T
        probe_steps = sorted({
            max(1, int(T * f)) for f in [0.8, 0.6, 0.4, 0.2]
        }, reverse=True)

        # Grab a batch from val_loader
        try:
            probe_batch = next(iter(self.val_loader))
        except StopIteration:
            return

        tokens = {k: v[:n_probe].to(self.device) for k, v in probe_batch['tokens'].items()}
        aux_trunc = {}
        for tl, trunc_dict in probe_batch.get('aux_trunc_tokens', {}).items():
            aux_trunc[tl] = {k: v[:n_probe].to(self.device) for k, v in trunc_dict.items()}

        if not aux_trunc:
            return

        # Run sample() with snapshot recording
        result = self.diffusion.sample(
            batch_size=n_probe,
            denoising_network=self.denoiser,
            device=self.device,
            snapshot_steps=probe_steps,
        )

        if not isinstance(result, tuple):
            return
        final_tokens, trajectory = result

        # Temporal keys only (matching roundtrip loss scope)
        temporal_keys = [
            k for k in final_tokens
            if self.diffusion.category_level_info.get(k, {}).get('family') == 'temporal'
        ]
        if not temporal_keys:
            return

        trunc_levels = self.roundtrip_loss.truncation_levels

        # For each snapshot step, find best-matching truncation and compute agreement
        probe_parts = []
        for step in sorted(trajectory.keys(), reverse=True):
            snapshot = trajectory[step]

            # Compute noise fraction → truncation index (same mapping as roundtrip loss)
            noise_frac = step / T
            inv_frac = 1.0 - noise_frac
            trunc_idx = 0
            for boundary in self.roundtrip_loss._boundaries:
                if inv_frac > boundary:
                    trunc_idx += 1
            trunc_idx = min(trunc_idx, len(trunc_levels) - 1)
            matched_trunc = trunc_levels[trunc_idx]

            # Get GT tokens at this truncation level
            gt = aux_trunc.get(matched_trunc, {})
            if not gt:
                continue

            # Per-position agreement across temporal keys
            n_agree = 0
            n_total = 0
            for key in temporal_keys:
                if key in snapshot and key in gt:
                    n_agree += (snapshot[key] == gt[key]).sum().item()
                    n_total += gt[key].numel()

            if n_total > 0:
                agree = n_agree / n_total
                probe_parts.append(f"t={step}→T{matched_trunc} agree={agree:.3f}")
                metrics[f'probe_t{step}_agree'] = agree

        if probe_parts:
            logger.info(f"  Trajectory probe: {', '.join(probe_parts)}")

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_epoch': self.global_epoch,
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
            path = self.output_dir / f"{prefix}_epoch{self.global_epoch}.pt"

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
        self.global_epoch = checkpoint.get('global_epoch', checkpoint['epoch'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        logger.info(f"Checkpoint loaded: {checkpoint_path}, epoch={self.current_epoch}")
