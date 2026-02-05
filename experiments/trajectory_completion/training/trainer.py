"""Trainer for trajectory completion model."""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from tqdm import tqdm

from experiments.common.training.trainer import BaseExperimentTrainer


class CompletionTrainer(BaseExperimentTrainer):
    """Trainer for trajectory completion model."""

    def __init__(self, model, train_loader, val_loader, config, output_dir, vqvae):
        super().__init__(model, train_loader, val_loader, config, output_dir)
        self.vqvae = vqvae

    def _compute_batch_loss_and_accuracy(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and accuracy for a batch.

        Args:
            outputs: Model outputs dict with 'logits' and 'predictions'
            targets: [batch, seq_len] ground truth token indices
            mask_target: [batch, seq_len] bool mask for target positions

        Returns:
            (loss, accuracy) both scalar tensors
        """
        logits = outputs['logits']  # [batch, seq_len, max_vocab_size]
        predictions = outputs['predictions']  # [batch, seq_len]

        loss = 0.0
        accuracy = 0.0
        num_positions = 0

        # Get codebook sizes per level
        num_levels = 3
        codebook_sizes = self.vqvae.codebook_sizes

        for i in range(logits.shape[1]):
            if mask_target[:, i].any():
                # Get vocabulary size for this position
                level_idx = i % num_levels
                vocab_size = codebook_sizes[level_idx]

                # Loss for this position (only consider valid vocab range)
                loss_i = F.cross_entropy(
                    logits[:, i, :vocab_size],
                    targets[:, i],
                    reduction='none'
                )
                loss_i = (loss_i * mask_target[:, i]).sum() / mask_target[:, i].sum()
                loss = loss + loss_i

                # Accuracy for this position
                correct = (predictions[:, i] == targets[:, i]) & mask_target[:, i]
                accuracy += correct.float().sum() / mask_target[:, i].sum()

                num_positions += 1

        # Average over positions
        loss = loss / num_positions if num_positions > 0 else loss
        accuracy = accuracy / num_positions if num_positions > 0 else accuracy

        return loss, accuracy

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        for batch in pbar:
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                tokens_observed=batch['tokens_observed'].to(self.device),
                mask_observed=batch['mask_observed'].to(self.device),
                mask_target=batch['mask_target'].to(self.device)
            )

            # Compute loss and accuracy
            loss, accuracy = self._compute_batch_loss_and_accuracy(
                outputs=outputs,
                targets=batch['tokens_full'].to(self.device),
                mask_target=batch['mask_target'].to(self.device)
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item(), 'acc': accuracy.item()})

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_reconstruction_error = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
            for batch in pbar:
                # Forward pass
                outputs = self.model(
                    tokens_observed=batch['tokens_observed'].to(self.device),
                    mask_observed=batch['mask_observed'].to(self.device),
                    mask_target=batch['mask_target'].to(self.device)
                )

                # Compute loss and accuracy
                loss, accuracy = self._compute_batch_loss_and_accuracy(
                    outputs=outputs,
                    targets=batch['tokens_full'].to(self.device),
                    mask_target=batch['mask_target'].to(self.device)
                )

                # Reconstruction error (decode completed tokens)
                tokens_completed = outputs['tokens_completed']
                features_recon = self.vqvae.decode(tokens_completed)
                features_true = batch['features_full'].to(self.device)
                recon_error = F.mse_loss(features_recon, features_true)

                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_reconstruction_error += recon_error.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': accuracy.item(),
                    'recon': recon_error.item()
                })

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'reconstruction_error': total_reconstruction_error / num_batches
        }
