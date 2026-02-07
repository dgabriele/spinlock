"""VQ Tokenizer training orchestration.

Handles the complete training loop for JointHierarchicalVQVAE including:
- Multi-family data loading and batching
- Variable-length sequence handling
- Training and validation loops
- Dead code reset
- Early stopping
- Checkpoint management
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import JointHierarchicalVQVAE
from .losses import VQVAELoss
from .config import TokenizerConfig
from .checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class VQTokenizerTrainer:
    """Training orchestrator for VQ tokenizer.

    Manages the complete training pipeline including data loading,
    optimization, validation, and checkpointing.

    Args:
        model: JointHierarchicalVQVAE instance
        config: Complete tokenizer configuration
        group_indices: Dict mapping family_category → feature indices
        normalization_stats: Optional normalization statistics

    Example:
        >>> config = TokenizerConfig(...)
        >>> model = JointHierarchicalVQVAE(config, group_indices)
        >>> trainer = VQTokenizerTrainer(model, config, group_indices)
        >>> trainer.train(dataset, output_dir="checkpoints/")
    """

    def __init__(
        self,
        model: JointHierarchicalVQVAE,
        config: TokenizerConfig,
        group_indices: Dict[str, list],
        normalization_stats: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.group_indices = group_indices
        self.normalization_stats = normalization_stats

        # Determine device
        if config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.training.device)

        self.model.to(self.device)

        # Loss function
        self.loss_fn = VQVAELoss(config.loss)

        # Optimizer
        if config.training.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        elif config.training.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

        # Learning rate scheduler (optional)
        self.scheduler = None
        if config.training.use_scheduler:
            if config.training.scheduler_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.training.num_epochs - config.training.warmup_epochs,
                )
            elif config.training.scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.training.num_epochs // 3,
                    gamma=0.1,
                )
            elif config.training.scheduler_type == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.95
                )

        # Model compilation (optional)
        if config.training.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': [],
        }

    def train(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        output_dir: Path = Path("checkpoints"),
        checkpoint_prefix: str = "vq_tokenizer",
    ) -> Dict[str, Any]:
        """Run complete training loop.

        Args:
            temporal_features: Temporal sequences [N, T, D_t] (optional)
            initial_manual: Manual initial features [N, D_i] (optional)
            initial_raw: Raw initial conditions [N, C, H, W] (optional)
            temporal_mask: Validity mask for temporal [N, T] (optional)
            temporal_lengths: Actual sequence lengths [N] (optional)
            output_dir: Directory to save checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames

        Returns:
            Training history dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create train/val split
        train_loader, val_loader = self._create_dataloaders(
            temporal_features,
            initial_manual,
            initial_raw,
            temporal_mask,
            temporal_lengths,
        )

        logger.info(f"Training VQ Tokenizer on {self.device}")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")

        # Training loop
        val_loss = None  # Initialize for final checkpoint saving
        for epoch in range(self.config.training.num_epochs):
            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            self.training_history['train_losses'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)

            # Validate every N epochs
            if (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                val_metrics = self._validate_epoch(val_loader)
                self.training_history['val_losses'].append(val_metrics['loss'])
                self.training_history['val_metrics'].append(val_metrics)

                val_loss = val_metrics['loss']

                # Check for improvement
                if val_loss < self.best_val_loss - self.config.training.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0

                    # Save best checkpoint
                    best_path = output_dir / f"{checkpoint_prefix}_best.pt"
                    self._save_checkpoint(best_path, epoch, val_loss)
                    logger.info(f"New best model saved: val_loss={val_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1

                # Early stopping check
                if self.epochs_without_improvement >= self.config.training.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"({self.epochs_without_improvement} epochs without improvement)"
                    )
                    break

            # Learning rate scheduler
            if self.scheduler is not None and epoch >= self.config.training.warmup_epochs:
                self.scheduler.step()

            # Dead code reset (periodically)
            if (epoch + 1) % self.config.training.dead_code_reset_interval == 0:
                self._reset_dead_codes()

            # Log progress
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                log_msg = (
                    f"Epoch {epoch+1}/{self.config.training.num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"LR: {lr:.6f}"
                )
                if (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                    perplexity = val_metrics['perplexity']
                    # Compute average codebook size across all quantizers
                    total_codes = sum(q.num_embeddings for q in self.model.quantizers.values())
                    num_quantizers = len(self.model.quantizers)
                    avg_codebook_size = total_codes / num_quantizers if num_quantizers > 0 else 1
                    util_pct = (perplexity / avg_codebook_size) * 100
                    log_msg += (
                        f" | Val Loss: {val_metrics['loss']:.6f} "
                        f"(recon={val_metrics['reconstruction']:.4f}, "
                        f"vq={val_metrics['vq']:.4f}, "
                        f"topo={val_metrics['topographic']:.4f} "
                        f"[pre={val_metrics['topo_pre']:.3f}, post={val_metrics['topo_post']:.3f}], "
                        f"util={util_pct:.1f}%, avg_codes={avg_codebook_size:.1f})"
                    )
                logger.info(log_msg)

        # Save final checkpoint
        final_path = output_dir / f"{checkpoint_prefix}_final.pt"
        self._save_checkpoint(final_path, epoch, val_loss if val_loss else None)
        logger.info(f"Final model saved to {final_path}")

        return self.training_history

    def _create_dataloaders(
        self,
        temporal_features: Optional[torch.Tensor],
        initial_manual: Optional[torch.Tensor],
        initial_raw: Optional[torch.Tensor],
        temporal_mask: Optional[torch.Tensor],
        temporal_lengths: Optional[torch.Tensor],
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders.

        Args:
            Same as train()

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Collect all tensors and track their order
        tensors = []
        tensor_map = {}  # Maps tensor type to batch index

        if temporal_features is not None:
            tensor_map['temporal_features'] = len(tensors)
            tensors.append(temporal_features)
        if initial_manual is not None:
            tensor_map['initial_manual'] = len(tensors)
            tensors.append(initial_manual)
        if initial_raw is not None:
            tensor_map['initial_raw'] = len(tensors)
            tensors.append(initial_raw)
        if temporal_mask is not None:
            tensor_map['temporal_mask'] = len(tensors)
            tensors.append(temporal_mask)
        if temporal_lengths is not None:
            tensor_map['temporal_lengths'] = len(tensors)
            tensors.append(temporal_lengths)

        # Store tensor map for batch unpacking
        self.tensor_map = tensor_map

        # Create dataset
        dataset = TensorDataset(*tensors)

        # Train/val split
        val_size = int(len(dataset) * self.config.training.val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(
                self.config.random_seed if self.config.random_seed else 42
            ),
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        return train_loader, val_loader

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_ortho = 0.0
        total_info = 0.0
        num_batches = 0

        for batch in loader:
            # Unpack batch using tensor_map to handle variable tensor order
            temporal_feats = (
                batch[self.tensor_map['temporal_features']].to(self.device)
                if 'temporal_features' in self.tensor_map else None
            )
            initial_man = (
                batch[self.tensor_map['initial_manual']].to(self.device)
                if 'initial_manual' in self.tensor_map else None
            )
            initial_r = (
                batch[self.tensor_map['initial_raw']].to(self.device)
                if 'initial_raw' in self.tensor_map else None
            )
            temp_mask = (
                batch[self.tensor_map['temporal_mask']].to(self.device)
                if 'temporal_mask' in self.tensor_map else None
            )
            temp_lens = (
                batch[self.tensor_map['temporal_lengths']].to(self.device)
                if 'temporal_lengths' in self.tensor_map else None
            )

            # Forward pass
            outputs = self.model(
                temporal_features=temporal_feats,
                initial_manual=initial_man,
                initial_raw=initial_r,
                temporal_mask=temp_mask,
                temporal_lengths=temp_lens,
            )

            # Extract category embeddings for auxiliary losses
            category_embeddings = self._extract_category_embeddings(outputs)

            # Compute loss
            # Extract codebooks for topographic loss
            codebooks = {
                key: quantizer.embedding.weight
                for key, quantizer in self.model.quantizers.items()
            }

            losses = self.loss_fn(
                original=outputs['original_encoded'],
                reconstructed=outputs['reconstructed'],
                vq_loss=outputs['vq_loss'],
                category_embeddings=category_embeddings,
                quantized_vectors=outputs.get('encodings'),
                token_indices=outputs.get('token_indices'),
                codebooks=codebooks,
                latent_vectors=outputs.get('latents'),
            )

            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional)
            if self.config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon += losses['reconstruction'].item()
            total_vq += losses['vq'].item()
            total_ortho += losses['orthogonality'].item()
            total_info += losses['informativeness'].item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'reconstruction': total_recon / num_batches,
            'vq': total_vq / num_batches,
            'orthogonality': total_ortho / num_batches,
            'informativeness': total_info / num_batches,
        }

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch.

        Args:
            loader: Validation data loader

        Returns:
            Dict of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_ortho = 0.0
        total_info = 0.0
        total_topo = 0.0
        total_topo_pre = 0.0
        total_topo_post = 0.0
        total_perplexity = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                # Unpack batch using tensor_map to handle variable tensor order
                temporal_feats = (
                    batch[self.tensor_map['temporal_features']].to(self.device)
                    if 'temporal_features' in self.tensor_map else None
                )
                initial_man = (
                    batch[self.tensor_map['initial_manual']].to(self.device)
                    if 'initial_manual' in self.tensor_map else None
                )
                initial_r = (
                    batch[self.tensor_map['initial_raw']].to(self.device)
                    if 'initial_raw' in self.tensor_map else None
                )
                temp_mask = (
                    batch[self.tensor_map['temporal_mask']].to(self.device)
                    if 'temporal_mask' in self.tensor_map else None
                )
                temp_lens = (
                    batch[self.tensor_map['temporal_lengths']].to(self.device)
                    if 'temporal_lengths' in self.tensor_map else None
                )

                # Forward pass
                outputs = self.model(
                    temporal_features=temporal_feats,
                    initial_manual=initial_man,
                    initial_raw=initial_r,
                    temporal_mask=temp_mask,
                    temporal_lengths=temp_lens,
                )

                # Extract category embeddings
                category_embeddings = self._extract_category_embeddings(outputs)

                # Compute loss
                # Extract codebooks for topographic loss
                codebooks = {
                    key: quantizer.embeddings.weight
                    for key, quantizer in self.model.quantizers.items()
                }

                losses = self.loss_fn(
                    original=outputs['original_encoded'],
                    reconstructed=outputs['reconstructed'],
                    vq_loss=outputs['vq_loss'],
                    category_embeddings=category_embeddings,
                    quantized_vectors=outputs.get('encodings'),
                    token_indices=outputs.get('token_indices'),
                    codebooks=codebooks,
                    latent_vectors=outputs.get('latents'),
                )

                # Accumulate metrics
                total_loss += losses['total'].item()
                total_recon += losses['reconstruction'].item()
                total_vq += losses['vq'].item()
                total_ortho += losses['orthogonality'].item()
                total_info += losses['informativeness'].item()
                total_topo += losses['topographic'].item()
                total_topo_pre += losses['topo_pre']
                total_topo_post += losses['topo_post']
                total_perplexity += outputs['perplexity'].item()
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'reconstruction': total_recon / num_batches,
            'vq': total_vq / num_batches,
            'orthogonality': total_ortho / num_batches,
            'informativeness': total_info / num_batches,
            'topographic': total_topo / num_batches,
            'topo_pre': total_topo_pre / num_batches,
            'topo_post': total_topo_post / num_batches,
            'perplexity': total_perplexity / num_batches,
        }

    def _extract_category_embeddings(
        self, outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Extract per-category embeddings from model outputs.

        Args:
            outputs: Model forward outputs

        Returns:
            Dict mapping category → embeddings [B, D_cat]
        """
        # For now, extract from original_encoded using group_indices
        original_encoded = outputs['original_encoded']

        category_embeddings = {}
        for family_cat, indices in self.group_indices.items():
            cat_emb = original_encoded[:, indices]
            category_embeddings[family_cat] = cat_emb

        return category_embeddings

    def _reset_dead_codes(self):
        """Reset dead codebook entries that are rarely used."""
        threshold = self.config.training.dead_code_threshold

        for name, quantizer in self.model.quantizers.items():
            if hasattr(quantizer, 'ema_cluster_size') and quantizer.use_ema:
                # EMA quantizers track cluster sizes
                cluster_sizes = quantizer.ema_cluster_size.data
                total_usage = cluster_sizes.sum()

                if total_usage > 0:
                    usage_freq = cluster_sizes / total_usage
                    dead_mask = usage_freq < threshold

                    num_dead = dead_mask.sum().item()
                    if num_dead > 0:
                        # Reset dead codes to random live codes
                        live_indices = (~dead_mask).nonzero(as_tuple=True)[0]
                        if len(live_indices) > 0:
                            for dead_idx in dead_mask.nonzero(as_tuple=True)[0]:
                                random_live = live_indices[
                                    torch.randint(0, len(live_indices), (1,))
                                ]
                                quantizer.embedding.weight.data[dead_idx] = (
                                    quantizer.embedding.weight.data[random_live]
                                    + torch.randn_like(
                                        quantizer.embedding.weight.data[random_live]
                                    ) * 0.01
                                )
                                quantizer.ema_cluster_size.data[dead_idx] = (
                                    quantizer.ema_cluster_size.data[random_live] * 0.5
                                )

                        logger.info(f"Reset {num_dead} dead codes in {name}")

    def _save_checkpoint(
        self, path: Path, epoch: int, val_loss: Optional[float]
    ):
        """Save training checkpoint.

        Args:
            path: Output path for checkpoint
            epoch: Current epoch number
            val_loss: Current validation loss (optional)
        """
        save_checkpoint(
            path=path,
            model=self.model,
            config=self.config,
            group_indices=self.group_indices,
            normalization_stats=self.normalization_stats,
            optimizer=self.optimizer,
            epoch=epoch,
            val_loss=val_loss,
            metadata={'training_history': self.training_history},
        )
