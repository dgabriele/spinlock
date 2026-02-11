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
        feature_metadata: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.group_indices = group_indices
        self.normalization_stats = normalization_stats
        self.feature_metadata = feature_metadata

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
        theta_features: Optional[torch.Tensor] = None,
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
            theta_features: Operator parameters [N, param_dim] (optional)
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
            theta_features,
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
                    # Compute average codebook size for util_epoch (but don't log it)
                    total_codes = sum(q.num_embeddings for q in self.model.quantizers.values())
                    num_quantizers = len(self.model.quantizers)
                    avg_codebook_size = total_codes / num_quantizers if num_quantizers > 0 else 1
                    util_pct = (perplexity / avg_codebook_size) * 100
                    log_msg += (
                        f" | Val Loss: {val_metrics['loss']:.6f} "
                        f"(recon={val_metrics['reconstruction']:.4f}, "
                        f"vq={val_metrics['vq']:.4f}"
                    )
                    if 'roundtrip' in val_metrics:
                        log_msg += f", roundtrip={val_metrics['roundtrip']:.4f}"
                    log_msg += (
                        f", topo={val_metrics['topographic']:.4f} "
                        f"[pre={val_metrics['topo_pre']:.3f}, post={val_metrics['topo_post']:.3f}], "
                        f"util_epoch={util_pct:.1f}%)"
                    )
                logger.info(log_msg)

        # Compute final validation metrics (post-convergence)
        logger.info("Computing final validation metrics...")
        final_metrics = self._compute_final_validation_metrics(val_loader)
        self.training_history['final_metrics'] = final_metrics

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
        theta_features: Optional[torch.Tensor],
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
        if theta_features is not None:
            tensor_map['theta_features'] = len(tensors)
            tensors.append(theta_features)
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
            theta_feats = (
                batch[self.tensor_map['theta_features']].to(self.device)
                if 'theta_features' in self.tensor_map else None
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
                theta_features=theta_feats,
                temporal_mask=temp_mask,
                temporal_lengths=temp_lens,
            )

            # VALIDATION: Ensure dimensions match if roundtrip loss is enabled
            if self.loss_fn.roundtrip_loss is not None and initial_man is not None:
                from spinlock.tokens.encoders.initial import InitialHybridEncoder
                if isinstance(self.model.initial_encoder, InitialHybridEncoder):
                    expected_dim = self.model.initial_encoder.manual_encoder[0].in_features
                    actual_dim = initial_man.shape[1]
                    if expected_dim != actual_dim:
                        raise RuntimeError(
                            f"Feature dimension mismatch: model expects {expected_dim}D initial features "
                            f"but batch provides {actual_dim}D. This indicates an inconsistency between "
                            f"dataset feature extraction and model initialization. "
                            f"Check that InitialManualExtractor is not being used during roundtrip loss."
                        )

            # Extract category embeddings for auxiliary losses
            category_embeddings = self._extract_category_embeddings(outputs)

            # Compute loss
            # Extract codebooks for topographic loss
            codebooks = {
                key: quantizer.embedding.weight
                for key, quantizer in self.model.quantizers.items()
            }

            # Prepare roundtrip loss inputs (if enabled)
            tokens = None
            decoded = outputs.get('decoded')
            if self.loss_fn.roundtrip_loss is not None and decoded is not None:
                # Extract token indices from quantized outputs
                tokens = outputs.get('token_indices')

            losses = self.loss_fn(
                original=outputs['original_encoded'],
                reconstructed=outputs['reconstructed'],
                vq_loss=outputs['vq_loss'],
                category_embeddings=category_embeddings,
                quantized_vectors=outputs.get('encodings'),
                token_indices=outputs.get('token_indices'),
                codebooks=codebooks,
                latent_vectors=outputs.get('latents'),
                model=self.model,  # NEW: for roundtrip loss
                tokens=tokens,  # NEW: for roundtrip loss
                decoded=decoded,  # NEW: for roundtrip loss
                initial_manual=initial_man,  # NEW: for roundtrip loss
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

        metrics = {
            'loss': total_loss / num_batches,
            'reconstruction': total_recon / num_batches,
            'vq': total_vq / num_batches,
            'orthogonality': total_ortho / num_batches,
            'informativeness': total_info / num_batches,
        }

        # Add roundtrip metric if available (computed in last batch)
        if 'roundtrip/total' in losses:
            metrics['roundtrip'] = losses['roundtrip/total']

        return metrics

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
        total_roundtrip = 0.0  # NEW: for roundtrip loss
        num_batches = 0

        # Track token frequencies for per-quantizer utilization
        token_frequencies = {}
        for quantizer_name in self.model.quantizers.keys():
            num_codes = self.model.quantizers[quantizer_name].num_embeddings
            token_frequencies[quantizer_name] = torch.zeros(num_codes, dtype=torch.long)

        # Track per-category reconstruction MSE
        category_mse = {}
        category_counts = {}

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
                theta_feats = (
                    batch[self.tensor_map['theta_features']].to(self.device)
                    if 'theta_features' in self.tensor_map else None
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
                    theta_features=theta_feats,
                    temporal_mask=temp_mask,
                    temporal_lengths=temp_lens,
                )

                # Extract category embeddings
                category_embeddings = self._extract_category_embeddings(outputs)

                # Compute loss
                # Extract codebooks for topographic loss
                codebooks = {
                    key: quantizer.embedding.weight
                    for key, quantizer in self.model.quantizers.items()
                }

                # Prepare roundtrip loss inputs (if enabled)
                tokens = None
                decoded = outputs.get('decoded')
                if self.loss_fn.roundtrip_loss is not None and decoded is not None:
                    tokens = outputs.get('token_indices')

                losses = self.loss_fn(
                    original=outputs['original_encoded'],
                    reconstructed=outputs['reconstructed'],
                    vq_loss=outputs['vq_loss'],
                    category_embeddings=category_embeddings,
                    quantized_vectors=outputs.get('encodings'),
                    token_indices=outputs.get('token_indices'),
                    codebooks=codebooks,
                    latent_vectors=outputs.get('latents'),
                    model=self.model,  # NEW: for roundtrip loss
                    tokens=tokens,  # NEW: for roundtrip loss
                    decoded=decoded,  # NEW: for roundtrip loss
                    initial_manual=initial_man,  # NEW: for roundtrip loss
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
                if 'roundtrip/total' in losses:  # NEW: accumulate roundtrip loss
                    total_roundtrip += losses['roundtrip/total']
                num_batches += 1

                # Track token frequencies (post-convergence utilization)
                if 'token_indices' in outputs:
                    for quantizer_name, token_idxs in outputs['token_indices'].items():
                        # token_idxs: [B] or [B, T] - flatten to count all tokens
                        flat_tokens = token_idxs.flatten()
                        # Use bincount to count occurrences
                        counts = torch.bincount(
                            flat_tokens,
                            minlength=token_frequencies[quantizer_name].shape[0]
                        )
                        token_frequencies[quantizer_name] += counts.cpu()

                # Track per-category reconstruction MSE
                original = outputs['original_encoded']
                reconstructed = outputs['reconstructed']
                for family_cat, indices in self.group_indices.items():
                    cat_orig = original[:, indices]
                    cat_recon = reconstructed[:, indices]
                    cat_mse = torch.mean((cat_orig - cat_recon) ** 2).item()

                    if family_cat not in category_mse:
                        category_mse[family_cat] = 0.0
                        category_counts[family_cat] = 0

                    category_mse[family_cat] += cat_mse
                    category_counts[family_cat] += 1

        # Compute per-quantizer utilization from token frequencies
        per_quantizer_utilization = self._compute_token_utilization(token_frequencies)

        # Check for codebook collapse (log warnings for very low utilization)
        self._check_codebook_collapse(per_quantizer_utilization)

        # Compute embedding-based utilization (true codebook diversity)
        embed_util = self._compute_embedding_utilization()

        # Average per-category MSE
        per_category_mse = {
            cat: category_mse[cat] / category_counts[cat]
            for cat in category_mse.keys()
        }

        metrics = {
            'loss': total_loss / num_batches,
            'reconstruction': total_recon / num_batches,
            'vq': total_vq / num_batches,
            'orthogonality': total_ortho / num_batches,
            'informativeness': total_info / num_batches,
            'topographic': total_topo / num_batches,
            'topo_pre': total_topo_pre / num_batches,
            'topo_post': total_topo_post / num_batches,
            'perplexity': total_perplexity / num_batches,
            'embedding_utilization': embed_util,
            'per_quantizer_utilization': per_quantizer_utilization,
            'per_category_mse': per_category_mse,
        }

        # Add roundtrip metric if enabled
        if self.loss_fn.roundtrip_loss is not None:
            metrics['roundtrip'] = total_roundtrip / num_batches

        return metrics

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

    def _compute_token_utilization(
        self, token_frequencies: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute utilization from token frequency counts.

        Utilization = (codes used at least once) / codebook_size

        This measures real post-convergence usage, not EMA artifacts from
        training transients.

        Args:
            token_frequencies: Dict mapping quantizer_name → frequency counts [num_codes]

        Returns:
            Dict mapping quantizer_name → utilization (0-100%)
        """
        utilizations = {}
        for quantizer_name, frequencies in token_frequencies.items():
            num_used = (frequencies > 0).sum().item()
            codebook_size = len(frequencies)
            utilizations[quantizer_name] = (num_used / codebook_size) * 100.0
        return utilizations

    def _check_codebook_collapse(
        self, per_quantizer_utilization: Dict[str, float]
    ) -> None:
        """Check for codebook collapse and log warnings.

        Warns when quantizers show very low utilization, which indicates
        the codebook is collapsing and not learning diverse representations.

        Args:
            per_quantizer_utilization: Dict mapping quantizer_name → utilization (0-100%)
        """
        COLLAPSE_THRESHOLD = 5.0  # Warn if utilization < 5%
        CRITICAL_THRESHOLD = 2.0  # Critical warning if < 2%

        collapsed_quantizers = []
        critical_quantizers = []

        for quantizer_name, utilization in per_quantizer_utilization.items():
            if utilization < CRITICAL_THRESHOLD:
                critical_quantizers.append((quantizer_name, utilization))
            elif utilization < COLLAPSE_THRESHOLD:
                collapsed_quantizers.append((quantizer_name, utilization))

        # Log warnings
        if critical_quantizers:
            logger.warning(
                f"⚠️  CRITICAL CODEBOOK COLLAPSE: {len(critical_quantizers)} quantizers "
                f"with <{CRITICAL_THRESHOLD}% utilization"
            )
            for name, util in critical_quantizers[:3]:  # Show first 3
                logger.warning(f"    {name}: {util:.2f}%")
            if len(critical_quantizers) > 3:
                logger.warning(f"    ... and {len(critical_quantizers) - 3} more")

        elif collapsed_quantizers:
            logger.warning(
                f"⚠️  Codebook collapse detected: {len(collapsed_quantizers)} quantizers "
                f"with <{COLLAPSE_THRESHOLD}% utilization"
            )
            if self.config.verbose:
                for name, util in collapsed_quantizers[:5]:  # Show first 5 in verbose
                    logger.warning(f"    {name}: {util:.2f}%")

    def _compute_embedding_utilization(self) -> float:
        """Compute true codebook utilization based on non-zero embeddings.

        This measures the percentage of codebook entries with non-negligible
        norms, which reflects actual learned diversity (not just epoch-wise
        selection rate).

        Returns:
            Average utilization percentage across all quantizers
        """
        utilizations = []

        for name, quantizer in self.model.quantizers.items():
            # Get embedding weights
            embeddings = quantizer.embedding.weight  # [num_embeddings, embedding_dim]

            # Compute norms
            norms = torch.norm(embeddings, dim=1)

            # Count non-zero embeddings (threshold to avoid numerical issues)
            num_active = (norms > 1e-6).sum().item()
            num_total = embeddings.shape[0]

            # Utilization for this quantizer
            util = (num_active / num_total) * 100
            utilizations.append(util)

        # Return average across all quantizers
        return sum(utilizations) / len(utilizations) if utilizations else 0.0

    def _compute_final_validation_metrics(
        self, loader: DataLoader
    ) -> Dict[str, float]:
        """Compute comprehensive post-training validation metrics.

        This runs after training completes to capture post-convergence behavior
        without training transients. Metrics are formatted for visualization
        dashboard compatibility.

        Args:
            loader: Validation data loader

        Returns:
            Dict with visualization-compatible keys:
            - "{category}/level_{level}/utilization": float (0-100)
            - "{category}/reconstruction_mse": float
        """
        logger.info("Computing final validation metrics (post-convergence)...")

        # Run one final validation pass to collect metrics
        val_metrics = self._validate_epoch(loader)

        # Extract per-quantizer utilization
        per_quantizer_util = val_metrics['per_quantizer_utilization']
        per_category_mse = val_metrics['per_category_mse']

        # Format for visualization compatibility
        final_metrics = {}

        # Parse quantizer keys like "temporal_group_1_L0" → category + level
        for quantizer_name, utilization in per_quantizer_util.items():
            # Parse format: "{family}_{category}_L{level}"
            parts = quantizer_name.split('_')
            if len(parts) >= 3 and parts[-1].startswith('L'):
                level_str = parts[-1]  # "L0", "L1", "L2"
                level_num = int(level_str[1:])  # Extract level number
                category = '_'.join(parts[:-1])  # Everything before level

                # Format: "{category}/level_{level}/utilization"
                key = f"{category}/level_{level_num}/utilization"
                final_metrics[key] = utilization
            else:
                # Fallback for non-hierarchical quantizers
                final_metrics[f"{quantizer_name}/utilization"] = utilization

        # Add per-category reconstruction MSE
        for category, mse in per_category_mse.items():
            final_metrics[f"{category}/reconstruction_mse"] = mse

        # Log summary
        if self.config.verbose:
            logger.info("\nFinal Metrics Summary:")
            logger.info(f"  Validation Loss: {val_metrics['loss']:.6f}")
            logger.info(f"  Reconstruction MSE: {val_metrics['reconstruction']:.6f}")
            logger.info(f"  Average Token Utilization: {sum(per_quantizer_util.values()) / len(per_quantizer_util):.2f}%")
            logger.info(f"  Total Metrics Captured: {len(final_metrics)}")

        return final_metrics

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
            temporal_input_dim=self.model.temporal_input_dim,
            initial_input_dim=self.model.initial_input_dim,
            feature_metadata=self.feature_metadata,
        )
