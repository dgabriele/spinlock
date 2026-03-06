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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from .model import JointHierarchicalVQVAE
from .losses import VQVAELoss
from .config import TokenizerConfig
from .checkpoint import save_checkpoint
from .schedules import ParameterSchedule

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
        replayer=None,
    ):
        self.model = model
        self.config = config
        self.group_indices = group_indices
        self.normalization_stats = normalization_stats
        self.feature_metadata = feature_metadata
        self.replayer = replayer  # CNOReplayer for on-the-fly trajectory generation

        # Determine device
        if config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.training.device)

        self.model.to(self.device)

        # Loss function (pass aux_config for temporal-only mode)
        aux_config = config.aux_heads if config.temporal_only else None
        self.loss_fn = VQVAELoss(config.loss, aux_config=aux_config)

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

        # Variable-length training support
        self.length_sampler = None
        vl_config = self.config.encoder.temporal.variable_length
        if vl_config:  # Handle both bool and VariableLengthConfig
            from spinlock.encoding.variable_length_utils import create_length_sampler

            # Handle Union[bool, VariableLengthConfig]
            if isinstance(vl_config, bool):
                if vl_config:
                    logger.warning(
                        "variable_length=True without VariableLengthConfig. "
                        "No random length sampling will occur (no length_bins provided)."
                    )
            else:
                # VariableLengthConfig object or dict
                if hasattr(vl_config, 'model_dump'):
                    vl_dict = vl_config.model_dump()
                else:
                    vl_dict = vl_config

                if vl_dict.get("enabled", False):
                    self.length_sampler = create_length_sampler(vl_dict)
                    logger.info(
                        f"Variable-length training enabled: "
                        f"strategy={vl_dict.get('sampling_strategy', 'uniform')}, "
                        f"bins={vl_dict.get('length_bins', 'N/A')}"
                    )

        # Parameter schedules (dropout annealing, weight scheduling)
        self._dropout_schedule = (
            ParameterSchedule(config.training.dropout_schedule)
            if config.training.dropout_schedule else None
        )
        self._weight_schedules: Dict[str, tuple] = {}
        if config.training.weight_schedules:
            for name, sched_cfg in config.training.weight_schedules.items():
                if hasattr(config.loss, name):
                    target = "loss"
                elif config.aux_heads and hasattr(config.aux_heads, name):
                    target = "aux"
                else:
                    raise ValueError(
                        f"weight_schedules key '{name}' not found in LossConfig or AuxHeadConfig"
                    )
                self._weight_schedules[name] = (ParameterSchedule(sched_cfg), target)

        # Length curriculum: start with longer trajectories, decay min_T over training
        self._curriculum_start_min: Optional[int] = None
        self._curriculum_end_min: Optional[int] = None
        if (
            self.length_sampler is not None
            and hasattr(vl_config, 'curriculum_start_min')
            and vl_config.curriculum_start_min is not None
        ):
            self._curriculum_start_min = vl_config.curriculum_start_min
            self._curriculum_end_min = vl_config.min_timesteps
            # Set initial min_T to curriculum start
            self.length_sampler.min_T = self._curriculum_start_min
            logger.info(
                f"Length curriculum: min_T {self._curriculum_start_min} → "
                f"{self._curriculum_end_min} (cosine schedule)"
            )

        # Batch unpacking mode: False = TensorDataset (index-based), True = dict batches
        self._dict_batch_mode = False

        # Intra-epoch convergence stopping state
        self._convergence_stopped = False

        # Tracking
        self.best_val_loss = float('inf')       # Best-model saving (no min_delta)
        self._best_es_metric = float('inf')     # Early-stopping patience (uses min_delta)
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
        dataset: Optional[Dataset] = None,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        """Run complete training loop.

        Two data paths:
        1. Tensor path (default): Pass pre-loaded tensors. Creates TensorDataset
           internally. Used by manual-mode and train_on_features().
        2. Dataset path: Pass a PyTorch Dataset directly (e.g. SpinlockDataset
           with lazy_ics=True). Batches are dicts with 'ic', 'params' keys.
           Used by learned-mode to avoid loading all ICs into RAM.

        Args:
            temporal_features: Temporal sequences [N, T, D_t] (optional)
            initial_manual: Manual initial features [N, D_i] (optional)
            initial_raw: Raw initial conditions [N, C, H, W] (optional)
            theta_features: Operator parameters [N, param_dim] (optional)
            temporal_mask: Validity mask for temporal [N, T] (optional)
            temporal_lengths: Actual sequence lengths [N] (optional)
            output_dir: Directory to save checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames
            dataset: Optional PyTorch Dataset (overrides tensor args when provided)
            start_epoch: Epoch to start from (>0 when resuming from checkpoint)

        Returns:
            Training history dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store for periodic checkpoint saves in _train_epoch
        self._output_dir = output_dir
        self._checkpoint_prefix = checkpoint_prefix

        # Create train/val split — dataset path or tensor path
        if dataset is not None:
            train_loader, val_loader = self._create_dataloaders_from_dataset(dataset)
        else:
            train_loader, val_loader = self._create_dataloaders(
                temporal_features,
                initial_manual,
                initial_raw,
                theta_features,
                temporal_mask,
                temporal_lengths,
            )

        logger.info(f"Training VQ Tokenizer on {self.device}")
        if start_epoch > 0:
            logger.info(f"  Resuming from epoch: {start_epoch}")
        logger.info(f"  Epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size: {self.config.training.batch_size}")
        if self.config.training.gradient_accumulation_steps > 1:
            eff = self.config.training.batch_size * self.config.training.gradient_accumulation_steps
            logger.info(f"  Gradient accumulation: {self.config.training.gradient_accumulation_steps} steps (effective batch size: {eff})")
        logger.info(f"  Learning rate: {self.config.training.learning_rate}")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        if self.device.type == "cuda":
            _alloc = torch.cuda.memory_allocated() / 1024**2
            _resv = torch.cuda.memory_reserved() / 1024**2
            logger.info(
                f"  GPU memory before training: alloc={_alloc:.0f}MB reserved={_resv:.0f}MB"
            )

        # OPQ calibration: compute rotation before training begins
        # Skip in pyramid_first mode — learned projection subsumes OPQ
        if (
            self.config.training.opq_enabled
            and self.model.temporal_rotation_matrix is None
            and self.model.temporal_cnn_encoder is not None
            and self.model.pyramid_first_encoder is None
            and self.config.training.opq_warmup_epochs == 0
        ):
            self._calibrate_opq(train_loader)

        # Log active schedules
        if self._dropout_schedule:
            logger.info(f"  Dropout schedule: {self._dropout_schedule}")
        if self._weight_schedules:
            for name, (sched, _) in self._weight_schedules.items():
                logger.info(f"  Weight schedule [{name}]: {sched}")

        # Training loop
        val_loss = None  # Initialize for final checkpoint saving
        for epoch in range(start_epoch, self.config.training.num_epochs):
            self._apply_schedules(epoch)

            # Train
            train_metrics = self._train_epoch(train_loader, epoch)
            self.training_history['train_losses'].append(train_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)

            # Intra-epoch convergence stopping triggered — save and exit
            # Skip validation: the convergence criteria already measured quality
            # over a 100-batch rolling window. Full validation with on-the-fly
            # replay can take hours and provides redundant quality assurance.
            if self._convergence_stopped:
                best_path = output_dir / f"{checkpoint_prefix}_best.pt"
                self._save_checkpoint(best_path, epoch, val_loss=None)
                logger.info(
                    "Convergence checkpoint saved to %s (validation skipped)",
                    best_path,
                )
                break

            # Deferred OPQ calibration (after warmup epochs)
            # Skip in pyramid_first mode — learned projection subsumes OPQ
            if (
                self.config.training.opq_enabled
                and self.model.temporal_rotation_matrix is None
                and self.model.temporal_cnn_encoder is not None
                and self.model.pyramid_first_encoder is None
                and self.config.training.opq_warmup_epochs > 0
                and (epoch + 1) == self.config.training.opq_warmup_epochs
            ):
                self._calibrate_opq(train_loader)

            # Validate every N epochs
            if (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                val_metrics = self._validate_epoch(val_loader)
                self.training_history['val_losses'].append(val_metrics['loss'])
                self.training_history['val_metrics'].append(val_metrics)

                val_loss = val_metrics['loss']

                # Compute custom "best model" metric that prioritizes:
                # 1. Reconstruction quality (recon)
                # 2. Roundtrip consistency (roundtrip)
                # 3. Codebook diversity (util_epoch: perplexity / avg_codebook_size)
                # 4. De-emphasize topographic loss (topo)
                recon = val_metrics['reconstruction']
                roundtrip = val_metrics.get('roundtrip', 0.0)  # 0 if not enabled
                topo = val_metrics['topographic']
                perplexity = val_metrics['perplexity']

                # Compute util_epoch (same as logging)
                total_codes = sum(q.num_embeddings for q in self.model.quantizers.values())
                num_quantizers = len(self.model.quantizers)
                avg_codebook_size = total_codes / num_quantizers if num_quantizers > 0 else 1
                util_epoch = (perplexity / avg_codebook_size) * 100  # Percentage

                # Custom metric: lower is better.  Includes all validation
                # components that indicate token quality.
                vq = val_metrics['vq']
                es_metric = recon + roundtrip + vq + 0.1 * topo
                best_metric = es_metric - 0.0001 * util_epoch

                # Always save if metric is strictly better (no min_delta gate).
                # Early-stopping patience uses min_delta separately below.
                if best_metric < self.best_val_loss:
                    self.best_val_loss = best_metric

                    # Save best checkpoint
                    best_path = output_dir / f"{checkpoint_prefix}_best.pt"
                    self._save_checkpoint(best_path, epoch, val_loss)
                    logger.info(
                        f"New best model saved: metric={best_metric:.6f} "
                        f"(recon={recon:.4f}, roundtrip={roundtrip:.4f}, "
                        f"vq={vq:.4f}, topo={topo:.4f}, util_epoch={util_epoch:.1f}%)"
                    )

                # Early stopping uses loss-only metric (no utilization bonus)
                # to prevent util oscillations from resetting patience.
                if es_metric < self._best_es_metric - self.config.training.early_stopping_min_delta:
                    self._best_es_metric = es_metric
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.config.verbose:
                    logger.info(
                        f"  ES: loss_metric={es_metric:.6f}, best_es={self._best_es_metric:.6f}, "
                        f"patience={self.epochs_without_improvement}/{self.config.training.early_stopping_patience}"
                    )

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

            # Save latest checkpoint every validation epoch (survives kill)
            if (epoch + 1) % self.config.training.val_every_n_epochs == 0:
                latest_path = output_dir / f"{checkpoint_prefix}_latest.pt"
                self._save_checkpoint(latest_path, epoch, val_loss)

        # Compute final validation metrics (post-convergence)
        logger.info("Computing final validation metrics...")
        final_metrics = self._compute_final_validation_metrics(val_loader)
        self.training_history['final_metrics'] = final_metrics

        # Save final checkpoint
        final_path = output_dir / f"{checkpoint_prefix}_final.pt"
        self._save_checkpoint(final_path, epoch, val_loss if val_loss else None)
        logger.info(f"Final model saved to {final_path}")

        return self.training_history

    def train_on_features(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Fine-tune on pre-extracted features (for ADAPT phase).

        Unlike train(), this does NOT save checkpoints, run early stopping,
        or compute final metrics. It runs a short training loop and returns
        the history for the caller to evaluate.

        Args:
            temporal_features: [N, T, D_t] temporal features (optional)
            initial_manual: [N, D_i] manual initial features (optional)
            initial_raw: [N, C, H, W] raw initial conditions (optional)
            theta_features: [N, param_dim] operator parameters (optional)
            temporal_mask: [N, T] validity mask (optional)
            temporal_lengths: [N] sequence lengths (optional)
            num_epochs: Override config epoch count (for short fine-tuning)

        Returns:
            Dict with 'train_losses' and 'val_losses' lists
        """
        epochs = num_epochs or self.config.training.num_epochs

        train_loader, val_loader = self._create_dataloaders(
            temporal_features, initial_manual, initial_raw,
            theta_features, temporal_mask, temporal_lengths,
        )

        logger.info(
            f"ADAPT fine-tuning: {epochs} epochs, "
            f"lr={self.optimizer.param_groups[0]['lr']:.1e}, "
            f"train_batches={len(train_loader)}"
        )

        history = {'train_losses': [], 'val_losses': []}

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            history['train_losses'].append(train_metrics['loss'])

            if (epoch + 1) % max(1, self.config.training.val_every_n_epochs) == 0:
                val_metrics = self._validate_epoch(val_loader)
                history['val_losses'].append(val_metrics['loss'])

            if self.scheduler is not None and epoch >= self.config.training.warmup_epochs:
                self.scheduler.step()

        return history

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

        # Sequential split preserving Sobol ordering
        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, len(dataset)))

        # Create dataloaders (shuffle from config, default False to preserve Sobol ordering)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
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

    def _create_dataloaders_from_dataset(
        self,
        dataset: Dataset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders from a PyTorch Dataset.

        Used by learned-mode with lazy HDF5 IC loading. Batches are dicts
        (collated by default dict collation) rather than indexed tuples.

        Preserves Sobol ordering: train = first (1-val_split) samples,
        val = last val_split samples. No shuffling — Sobol sequences are
        quasi-random by construction, so sequential access already provides
        uniform parameter space coverage per batch.

        Args:
            dataset: PyTorch Dataset returning dicts with 'ic', 'params', etc.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        self._dict_batch_mode = True
        self.tensor_map = {}  # Not used in dict mode, but avoids AttributeError

        n = len(dataset)
        val_size = int(n * self.config.training.val_split)
        train_size = n - val_size

        # Sequential split: train = indices [0, train_size), val = [train_size, n)
        # Preserves Sobol quasi-random ordering in both splits.
        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, n))

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=self.config.training.shuffle,
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

    def _unpack_batch(self, batch) -> Dict[str, Optional[torch.Tensor]]:
        """Unpack a batch into a standard dict of named tensors.

        Handles both batch modes:
        - dict mode (lazy Dataset): batch is a dict with 'ic', 'params' keys
        - tensor mode (TensorDataset): batch is a list, indexed via tensor_map

        Returns:
            Dict with keys: temporal_features, initial_manual, initial_raw,
            theta_features, temporal_mask, temporal_lengths — each either a
            device tensor or None.
        """
        if self._dict_batch_mode:
            return {
                'temporal_features': None,
                'initial_manual': (
                    batch['initial_manual'].to(self.device)
                    if 'initial_manual' in batch else None
                ),
                'initial_raw': batch['ic'].to(self.device),
                'theta_features': batch['params'].to(self.device),
                'temporal_mask': None,
                'temporal_lengths': None,
            }
        else:
            return {
                key: (
                    batch[self.tensor_map[key]].to(self.device)
                    if key in self.tensor_map else None
                )
                for key in [
                    'temporal_features', 'initial_manual', 'initial_raw',
                    'theta_features', 'temporal_mask', 'temporal_lengths',
                ]
            }

    def _generate_trajectories(
        self,
        theta_feats: torch.Tensor,
        initial_raw: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Generate trajectories on-the-fly using the replayer.

        Prefers batched rollout (LeniaReplayAdapter) when available,
        falling back to per-sample rollout (CNOReplayer) otherwise.

        Returns:
            [B, T+1, C, H, W] on CPU, or None if replayer unavailable.
        """
        if self.replayer is None or theta_feats is None or initial_raw is None:
            return None

        # Determine timesteps
        if self.length_sampler is not None:
            batch_len = self.length_sampler.sample_lengths(1)  # [1]
            timesteps = int(batch_len[0].item())
        else:
            timesteps = self.config.generation_timesteps or 64

        # Batched path: replayer supports rollout_batch (e.g. LeniaReplayAdapter)
        if hasattr(self.replayer, 'rollout_batch'):
            with torch.no_grad():
                return self.replayer.rollout_batch(
                    params_batch=theta_feats.cpu(),
                    ics=initial_raw,
                    timesteps=timesteps,
                    return_all_steps=True,
                )  # [B, T+1, C, H, W] on CPU

        # Per-sample path: replayer only has rollout() (e.g. CNOReplayer)
        B = theta_feats.shape[0]
        trajectories = []
        with torch.no_grad():
            for i in range(B):
                traj = self.replayer.rollout(
                    params_vector=theta_feats[i].cpu(),
                    ic=initial_raw[i],
                    timesteps=timesteps,
                    return_all_steps=True,
                )  # [1, T+1, C, H, W]
                trajectories.append(traj.squeeze(0).cpu())
        return torch.stack(trajectories, dim=0)  # [B, T+1, C, H, W]

    def _compute_trajectory_targets(
        self, temporal_raw: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Subsample K keyframes from raw trajectories for trajectory prototype loss.

        Picks num_keyframes evenly-spaced frames from the T+1 trajectory.
        Returns None if the trajectory prototype head is not active or no
        trajectory data is available.

        Args:
            temporal_raw: [B, T+1, C, H, W] raw trajectory from replayer (may be CPU).

        Returns:
            [B, K, C, H, W] keyframe targets on self.device, or None.
        """
        if (
            temporal_raw is None
            or not hasattr(self.model, 'traj_prototype_head')
            or self.model.traj_prototype_head is None
        ):
            return None

        T_plus_1 = temporal_raw.shape[1]  # T+1 frames (T timesteps + initial)
        K = self.model.traj_prototype_head.num_keyframes

        # Evenly-spaced indices across the full trajectory
        keyframe_indices = torch.linspace(0, T_plus_1 - 1, K).long()
        targets = temporal_raw[:, keyframe_indices]  # [B, K, C, H, W]
        return targets.to(self.device)

    def _calibrate_opq(self, train_loader: DataLoader) -> None:
        """Compute OPQ rotation from CNN features and register on the model.

        Collects CNN output features from a subset of training batches,
        flattens [N_batch, T, D] → [N_batch*T, D], subsamples to
        max_calibration_samples, then runs OPQ alternating optimization.
        The resulting rotation is registered as model buffers so it is
        automatically saved/loaded with checkpoints.
        """
        from spinlock.features.grouping.opq import compute_opq_rotation
        from spinlock.encoding.normalization import LinearTransform

        cfg = self.config.training
        learned_cfg = self.config.encoder.temporal.learned

        logger.info("Calibrating OPQ rotation from CNN features...")

        self.model.eval()
        all_features = []
        n_batches = 0

        with torch.no_grad():
            for batch in train_loader:
                unpacked = self._unpack_batch(batch)
                initial_r = unpacked['initial_raw']
                theta_feats = unpacked['theta_features']

                temporal_raw = self._generate_trajectories(theta_feats, initial_r)
                if temporal_raw is None:
                    logger.warning("OPQ calibration: no trajectories generated, skipping")
                    self.model.train()
                    return

                # CNN forward only (no grad)
                temporal_raw_dev = temporal_raw.to(self.device)
                cnn_features = self.model.temporal_cnn_encoder(temporal_raw_dev)  # [B, T, D]
                # Flatten to per-frame features and move to CPU
                B, T, D = cnn_features.shape
                all_features.append(cnn_features.reshape(B * T, D).cpu())

                n_batches += 1
                if n_batches >= cfg.opq_n_calibration_batches:
                    break

        self.model.train()

        if not all_features:
            logger.warning("OPQ calibration: no features collected, skipping")
            return

        X = torch.cat(all_features, dim=0).numpy()  # [N_total, D]
        logger.info(
            "OPQ calibration: collected %d frames from %d batches (D=%d)",
            X.shape[0], n_batches, X.shape[1],
        )

        # Subsample if needed
        if X.shape[0] > cfg.opq_max_samples:
            rng = np.random.RandomState(self.config.random_seed or 42)
            indices = rng.choice(X.shape[0], cfg.opq_max_samples, replace=False)
            X = X[indices]
            logger.info("  Subsampled to %d frames", X.shape[0])

        M = learned_cfg.num_groups
        d_sub = learned_cfg.embedding_dim // M

        # Log per-group variance BEFORE rotation
        group_vars_before = [
            X[:, g * d_sub:(g + 1) * d_sub].var()
            for g in range(M)
        ]
        logger.info(
            "  Pre-OPQ per-group var: min=%.4f, max=%.4f, ratio=%.1f",
            min(group_vars_before), max(group_vars_before),
            max(group_vars_before) / max(min(group_vars_before), 1e-12),
        )

        # Compute OPQ rotation
        R, mean = compute_opq_rotation(
            X, M=M, d_sub=d_sub,
            n_iter=cfg.opq_n_iter,
            n_codes=cfg.opq_n_codes,
            random_state=self.config.random_seed,
        )

        # Log per-group variance AFTER rotation
        X_rot = (X - mean) @ R.T
        group_vars_after = [
            X_rot[:, g * d_sub:(g + 1) * d_sub].var()
            for g in range(M)
        ]
        logger.info(
            "  Post-OPQ per-group var: min=%.4f, max=%.4f, ratio=%.1f",
            min(group_vars_after), max(group_vars_after),
            max(group_vars_after) / max(min(group_vars_after), 1e-12),
        )

        # Register on model
        transform = LinearTransform(mean=mean, components=R)
        self.model.set_temporal_rotation(transform)

    def _apply_schedules(self, epoch: int) -> None:
        """Apply parameter schedules at epoch start.

        Computes training progress as epoch / max(num_epochs - 1, 1) so that
        the schedule reaches its ``end`` value at the final epoch.  Updates
        nn.Dropout.p for dropout annealing and mutates LossConfig / AuxHeadConfig
        fields for weight scheduling.
        """
        num_epochs = self.config.training.num_epochs
        progress = epoch / max(num_epochs - 1, 1)

        if self._dropout_schedule is not None:
            new_p = self._dropout_schedule(progress)
            for m in self.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = new_p
            logger.info(f"  [Schedule] dropout={new_p:.4f} (progress={progress:.3f})")

        for name, (sched, target) in self._weight_schedules.items():
            new_val = sched(progress)
            cfg_obj = self.config.loss if target == "loss" else self.config.aux_heads
            setattr(cfg_obj, name, new_val)
            logger.info(f"  [Schedule] {name}={new_val:.4f} (progress={progress:.3f})")

        # Length curriculum: cosine decay of min_T from start_min → end_min
        if self._curriculum_start_min is not None and self.length_sampler is not None:
            import math
            s, e = self._curriculum_start_min, self._curriculum_end_min
            new_min = int(e + (s - e) * 0.5 * (1.0 + math.cos(math.pi * progress)))
            self.length_sampler.min_T = new_min
            # Log which bins are active
            if hasattr(self.length_sampler, 'bins'):
                active = [b for b in self.length_sampler.bins if b >= new_min]
                logger.info(
                    f"  [Schedule] min_T={new_min} (progress={progress:.3f}), "
                    f"active bins={active}"
                )
            else:
                logger.info(f"  [Schedule] min_T={new_min} (progress={progress:.3f})")

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

        # Convergence stopping rolling buffers
        tc = self.config.training
        _conv_enabled = tc.convergence_stop_enabled
        _conv_window = tc.convergence_stop_window if _conv_enabled else 0
        _conv_vq_buf: list = []
        _conv_recon_buf: list = []
        _conv_info_buf: list = []
        _conv_topo_post_buf: list = []

        dead_code_interval = self.config.training.dead_code_reset_interval
        num_batches_total = len(loader)
        accum_steps = self.config.training.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            unpacked = self._unpack_batch(batch)
            temporal_feats = unpacked['temporal_features']
            initial_man = unpacked['initial_manual']
            initial_r = unpacked['initial_raw']
            theta_feats = unpacked['theta_features']
            temp_mask = unpacked['temporal_mask']
            temp_lens = unpacked['temporal_lengths']

            # On-the-fly trajectory generation for learned mode
            temporal_raw = self._generate_trajectories(theta_feats, initial_r)

            # Subsample trajectory keyframes for trajectory prototype head
            trajectory_targets = self._compute_trajectory_targets(temporal_raw)

            # Random length sampling for variable-length training (manual mode only)
            if self.length_sampler is not None and temporal_feats is not None:
                B, T, D = temporal_feats.shape

                # Sample random lengths for this batch
                sampled_lengths = self.length_sampler.sample_lengths(B)  # [B]

                # Create mask from sampled lengths
                sampled_mask = self.length_sampler.create_mask(sampled_lengths, T)  # [B, T]

                # Override the all-True mask from dataset
                temp_mask = sampled_mask.to(self.device)
                temp_lens = sampled_lengths.to(self.device)

            # Memory profiling (first batch only)
            _do_mem_log = self.config.verbose and batch_idx == 0 and epoch == 0
            if _do_mem_log and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                logger.info(
                    f"  [MEM pre-fwd] alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB"
                )

            # Forward pass — in temporal-only mode, don't pass theta/initial to model
            # (they have no encoders), but keep them for aux loss computation.
            if self.config.temporal_only:
                outputs = self.model(
                    temporal_features=temporal_feats,
                    temporal_mask=temp_mask,
                    temporal_lengths=temp_lens,
                    temporal_raw=temporal_raw,
                )
            else:
                outputs = self.model(
                    temporal_features=temporal_feats,
                    initial_manual=initial_man,
                    initial_raw=initial_r,
                    theta_features=theta_feats,
                    temporal_mask=temp_mask,
                    temporal_lengths=temp_lens,
                    temporal_raw=temporal_raw,
                )

            if _do_mem_log and self.device.type == "cuda":
                logger.info(
                    f"  [MEM post-fwd] alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                    f"peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB"
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

            # Determine num_groups for group balance loss (dynamic from model)
            _num_groups = sum(
                1 for k in self.group_indices if k.startswith("temporal_")
            ) if outputs.get('cnn_features') is not None else None

            losses = self.loss_fn(
                original=outputs['original_encoded'],
                reconstructed=outputs['reconstructed'],
                vq_loss=outputs['vq_loss'],
                category_embeddings=category_embeddings,
                quantized_vectors=outputs.get('encodings'),
                token_indices=outputs.get('token_indices'),
                codebooks=codebooks,
                latent_vectors=outputs.get('latents'),
                model=self.model,
                tokens=tokens,
                decoded=decoded,
                initial_manual=initial_man,
                original_theta=theta_feats,
                original_initial=initial_r,
                cnn_features=outputs.get('cnn_features'),
                num_groups=_num_groups,
                gate_values=outputs.get('gate_values'),
                trajectory_targets=trajectory_targets,
            )

            loss = losses['total']

            if _do_mem_log and self.device.type == "cuda":
                logger.info(
                    f"  [MEM post-loss] alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                    f"peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB"
                )

            # Backward pass (gradient accumulation: scale loss by 1/N)
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            if _do_mem_log and self.device.type == "cuda":
                logger.info(
                    f"  [MEM post-bwd] alloc={torch.cuda.memory_allocated()/1024**2:.0f}MB "
                    f"peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB"
                )

            # Step optimizer every accum_steps micro-batches (or at last batch)
            is_accum_step = (batch_idx + 1) % accum_steps == 0
            is_last_batch = batch_idx == num_batches_total - 1
            if is_accum_step or is_last_batch:
                # Gradient clipping (optional)
                if self.config.training.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Dead code reset: fires every N batches (not epochs) so that
            # underused codebook entries get reseeded from fresh latents.
            if dead_code_interval > 0 and (batch_idx + 1) % dead_code_interval == 0:
                latents_dict = outputs.get("latents", {})
                total_reset = 0
                for qkey, quantizer in self.model.quantizers.items():
                    if qkey in latents_dict:
                        n = quantizer.reset_dead_codes(
                            latents_dict[qkey].detach(),
                            percentile_threshold=10.0,
                            max_reset_fraction=0.25,
                        )
                        total_reset += n
                if total_reset > 0:
                    logger.info(
                        "Dead code reset (epoch %d, batch %d): "
                        "%d codes reset across %d quantizers",
                        epoch + 1, batch_idx + 1,
                        total_reset, len(self.model.quantizers),
                    )

            # Periodic checkpoint: save latest every 500 batches so we never
            # lose more than ~500 batches of work on crash or kill.
            if (batch_idx + 1) % 500 == 0:
                latest_path = self._output_dir / f"{self._checkpoint_prefix}_latest.pt"
                self._save_checkpoint(latest_path, epoch, val_loss=None)
                logger.info(
                    "Periodic checkpoint saved (E%d B%d) → %s",
                    epoch + 1, batch_idx + 1, latest_path,
                )

            # Accumulate metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            total_recon += losses['reconstruction'].item()
            total_vq += losses['vq'].item()
            total_ortho += losses['orthogonality'].item()
            total_info += losses['informativeness'].item()
            num_batches += 1

            # Per-batch logging
            if self.config.verbose:
                recon_val = losses['reconstruction']
                recon_str = f"{recon_val.item():.4f}" if hasattr(recon_val, 'item') else f"{recon_val:.4f}"
                parts = [
                    f"  [E{epoch+1} B{batch_idx+1}/{num_batches_total}]",
                    f"loss={batch_loss:.4f}",
                    f"recon={recon_str}",
                ]
                # Show per-family inverse reconstruction if available
                if 'recon/theta' in losses:
                    parts.append(f"θ={losses['recon/theta']:.4f}")
                if 'recon/initial' in losses:
                    parts.append(f"ic={losses['recon/initial']:.4f}")
                # Show aux head losses (temporal-only mode)
                if 'aux/theta' in losses:
                    parts.append(f"aux_θ={losses['aux/theta']:.4f}")
                if 'aux/initial' in losses:
                    parts.append(f"aux_ic={losses['aux/initial']:.4f}")
                if 'aux/theta_probe' in losses:
                    parts.append(f"probe_θ={losses['aux/theta_probe']:.4f}")
                if 'aux/initial_probe' in losses:
                    parts.append(f"probe_ic={losses['aux/initial_probe']:.4f}")
                if 'aux/trajectory' in losses:
                    parts.append(f"aux_traj={losses['aux/trajectory']:.4f}")
                parts.extend([
                    f"vq={losses['vq'].item():.4f}",
                    f"ortho={losses['orthogonality'].item():.4f}",
                    f"info={losses['informativeness'].item():.4f}",
                ])
                if 'roundtrip/total' in losses:
                    parts.append(f"rt={losses['roundtrip/total']:.4f}")
                    # Mean roundtrip token accuracy across all quantizers
                    rt_accs = [v for k, v in losses.items() if k.startswith('roundtrip_acc/')]
                    if rt_accs:
                        parts.append(f"rt_acc={sum(rt_accs)/len(rt_accs):.3f}")
                if 'topographic' in losses:
                    parts.append(f"topo={losses['topographic'].item():.4f}")
                    if 'topo_pre' in losses and 'topo_post' in losses:
                        pre = losses['topo_pre']
                        post = losses['topo_post']
                        pre_val = pre.item() if hasattr(pre, 'item') else pre
                        post_val = post.item() if hasattr(post, 'item') else post
                        parts.append(f"pre={pre_val:.3f}")
                        parts.append(f"post={post_val:.3f}")
                if losses.get('group_balance') is not None:
                    bal = losses['group_balance']
                    bal_val = bal.item() if hasattr(bal, 'item') else bal
                    if bal_val > 0:
                        parts.append(f"bal={bal_val:.4f}")
                # Gate stats: show mean gate value and how many gates are "open" (>0.5)
                gate_vals = outputs.get('gate_values')
                if gate_vals is not None:
                    g_mean = gate_vals.mean().item()
                    g_open = (gate_vals > 0.5).sum().item()
                    g_total = gate_vals.numel()
                    parts.append(f"gates={g_open:.0f}/{g_total}({g_mean:.3f})")
                # Codebook utilization: unique codes used / codebook size, per-level and overall
                token_indices = outputs.get('token_indices')
                if token_indices:
                    level_utils: dict = {}  # level_idx -> list of utilizations
                    for qname, tidx in token_indices.items():
                        n_unique = tidx.flatten().unique().numel()
                        n_codes = self.model.quantizers[qname].num_embeddings
                        util = n_unique / n_codes
                        # Parse level from key like "temporal_group_0_L0"
                        level_str = qname.rsplit('_', 1)[-1]  # "L0", "L1", "L2"
                        if level_str.startswith('L') and level_str[1:].isdigit():
                            lvl = int(level_str[1:])
                            level_utils.setdefault(lvl, []).append(util)
                    # Log per-level mean utilization
                    for lvl in sorted(level_utils.keys()):
                        lvl_mean = sum(level_utils[lvl]) / len(level_utils[lvl])
                        parts.append(f"util_L{lvl}={lvl_mean:.3f}")
                    # Overall mean
                    all_utils = [u for us in level_utils.values() for u in us]
                    if all_utils:
                        parts.append(f"cb_util={sum(all_utils)/len(all_utils):.3f}")
                logger.info(" ".join(parts))

            # ── Intra-epoch convergence stopping ──────────────────────
            if _conv_enabled and num_batches >= tc.convergence_stop_min_batches:
                # Append to rolling buffers
                _conv_vq_buf.append(losses['vq'].item())
                _conv_recon_buf.append(losses['reconstruction'].item())
                _conv_info_buf.append(losses['informativeness'].item())
                topo_post_val = losses.get('topo_post')
                if topo_post_val is not None:
                    _p = topo_post_val.item() if hasattr(topo_post_val, 'item') else topo_post_val
                    _conv_topo_post_buf.append(_p)

                # Trim to window size
                if len(_conv_vq_buf) > _conv_window:
                    _conv_vq_buf = _conv_vq_buf[-_conv_window:]
                    _conv_recon_buf = _conv_recon_buf[-_conv_window:]
                    _conv_info_buf = _conv_info_buf[-_conv_window:]
                    if _conv_topo_post_buf:
                        _conv_topo_post_buf = _conv_topo_post_buf[-_conv_window:]

                # Check once buffer is full
                if len(_conv_vq_buf) >= _conv_window:
                    all_met = True
                    details = []

                    if tc.convergence_stop_vq is not None:
                        mean_vq = sum(_conv_vq_buf) / len(_conv_vq_buf)
                        met = mean_vq < tc.convergence_stop_vq
                        all_met &= met
                        details.append(f"vq={mean_vq:.5f}<{tc.convergence_stop_vq}")

                    if tc.convergence_stop_recon is not None:
                        mean_recon = sum(_conv_recon_buf) / len(_conv_recon_buf)
                        met = mean_recon < tc.convergence_stop_recon
                        all_met &= met
                        details.append(f"recon={mean_recon:.5f}<{tc.convergence_stop_recon}")

                    if tc.convergence_stop_info is not None:
                        mean_info = sum(_conv_info_buf) / len(_conv_info_buf)
                        met = mean_info < tc.convergence_stop_info
                        all_met &= met
                        details.append(f"info={mean_info:.5f}<{tc.convergence_stop_info}")

                    if tc.convergence_stop_topo_post is not None and _conv_topo_post_buf:
                        mean_topo = sum(_conv_topo_post_buf) / len(_conv_topo_post_buf)
                        met = mean_topo >= tc.convergence_stop_topo_post
                        all_met &= met
                        details.append(f"topo_post={mean_topo:.5f}>={tc.convergence_stop_topo_post}")

                    if all_met:
                        logger.info(
                            f"  Convergence stopping triggered at E{epoch+1} "
                            f"B{batch_idx+1}/{num_batches_total} "
                            f"(window={_conv_window}): {', '.join(details)}"
                        )
                        self._convergence_stopped = True
                        break

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
                unpacked = self._unpack_batch(batch)
                temporal_feats = unpacked['temporal_features']
                initial_man = unpacked['initial_manual']
                initial_r = unpacked['initial_raw']
                theta_feats = unpacked['theta_features']
                temp_mask = unpacked['temporal_mask']
                temp_lens = unpacked['temporal_lengths']

                # On-the-fly trajectory generation for learned mode
                temporal_raw = self._generate_trajectories(theta_feats, initial_r)

                # Subsample trajectory keyframes for trajectory prototype head
                trajectory_targets = self._compute_trajectory_targets(temporal_raw)

                # Random length sampling for variable-length training (manual mode only)
                if self.length_sampler is not None and temporal_feats is not None:
                    B, T, D = temporal_feats.shape

                    # Sample random lengths for this batch (same distribution as training)
                    sampled_lengths = self.length_sampler.sample_lengths(B)  # [B]

                    # Create mask from sampled lengths
                    sampled_mask = self.length_sampler.create_mask(sampled_lengths, T)  # [B, T]

                    # Override the all-True mask from dataset
                    temp_mask = sampled_mask.to(self.device)
                    temp_lens = sampled_lengths.to(self.device)

                # Forward pass — temporal-only mode skips theta/initial inputs
                if self.config.temporal_only:
                    outputs = self.model(
                        temporal_features=temporal_feats,
                        temporal_mask=temp_mask,
                        temporal_lengths=temp_lens,
                        temporal_raw=temporal_raw,
                    )
                else:
                    outputs = self.model(
                        temporal_features=temporal_feats,
                        initial_manual=initial_man,
                        initial_raw=initial_r,
                        theta_features=theta_feats,
                        temporal_mask=temp_mask,
                        temporal_lengths=temp_lens,
                        temporal_raw=temporal_raw,
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

                # Determine num_groups for group balance loss (dynamic from model)
                _num_groups = sum(
                    1 for k in self.group_indices if k.startswith("temporal_")
                ) if outputs.get('cnn_features') is not None else None

                losses = self.loss_fn(
                    original=outputs['original_encoded'],
                    reconstructed=outputs['reconstructed'],
                    vq_loss=outputs['vq_loss'],
                    category_embeddings=category_embeddings,
                    quantized_vectors=outputs.get('encodings'),
                    token_indices=outputs.get('token_indices'),
                    codebooks=codebooks,
                    latent_vectors=outputs.get('latents'),
                    model=self.model,
                    tokens=tokens,
                    decoded=decoded,
                    initial_manual=initial_man,
                    original_theta=theta_feats,
                    original_initial=initial_r,
                    cnn_features=outputs.get('cnn_features'),
                    num_groups=_num_groups,
                    gate_values=outputs.get('gate_values'),
                    trajectory_targets=trajectory_targets,
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

        Uses ``per_group_encoded`` (direct per-group encoded tensors) when
        available, avoiding stale index-based slicing into the concatenated
        ``original_encoded`` tensor which has a different layout.

        Args:
            outputs: Model forward outputs

        Returns:
            Dict mapping category → embeddings [B, D_cat]
        """
        per_group = outputs.get("per_group_encoded")
        if per_group:
            return per_group

        # Fallback for older checkpoints without per_group_encoded
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

    def _build_resume_metadata(self) -> Dict[str, Any]:
        """Build metadata dict containing all state needed for training resume."""
        return {
            'training_history': self.training_history,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_ema_state': dict(self.loss_fn._ema),
            'best_val_loss': self.best_val_loss,
            'best_es_metric': self._best_es_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
        }

    def resume_state(self, checkpoint: 'TokenizerCheckpoint') -> int:
        """Restore training state from a checkpoint.

        Loads optimizer, scheduler, loss EMA, and tracking state.
        Model state dict should be loaded before constructing the trainer
        (so the optimizer wraps the correct parameters).

        Args:
            checkpoint: TokenizerCheckpoint from load_checkpoint()

        Returns:
            start_epoch: Epoch to resume from (checkpoint.epoch + 1)
        """
        # Optimizer
        if checkpoint.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Restored optimizer state")

        # Scheduler, loss EMA, tracking — stored in metadata
        meta = checkpoint.metadata or {}

        if self.scheduler is not None and meta.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(meta['scheduler_state_dict'])
            logger.info("Restored LR scheduler state")

        if 'loss_ema_state' in meta:
            self.loss_fn._ema.update(meta['loss_ema_state'])
            logger.info("Restored loss EMA normalization state")

        self.best_val_loss = meta.get('best_val_loss', float('inf'))
        self._best_es_metric = meta.get('best_es_metric', float('inf'))
        self.epochs_without_improvement = meta.get('epochs_without_improvement', 0)

        if 'training_history' in meta:
            self.training_history = meta['training_history']

        start_epoch = (checkpoint.epoch or 0) + 1
        logger.info(
            f"Resuming from epoch {start_epoch}/{self.config.training.num_epochs}, "
            f"best_val={self.best_val_loss:.6f}, "
            f"patience={self.epochs_without_improvement}"
        )
        return start_epoch

    def _save_checkpoint(
        self, path: Path, epoch: int, val_loss: Optional[float]
    ):
        """Save training checkpoint with full resume state.

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
            metadata=self._build_resume_metadata(),
            temporal_input_dim=self.model.temporal_input_dim,
            initial_input_dim=self.model.initial_input_dim,
            feature_metadata=self.feature_metadata,
        )
