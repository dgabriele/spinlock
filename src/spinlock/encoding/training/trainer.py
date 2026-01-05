"""VQ-VAE trainer for operator feature tokenization.

Simple training loop with:
- 5-component loss function
- Early stopping
- Dead code reset
- Checkpointing
- Validation every N epochs

Ported from unisim.system.training.trainer (simplified, removed multimodal/IC support).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import time
import logging

from ..categorical_vqvae import CategoricalHierarchicalVQVAE
from .losses import compute_total_loss
from .callbacks import EarlyStopping, DeadCodeReset, SmartDeadCodeReset, Checkpointer

logger = logging.getLogger(__name__)


class VQVAETrainer:
    """Trainer for categorical hierarchical VQ-VAE."""

    def __init__(
        self,
        model: CategoricalHierarchicalVQVAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        device: str = "cuda",
        # Loss weights
        orthogonality_weight: float = 0.1,
        informativeness_weight: float = 0.1,
        topo_weight: float = 0.02,
        topo_samples: int = 64,
        # Callbacks
        early_stopping_patience: int = 100,
        early_stopping_min_delta: float = 0.01,
        dead_code_reset_interval: int = 100,
        dead_code_threshold: float = 10.0,
        dead_code_max_reset_fraction: float = 0.25,
        use_smart_reset: bool = False,
        checkpoint_dir: Optional[Path] = None,
        # Optimization
        use_torch_compile: bool = True,
        val_every_n_epochs: int = 5,
        # Logging
        verbose: bool = True,
    ):
        """Initialize trainer.

        Args:
            model: CategoricalHierarchicalVQVAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate
            device: Device to use
            orthogonality_weight: Weight for orthogonality loss
            informativeness_weight: Weight for informativeness loss
            topo_weight: Weight for topographic loss
            topo_samples: Number of samples for topographic loss
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Min delta for early stopping
            dead_code_reset_interval: Interval for dead code reset (0 to disable, only for legacy mode)
            dead_code_threshold: Percentile threshold for dead code detection
            dead_code_max_reset_fraction: Max fraction of codebook to reset at once
            use_smart_reset: Use intelligent SmartDeadCodeReset instead of fixed-interval resets
            checkpoint_dir: Directory for checkpoints (None to disable)
            use_torch_compile: Use torch.compile() for JIT compilation
            val_every_n_epochs: Validate every N epochs
            verbose: Whether to print progress
        """
        # Configure logging if verbose
        if verbose and not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s',
                force=True
            )

        # Configure CUDA optimizations
        if device == "cuda":
            # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
            torch.set_float32_matmul_precision("high")
            if verbose:
                logger.info("Enabled TF32 matmul for faster training")

        self.model = model.to(device)

        # Apply torch.compile() for speedup (PyTorch 2.0+)
        if use_torch_compile and device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="default")
                if verbose:
                    logger.info(
                        "Applied torch.compile() - expect 30-40% speedup after warmup"
                    )
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"torch.compile() failed: {e}, continuing without compilation"
                    )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.verbose = verbose
        self.val_every_n_epochs = val_every_n_epochs

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss weights
        self.orthogonality_weight = orthogonality_weight
        self.informativeness_weight = informativeness_weight
        self.topo_weight = topo_weight
        self.topo_samples = topo_samples

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            verbose=verbose,
        )

        # Dead code reset: use smart reset if requested, otherwise legacy
        if use_smart_reset:
            self.dead_code_reset = SmartDeadCodeReset(
                base_threshold=dead_code_threshold,
                utilization_threshold=0.25,
                min_interval=50,
                lookback_window=10,
                verbose=verbose,
            )
            if verbose:
                logger.info("Using SmartDeadCodeReset (intelligent, condition-based)")
        else:
            self.dead_code_reset = DeadCodeReset(
                interval=dead_code_reset_interval,
                threshold=dead_code_threshold,
                max_reset_fraction=dead_code_max_reset_fraction,
                verbose=verbose,
            )
            if verbose:
                logger.info(f"Using DeadCodeReset (legacy, fixed interval={dead_code_reset_interval})")
        self.checkpointer = (
            Checkpointer(checkpoint_dir, verbose=verbose)
            if checkpoint_dir is not None
            else None
        )

        # Training history
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

    def train_epoch(self):
        """Train for one epoch.

        Returns:
            Tuple of (average training loss, last batch features, last batch raw_ics)
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        last_batch = None
        last_raw_ics = None

        for batch in self.train_loader:
            features = batch["features"].to(self.device)

            # Handle raw_ics for hybrid INITIAL encoder (end-to-end CNN training)
            raw_ics = batch.get("raw_ics")
            if raw_ics is not None:
                raw_ics = raw_ics.to(self.device)

            # Forward pass (pass raw_ics if model supports hybrid INITIAL)
            if raw_ics is not None and hasattr(self.model, 'initial_encoder'):
                outputs = self.model(features, raw_ics=raw_ics)
            elif raw_ics is not None and hasattr(self.model, '_orig_mod') and hasattr(self.model._orig_mod, 'initial_encoder'):
                # Handle torch.compile wrapped model
                outputs = self.model(features, raw_ics=raw_ics)
            else:
                outputs = self.model(features)

            # Compute loss
            # For hybrid INITIAL models, use the expanded input_features as target
            # (includes CNN embeddings that must be reconstructed)
            if "input_features" in outputs:
                targets = {"features": outputs["input_features"]}
            else:
                targets = {"features": features}
            losses = compute_total_loss(
                outputs,
                targets,
                self.model,
                orthogonality_weight=self.orthogonality_weight,
                informativeness_weight=self.informativeness_weight,
                topo_weight=self.topo_weight,
                topo_samples=self.topo_samples,
            )

            loss = losses["total"]

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Save last batch for dead code reset (need both features and raw_ics for hybrid models)
            last_batch = features
            last_raw_ics = raw_ics

        avg_loss = total_loss / n_batches
        return avg_loss, last_batch, last_raw_ics

    def validate(self):
        """Validate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)

                # Handle raw_ics for hybrid INITIAL encoder
                raw_ics = batch.get("raw_ics")
                if raw_ics is not None:
                    raw_ics = raw_ics.to(self.device)

                # Forward pass (pass raw_ics if model supports hybrid INITIAL)
                if raw_ics is not None and hasattr(self.model, 'initial_encoder'):
                    outputs = self.model(features, raw_ics=raw_ics)
                elif raw_ics is not None and hasattr(self.model, '_orig_mod') and hasattr(self.model._orig_mod, 'initial_encoder'):
                    # Handle torch.compile wrapped model
                    outputs = self.model(features, raw_ics=raw_ics)
                else:
                    outputs = self.model(features)

                # Compute loss
                # For hybrid INITIAL models, use the expanded input_features as target
                if "input_features" in outputs:
                    targets = {"features": outputs["input_features"]}
                else:
                    targets = {"features": features}
                losses = compute_total_loss(
                    outputs,
                    targets,
                    self.model,
                    orthogonality_weight=self.orthogonality_weight,
                    informativeness_weight=self.informativeness_weight,
                    topo_weight=self.topo_weight,
                    topo_samples=self.topo_samples,
                )

                loss = losses["total"]
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute validation metrics.

        Returns:
            Dict with utilization, quality, and detailed per-category metrics
        """
        from .metrics import (
            compute_per_category_metrics,
            compute_reconstruction_error,
            compute_quality_score
        )

        # Unwrap compiled model if using torch.compile
        model_for_metrics = self.model
        if hasattr(self.model, '_orig_mod'):
            model_for_metrics = self.model._orig_mod

        # Compute reconstruction error and quality
        reconstruction_error = compute_reconstruction_error(
            model_for_metrics,
            self.val_loader,
            device=self.device
        )
        quality = compute_quality_score(reconstruction_error)

        # Compute detailed metrics on validation set
        detailed_metrics = compute_per_category_metrics(
            model_for_metrics,
            self.val_loader,
            device=self.device,
            max_batches=None  # Use full val set
        )

        # Extract average utilization across all category-levels
        utilization_metrics = [
            v for k, v in detailed_metrics.items()
            if "utilization" in k and "level" in k
        ]

        if utilization_metrics:
            avg_utilization = sum(utilization_metrics) / len(utilization_metrics)
        else:
            avg_utilization = 0.0

        # Return both aggregate and detailed metrics
        result = {
            "utilization": avg_utilization,
            "reconstruction_error": reconstruction_error,
            "quality": quality
        }
        result.update(detailed_metrics)  # Include all detailed metrics

        return result

    def train(self, epochs: int):
        """Train for specified number of epochs.

        Args:
            epochs: Number of epochs to train

        Returns:
            Training history dict
        """
        if self.verbose:
            logger.info("=" * 70)
            logger.info("VQ-VAE TRAINING")
            logger.info("=" * 70)
            logger.info(f"Epochs: {epochs}")
            logger.info(f"Training samples: {len(self.train_loader.dataset)}")
            logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Validation frequency: every {self.val_every_n_epochs} epochs")

        start_time = time.time()
        last_val_loss = None

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, last_batch, last_raw_ics = self.train_epoch()
            self.history["train_loss"].append(train_loss)

            # Validate (only every N epochs or last epoch)
            should_validate = (epoch % self.val_every_n_epochs == 0) or (epoch == epochs)
            if should_validate:
                val_loss = self.validate()
                last_val_loss = val_loss
                self.history["val_loss"].append(val_loss)
            else:
                # Skip validation but append last value for history continuity
                if last_val_loss is not None:
                    self.history["val_loss"].append(last_val_loss)
                else:
                    # First epoch - must validate
                    val_loss = self.validate()
                    last_val_loss = val_loss
                    self.history["val_loss"].append(val_loss)

            # Compute metrics
            metrics = self.compute_metrics()
            self.history["metrics"].append(metrics)

            epoch_time = time.time() - epoch_start

            # Logging
            if self.verbose:
                msg = f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s): "
                msg += f"train_loss={train_loss:.6f}"
                if should_validate:
                    msg += f", val_loss={val_loss:.6f}"
                else:
                    msg += f", val_loss={last_val_loss:.6f} (cached)"

                # Add utilization and quality to log
                util = metrics.get("utilization", 0.0)
                quality = metrics.get("quality", 0.0)
                msg += f", util={util:.1%}, quality={quality:.4f}"

                logger.info(msg)

            # Callbacks
            # 1. Dead code reset
            if last_batch is not None:
                # Check if using SmartDeadCodeReset (needs additional params)
                if isinstance(self.dead_code_reset, SmartDeadCodeReset):
                    current_util = metrics.get("utilization", 0.0)

                    # Extract per-category utilization for smarter resets
                    per_category_utils = {}
                    for key, val in metrics.items():
                        if "/utilization" in key and "/level_" in key:
                            # Extract category name from "cluster_1/level_0/utilization"
                            category = key.split("/")[0]
                            if category not in per_category_utils:
                                per_category_utils[category] = []
                            per_category_utils[category].append(val)

                    # Average utilization across levels for each category
                    per_category_utils = {
                        cat: sum(utils) / len(utils)
                        for cat, utils in per_category_utils.items()
                    }

                    # Extract feature counts per category from model config
                    per_category_feature_counts = {}
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'group_indices'):
                        # Unwrap compiled model if needed
                        model_for_config = self.model
                        if hasattr(self.model, '_orig_mod'):
                            model_for_config = self.model._orig_mod

                        if hasattr(model_for_config.config, 'group_indices'):
                            for category, feature_indices in model_for_config.config.group_indices.items():
                                per_category_feature_counts[category] = len(feature_indices)

                    self.dead_code_reset(
                        self.model,
                        last_batch,
                        epoch,
                        current_util,
                        val_loss,
                        self.early_stopping.counter,
                        per_category_utils,
                        per_category_feature_counts,
                        raw_ics=last_raw_ics,
                    )
                else:
                    # Legacy DeadCodeReset (fixed interval)
                    self.dead_code_reset(self.model, last_batch, epoch, raw_ics=last_raw_ics)

            # 2. Checkpointing (only when we validated)
            if should_validate and self.checkpointer is not None:
                self.checkpointer(self.model, self.optimizer, val_loss, epoch, metrics)

            # 3. Early stopping (only when we validated)
            if should_validate and self.early_stopping(val_loss, epoch):
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final metrics
        final_metrics = self.compute_metrics()
        self.history["final_metrics"] = final_metrics

        elapsed = time.time() - start_time

        if self.verbose:
            logger.info("=" * 70)
            logger.info("TRAINING COMPLETE")
            logger.info(f"Total time: {elapsed:.1f}s")
            logger.info("Final metrics:")
            for key, val in final_metrics.items():
                if isinstance(val, float):
                    logger.info(f"  {key}: {val:.4f}")
            logger.info("=" * 70)

        return self.history
