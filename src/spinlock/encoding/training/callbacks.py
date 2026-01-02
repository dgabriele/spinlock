"""Training callbacks for VQ-VAE training.

Callbacks:
- Early stopping (patience, min_delta)
- Dead code reset (periodic codebook reinitialization)
- Checkpointing (save best model)

Ported from unisim.system.training.callbacks (100% generic).
"""

import torch
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback.

    Stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 100, min_delta: float = 0.01, verbose: bool = True):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Validation loss
            epoch: Current epoch

        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"  [EarlyStopping] Improvement at epoch {epoch}, new best loss: {val_loss:.6f}"
                )
        else:
            # No improvement
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                logger.info(
                    f"  [EarlyStopping] No improvement for {self.counter} epochs (patience: {self.patience})"
                )

            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(
                        f"  [EarlyStopping] Stopping at epoch {epoch}, best loss: {self.best_loss:.6f}"
                    )
                self.should_stop = True
                return True

        return False

    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False


class DeadCodeReset:
    """Dead code reset callback.

    Resets unused codebook entries to high-error samples using EMA percentile thresholds.
    """

    def __init__(
        self,
        interval: int = 100,
        threshold: float = 10.0,
        max_reset_fraction: float = 0.25,
        verbose: bool = True,
    ):
        """Initialize dead code reset.

        Args:
            interval: Reset every N epochs (0 to disable)
            threshold: Percentile threshold (0-100). Codes with EMA cluster sizes
                      below this percentile are reset. Default 10.0 = bottom 10%.
            max_reset_fraction: Maximum fraction of each codebook to reset at once
            verbose: Whether to print messages
        """
        self.interval = interval
        self.threshold = threshold
        self.max_reset_fraction = max_reset_fraction
        self.verbose = verbose

    def __call__(self, model, training_batch, epoch: int) -> int:
        """Reset dead codes if at interval.

        Args:
            model: CategoricalHierarchicalVQVAE model
            training_batch: Recent training batch [batch, input_dim]
            epoch: Current epoch

        Returns:
            Number of dead codes reset
        """
        if self.interval > 0 and epoch > 0 and epoch % self.interval == 0:
            n_reset = model.reset_dead_codes(training_batch, self.threshold)
            if self.verbose:
                if n_reset > 0:
                    logger.info(f"  [DeadCodeReset] Epoch {epoch}: Reset {n_reset} dead codes")
                else:
                    logger.info(f"  [DeadCodeReset] Epoch {epoch}: No dead codes found")
            return n_reset
        return 0


class Checkpointer:
    """Checkpointing callback.

    Saves best model based on validation loss.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        save_every: Optional[int] = None,
        verbose: bool = True,
    ):
        """Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs (None = only save best)
            verbose: Whether to print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.verbose = verbose

        self.best_loss = float("inf")

    def __call__(self, model, optimizer, val_loss: float, epoch: int, metrics: dict):
        """Save checkpoint if best or at interval.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            val_loss: Validation loss
            epoch: Current epoch
            metrics: Metrics dict
        """
        # Check if this is the best model
        is_best = False
        if val_loss < self.best_loss:
            is_best = True
            self.best_loss = val_loss
            if self.verbose:
                logger.info(
                    f"  [Checkpointer] Saved best model at epoch {epoch} (loss: {val_loss:.6f})"
                )

        # Save best model
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            self._save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss, metrics)

        # Save periodic checkpoint
        if self.save_every is not None and epoch % self.save_every == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            self._save_checkpoint(checkpoint_path, model, optimizer, epoch, val_loss, metrics)
            if self.verbose:
                logger.info(f"  [Checkpointer] Saved checkpoint at epoch {epoch}")

    def _save_checkpoint(
        self, path: Path, model, optimizer, epoch: int, val_loss: float, metrics: dict
    ):
        """Save checkpoint to disk.

        Args:
            path: Path to save checkpoint
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            val_loss: Validation loss
            metrics: Metrics dict
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
        }

        # Save model config for categorical models
        if hasattr(model, "config"):
            config_dict = {
                "input_dim": model.config.input_dim,
                "group_indices": model.config.group_indices,
                "group_embedding_dim": model.config.group_embedding_dim,
                "group_hidden_dim": model.config.group_hidden_dim,
            }

            # Add either category_levels or levels
            if (
                hasattr(model.config, "category_levels")
                and model.config.category_levels is not None
            ):
                config_dict["category_levels"] = model.config.category_levels
            elif hasattr(model.config, "levels") and model.config.levels is not None:
                config_dict["levels"] = model.config.levels

            checkpoint["config"] = config_dict

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path, model, optimizer=None):
        """Load checkpoint from disk.

        Args:
            path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Checkpoint dict
        """
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
