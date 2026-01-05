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
    """Dead code reset callback (legacy fixed-interval version).

    Resets unused codebook entries to high-error samples using EMA percentile thresholds.

    NOTE: This is the legacy fixed-interval version. For production training,
    use SmartDeadCodeReset instead which adapts to training dynamics.
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

    def __call__(self, model, training_batch, epoch: int, raw_ics=None) -> int:
        """Reset dead codes if at interval.

        Args:
            model: CategoricalHierarchicalVQVAE model
            training_batch: Recent training batch [batch, input_dim]
            epoch: Current epoch
            raw_ics: Raw initial conditions [batch, C, H, W] (for hybrid models)

        Returns:
            Number of dead codes reset
        """
        if self.interval > 0 and epoch > 0 and epoch % self.interval == 0:
            n_reset = model.reset_dead_codes(training_batch, self.threshold, raw_ics=raw_ics)
            if self.verbose:
                if n_reset > 0:
                    logger.info(f"  [DeadCodeReset] Epoch {epoch}: Reset {n_reset} dead codes")
                else:
                    logger.info(f"  [DeadCodeReset] Epoch {epoch}: No dead codes found")
            return n_reset
        return 0


class SmartDeadCodeReset:
    """Intelligent dead code reset callback.

    Unlike fixed-interval resets, this callback only resets when beneficial:
    - Utilization is actually low and declining
    - Training is not actively improving
    - Respects adaptive intervals (longer as training progresses)
    - Uses gentler reset magnitudes in late training
    - Stops resetting if model has converged

    This prevents the destructive resets that occur with fixed intervals
    when the model has already converged.
    """

    def __init__(
        self,
        base_threshold: float = 10.0,
        utilization_threshold: float = 0.25,
        min_interval: int = 50,
        lookback_window: int = 10,
        verbose: bool = True,
    ):
        """Initialize smart dead code reset.

        Args:
            base_threshold: Base percentile threshold for dead code detection
            utilization_threshold: Minimum utilization to trigger reset consideration
            min_interval: Minimum epochs between resets
            lookback_window: Number of epochs to look back for trend analysis
            verbose: Whether to print messages
        """
        self.base_threshold = base_threshold
        self.utilization_threshold = utilization_threshold
        self.min_interval = min_interval
        self.lookback_window = lookback_window
        self.verbose = verbose

        # State tracking
        self.last_reset_epoch = 0
        self.utilization_history = []
        self.val_loss_history = []
        self.reset_count = 0

    def should_reset(self, epoch: int, current_util: float, current_val_loss: float, early_stopping_counter: int, per_category_utils: dict = None, per_category_feature_counts: dict = None) -> bool:
        """Determine if reset is beneficial based on training state.

        Args:
            epoch: Current epoch
            current_util: Current average codebook utilization
            current_val_loss: Current validation loss
            early_stopping_counter: Current early stopping counter
            per_category_utils: Optional dict of per-category utilization (e.g., {"cluster_1": 0.15, ...})
            per_category_feature_counts: Optional dict of feature counts per category (e.g., {"cluster_1": 10, ...})

        Returns:
            True if reset should be performed
        """
        # Track history
        self.utilization_history.append(current_util)
        self.val_loss_history.append(current_val_loss)

        # CRITICAL: Check per-category utilization if provided
        # Reset if ANY category is critically low (considering feature count)
        if per_category_utils is not None:
            # Use adaptive interval for category resets
            adaptive_interval = self._get_adaptive_interval(epoch)
            min_interval_category = max(adaptive_interval // 2, 25)  # At least 25 epochs

            if epoch - self.last_reset_epoch >= min_interval_category:
                # Check each category with size-aware threshold
                for category, util in per_category_utils.items():
                    # Get feature count for this category (if available)
                    feature_count = None
                    if per_category_feature_counts is not None:
                        feature_count = per_category_feature_counts.get(category)

                    # Compute adaptive threshold based on feature count
                    # Small categories (3-5 features): 5% threshold
                    # Medium categories (10-20 features): 10% threshold
                    # Large categories (50+ features): 15% threshold
                    if feature_count is not None:
                        if feature_count <= 5:
                            threshold = 0.05
                        elif feature_count <= 15:
                            # Interpolate: 5 → 0.05, 15 → 0.10
                            threshold = 0.05 + (feature_count - 5) * 0.005
                        elif feature_count <= 50:
                            # Interpolate: 15 → 0.10, 50 → 0.15
                            threshold = 0.10 + (feature_count - 15) * 0.0014
                        else:
                            threshold = 0.15
                    else:
                        # Fallback: fixed 10% threshold
                        threshold = 0.10

                    # Check if this category is below its adaptive threshold
                    if util < threshold:
                        if self.verbose:
                            logger.info(
                                f"  [SmartReset] CATEGORY EMERGENCY: '{category}' at {util:.3f} util "
                                f"(threshold={threshold:.3f}, {feature_count or '?'} features)"
                            )
                        return True

        # CRITICAL: Detect catastrophic collapse (rapid sustained decline)
        # Override all other conditions if collapse detected
        if len(self.utilization_history) >= 20 and epoch > 20:
            # Check for sustained decline over last 20 epochs
            recent_20 = self.utilization_history[-20:]
            first_10_avg = sum(recent_20[:10]) / 10
            last_10_avg = sum(recent_20[10:]) / 10
            decline = first_10_avg - last_10_avg

            # Catastrophic collapse: >5% absolute decline in 10 epochs
            if decline > 0.05:
                # Respect minimum interval of 10 epochs during collapse
                if epoch - self.last_reset_epoch >= 10:
                    if self.verbose:
                        logger.info(
                            f"  [SmartReset] CATASTROPHIC COLLAPSE detected: "
                            f"util declined {decline:.3f} ({first_10_avg:.3f} → {last_10_avg:.3f})"
                        )
                    return True

        # Condition 1: Respect minimum interval since last reset
        epochs_since_reset = epoch - self.last_reset_epoch
        adaptive_interval = self._get_adaptive_interval(epoch)
        if epochs_since_reset < adaptive_interval:
            return False

        # Condition 2: Utilization is critically low (<15% is emergency)
        if current_util < 0.15:
            # Emergency reset for critically low utilization
            if self.verbose:
                logger.info(f"  [SmartReset] EMERGENCY: Critically low utilization ({current_util:.3f})")
            return True

        # Condition 3: Utilization is low (but not critical)
        if current_util > self.utilization_threshold:
            if self.verbose and epochs_since_reset >= adaptive_interval:
                logger.debug(f"  [SmartReset] Skip: Utilization healthy ({current_util:.3f} > {self.utilization_threshold})")
            return False

        # Condition 4: Utilization is declining OR stable low (changed from AND)
        # We want to reset when low AND (declining OR stable)
        # Previously required declining - now we reset on stable low too
        if len(self.utilization_history) >= 2 * self.lookback_window:
            recent_util = sum(self.utilization_history[-self.lookback_window:]) / self.lookback_window
            prev_util = sum(self.utilization_history[-2*self.lookback_window:-self.lookback_window]) / self.lookback_window

            # If stable low, still proceed (removed the early return)
            is_declining = recent_util < prev_util - 0.01
            is_stable_low = recent_util < self.utilization_threshold and abs(recent_util - prev_util) < 0.01

            if not is_declining and not is_stable_low:
                if self.verbose:
                    logger.debug(f"  [SmartReset] Skip: Utilization improving ({recent_util:.3f} vs {prev_util:.3f})")
                return False

        # Condition 5: Training is not actively improving
        if len(self.val_loss_history) >= 2 * self.lookback_window:
            recent_losses = self.val_loss_history[-self.lookback_window:]
            prev_losses = self.val_loss_history[-2*self.lookback_window:-self.lookback_window]
            recent_avg = sum(recent_losses) / len(recent_losses)
            prev_avg = sum(prev_losses) / len(prev_losses)
            improvement = prev_avg - recent_avg

            # Relaxed threshold: 0.005 instead of 0.01
            if improvement > 0.005:  # Still improving significantly
                if self.verbose:
                    logger.debug(f"  [SmartReset] Skip: Still improving (Δloss={improvement:.4f})")
                return False

        # Condition 6: Not in late convergence (would be disruptive)
        if early_stopping_counter > 30:
            if self.verbose:
                logger.debug(f"  [SmartReset] Skip: In late convergence (ES counter={early_stopping_counter})")
            return False

        # All conditions met - reset is beneficial
        return True

    def _get_adaptive_interval(self, epoch: int) -> int:
        """Get adaptive minimum interval based on training progress.

        Args:
            epoch: Current epoch

        Returns:
            Minimum epochs between resets
        """
        if epoch < 100:
            return self.min_interval  # e.g., 50 early
        elif epoch < 200:
            return self.min_interval * 2  # e.g., 100 mid
        else:
            return self.min_interval * 4  # e.g., 200 late

    def _get_adaptive_threshold(self, epoch: int) -> float:
        """Get adaptive reset threshold based on training progress.

        Gentler resets (higher percentile threshold) as training progresses.

        Args:
            epoch: Current epoch

        Returns:
            Percentile threshold for dead code detection
        """
        if epoch < 100:
            return self.base_threshold  # e.g., 10.0 (bottom 10%)
        elif epoch < 200:
            return self.base_threshold * 0.7  # e.g., 7.0 (bottom 7%)
        else:
            return self.base_threshold * 0.5  # e.g., 5.0 (bottom 5%)

    def __call__(self, model, training_batch, epoch: int, current_util: float, current_val_loss: float, early_stopping_counter: int, per_category_utils: dict = None, per_category_feature_counts: dict = None, raw_ics=None) -> int:
        """Conditionally reset dead codes based on training state.

        Args:
            model: CategoricalHierarchicalVQVAE model
            training_batch: Recent training batch [batch, input_dim]
            epoch: Current epoch
            current_util: Current codebook utilization
            current_val_loss: Current validation loss
            early_stopping_counter: Current early stopping counter
            per_category_utils: Optional dict of per-category utilization
            per_category_feature_counts: Optional dict of feature counts per category
            raw_ics: Raw initial conditions [batch, C, H, W] (for hybrid models)

        Returns:
            Number of dead codes reset
        """
        if epoch == 0:
            return 0

        # Check if reset is beneficial
        if not self.should_reset(epoch, current_util, current_val_loss, early_stopping_counter, per_category_utils, per_category_feature_counts):
            return 0

        # Perform reset with adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold(epoch)
        n_reset = model.reset_dead_codes(training_batch, adaptive_threshold, raw_ics=raw_ics)

        # Update state
        self.last_reset_epoch = epoch
        self.reset_count += 1

        if self.verbose:
            if n_reset > 0:
                logger.info(
                    f"  [SmartReset] Epoch {epoch}: Reset {n_reset} dead codes "
                    f"(util={current_util:.3f}, threshold={adaptive_threshold:.1f}%)"
                )
            else:
                logger.info(f"  [SmartReset] Epoch {epoch}: No dead codes found")

        return n_reset


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
