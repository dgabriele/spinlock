"""Base trainer for experiments with common training loop."""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
import json
from tqdm import tqdm


class BaseExperimentTrainer:
    """Base trainer for experiments with common training loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        output_dir: Path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir

        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Override in subclasses."""
        raise NotImplementedError

    def validate(self) -> Dict[str, float]:
        """Validate on validation set. Override in subclasses."""
        raise NotImplementedError

    def train(self, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """Main training loop."""
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_metrics"].append(train_metrics)

            # Validation
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_metrics"].append(val_metrics)

            # Logging
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # Checkpointing
            if (epoch + 1) % self.config.training.save_every == 0:
                self._save_checkpoint(epoch, val_metrics["loss"])

        # Save final history
        self._save_history()
        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "history": self.history,
            "config": self.config.model_dump()
        }

        path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
