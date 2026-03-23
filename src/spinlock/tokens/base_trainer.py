"""Base class for tokenizer trainers.

Provides shared training infrastructure: device selection, optimizer/scheduler
creation, data loading (train/val split), batch unpacking, and checkpoint
management. Concrete trainers (VQ, NL) implement the epoch-level training
and validation logic.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

logger = logging.getLogger(__name__)


class BaseTokenizerTrainer(ABC):
    """Abstract base for tokenizer training orchestration.

    Subclasses must implement:
        - _train_epoch(loader, epoch) → metrics dict
        - _validate_epoch(loader) → metrics dict
        - _save_checkpoint(path, epoch, val_loss)
        - _build_resume_metadata() → dict

    Provides:
        - Device selection and model placement
        - Optimizer and LR scheduler creation
        - Per-batch linear LR warmup
        - Train/val dataloader creation (tensor and dataset paths)
        - Batch unpacking to standardized dict
        - Early stopping tracking
        - Training history accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        config,
        group_indices: Dict[str, list],
        normalization_stats: Optional[Dict] = None,
        feature_metadata: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        self.group_indices = group_indices
        self.normalization_stats = normalization_stats
        self.feature_metadata = feature_metadata

        # ── Device ──
        device_cfg = getattr(config.training, "device", "auto")
        if device_cfg == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_cfg)
        self.model.to(self.device)

        # ── Optimizer ──
        opt_name = getattr(config.training, "optimizer", "adam")
        lr = config.training.learning_rate
        wd = getattr(config.training, "weight_decay", 0.0)
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd,
            )
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # ── LR Scheduler ──
        self.scheduler = None
        if getattr(config.training, "use_scheduler", False):
            stype = getattr(config.training, "scheduler_type", "cosine")
            warmup_epochs = getattr(config.training, "warmup_epochs", 0)
            if stype == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.training.num_epochs - warmup_epochs,
                )
            elif stype == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.training.num_epochs // 3,
                    gamma=0.1,
                )
            elif stype == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=0.95,
                )

        # ── Per-batch linear warmup ──
        self._warmup_scheduler = None
        self._warmup_steps_done = 0
        accum = getattr(config.training, "gradient_accumulation_steps", 1)
        warmup_batches = getattr(config.training, "warmup_batches", 0)
        warmup_optim_steps = (
            max(1, (warmup_batches + accum - 1) // accum)
            if warmup_batches > 0
            else 0
        )
        self._warmup_batches = warmup_optim_steps
        if self._warmup_batches > 0:
            self._warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self._warmup_batches,
            )

        # ── Batch unpacking mode ──
        self._dict_batch_mode = False
        self.tensor_map: Dict[str, int] = {}

        # ── Tracking ──
        self.best_val_loss = float("inf")
        self._best_es_metric = float("inf")
        self.epochs_without_improvement = 0
        self.training_history: Dict[str, list] = {
            "train_losses": [],
            "val_losses": [],
            "train_metrics": [],
            "val_metrics": [],
        }

    # ──────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────

    def _create_dataloaders(
        self,
        *,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders from tensors.

        Sequential split (not random) to preserve Sobol ordering.
        """
        self._dict_batch_mode = False

        tensors = []
        self.tensor_map = {}
        tensor_names = [
            ("temporal_features", temporal_features),
            ("initial_manual", initial_manual),
            ("initial_raw", initial_raw),
            ("theta_features", theta_features),
            ("temporal_mask", temporal_mask),
            ("temporal_lengths", temporal_lengths),
        ]
        for name, t in tensor_names:
            if t is not None:
                self.tensor_map[name] = len(tensors)
                tensors.append(t)

        dataset = TensorDataset(*tensors)
        n = len(dataset)
        train_size = int(n * (1.0 - self.config.training.val_split))
        train_ds = Subset(dataset, list(range(train_size)))
        val_ds = Subset(dataset, list(range(train_size, n)))

        bs = self.config.training.batch_size
        shuffle = getattr(self.config.training, "shuffle", False)
        pin = self.device.type == "cuda"

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, pin_memory=pin,
        )
        logger.info(
            "Dataloaders: train=%d val=%d (batch_size=%d)",
            len(train_ds), len(val_ds), bs,
        )
        return train_loader, val_loader

    def _create_dataloaders_from_dataset(
        self, dataset: Dataset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders from a PyTorch Dataset (dict batches)."""
        self._dict_batch_mode = True
        self.tensor_map = {}

        n = len(dataset)
        train_size = int(n * (1.0 - self.config.training.val_split))
        train_ds = Subset(dataset, list(range(train_size)))
        val_ds = Subset(dataset, list(range(train_size, n)))

        bs = self.config.training.batch_size
        shuffle = getattr(self.config.training, "shuffle", False)
        pin = self.device.type == "cuda"

        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs, shuffle=False, pin_memory=pin,
        )
        logger.info(
            "Dataloaders (dataset): train=%d val=%d (batch_size=%d)",
            len(train_ds), len(val_ds), bs,
        )
        return train_loader, val_loader

    def _unpack_batch(self, batch) -> Dict[str, Optional[torch.Tensor]]:
        """Convert batch (tuple or dict) to standardized dict on device."""
        result: Dict[str, Optional[torch.Tensor]] = {
            "temporal_features": None,
            "initial_manual": None,
            "initial_raw": None,
            "theta_features": None,
            "temporal_mask": None,
            "temporal_lengths": None,
        }

        if self._dict_batch_mode:
            # Dict-mode batches from SpinlockDataset
            if "ic" in batch:
                result["initial_raw"] = batch["ic"].to(self.device, non_blocking=True)
            if "params" in batch:
                result["theta_features"] = batch["params"].to(
                    self.device, non_blocking=True,
                )
            if "initial_manual" in batch:
                result["initial_manual"] = batch["initial_manual"].to(
                    self.device, non_blocking=True,
                )
        else:
            # Tensor-mode batches from TensorDataset
            for name, idx in self.tensor_map.items():
                result[name] = batch[idx].to(self.device, non_blocking=True)

        return result

    # ──────────────────────────────────────────────────────────────
    # Abstract methods
    # ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch. Returns metrics dict."""
        ...

    @abstractmethod
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch. Returns metrics dict."""
        ...

    @abstractmethod
    def _save_checkpoint(
        self, path: Path, epoch: int, val_loss: float,
    ) -> None:
        """Save checkpoint to disk."""
        ...

    @abstractmethod
    def _build_resume_metadata(self) -> Dict[str, Any]:
        """Build metadata dict for checkpoint resume."""
        ...
