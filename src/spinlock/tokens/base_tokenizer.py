"""Base class for tokenizer high-level interfaces.

Defines the shared API contract that both VQTokenizer and NLTokenizer
implement. Concrete tokenizers handle model creation and training
delegation; this base provides the shared feature pipeline utilities.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from spinlock.data import SpinlockDataset


class BaseTokenizer(ABC):
    """Abstract base for tokenizer high-level interfaces.

    Subclasses must implement:
        - train(): Full training pipeline
        - from_checkpoint(): Load from saved checkpoint

    Provides:
        - Shared __init__ pattern (config, model, group_indices)
        - Normalization and feature metadata bookkeeping
    """

    def __init__(
        self,
        config,
        model=None,
        group_indices: Optional[Dict[str, list]] = None,
    ):
        self.config = config
        self.model = model
        self.group_indices = group_indices
        self.normalization_stats = None
        self.feature_metadata = None

    @abstractmethod
    def train(
        self,
        dataset: Union[SpinlockDataset, str, "Path"],
        output_dir: Union[str, "Path"] = "checkpoints",
        **kwargs,
    ) -> Dict[str, Any]:
        """Train tokenizer on dataset.

        Returns:
            Training history dict
        """
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, "Path"], **kwargs):
        """Load tokenizer from saved checkpoint."""
        ...
