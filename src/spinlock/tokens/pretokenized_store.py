"""Shared pretokenized token store for indexed access to HDF5 token datasets.

Provides a simple in-memory token store that loads all tokens from an HDF5 file
and supports fast batch/sample indexing. Used by:
- PretokenizedDiffusionDataset (D3PM training)
- RoundtripConsistencyLoss (MNO roundtrip loss, Mode A)

HDF5 format: /tokens/{quantizer_key} -> [N] int32
Keys follow the convention: "{family}_{category}_L{level}"
"""

import logging
from pathlib import Path
from typing import Dict, List

import h5py
import torch

logger = logging.getLogger(__name__)


class PretokenizedTokenStore:
    """Load pretokenized tokens from HDF5 and provide indexed access.

    All tokens are loaded into CPU memory for fast batch indexing.
    Typical memory footprint: ~50MB for 50K samples x 30 quantizers x 4 bytes.

    Args:
        path: Path to HDF5 file with /tokens group
        device: Device for returned tensors (default: "cpu")
    """

    def __init__(self, path: Path, device: str = "cpu"):
        self.tokens: Dict[str, torch.Tensor] = {}
        self.keys: List[str] = []
        self.num_samples: int = 0
        self._device = device
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        """Read /tokens/ group from HDF5, converting each key to long tensor."""
        with h5py.File(path, "r") as f:
            tokens_group = f["tokens"]
            self.keys = sorted(tokens_group.keys())

            for key in self.keys:
                self.tokens[key] = torch.from_numpy(tokens_group[key][:]).long()

            self.num_samples = self.tokens[self.keys[0]].shape[0]

        logger.info(
            f"PretokenizedTokenStore: {self.num_samples:,} samples, "
            f"{len(self.keys)} quantizer keys from {path}"
        )

    def get_batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Index tokens for a batch.

        Args:
            indices: [B] integer indices into the token arrays

        Returns:
            Dict mapping quantizer key -> [B] long tensor
        """
        return {key: self.tokens[key][indices] for key in self.keys}

    def get_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Single sample access.

        Args:
            idx: Sample index

        Returns:
            Dict mapping quantizer key -> scalar long tensor
        """
        return {key: self.tokens[key][idx] for key in self.keys}

    @property
    def schema(self) -> "TokenSchema":
        """Infer TokenSchema from loaded tokens (max+1 per key for vocab size)."""
        from spinlock.tokens.schema import TokenSchema

        return TokenSchema.from_pretokenized_store(self)
