"""Dataset for Token-Conditioned CVAE training.

This module provides a dataset that pairs pre-computed temporal token indices
with ground truth (theta, initial_grids) from the dataset.

Unlike TokenToRolloutDataset, this dataset:
- Filters to temporal-family keys only (removes initial/theta tokens)
- Uses generic naming (no operator-specific names like "cno")
- Supports the CVAE's temporal-only conditioning
- Handles realization-expanded datasets (N params × M realizations = N*M tokens)
- Supports truncation_length filtering for multi-truncation pretokenized datasets
- Preloads grids as uint8 for fast random access (shared via fork COW)
- Preserves Sobol ordering (sequential split, no shuffle)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from spinlock.tokens.schema import TokenSchema, _TRUNC_RE, strip_trunc_suffix

logger = logging.getLogger(__name__)


class CVAEDataset(Dataset):
    """Dataset that pairs temporal tokens with ground truth (theta, initial_grids).

    This dataset loads:
    - Token indices from a pre-tokenized dataset, filtered to temporal-family keys
    - Theta parameters from the dataset (/parameters/params)
    - Initial condition grids from the dataset (/inputs/fields)

    Supports two modes:

    **Standard mode** (token count == param count):
        ICs are averaged across realizations. All data in memory.

    **Realization-expanded mode** (token count == param count × num_realizations):
        Each realization is a separate token sample. Theta is repeated via index
        mapping. ICs are loaded lazily from HDF5 per-worker (fork-safe).

    h5py handles are opened lazily in __getitem__ so each DataLoader worker
    gets its own handle after fork(). This enables num_workers > 0.

    Args:
        dataset_path: Path to dataset HDF5 (e.g., ds_lenia_v1.h5)
        tokenized_dataset_path: Path to pre-tokenized dataset
        temporal_keys_only: If True, filter to temporal-family keys only
        truncation_length: If set, filter to this truncation and remap keys
            to base form (e.g., temporal_group_0_trunc_T512_L0 → temporal_group_0_L0)
        max_samples: If set, limit to this many samples
        indices: Optional subset of indices to use (for train/val split)
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenized_dataset_path: Path,
        temporal_keys_only: bool = True,
        truncation_length: Optional[int] = None,
        max_samples: Optional[int] = None,
        indices: np.ndarray | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.tokenized_dataset_path = Path(tokenized_dataset_path)
        self._truncation_length = truncation_length

        # Load theta (always fits in memory)
        with h5py.File(self.dataset_path, "r") as f:
            self._raw_theta = torch.from_numpy(f["/parameters/params"][:]).float()  # [N, P]
            fields_shape = f["/inputs/fields"].shape  # [N, M, C, H, W]
            self._num_params = fields_shape[0]
            self._num_realizations = fields_shape[1]
            self._grid_channels = fields_shape[2]
            self._grid_h = fields_shape[3]
            self._grid_w = fields_shape[4]

        # Load and filter token keys
        trunc_tag = (
            f"_trunc_T{truncation_length:03d}_"
            if truncation_length is not None
            else None
        )

        with h5py.File(self.tokenized_dataset_path, "r") as f:
            tokens_group = f["tokens"]
            all_keys = sorted(list(tokens_group.keys()))

            # Filter by truncation length
            if trunc_tag is not None:
                filtered_keys = [
                    k for k in all_keys
                    if trunc_tag in k or not _TRUNC_RE.match(k)
                ]
            else:
                filtered_keys = all_keys

            # Filter to temporal family
            if temporal_keys_only:
                kept_keys = []
                for k in filtered_keys:
                    base_k = strip_trunc_suffix(k) if trunc_tag is not None else k
                    try:
                        info = TokenSchema.parse_key(base_k)
                        if info.family == "temporal":
                            kept_keys.append(k)
                    except ValueError:
                        continue
                filtered_keys = kept_keys

            # Determine key mapping (truncated → base)
            self.token_keys = []
            self._key_remap = {}  # full_key → base_key
            for k in filtered_keys:
                base_k = strip_trunc_suffix(k) if trunc_tag is not None else k
                self.token_keys.append(base_k)
                self._key_remap[k] = base_k

            # Load token indices
            self._num_token_samples = tokens_group[filtered_keys[0]].shape[0]
            self.tokens = {}
            for full_key in filtered_keys:
                base_key = self._key_remap[full_key]
                self.tokens[base_key] = torch.from_numpy(
                    tokens_group[full_key][:]
                ).long()

        self.token_keys = sorted(self.tokens.keys())

        # Detect realization mode
        if self._num_token_samples == self._num_params:
            self._realization_expanded = False
            logger.info("CVAEDataset: standard mode (token count == param count)")
        elif self._num_token_samples == self._num_params * self._num_realizations:
            self._realization_expanded = True
            logger.info(
                f"CVAEDataset: realization-expanded mode "
                f"({self._num_params} params × {self._num_realizations} realizations "
                f"= {self._num_token_samples} token samples)"
            )
        else:
            raise ValueError(
                f"Token sample count ({self._num_token_samples}) does not match "
                f"param count ({self._num_params}) or param × realizations "
                f"({self._num_params * self._num_realizations})"
            )

        # Determine effective dataset size
        self._effective_size = self._num_token_samples

        # Apply index subset
        if indices is not None:
            self._indices = indices
            self.tokens = {key: val[indices] for key, val in self.tokens.items()}
            self._effective_size = len(indices)
        else:
            self._indices = np.arange(self._effective_size)

        # Apply max_samples limit
        if max_samples is not None and max_samples < self._effective_size:
            self._indices = self._indices[:max_samples]
            self.tokens = {key: val[:max_samples] for key, val in self.tokens.items()}
            self._effective_size = max_samples

        # Load grids into memory for fast random access.
        # Standard mode: average over realizations.
        # Realization-expanded mode: preload the needed param range as uint8
        # (quantized from [0,1] float32). Workers share via fork copy-on-write.
        if not self._realization_expanded:
            with h5py.File(self.dataset_path, "r") as f:
                fields = torch.from_numpy(f["/inputs/fields"][:]).float()
                self.grids = fields.mean(dim=1)  # [N, C, H, W]
            self.theta = self._raw_theta.clone()
            if indices is not None:
                self.theta = self.theta[indices]
                self.grids = self.grids[indices]
            if max_samples is not None and max_samples < len(self.theta):
                self.theta = self.theta[:max_samples]
                self.grids = self.grids[:max_samples]
            self._grids_uint8 = None
        else:
            self.theta = self._raw_theta  # [N_params, P], shared
            self.grids = None

            # Preload grids as uint8 (eliminates per-sample h5py decompression)
            min_expanded = int(self._indices.min())
            max_expanded = int(self._indices.max())
            min_param = min_expanded // self._num_realizations
            max_param = max_expanded // self._num_realizations
            num_params_needed = max_param - min_param + 1
            self._grid_param_offset = min_param

            grid_mem_gb = (
                num_params_needed * self._num_realizations
                * self._grid_channels * self._grid_h * self._grid_w
            ) / (1024 ** 3)
            logger.info(
                f"Preloading grids [{min_param}:{max_param + 1}] "
                f"as uint8 ({grid_mem_gb:.1f} GB)..."
            )

            # Read in chunks to avoid float32 memory spike
            self._grids_uint8 = torch.empty(
                num_params_needed, self._num_realizations,
                self._grid_channels, self._grid_h, self._grid_w,
                dtype=torch.uint8,
            )
            chunk_size = 5000
            with h5py.File(self.dataset_path, "r") as f:
                fields_ds = f["/inputs/fields"]
                for start in range(min_param, max_param + 1, chunk_size):
                    end = min(start + chunk_size, max_param + 1)
                    local_start = start - min_param
                    local_end = local_start + (end - start)
                    chunk = fields_ds[start:end]  # [chunk, M, C, H, W] float32
                    self._grids_uint8[local_start:local_end] = torch.from_numpy(
                        (chunk * 255).clip(0, 255).astype(np.uint8)
                    )
            logger.info("Grid preload complete.")

        logger.info(
            f"CVAEDataset: {self._effective_size} samples, "
            f"{len(self.token_keys)} temporal keys, "
            f"theta_dim={self._raw_theta.shape[1]}, "
            f"grid_shape=({self._grid_channels}, {self._grid_h}, {self._grid_w})"
        )

    def _expanded_to_param_realization(self, idx: int) -> Tuple[int, int]:
        """Map expanded sample index to (param_index, realization_index)."""
        expanded_idx = self._indices[idx] if self._realization_expanded else idx
        param_idx = expanded_idx // self._num_realizations
        real_idx = expanded_idx % self._num_realizations
        return param_idx, real_idx

    def __len__(self) -> int:
        return self._effective_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
            - tokens: Dict[str, Tensor] of temporal token indices
            - theta: Tensor [theta_dim] operator parameters
            - grids: Tensor [C, H, W] initial condition grids
        """
        tokens = {key: self.tokens[key][idx] for key in self.token_keys}

        if self._realization_expanded:
            param_idx, real_idx = self._expanded_to_param_realization(idx)
            theta = self.theta[param_idx]
            # uint8 preloaded → dequantize to float32
            local_idx = param_idx - self._grid_param_offset
            grid = self._grids_uint8[local_idx, real_idx].float() / 255.0
        else:
            theta = self.theta[idx]
            grid = self.grids[idx]

        return {
            "tokens": tokens,
            "theta": theta,
            "grids": grid,
        }

    @property
    def theta_dim(self) -> int:
        """Dimensionality of theta parameters."""
        return self._raw_theta.shape[1]

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of initial condition grids (C, H, W)."""
        return (self._grid_channels, self._grid_h, self._grid_w)

    @property
    def num_temporal_keys(self) -> int:
        """Number of temporal token keys."""
        return len(self.token_keys)

    @staticmethod
    def create_splits(
        dataset_path: Path,
        tokenized_dataset_path: Path,
        temporal_keys_only: bool = True,
        truncation_length: Optional[int] = None,
        max_samples: Optional[int] = None,
        train_split: float = 0.9,
        seed: int = 42,
    ) -> Tuple["CVAEDataset", "CVAEDataset"]:
        """Create train/val split datasets with sequential ordering.

        Uses sequential (non-shuffled) splits to preserve Sobol low-discrepancy
        properties of the dataset. The first train_split fraction of samples
        goes to training, the rest to validation.

        Args:
            dataset_path: Path to dataset HDF5
            tokenized_dataset_path: Path to pre-tokenized dataset
            temporal_keys_only: Filter to temporal-family keys only
            truncation_length: Truncation length to filter to
            max_samples: If set, limit total samples before splitting
            train_split: Fraction of data for training
            seed: Random seed (unused, kept for API compatibility)

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Determine effective dataset size
        with h5py.File(tokenized_dataset_path, "r") as f:
            first_key = sorted(f["tokens"].keys())[0]
            total_token_samples = f["tokens"][first_key].shape[0]

        effective_size = total_token_samples
        if max_samples is not None:
            effective_size = min(effective_size, max_samples)

        # Sequential split (preserves Sobol ordering)
        indices = np.arange(effective_size)
        split_idx = int(effective_size * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_dataset = CVAEDataset(
            dataset_path, tokenized_dataset_path,
            temporal_keys_only=temporal_keys_only,
            truncation_length=truncation_length,
            indices=train_indices,
        )
        val_dataset = CVAEDataset(
            dataset_path, tokenized_dataset_path,
            temporal_keys_only=temporal_keys_only,
            truncation_length=truncation_length,
            indices=val_indices,
        )

        return train_dataset, val_dataset


def collate_cvae_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for CVAEDataset.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Batched dictionary with:
        - tokens: Dict[str, Tensor] of shape [B] for each token key
        - theta: Tensor [B, theta_dim]
        - grids: Tensor [B, C, H, W]
    """
    token_keys = batch[0]["tokens"].keys()
    tokens_batched = {
        key: torch.stack([sample["tokens"][key] for sample in batch])
        for key in token_keys
    }

    theta_batched = torch.stack([sample["theta"] for sample in batch])
    grids_batched = torch.stack([sample["grids"] for sample in batch])

    return {
        "tokens": tokens_batched,
        "theta": theta_batched,
        "grids": grids_batched,
    }
