"""Dataset for Token-Conditioned CVAE training.

This module provides a dataset that pairs pre-computed temporal token indices
with ground truth (theta, initial_grids) from the dataset.

Unlike TokenToRolloutDataset, this dataset:
- Filters to temporal-family keys only (removes initial/theta tokens)
- Uses generic naming (no operator-specific names like "cno")
- Supports the CVAE's temporal-only conditioning
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from spinlock.tokens.schema import TokenSchema


class CVAEDataset(Dataset):
    """Dataset that pairs temporal tokens with ground truth (theta, initial_grids).

    This dataset loads:
    - Token indices from a pre-tokenized dataset, filtered to temporal-family keys
    - Theta parameters from the dataset (/parameters/params)
    - Initial condition grids from the dataset (/inputs/fields)

    The initial grids are averaged across M realizations to create a single
    representative IC per sample.

    Args:
        dataset_path: Path to dataset HDF5 (e.g., 50k_baseline.h5)
        tokenized_dataset_path: Path to pre-tokenized dataset
        temporal_keys_only: If True, filter to temporal-family keys only
        indices: Optional subset of indices to use (for train/val split)
    """

    def __init__(
        self,
        dataset_path: Path,
        tokenized_dataset_path: Path,
        temporal_keys_only: bool = True,
        indices: np.ndarray | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.tokenized_dataset_path = Path(tokenized_dataset_path)

        # Load dataset (theta + grids)
        with h5py.File(self.dataset_path, "r") as f:
            self.theta = torch.from_numpy(f["/parameters/params"][:]).float()  # [N, P]
            fields = torch.from_numpy(f["/inputs/fields"][:]).float()  # [N, M, C, H, W]
            # Average over M realizations to get representative IC
            self.grids = fields.mean(dim=1)  # [N, C, H, W]

        # Load token keys from tokenized dataset, optionally filtering to temporal
        with h5py.File(self.tokenized_dataset_path, "r") as f:
            tokens_group = f["tokens"]
            all_keys = sorted(list(tokens_group.keys()))

            if temporal_keys_only:
                self.token_keys = [
                    k for k in all_keys
                    if TokenSchema.parse_key(k).family == "temporal"
                ]
            else:
                self.token_keys = all_keys

            # Load all token indices into memory
            self.tokens = {
                key: torch.from_numpy(tokens_group[key][:]).long()
                for key in self.token_keys
            }

        # Apply index subset if provided
        if indices is not None:
            self.indices = indices
            self.theta = self.theta[indices]
            self.grids = self.grids[indices]
            self.tokens = {key: val[indices] for key, val in self.tokens.items()}
        else:
            self.indices = np.arange(len(self.theta))

        # Validation
        assert self.theta.shape[0] == self.grids.shape[0], "Theta and grids length mismatch"
        for key, val in self.tokens.items():
            assert val.shape[0] == self.theta.shape[0], f"Token {key} length mismatch"

    def __len__(self) -> int:
        return len(self.theta)

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
        return {
            "tokens": {key: self.tokens[key][idx] for key in self.token_keys},
            "theta": self.theta[idx],
            "grids": self.grids[idx],
        }

    @property
    def theta_dim(self) -> int:
        """Dimensionality of theta parameters."""
        return self.theta.shape[1]

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of initial condition grids (C, H, W)."""
        return tuple(self.grids.shape[1:])

    @property
    def num_temporal_keys(self) -> int:
        """Number of temporal token keys."""
        return len(self.token_keys)

    @staticmethod
    def create_splits(
        dataset_path: Path,
        tokenized_dataset_path: Path,
        temporal_keys_only: bool = True,
        train_split: float = 0.9,
        seed: int = 42,
    ) -> Tuple["CVAEDataset", "CVAEDataset"]:
        """Create train/val split datasets.

        Args:
            dataset_path: Path to dataset HDF5
            tokenized_dataset_path: Path to pre-tokenized dataset
            temporal_keys_only: Filter to temporal-family keys only
            train_split: Fraction of data for training
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Get total dataset size
        with h5py.File(dataset_path, "r") as f:
            total_size = f["/parameters/params"].shape[0]

        # Create random split
        rng = np.random.default_rng(seed)
        indices = np.arange(total_size)
        rng.shuffle(indices)

        split_idx = int(total_size * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_dataset = CVAEDataset(
            dataset_path, tokenized_dataset_path,
            temporal_keys_only=temporal_keys_only,
            indices=train_indices,
        )
        val_dataset = CVAEDataset(
            dataset_path, tokenized_dataset_path,
            temporal_keys_only=temporal_keys_only,
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
