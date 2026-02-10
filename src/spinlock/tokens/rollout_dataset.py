"""Dataset for Token-to-Rollout VAE training.

This module provides a dataset that pairs pre-computed token indices with
ground truth (theta, initial_grids) from the CNO dataset.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenToRolloutDataset(Dataset):
    """Dataset that pairs tokens with ground truth (theta, initial_grids).

    This dataset loads:
    - Token indices from a pre-tokenized dataset (96 token groups)
    - Theta parameters from CNO dataset (/parameters/params)
    - Initial condition grids from CNO dataset (/inputs/fields)

    The initial grids are averaged across M realizations to create a single
    representative IC per sample.

    Args:
        cno_dataset_path: Path to CNO dataset (e.g., 50k_baseline.h5)
        tokenized_dataset_path: Path to pre-tokenized dataset
        indices: Optional subset of indices to use (for train/val split)
    """

    def __init__(
        self,
        cno_dataset_path: Path,
        tokenized_dataset_path: Path,
        indices: np.ndarray | None = None,
    ):
        self.cno_dataset_path = Path(cno_dataset_path)
        self.tokenized_dataset_path = Path(tokenized_dataset_path)

        # Load CNO dataset metadata
        with h5py.File(self.cno_dataset_path, "r") as f:
            self.theta_shape = f["/parameters/params"].shape
            self.fields_shape = f["/inputs/fields"].shape  # [N, M, C, H, W]

            # Load theta and grids into memory (reasonable for 50K samples)
            self.theta = torch.from_numpy(f["/parameters/params"][:]).float()  # [N, 14]
            fields = torch.from_numpy(f["/inputs/fields"][:]).float()  # [N, M, C, H, W]

            # Average over M realizations to get representative IC
            self.grids = fields.mean(dim=1)  # [N, C, H, W]

        # Load token keys from tokenized dataset
        with h5py.File(self.tokenized_dataset_path, "r") as f:
            # Tokens are stored under the "tokens/" group
            tokens_group = f["tokens"]

            # Get all token group keys (e.g., "temporal_group_1_L0", "initial_group_1_L0", etc.)
            self.token_keys = sorted(list(tokens_group.keys()))

            # Load all token indices into memory
            self.tokens = {
                key: torch.from_numpy(tokens_group[key][:]).long() for key in self.token_keys
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
        assert len(self.token_keys) == 96, f"Expected 96 token groups, got {len(self.token_keys)}"
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
            - tokens: Dict[str, Tensor] of token indices (96 keys)
            - theta: Tensor [theta_dim] operator parameters
            - grids: Tensor [C, H, W] initial condition grids
        """
        return {
            "tokens": {key: self.tokens[key][idx] for key in self.token_keys},
            "theta": self.theta[idx],  # [theta_dim]
            "grids": self.grids[idx],  # [C, H, W]
        }

    @property
    def theta_dim(self) -> int:
        """Dimensionality of theta parameters."""
        return self.theta.shape[1]

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of initial condition grids (C, H, W)."""
        return tuple(self.grids.shape[1:])

    @staticmethod
    def create_splits(
        cno_dataset_path: Path,
        tokenized_dataset_path: Path,
        train_split: float = 0.9,
        seed: int = 42,
    ) -> Tuple["TokenToRolloutDataset", "TokenToRolloutDataset"]:
        """Create train/val split datasets.

        Args:
            cno_dataset_path: Path to CNO dataset
            tokenized_dataset_path: Path to pre-tokenized dataset
            train_split: Fraction of data for training
            seed: Random seed for reproducible splits

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Get total dataset size
        with h5py.File(cno_dataset_path, "r") as f:
            total_size = f["/parameters/params"].shape[0]

        # Create random split
        rng = np.random.default_rng(seed)
        indices = np.arange(total_size)
        rng.shuffle(indices)

        split_idx = int(total_size * train_split)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        # Create datasets
        train_dataset = TokenToRolloutDataset(
            cno_dataset_path, tokenized_dataset_path, indices=train_indices
        )
        val_dataset = TokenToRolloutDataset(
            cno_dataset_path, tokenized_dataset_path, indices=val_indices
        )

        return train_dataset, val_dataset


def collate_token_rollout_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for TokenToRolloutDataset.

    Args:
        batch: List of sample dicts from dataset

    Returns:
        Batched dictionary with:
        - tokens: Dict[str, Tensor] of shape [B] for each token key
        - theta: Tensor [B, theta_dim]
        - grids: Tensor [B, C, H, W]
    """
    # Stack tokens (each key independently)
    token_keys = batch[0]["tokens"].keys()
    tokens_batched = {
        key: torch.stack([sample["tokens"][key] for sample in batch]) for key in token_keys
    }

    # Stack theta and grids
    theta_batched = torch.stack([sample["theta"] for sample in batch])
    grids_batched = torch.stack([sample["grids"] for sample in batch])

    return {
        "tokens": tokens_batched,
        "theta": theta_batched,
        "grids": grids_batched,
    }
