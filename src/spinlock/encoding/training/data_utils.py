"""
Data utilities for VQ-VAE training.

Provides dataset classes and data loader creation functions.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, Tuple


class FeatureDataset(Dataset):
    """Dataset for VQ-VAE training with optional auxiliary data.

    Args:
        features: Pre-encoded features [N, D] or [N, T, D]
        raw_ics: Optional raw initial conditions [N, C, H, W] for hybrid INITIAL
        reference_features: Optional reference solver features [N, D] for regularization
        is_interpolated: Optional boolean mask [N] for interpolated samples
        raw_summary: Optional raw SUMMARY features [N, D] for MNO regularization
        encoded_initial_features: Optional pre-encoded initial features [N, D_initial] for variable-length mode
    """

    def __init__(
        self,
        features: np.ndarray,
        raw_ics: Optional[np.ndarray] = None,
        reference_features: Optional[np.ndarray] = None,
        is_interpolated: Optional[np.ndarray] = None,
        raw_summary: Optional[np.ndarray] = None,
        encoded_initial_features: Optional[np.ndarray] = None,
    ):
        self.features = torch.from_numpy(features).float()

        self.raw_ics = None
        if raw_ics is not None:
            self.raw_ics = torch.from_numpy(raw_ics).float()

        self.reference_features = None
        if reference_features is not None:
            self.reference_features = torch.from_numpy(reference_features).float()

        self.is_interpolated = None
        if is_interpolated is not None:
            self.is_interpolated = torch.from_numpy(is_interpolated).bool()

        self.raw_summary = None
        if raw_summary is not None:
            self.raw_summary = torch.from_numpy(raw_summary).float()

        self.encoded_initial_features = None
        if encoded_initial_features is not None:
            self.encoded_initial_features = torch.from_numpy(encoded_initial_features).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"features": self.features[idx]}

        if self.raw_ics is not None:
            item["raw_ics"] = self.raw_ics[idx]

        if self.reference_features is not None:
            item["reference_features"] = self.reference_features[idx]

        if self.is_interpolated is not None:
            item["is_interpolated"] = self.is_interpolated[idx]

        if self.raw_summary is not None:
            item["raw_summary"] = self.raw_summary[idx]

        if self.encoded_initial_features is not None:
            item["encoded_initial_features"] = self.encoded_initial_features[idx]

        return item


def create_train_val_dataloaders(
    features: np.ndarray,
    config: Dict[str, Any],
    raw_ics: Optional[np.ndarray] = None,
    reference_features: Optional[np.ndarray] = None,
    is_interpolated: Optional[np.ndarray] = None,
    raw_summary: Optional[np.ndarray] = None,
    length_sampler: Optional[Any] = None,
    encoded_initial_features: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders with optional variable-length support.

    Args:
        features: Pre-encoded features [N, D] or [N, T, D]
        config: Training configuration dict
        raw_ics: Optional raw initial conditions for hybrid INITIAL
        reference_features: Optional reference solver features
        is_interpolated: Optional interpolation mask
        raw_summary: Optional raw SUMMARY features
        length_sampler: Optional TrajectoryLengthSampler for variable-length mode
        verbose: Whether to print diagnostic info

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split into train/val (90/10)
    n_samples = len(features)
    n_train = int(0.9 * n_samples)

    # Shuffle
    rng = np.random.RandomState(config.get("random_seed", 42))
    indices = rng.permutation(n_samples)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split auxiliary data
    train_raw_ics = raw_ics[train_indices] if raw_ics is not None else None
    val_raw_ics = raw_ics[val_indices] if raw_ics is not None else None

    train_ref_features = reference_features[train_indices] if reference_features is not None else None
    val_ref_features = reference_features[val_indices] if reference_features is not None else None

    train_is_interpolated = is_interpolated[train_indices] if is_interpolated is not None else None
    val_is_interpolated = is_interpolated[val_indices] if is_interpolated is not None else None

    train_raw_summary = raw_summary[train_indices] if raw_summary is not None else None
    val_raw_summary = raw_summary[val_indices] if raw_summary is not None else None

    train_encoded_initial = encoded_initial_features[train_indices] if encoded_initial_features is not None else None
    val_encoded_initial = encoded_initial_features[val_indices] if encoded_initial_features is not None else None

    # Debug info
    if verbose:
        print(f"  Creating datasets with:")
        print(f"    train_ref_features: {train_ref_features.shape if train_ref_features is not None else None}")
        print(f"    val_ref_features: {val_ref_features.shape if val_ref_features is not None else None}")
        print(f"    train_raw_summary: {train_raw_summary.shape if train_raw_summary is not None else None}")
        print(f"    val_raw_summary: {val_raw_summary.shape if val_raw_summary is not None else None}")
        if length_sampler is not None:
            print(f"    length_sampler: {length_sampler}")

    # Create datasets
    if length_sampler is not None:
        # Variable-length mode: wrap dataset with length sampling
        from ..variable_length_utils import augment_dataset_with_lengths

        train_dataset = augment_dataset_with_lengths(
            FeatureDataset,
            features[train_indices],
            length_sampler,
            raw_ics=train_raw_ics,
            reference_features=train_ref_features,
            is_interpolated=train_is_interpolated,
            raw_summary=train_raw_summary,
            encoded_initial_features=train_encoded_initial,
        )

        val_dataset = augment_dataset_with_lengths(
            FeatureDataset,
            features[val_indices],
            length_sampler,
            raw_ics=val_raw_ics,
            reference_features=val_ref_features,
            is_interpolated=val_is_interpolated,
            raw_summary=val_raw_summary,
            encoded_initial_features=val_encoded_initial,
        )
    else:
        # Standard mode
        train_dataset = FeatureDataset(
            features[train_indices],
            train_raw_ics,
            train_ref_features,
            train_is_interpolated,
            train_raw_summary,
            train_encoded_initial,
        )

        val_dataset = FeatureDataset(
            features[val_indices],
            val_raw_ics,
            val_ref_features,
            val_is_interpolated,
            val_raw_summary,
            val_encoded_initial,
        )

    # Create data loaders
    batch_size = config.get("batch_size", 512)
    val_batch_size = config.get("val_batch_size", batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader
