"""Shared feature preprocessing for VQ-VAE and NOA pipelines.

This module provides consistent feature cleaning (NaN removal, masking) for both:
- VQ-VAE training on pre-computed dataset features
- NOA training with on-the-fly feature extraction

The FeaturePreprocessor detects 100% NaN features in a dataset and provides
a consistent API for filtering them from both ground-truth and predicted features.

Dataset NaN Features (100k_full_features.h5):
- summary/per_trajectory: indices 110-119 (operator_sensitivity) are 100% NaN
- summary/aggregated: indices 110-119, 230-239, 350-359 (same 10 × 3 aggregations)
- temporal: No NaN features (63 features all valid)

Usage:
    >>> preprocessor = FeaturePreprocessor.from_dataset("datasets/100k_full_features.h5")
    >>> print(f"Valid summary features: {preprocessor.get_valid_count('summary_per_trajectory')}")
    Valid summary features: 110
    >>> clean_features = preprocessor.clean_features(raw_features, 'summary_per_trajectory')
"""

import h5py
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from pathlib import Path


class FeaturePreprocessor:
    """Shared feature preprocessing for VQ-VAE and NOA pipelines.

    Handles:
    - NaN feature detection and masking
    - Feature index mapping (raw → clean)
    - Consistent feature ordering across pipelines

    The preprocessor is initialized from a dataset file and can then be used
    to clean features from that dataset or from newly-extracted features.

    Attributes:
        nan_masks: Dict mapping family names to boolean arrays of NaN positions
        feature_counts: Dict mapping family names to total feature counts
        _valid_indices: Dict mapping family names to arrays of valid indices
    """

    def __init__(
        self,
        nan_masks: Dict[str, np.ndarray],
        feature_counts: Dict[str, int],
    ):
        """Initialize preprocessor with pre-computed NaN masks.

        Args:
            nan_masks: Dict mapping family name to boolean array where True = NaN
            feature_counts: Dict mapping family name to total feature count
        """
        self.nan_masks = nan_masks
        self.feature_counts = feature_counts
        self._valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self) -> Dict[str, np.ndarray]:
        """Compute indices of valid (non-NaN) features for each family."""
        return {
            family: np.where(~mask)[0]
            for family, mask in self.nan_masks.items()
        }

    def clean_features(
        self,
        features: torch.Tensor,
        family: str,
    ) -> torch.Tensor:
        """Remove NaN features, keeping only valid indices.

        Works with any tensor shape where the last dimension is the feature dimension.

        Args:
            features: Tensor with shape [..., D] where D is feature dimension
            family: Feature family name ('summary_per_trajectory', 'summary_aggregated', 'temporal')

        Returns:
            Tensor with shape [..., D_clean] where D_clean <= D
        """
        if family not in self._valid_indices:
            return features

        valid_idx = self._valid_indices[family]
        # Convert to torch tensor for indexing if numpy
        valid_idx_tensor = torch.tensor(valid_idx, device=features.device, dtype=torch.long)
        return features[..., valid_idx_tensor]

    def get_valid_count(self, family: str) -> int:
        """Get number of valid features after cleaning.

        Args:
            family: Feature family name

        Returns:
            Number of valid (non-NaN) features
        """
        return len(self._valid_indices.get(family, []))

    def get_nan_count(self, family: str) -> int:
        """Get number of NaN features.

        Args:
            family: Feature family name

        Returns:
            Number of NaN features
        """
        mask = self.nan_masks.get(family, np.array([]))
        return int(np.sum(mask))

    def get_nan_indices(self, family: str) -> List[int]:
        """Get indices of NaN features (for debugging/logging).

        Args:
            family: Feature family name

        Returns:
            List of indices that are 100% NaN
        """
        mask = self.nan_masks.get(family, np.array([]))
        return np.where(mask)[0].tolist()

    def get_valid_indices(self, family: str) -> np.ndarray:
        """Get indices of valid features.

        Args:
            family: Feature family name

        Returns:
            Array of valid feature indices
        """
        return self._valid_indices.get(family, np.array([]))

    def get_info(self) -> Dict[str, Any]:
        """Get summary information about the preprocessor.

        Returns:
            Dict with counts and indices for all families
        """
        info = {}
        for family in self.nan_masks.keys():
            info[family] = {
                'total': self.feature_counts.get(family, 0),
                'valid': self.get_valid_count(family),
                'nan': self.get_nan_count(family),
                'nan_indices': self.get_nan_indices(family),
            }
        return info

    @classmethod
    def from_dataset(cls, dataset_path: str, sample_size: int = 1000) -> 'FeaturePreprocessor':
        """Create preprocessor by detecting NaN features in dataset.

        Scans the dataset to find features that are 100% NaN across all samples.
        Uses a sample for efficiency on large datasets.

        Args:
            dataset_path: Path to HDF5 dataset file
            sample_size: Number of samples to check (0 = all samples)

        Returns:
            Configured FeaturePreprocessor instance
        """
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        nan_masks = {}
        feature_counts = {}

        with h5py.File(path, 'r') as f:
            # Determine sample indices to check
            total_samples = f['inputs/fields'].shape[0]
            if sample_size > 0 and sample_size < total_samples:
                # Use evenly spaced samples
                indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
            else:
                indices = slice(None)  # All samples

            # Detect NaN in summary/per_trajectory
            if 'features/summary/per_trajectory/features' in f:
                data = f['features/summary/per_trajectory/features'][indices]
                # NaN if ALL values across samples and realizations are NaN
                nan_per_feature = np.isnan(data).all(axis=(0, 1))
                nan_masks['summary_per_trajectory'] = nan_per_feature
                feature_counts['summary_per_trajectory'] = data.shape[-1]

            # Detect NaN in summary/aggregated
            if 'features/summary/aggregated/features' in f:
                data = f['features/summary/aggregated/features'][indices]
                # NaN if ALL values across samples are NaN
                nan_per_feature = np.isnan(data).all(axis=0)
                nan_masks['summary_aggregated'] = nan_per_feature
                feature_counts['summary_aggregated'] = data.shape[-1]

            # Detect NaN in temporal
            if 'features/temporal/features' in f:
                data = f['features/temporal/features'][indices]
                # NaN if ALL values across samples and timesteps are NaN
                nan_per_feature = np.isnan(data).all(axis=(0, 1))
                nan_masks['temporal'] = nan_per_feature
                feature_counts['temporal'] = data.shape[-1]

            # Detect NaN in initial
            if 'features/initial/aggregated/features' in f:
                data = f['features/initial/aggregated/features'][indices]
                nan_per_feature = np.isnan(data).all(axis=0)
                nan_masks['initial'] = nan_per_feature
                feature_counts['initial'] = data.shape[-1]

        return cls(nan_masks, feature_counts)

    @classmethod
    def create_identity(cls, feature_counts: Dict[str, int]) -> 'FeaturePreprocessor':
        """Create a no-op preprocessor that keeps all features.

        Useful for testing or when no NaN filtering is needed.

        Args:
            feature_counts: Dict mapping family names to feature counts

        Returns:
            Preprocessor that keeps all features
        """
        nan_masks = {
            family: np.zeros(count, dtype=bool)
            for family, count in feature_counts.items()
        }
        return cls(nan_masks, feature_counts)

    def __repr__(self) -> str:
        families = list(self.nan_masks.keys())
        return f"FeaturePreprocessor(families={families})"
