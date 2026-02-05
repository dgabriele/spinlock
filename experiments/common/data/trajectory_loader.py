"""Load trajectories and features from MNO-generated datasets."""

import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


class TrajectoryDataLoader:
    """
    Load trajectories and features from MNO-generated datasets.

    Flexible framework interface: adapts to different dataset configurations.
    Only loads features that are actually present in the dataset.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.available_features = self._discover_features()

    def _discover_features(self) -> List[str]:
        """Discover which feature families are present in dataset."""
        available = []
        with h5py.File(self.dataset_path, 'r') as f:
            # Check for feature families (summary is deprecated but check anyway)
            feature_families = ['initial', 'temporal', 'summary']
            for family in feature_families:
                family_path = f'features/{family}'
                if family_path in f:
                    available.append(family)
        return available

    def load_features(
        self,
        feature_families: Optional[List[str]] = None,
        indices: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Load features for given indices.

        Args:
            feature_families: Which families to load (None = all available)
            indices: Sample indices to load (None = all)

        Returns:
            Dictionary mapping feature family name to tensor:
            {
                'initial': [N, D_initial],
                'temporal': [N, T, D_temporal],
                'summary': [N, D_summary]  # Only if present
            }
        """
        if feature_families is None:
            feature_families = self.available_features

        features = {}

        with h5py.File(self.dataset_path, 'r') as f:
            for family in feature_families:
                if family not in self.available_features:
                    continue  # Skip unavailable features

                # Load based on family structure
                if family == 'initial':
                    path = 'features/initial/aggregated/features'
                    if path in f:
                        data = f[path][:] if indices is None else f[path][indices]
                        features['initial'] = torch.from_numpy(data)

                elif family == 'temporal':
                    path = 'features/temporal/features'
                    if path in f:
                        data = f[path][:] if indices is None else f[path][indices]
                        features['temporal'] = torch.from_numpy(data)

                elif family == 'summary':
                    # Summary features deprecated but support if present
                    path = 'features/summary/aggregated/features'
                    if path in f:
                        data = f[path][:] if indices is None else f[path][indices]
                        features['summary'] = torch.from_numpy(data)

        return features

    def get_num_samples(self) -> int:
        """Get total number of samples in dataset."""
        with h5py.File(self.dataset_path, 'r') as f:
            # Use first available feature family to determine sample count
            for family in self.available_features:
                if family == 'initial':
                    return f['features/initial/aggregated/features'].shape[0]
                elif family == 'temporal':
                    return f['features/temporal/features'].shape[0]
                elif family == 'summary':
                    return f['features/summary/aggregated/features'].shape[0]
        raise ValueError("No feature families found in dataset")
