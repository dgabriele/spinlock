"""NOA Dataset - Load ICs and ground-truth features from HDF5.

Provides datasets for NOA Phase 1 training on real data.
Loads pre-computed features from the dataset HDF5 file.

ALL DIMENSIONS ARE RESOLVED DYNAMICALLY AT RUNTIME.
No hardcoded feature counts - the system adapts to whatever the dataset contains.

NaN Handling:
    The dataset can optionally use a FeaturePreprocessor to clean NaN features.
    This ensures consistency with VQ-VAE training, which also filters NaN features.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
import h5py

if TYPE_CHECKING:
    from spinlock.features.preprocessing import FeaturePreprocessor


class NOARealDataset(Dataset):
    """Dataset loading ICs and ground-truth features from HDF5.

    Loads:
    - Initial conditions from inputs/fields
    - Ground-truth SUMMARY features from features/summary/
    - Ground-truth TEMPORAL features from features/temporal/
    - Ground-truth INITIAL features from features/initial/

    All dimensions are determined dynamically from the HDF5 file.

    Args:
        hdf5_path: Path to dataset HDF5 file
        n_samples: Number of samples to load (None = all)
        use_per_trajectory: If True, use per_trajectory features instead of aggregated
        realization_idx: Which realization to use for ICs (default: 0)
        preprocessor: Optional FeaturePreprocessor to clean NaN features

    Example:
        >>> from spinlock.features import FeaturePreprocessor
        >>> preprocessor = FeaturePreprocessor.from_dataset("datasets/100k_full_features.h5")
        >>> dataset = NOARealDataset("datasets/100k_full_features.h5", n_samples=1000, preprocessor=preprocessor)
        >>> sample = dataset[0]
        >>> sample['ic'].shape  # [C, H, W] - determined from file
        >>> sample['summary'].shape  # [D_summary] - NaN features filtered
        >>> sample['temporal'].shape  # [T, D_temporal] - NaN features filtered
    """

    def __init__(
        self,
        hdf5_path: str,
        n_samples: Optional[int] = None,
        use_per_trajectory: bool = False,
        realization_idx: int = 0,
        preprocessor: Optional['FeaturePreprocessor'] = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.use_per_trajectory = use_per_trajectory
        self.realization_idx = realization_idx
        self.preprocessor = preprocessor

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")

        # Load data and discover dimensions dynamically
        self._load_data(n_samples)

    def _load_data(self, n_samples: Optional[int]):
        """Load data from HDF5 file into memory."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Determine sample count dynamically
            total_samples = f['inputs/fields'].shape[0]
            if n_samples is not None:
                n_samples = min(n_samples, total_samples)
            else:
                n_samples = total_samples

            # Load initial conditions - shape determined from file
            # [N, M, H, W] → [N, 1, H, W] using specified realization
            ics = f['inputs/fields'][:n_samples, self.realization_idx:self.realization_idx+1]
            self.ics = torch.from_numpy(ics).float()

            # Load SUMMARY features - shape determined from file
            if self.use_per_trajectory:
                per_traj = f['features/summary/per_trajectory/features'][:n_samples]
                self.summary = torch.from_numpy(per_traj[:, self.realization_idx]).float()
                summary_family = 'summary_per_trajectory'
            else:
                aggregated = f['features/summary/aggregated/features'][:n_samples]
                self.summary = torch.from_numpy(aggregated).float()
                summary_family = 'summary_aggregated'

            # Load TEMPORAL features - shape determined from file
            temporal = f['features/temporal/features'][:n_samples]
            self.temporal = torch.from_numpy(temporal).float()

            # Load INITIAL features - shape determined from file
            initial = f['features/initial/aggregated/features'][:n_samples]
            self.initial = torch.from_numpy(initial).float()

            # Store sample count
            self.n_samples = n_samples

            # Store metadata discovered from file
            self._metadata = self._extract_metadata(f)

        # Apply preprocessing if provided (clean NaN features)
        if self.preprocessor is not None:
            self.summary = self.preprocessor.clean_features(self.summary, summary_family)
            self.temporal = self.preprocessor.clean_features(self.temporal, 'temporal')
            self.initial = self.preprocessor.clean_features(self.initial, 'initial')
            # Update metadata with cleaned dimensions
            self._metadata['summary_dim_raw'] = self._metadata.get('summary_dim', 0)
            self._metadata['summary_dim'] = self.summary.shape[-1]
            self._metadata['temporal_dim_raw'] = self._metadata.get('temporal_shape', (0, 0))[-1]
            self._metadata['temporal_shape'] = self.temporal.shape[1:]

        # Replace any remaining NaN values with 0
        # This handles edge cases where NaN wasn't filtered by preprocessor
        self.summary = torch.nan_to_num(self.summary, nan=0.0)
        self.temporal = torch.nan_to_num(self.temporal, nan=0.0)
        self.initial = torch.nan_to_num(self.initial, nan=0.0)

    def _extract_metadata(self, f: h5py.File) -> Dict[str, Any]:
        """Extract metadata from HDF5 file."""
        metadata = {
            'total_samples': f['inputs/fields'].shape[0],
            'ic_shape': self.ics.shape[1:],
            'summary_dim': self.summary.shape[-1],
            'temporal_shape': self.temporal.shape[1:],
            'initial_dim': self.initial.shape[-1],
        }

        # Extract config from attributes if available
        if 'features/summary' in f:
            summary_grp = f['features/summary']
            if 'num_features' in summary_grp.attrs:
                metadata['summary_num_features'] = int(summary_grp.attrs['num_features'])
            if 'extraction_config' in summary_grp.attrs:
                metadata['summary_config'] = summary_grp.attrs['extraction_config']

        return metadata

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary with:
                'ic': Initial condition [C, H, W]
                'summary': Ground-truth SUMMARY features [D_summary]
                'temporal': Ground-truth TEMPORAL features [T, D_temporal]
                'initial': Ground-truth INITIAL features [D_initial]
        """
        return {
            'ic': self.ics[idx],
            'summary': self.summary[idx],
            'temporal': self.temporal[idx],
            'initial': self.initial[idx],
        }

    @property
    def ic_shape(self) -> Tuple[int, ...]:
        """Shape of initial conditions [C, H, W] - determined from file."""
        return tuple(self.ics.shape[1:])

    @property
    def summary_dim(self) -> int:
        """Dimension of SUMMARY features - determined from file."""
        return self.summary.shape[-1]

    @property
    def temporal_shape(self) -> Tuple[int, int]:
        """Shape of TEMPORAL features [T, D] - determined from file."""
        return tuple(self.temporal.shape[1:])

    @property
    def temporal_steps(self) -> int:
        """Number of timesteps in TEMPORAL features."""
        return self.temporal.shape[1]

    @property
    def temporal_dim(self) -> int:
        """Per-timestep feature dimension."""
        return self.temporal.shape[2]

    @property
    def initial_dim(self) -> int:
        """Dimension of INITIAL features - determined from file."""
        return self.initial.shape[-1]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dataset metadata discovered from HDF5 file."""
        return self._metadata

    def get_dimension_info(self) -> Dict[str, Any]:
        """Get all dimension information for this dataset.

        Useful for configuring downstream models.

        Returns:
            Dictionary with all dimension info
        """
        return {
            'n_samples': self.n_samples,
            'ic_shape': self.ic_shape,
            'ic_channels': self.ic_shape[0],
            'ic_height': self.ic_shape[1],
            'ic_width': self.ic_shape[2],
            'summary_dim': self.summary_dim,
            'temporal_steps': self.temporal_steps,
            'temporal_dim': self.temporal_dim,
            'initial_dim': self.initial_dim,
            'use_per_trajectory': self.use_per_trajectory,
        }


class NOARealDatasetStreaming(Dataset):
    """Streaming dataset for large HDF5 files.

    Unlike NOARealDataset, this version reads from HDF5 on-the-fly
    without loading everything into memory. Slower but memory-efficient.

    All dimensions are determined dynamically from the HDF5 file.

    Args:
        hdf5_path: Path to dataset HDF5 file
        n_samples: Number of samples to use (None = all)
        use_per_trajectory: If True, use per_trajectory features instead of aggregated
        realization_idx: Which realization to use for ICs
    """

    def __init__(
        self,
        hdf5_path: str,
        n_samples: Optional[int] = None,
        use_per_trajectory: bool = False,
        realization_idx: int = 0,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.use_per_trajectory = use_per_trajectory
        self.realization_idx = realization_idx

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")

        # Discover dimensions and sample count
        with h5py.File(self.hdf5_path, 'r') as f:
            total = f['inputs/fields'].shape[0]
            self.n_samples = min(n_samples, total) if n_samples else total

            # Discover shapes from first sample
            self._ic_shape = f['inputs/fields'].shape[2:]  # [H, W]
            self._summary_shape = f['features/summary/aggregated/features'].shape[1:]
            self._temporal_shape = f['features/temporal/features'].shape[1:]
            self._initial_shape = f['features/initial/aggregated/features'].shape[1:]

        self._file = None

    def _get_file(self):
        """Get HDF5 file handle (lazy initialization)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self._file

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from HDF5."""
        f = self._get_file()

        # Load IC
        ic = f['inputs/fields'][idx, self.realization_idx:self.realization_idx+1]
        ic = torch.from_numpy(ic).float()

        # Load SUMMARY
        if self.use_per_trajectory:
            summary = f['features/summary/per_trajectory/features'][idx, self.realization_idx]
        else:
            summary = f['features/summary/aggregated/features'][idx]
        summary = torch.from_numpy(summary).float()

        # Load TEMPORAL
        temporal = f['features/temporal/features'][idx]
        temporal = torch.from_numpy(temporal).float()

        # Load INITIAL
        initial = f['features/initial/aggregated/features'][idx]
        initial = torch.from_numpy(initial).float()

        return {
            'ic': ic,
            'summary': summary,
            'temporal': temporal,
            'initial': initial,
        }

    @property
    def ic_shape(self) -> Tuple[int, ...]:
        """Shape of initial conditions [C, H, W]."""
        return (1,) + self._ic_shape

    @property
    def summary_dim(self) -> int:
        """Dimension of SUMMARY features."""
        return self._summary_shape[0]

    @property
    def temporal_shape(self) -> Tuple[int, int]:
        """Shape of TEMPORAL features [T, D]."""
        return self._temporal_shape

    @property
    def temporal_steps(self) -> int:
        """Number of timesteps."""
        return self._temporal_shape[0]

    @property
    def temporal_dim(self) -> int:
        """Per-timestep feature dimension."""
        return self._temporal_shape[1]

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if self._file is not None:
            self._file.close()
