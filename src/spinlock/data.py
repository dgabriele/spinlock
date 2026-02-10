"""Dataset utilities for Spinlock.

Provides a unified interface for loading and accessing Spinlock HDF5 datasets.
"""

from pathlib import Path
from typing import Optional
import h5py


class SpinlockDataset:
    """Unified interface for Spinlock HDF5 datasets.

    Provides access to dataset features, inputs, parameters, and metadata.
    """

    def __init__(self, file_path: str):
        """Initialize dataset from HDF5 file.

        Args:
            file_path: Path to HDF5 dataset file
        """
        self.file_path = Path(file_path)
        self._file: Optional[h5py.File] = None
        self._features = None
        self._inputs = None
        self._parameters = None

    @classmethod
    def from_file(cls, file_path: str) -> "SpinlockDataset":
        """Load dataset from HDF5 file.

        Args:
            file_path: Path to HDF5 dataset file

        Returns:
            SpinlockDataset instance
        """
        return cls(file_path)

    def open(self):
        """Open the HDF5 file and provide access to datasets."""
        if self._file is None:
            self._file = h5py.File(self.file_path, 'r')
            # Lazy load features, inputs, and parameters
            if 'features' in self._file:
                self._features = _FeatureGroup(self._file['features'])
            if 'inputs' in self._file:
                self._inputs = _InputGroup(self._file['inputs'])
            if 'parameters' in self._file:
                self._parameters = _ParameterGroup(self._file['parameters'])
        return self

    def close(self):
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._features = None
            self._inputs = None
            self._parameters = None

    def __enter__(self):
        """Context manager entry."""
        return self.open()

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()

    @property
    def features(self):
        """Access feature datasets."""
        if self._features is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._features

    @property
    def inputs(self):
        """Access input datasets."""
        if self._inputs is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._inputs

    @property
    def parameters(self):
        """Access parameter datasets."""
        if self._file is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._parameters


class _FeatureGroup:
    """Wrapper for HDF5 features group."""

    def __init__(self, h5group):
        self._group = h5group

    def __hasattr__(self, name):
        return name in self._group

    @property
    def temporal(self):
        """Access temporal features dataset."""
        # Handle nested structure first (new format): features/temporal/features
        if 'temporal/features' in self._group:
            return _Dataset(self._group['temporal/features'])
        # Handle flat structure (legacy format): features/temporal
        elif 'temporal' in self._group:
            return _Dataset(self._group['temporal'])
        return None

    @property
    def initial(self):
        """Access initial features dataset."""
        # Handle nested structure first (new format): features/initial/aggregated/features
        if 'initial/aggregated/features' in self._group:
            return _Dataset(self._group['initial/aggregated/features'])
        # Handle flat structure (legacy formats)
        elif 'initial/aggregated' in self._group:
            return _Dataset(self._group['initial/aggregated'])
        elif 'initial' in self._group:
            return _Dataset(self._group['initial'])
        return None


class _InputGroup:
    """Wrapper for HDF5 inputs group."""

    def __init__(self, h5group):
        self._group = h5group if h5group is not None else {}

    def load_all(self):
        """Load all input data."""
        if 'fields' in self._group:
            return self._group['fields'][:]
        return None


class _ParameterGroup:
    """Wrapper for HDF5 parameters group."""

    def __init__(self, h5group):
        self._group = h5group if h5group is not None else {}

    @property
    def params(self):
        """Access parameter dataset (theta values)."""
        if 'params' in self._group:
            return _Dataset(self._group['params'])
        return None


class _Dataset:
    """Wrapper for HDF5 dataset."""

    def __init__(self, h5dataset):
        self._dataset = h5dataset

    def load_all(self):
        """Load entire dataset into memory."""
        return self._dataset[:]

    @property
    def shape(self):
        """Get dataset shape."""
        return self._dataset.shape
