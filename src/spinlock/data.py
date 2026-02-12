"""Dataset utilities for Spinlock.

Provides a unified interface for loading and accessing Spinlock HDF5 datasets.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import h5py

logger = logging.getLogger(__name__)


class _DatasetDimensionInferrer:
    """Internal helper for introspecting HDF5 dataset structure.

    Discovers dimensions from explicitly declared metadata.
    Requires all datasets to follow NMCHW format with metadata attributes.
    No heuristics, no guessing - fails fast with clear error messages.
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = str(dataset_path)

    def infer_dimensions(self) -> Dict[str, Any]:
        """Infer complete dataset structure."""
        with h5py.File(self.dataset_path, 'r') as f:
            result = {}

            # Number of samples
            if 'parameters/params' in f:
                result['num_samples'] = f['parameters/params'].shape[0]
            elif 'features/temporal/features' in f:
                result['num_samples'] = f['features/temporal/features'].shape[0]
            elif 'inputs/fields' in f:
                result['num_samples'] = f['inputs/fields'].shape[0]
            else:
                raise ValueError("Cannot determine dataset size")

            # Initial manual features
            if 'features/initial/aggregated/features' in f:
                initial_shape = f['features/initial/aggregated/features'].shape
                result['initial_manual_dim'] = initial_shape[1]
            else:
                result['initial_manual_dim'] = None

            # Temporal features
            if 'features/temporal/features' in f:
                temporal_shape = f['features/temporal/features'].shape  # [N, T, D]
                result['temporal_feature_dim'] = temporal_shape[2]
                result['temporal_timesteps'] = temporal_shape[1]
            else:
                result['temporal_feature_dim'] = None
                result['temporal_timesteps'] = None

            # Parameters/theta
            if 'parameters/params' in f:
                params_shape = f['parameters/params'].shape  # [N, P]
                result['theta_param_dim'] = params_shape[1]
            else:
                result['theta_param_dim'] = None

            # Raw initial conditions - REQUIRE explicit metadata
            if 'inputs/fields' not in f:
                raise ValueError(
                    f"Dataset missing required 'inputs/fields'. "
                    f"All Spinlock datasets must contain input fields."
                )

            fields = f['inputs/fields']
            result['initial_raw_shape'] = fields.shape

            # REQUIRE format metadata (no heuristics, no guessing)
            if 'format' not in fields.attrs:
                raise ValueError(
                    f"Dataset missing required 'format' attribute.\n"
                    f"All Spinlock datasets must explicitly declare their format.\n"
                    f"Expected format: 'NMCHW' [N, M, C, H, W]\n"
                    f"To fix: poetry run python scripts/dataset/add_format_metadata.py {self.dataset_path}"
                )

            format_str = str(fields.attrs['format'])
            if format_str != 'NMCHW':
                raise ValueError(
                    f"Unsupported format: '{format_str}'. "
                    f"Spinlock requires format='NMCHW' [N, M, C, H, W]"
                )

            # Read explicit dimensions from metadata
            if 'num_channels' not in fields.attrs or 'num_realizations' not in fields.attrs:
                raise ValueError(
                    f"Dataset missing required attributes 'num_channels' or 'num_realizations'.\n"
                    f"To fix: poetry run python scripts/dataset/add_format_metadata.py {self.dataset_path}"
                )

            result['initial_raw_channels'] = int(fields.attrs['num_channels'])
            result['num_realizations'] = int(fields.attrs['num_realizations'])

            # Validate shape matches metadata
            if len(fields.shape) != 5:
                raise ValueError(
                    f"Invalid shape {fields.shape}. "
                    f"Expected 5D tensor [N, M, C, H, W], got {len(fields.shape)}D"
                )

            N, M, C, H, W = fields.shape
            if M != result['num_realizations']:
                raise ValueError(
                    f"Shape/metadata mismatch: shape[1]={M} but num_realizations={result['num_realizations']}"
                )
            if C != result['initial_raw_channels']:
                raise ValueError(
                    f"Shape/metadata mismatch: shape[2]={C} but num_channels={result['initial_raw_channels']}"
                )

            return result


class SpinlockDataset:
    """Unified interface for Spinlock HDF5 datasets.

    Provides access to dataset features, inputs, parameters, and metadata.
    Automatically introspects dataset structure on open.
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
        self._introspector = None
        self._dimension_cache = None

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

            # Lazy create introspector and run dimension inference
            self._introspector = _DatasetDimensionInferrer(str(self.file_path))
            self._dimension_cache = self._introspector.infer_dimensions()
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

    # Inferion properties (delegated to DatasetInferor)
    @property
    def num_channels(self) -> Optional[int]:
        """Number of input channels (C dimension)."""
        return self._dimension_cache.get('initial_raw_channels') if self._dimension_cache else None

    @property
    def num_realizations(self) -> Optional[int]:
        """Number of realizations (M dimension)."""
        return self._dimension_cache.get('num_realizations') if self._dimension_cache else None

    @property
    def temporal_feature_dim(self) -> Optional[int]:
        """Dimensionality of temporal features."""
        return self._dimension_cache.get('temporal_feature_dim') if self._dimension_cache else None

    @property
    def initial_feature_dim(self) -> Optional[int]:
        """Dimensionality of initial/summary features."""
        return self._dimension_cache.get('initial_manual_dim') if self._dimension_cache else None

    @property
    def theta_param_dim(self) -> Optional[int]:
        """Dimensionality of theta parameters."""
        return self._dimension_cache.get('theta_param_dim') if self._dimension_cache else None

    @property
    def raw_input_shape(self) -> Optional[Tuple]:
        """Raw shape of inputs/fields dataset."""
        return self._dimension_cache.get('initial_raw_shape') if self._dimension_cache else None

    def get_dimension_inference_dict(self) -> Dict[str, Any]:
        """Get complete dimension inference results as dictionary."""
        return self._dimension_cache.copy() if self._dimension_cache else {}

    def get_encoder_config_overrides(self) -> Dict[str, Any]:
        """Get config overrides for encoder based on introspected dimensions."""
        if not self._dimension_cache:
            return {}

        info = self._dimension_cache
        overrides = {}

        # Initial encoder overrides
        if info.get('initial_manual_dim') is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['manual_dim'] = info['initial_manual_dim']

        if info.get('initial_raw_channels') is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['in_channels'] = info['initial_raw_channels']

        # Theta encoder overrides
        if info.get('theta_param_dim') is not None:
            overrides.setdefault('encoder', {}).setdefault('theta', {})['param_dim'] = info['theta_param_dim']

        # Temporal encoder overrides
        if info.get('temporal_timesteps') is not None:
            overrides.setdefault('encoder', {}).setdefault('temporal', {})['max_timesteps'] = max(
                info['temporal_timesteps'],
                overrides.get('encoder', {}).get('temporal', {}).get('max_timesteps', 256)
            )

        return overrides

    def infer_mno_dimensions(self) -> Dict[str, Any]:
        """Infer MNO model dimensions from dataset metadata and shapes.

        Returns MNO-specific dimension overrides for:
        - model.in_channels: Input channels (from dataset metadata)
        - model.out_channels: Output channels (same as input)
        - model.param_dim: Parameter dimension (from parameters/params)
        - model.spatial_dim: Spatial resolution (from field shape, if square)
        - training.max_timesteps: Maximum available timesteps (for validation)

        Returns:
            Dict with model dimension overrides:
            {
                'model': {
                    'in_channels': int,
                    'out_channels': int,
                    'param_dim': int,
                    'spatial_dim': int  # Optional, only if square spatial domain
                },
                'training': {
                    'max_timesteps': int
                }
            }

        Example:
            >>> dataset = SpinlockDataset("datasets/qbm_50k.h5").open()
            >>> mno_dims = dataset.infer_mno_dimensions()
            >>> print(mno_dims)
            {
                'model': {
                    'in_channels': 2,
                    'out_channels': 2,
                    'param_dim': 9,
                    'spatial_dim': 64
                },
                'training': {
                    'max_timesteps': 256
                }
            }
        """
        if not self._dimension_cache:
            return {}

        info = self._dimension_cache
        overrides = {}

        # Channel dimensions (input and output are the same for MNO)
        if info.get('initial_raw_channels') is not None:
            overrides.setdefault('model', {})
            overrides['model']['in_channels'] = info['initial_raw_channels']
            overrides['model']['out_channels'] = info['initial_raw_channels']

        # Parameter dimension
        if info.get('theta_param_dim') is not None:
            overrides.setdefault('model', {})['param_dim'] = info['theta_param_dim']

        # Spatial dimension (from inputs/fields shape [N, M, C, H, W])
        if info.get('initial_raw_shape') is not None:
            shape = info['initial_raw_shape']
            if len(shape) >= 2:
                H, W = shape[-2:]
                if H == W:  # Square spatial domain
                    overrides.setdefault('model', {})['spatial_dim'] = H

        # Max timesteps (for validation)
        if info.get('temporal_timesteps') is not None:
            overrides.setdefault('training', {})['max_timesteps'] = info['temporal_timesteps']

        return overrides

    @classmethod
    def infer_and_update_config(
        cls,
        config_dict: Dict,
        dataset_path: Path,
        verbose: bool = True
    ) -> Tuple['SpinlockDataset', Dict]:
        """Load dataset, introspect, and update config in one operation.

        Opens the dataset once, runs dimension inference, validates and updates config.
        Returns both the opened dataset and updated config.

        Args:
            config_dict: Configuration dictionary (will be deep-merged)
            dataset_path: Path to HDF5 dataset file
            verbose: Log dimension inference results

        Returns:
            (dataset, updated_config_dict) tuple

        Example:
            >>> config_dict = yaml.safe_load(open('config.yaml'))
            >>> dataset, config_dict = SpinlockDataset.infer_and_update_config(
            ...     config_dict, 'datasets/qbm_50k.h5'
            ... )
            >>> config = TokenizerConfig(**config_dict)
            >>> # dataset is already open and ready to use
        """

        # Open dataset (runs dimension inference automatically)
        dataset = cls.from_file(dataset_path).open()

        if verbose:
            logger.info(f"Infered dataset: {dataset_path}")

        # Get encoder config overrides
        overrides = dataset.get_encoder_config_overrides()

        # Deep merge helper
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        # Apply overrides
        config_dict = deep_update(config_dict, overrides)

        if verbose:
            logger.info("Applied dataset-detected dimensions to config")

        return dataset, config_dict


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
