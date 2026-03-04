"""Dataset utilities for Spinlock.

Provides a unified interface for loading and accessing Spinlock HDF5 datasets.
SpinlockDataset serves two complementary purposes:

1. **Introspection** (metadata-only): Use `infer_and_update_config()` or
   `open()`/`close()` to introspect HDF5 structure without loading data.

2. **PyTorch Dataset**: Pass `max_samples` / `sampling_strategy` to enable
   `__getitem__` / `__len__`. Eagerly loads ICs and params (small enough for RAM);
   optionally lazy-loads GT temporal features per sample from HDF5.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

import numpy as np
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
                result['has_temporal_features'] = True
            else:
                result['temporal_feature_dim'] = None
                result['temporal_timesteps'] = None
                result['has_temporal_features'] = False

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
    Automatically introspects dataset structure on construction.

    **Introspection-only usage** (no data loading):
        dataset = SpinlockDataset("datasets/50k_baseline.h5")
        dims = dataset.infer_mno_dimensions()

    **Training Dataset usage** (implements Dataset-like __len__/__getitem__):
        dataset = SpinlockDataset(
            "datasets/50k_baseline.h5",
            max_samples=10000,
            sampling_strategy="sequential",
            load_gt_temporal_features=True,
        )
        sample = dataset[0]
        # {'ic': Tensor[C,H,W], 'params': Tensor[P], 'sample_idx': int,
        #  'gt_raw_temporal': Tensor[T,D_raw]}  ← only if load_gt_temporal_features=True

    **Context manager usage** (for accessing HDF5 objects directly):
        with SpinlockDataset("datasets/50k_baseline.h5") as ds:
            data = ds.features.temporal.load_all()
    """

    def __init__(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        realization_idx: int = 0,
        sampling_strategy: str = "sequential",
        random_seed: int = 42,
        load_gt_temporal_features: bool = False,
        lazy_ics: bool = False,
        realization_mode: str = "single",
    ):
        """Initialize dataset with optional eager data loading.

        Dimension introspection always runs immediately. Data loading only
        happens when max_samples is provided (or if the dataset is used as
        a PyTorch Dataset).

        Args:
            file_path: Path to HDF5 dataset file.
            max_samples: If set, enables Dataset mode: loads ICs and params
                eagerly, enables __len__/__getitem__. None = introspection only.
            realization_idx: Which realization to use for IC (0 to M-1).
            sampling_strategy: How to subsample when max_samples < total.
                - "sequential": First n samples — preserves Sobol prefix-optimality.
                - "stratified": Uniformly spaced — breaks Sobol prefix-optimality.
                - "random": Reproducible random subset.
            random_seed: Seed for random sampling strategy.
            load_gt_temporal_features: If True and HDF5 contains
                'features/temporal/features', each __getitem__ sample will
                include 'gt_raw_temporal': [T, D_raw]. Loaded lazily per sample.
                Used for feature-space MSE supervision without CNO replayer.
            lazy_ics: When True, skip eager IC loading; read per-sample from
                HDF5 in __getitem__. Params are still loaded eagerly (small).
                This avoids loading all ICs into RAM for large datasets.
            realization_mode: How to handle M realizations per sample.
                - "single": Use realization_idx (existing behavior).
                - "mean": Compute mean across M realizations per sample.
        """
        import torch
        self._torch = torch

        self.file_path = Path(file_path)
        self._file: Optional[h5py.File] = None
        self._features = None
        self._inputs = None
        self._parameters = None

        # Lazy IC loading state
        self._lazy_ics = lazy_ics
        self._realization_mode = realization_mode
        self._realization_idx = realization_idx
        self._lazy_h5: Optional[h5py.File] = None

        # Always run dimension introspection immediately (brief file open)
        self._introspector = _DatasetDimensionInferrer(str(self.file_path))
        self._dimension_cache = self._introspector.infer_dimensions()

        # ── Dataset mode (optional) ───────────────────────────────────────
        # Only enabled when max_samples is provided.
        self._dataset_mode = max_samples is not None
        self.ics = None
        self.params = None
        self.indices = None
        self.n_samples = 0
        self._has_gt_features = False
        self._load_gt_temporal_features = load_gt_temporal_features

        if self._dataset_mode:
            self._load_dataset(
                max_samples=max_samples,
                realization_idx=realization_idx,
                sampling_strategy=sampling_strategy,
                random_seed=random_seed,
                load_gt_temporal_features=load_gt_temporal_features,
            )

    def _load_dataset(
        self,
        max_samples: int,
        realization_idx: int,
        sampling_strategy: str,
        random_seed: int,
        load_gt_temporal_features: bool,
    ) -> None:
        """Eagerly load params (and ICs unless lazy); detect GT feature availability."""
        import torch

        with h5py.File(self.file_path, "r") as f:
            total = f["inputs/fields"].shape[0]
            n = min(max_samples, total)

            if n == total:
                indices = np.arange(n)
            elif sampling_strategy == "sequential":
                indices = np.arange(n)
                print(f"  Using sequential sampling: first {n} samples from {total} total")
                print(f"  ✓ Sequential sampling preserves Sobol prefix-optimality (recommended)")
            elif sampling_strategy == "stratified":
                stride = total // n
                indices = np.arange(0, total, stride)[:n]
                print(f"  Using stratified sampling: {n} samples with stride {stride} from {total} total")
                print(f"  ⚠️  WARNING: Stride sampling breaks Sobol prefix-optimality!")
                print(f"  ⚠️  Consider using sampling_strategy='sequential' for better coverage.")
            elif sampling_strategy == "random":
                np.random.seed(random_seed)
                indices = np.random.choice(total, size=n, replace=False)
                indices = np.sort(indices)
                print(f"  Using random sampling: {n} samples from {total} total (seed={random_seed})")
            else:
                raise ValueError(
                    f"Unknown sampling_strategy: '{sampling_strategy}'. "
                    f"Must be one of: 'sequential', 'stratified', 'random'"
                )

            params = f["parameters/params"][indices]

            if self._lazy_ics:
                # Lazy mode: skip IC loading, validate population via params
                sample_norms = np.linalg.norm(params, axis=1)
                populated_mask = sample_norms > 1e-10
                num_populated = populated_mask.sum()
                if num_populated < params.shape[0]:
                    params = params[populated_mask]
                    indices = indices[populated_mask]
                    print(f"  Dataset pre-allocated to {len(indices)} but only {num_populated} populated")
                    print(f"  Trimmed to {num_populated} non-zero samples (checked via params)")

                self.ics = None  # ICs will be read per-sample in __getitem__

                if self._realization_mode == "all":
                    # Expand: each sample appears M times (one per realization)
                    M = f["inputs/fields"].shape[1]
                    self._realization_map = np.tile(np.arange(M), len(indices))
                    self.indices = np.repeat(indices, M)
                    self.params = torch.from_numpy(np.repeat(params, M, axis=0)).float()
                    self.n_samples = len(self.indices)
                else:
                    self._realization_map = None
                    self.params = torch.from_numpy(params).float()
                    self.indices = indices
                    self.n_samples = len(indices)

                ic_bytes = total * np.prod(f["inputs/fields"].shape[1:]) * 4
                print(
                    f"  Lazy IC mode: skipped {ic_bytes / 1e9:.1f} GB IC load, "
                    f"{self.n_samples} samples indexed, "
                    f"realization_mode='{self._realization_mode}'"
                )
            else:
                # Eager mode: load ICs into RAM
                inputs = f["inputs/fields"][indices, realization_idx, :, :, :]

                # Trim to populated (non-zero) samples — handles pre-allocated HDF5
                sample_norms = np.linalg.norm(inputs.reshape(inputs.shape[0], -1), axis=1)
                populated_mask = sample_norms > 1e-10
                num_populated = populated_mask.sum()
                if num_populated < inputs.shape[0]:
                    inputs = inputs[populated_mask]
                    params = params[populated_mask]
                    indices = indices[populated_mask]
                    print(f"  Dataset pre-allocated to {inputs.shape[0]} but only {num_populated} populated")
                    print(f"  Trimmed to {num_populated} non-zero samples")

                self.ics = torch.from_numpy(inputs).float()   # [N', C, H, W]
                self.params = torch.from_numpy(params).float()
                self.indices = indices
                self.n_samples = self.ics.shape[0]

            self._has_gt_features = (
                load_gt_temporal_features
                and "features/temporal/features" in f
            )
            if load_gt_temporal_features and not self._has_gt_features:
                print(
                    "  ⚠️  load_gt_temporal_features=True but 'features/temporal/features' "
                    "not found in HDF5 — gt_raw_temporal will be absent from batches"
                )
            elif self._has_gt_features:
                print(
                    f"  ✓ GT temporal features available "
                    f"(shape: {f['features/temporal/features'].shape})"
                )

    # ── PyTorch Dataset interface ─────────────────────────────────────────

    def __len__(self) -> int:
        if not self._dataset_mode:
            raise RuntimeError(
                "SpinlockDataset: __len__ requires Dataset mode. "
                "Pass max_samples= to __init__."
            )
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single training sample.

        Returns:
            dict with:
                'ic':               [C, H, W]   initial condition
                'params':           [P]          Sobol parameter vector
                'sample_idx':       int          original HDF5 index
                'gt_raw_temporal':  [T, D_raw]   (only when load_gt_temporal_features=True)
        """
        if not self._dataset_mode:
            raise RuntimeError(
                "SpinlockDataset: __getitem__ requires Dataset mode. "
                "Pass max_samples= to __init__."
            )

        orig_idx = int(self.indices[idx])

        # IC: lazy or eager
        if self._lazy_ics:
            self._ensure_h5_open()
            fields = self._lazy_h5["inputs/fields"]
            if self._realization_mode == "mean":
                # Read all M realizations and average → [C, H, W]
                ic_np = fields[orig_idx].mean(axis=0)  # [M,C,H,W] → [C,H,W]
            elif self._realization_mode == "all":
                real_idx = int(self._realization_map[idx])
                ic_np = fields[orig_idx, real_idx]  # [C, H, W]
            else:
                ic_np = fields[orig_idx, self._realization_idx]  # [C, H, W]
            ic = self._torch.from_numpy(ic_np.astype(np.float32))
        else:
            ic = self.ics[idx]

        result = {
            "ic": ic,
            "params": self.params[idx],
            "sample_idx": orig_idx,
        }
        if self._has_gt_features:
            self._ensure_h5_open()
            feat = self._lazy_h5["features/temporal/features"][orig_idx]
            result["gt_raw_temporal"] = self._torch.from_numpy(feat.astype(np.float32))
        return result

    # ── Lazy HDF5 handle management ──────────────────────────────────────

    def _ensure_h5_open(self):
        """Open persistent HDF5 handle for lazy reads (safe with num_workers=0)."""
        if self._lazy_h5 is None:
            self._lazy_h5 = h5py.File(self.file_path, "r")

    def close_lazy(self):
        """Close the persistent lazy HDF5 handle."""
        if self._lazy_h5 is not None:
            self._lazy_h5.close()
            self._lazy_h5 = None

    # ── Introspection / context manager interface ─────────────────────────

    @classmethod
    def from_file(cls, file_path: str) -> "SpinlockDataset":
        """Load dataset for introspection (no data loading)."""
        return cls(file_path)

    def open(self):
        """Open the HDF5 file for direct access to feature/input/parameter groups."""
        if self._file is None:
            self._file = h5py.File(self.file_path, 'r')
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
        return self.open()

    def __exit__(self, *args):
        self.close()

    @property
    def features(self):
        if self._features is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._features

    @property
    def inputs(self):
        if self._inputs is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._inputs

    @property
    def parameters(self):
        if self._file is None:
            raise RuntimeError("Dataset not opened. Use dataset.open() or 'with dataset:'")
        return self._parameters

    # ── Dimension properties ──────────────────────────────────────────────

    @property
    def num_channels(self) -> Optional[int]:
        return self._dimension_cache.get('initial_raw_channels')

    @property
    def num_realizations(self) -> Optional[int]:
        return self._dimension_cache.get('num_realizations')

    @property
    def temporal_feature_dim(self) -> Optional[int]:
        return self._dimension_cache.get('temporal_feature_dim')

    @property
    def initial_feature_dim(self) -> Optional[int]:
        return self._dimension_cache.get('initial_manual_dim')

    @property
    def theta_param_dim(self) -> Optional[int]:
        return self._dimension_cache.get('theta_param_dim')

    @property
    def raw_input_shape(self) -> Optional[Tuple]:
        return self._dimension_cache.get('initial_raw_shape')

    def get_dimension_inference_dict(self) -> Dict[str, Any]:
        return self._dimension_cache.copy()

    def get_encoder_config_overrides(self) -> Dict[str, Any]:
        """Get config overrides for encoder based on introspected dimensions."""
        info = self._dimension_cache
        overrides = {}

        if info.get('initial_manual_dim') is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['manual_dim'] = info['initial_manual_dim']

        if info.get('initial_raw_channels') is not None:
            overrides.setdefault('encoder', {}).setdefault('initial', {})['in_channels'] = info['initial_raw_channels']

        if info.get('initial_raw_shape') is not None:
            shape = info['initial_raw_shape']
            if len(shape) >= 2:
                H, W = shape[-2:]
                if H == W:
                    overrides.setdefault('encoder', {}).setdefault('initial', {})['spatial_size'] = H

        if info.get('theta_param_dim') is not None:
            overrides.setdefault('encoder', {}).setdefault('theta', {})['param_dim'] = info['theta_param_dim']

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
        """
        info = self._dimension_cache
        overrides = {}

        if info.get('initial_raw_channels') is not None:
            overrides.setdefault('model', {})
            overrides['model']['in_channels'] = info['initial_raw_channels']
            overrides['model']['out_channels'] = info['initial_raw_channels']

        if info.get('theta_param_dim') is not None:
            overrides.setdefault('model', {})['param_dim'] = info['theta_param_dim']

        if info.get('initial_raw_shape') is not None:
            shape = info['initial_raw_shape']
            if len(shape) >= 2:
                H, W = shape[-2:]
                if H == W:
                    overrides.setdefault('model', {})['spatial_dim'] = H

        if info.get('temporal_timesteps') is not None:
            overrides.setdefault('training', {})['max_timesteps'] = info['temporal_timesteps']

        return overrides

    @classmethod
    def infer_and_update_config(
        cls,
        config_dict: Dict,
        dataset_path: Path,
        verbose: bool = True,
        max_samples: Optional[int] = None,
    ) -> Tuple['SpinlockDataset', Dict]:
        """Load dataset, introspect, and update config in one operation.

        Args:
            config_dict: Configuration dictionary (will be deep-merged)
            dataset_path: Path to HDF5 dataset file
            verbose: Log dimension inference results
            max_samples: If set, limit dataset to first N samples

        Returns:
            (dataset, updated_config_dict) tuple
        """
        dataset = cls.from_file(dataset_path).open()
        if max_samples is not None:
            # Store on instance so downstream code can cap the dataset
            dataset._max_samples_override = max_samples

        if verbose:
            logger.info(f"Infered dataset: {dataset_path}")

        overrides = dataset.get_encoder_config_overrides()

        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

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
        if 'temporal/features' in self._group:
            return _Dataset(self._group['temporal/features'])
        elif 'temporal' in self._group:
            return _Dataset(self._group['temporal'])
        return None

    @property
    def initial(self):
        """Access initial features dataset."""
        if 'initial/aggregated/features' in self._group:
            return _Dataset(self._group['initial/aggregated/features'])
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
        if 'fields' in self._group:
            return self._group['fields'][:]
        return None


class _ParameterGroup:
    """Wrapper for HDF5 parameters group."""

    def __init__(self, h5group):
        self._group = h5group if h5group is not None else {}

    @property
    def params(self):
        if 'params' in self._group:
            return _Dataset(self._group['params'])
        return None


class _Dataset:
    """Wrapper for HDF5 dataset."""

    def __init__(self, h5dataset):
        self._dataset = h5dataset

    def load_all(self):
        return self._dataset[:]

    @property
    def shape(self):
        return self._dataset.shape
