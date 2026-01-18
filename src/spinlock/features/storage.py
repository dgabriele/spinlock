"""
HDF5 storage for extracted features.

Extends existing Spinlock datasets with feature groups following the established
HDF5 schema pattern. Per-timestep-only architecture for online perturbation-based NOA.

Schema (extends /features/ group in dataset):
    /features/
        @family_versions - {"temporal": "3.0", "initial": "1.0"}
        @extraction_timestamp - ISO timestamp
        @extraction_config - JSON config used
        /temporal/
            @version - "3.0"
            @feature_registry - JSON {name: index} mapping
            @num_features - Total feature dimension
            features [N, T, 193] - Per-timestep features
                - Spatial (24D): statistics, gradients, Laplacian, percentiles
                - Spectral (27D): FFT power, frequencies, bandwidth
                - Cross-channel (12D): correlation, mutual information, eigenvalues
                - Enhanced temporal (130D): instantaneous, local temporal, stability,
                  phase space, multi-scale features
        /initial/
            @version - "1.0"
            features [N, 42] - Initial condition features (extracted once from u₀)

Feature Families:
    TEMPORAL: Per-timestep features [N, T, 193] - all per-timestep dynamics
    INITIAL: Initial condition features [N, 42] - manual + CNN-encoded from u₀

Design follows src/spinlock/dataset/storage.py patterns.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, cast
from datetime import datetime
import json

from spinlock.features.registry import FeatureRegistry


class HDF5FeatureWriter:
    """
    HDF5 writer for extracted features.

    Adds feature groups to existing Spinlock datasets or creates new files.
    Supports chunked writes for memory efficiency. Per-timestep-only architecture.

    Example:
        ```python
        writer = HDF5FeatureWriter(
            dataset_path=Path("datasets/benchmark_10k.h5"),
            overwrite=False
        )

        with writer.open_for_writing():
            writer.create_temporal_group(
                num_samples=10000,
                num_timesteps=100,
                num_realizations=10,
                registry=registry,
                config=temporal_config
            )

            # Write batches
            for batch_idx in range(0, num_samples, batch_size):
                writer.write_temporal_batch(
                    batch_idx=batch_idx,
                    per_timestep=per_timestep_features  # [B, T, 193]
                )
        ```
    """

    def __init__(
        self,
        dataset_path: Path,
        overwrite: bool = False
    ):
        """
        Initialize HDF5 feature writer.

        Args:
            dataset_path: Path to HDF5 dataset (existing or new)
            overwrite: Whether to overwrite existing features
        """
        self.dataset_path = dataset_path
        self.overwrite = overwrite
        self.file: Optional[h5py.File] = None
        self._feature_groups_created: set = set()

    def open_for_writing(self):
        """
        Open HDF5 file for writing.

        Returns context manager for safe file handling.

        Returns:
            Self (context manager)
        """
        return self

    def __enter__(self):
        """Open HDF5 file."""
        # Open in append mode to preserve existing data
        self.file = h5py.File(self.dataset_path, 'a')

        # Create /features/ group if doesn't exist
        if 'features' not in self.file:
            features_group = self.file.create_group('features')
            features_group.attrs['extraction_timestamp'] = datetime.now().isoformat()
            features_group.attrs['family_versions'] = json.dumps({})
        else:
            features_group = self.file['features']

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def create_temporal_group(
        self,
        num_samples: int,
        num_timesteps: int,
        num_realizations: int,
        registry: FeatureRegistry,
        config: Any,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: int = 100,
        param_dim: int = 0,
    ) -> None:
        """
        Create TEMPORAL feature group structure.

        Creates:
        - /features/temporal/ - per-timestep features [N, T, 193]
        - /features/parameters/ - operator parameter vectors (if param_dim > 0)

        Args:
            num_samples: Number of samples (N)
            num_timesteps: Number of timesteps (T)
            num_realizations: Number of realizations (M) - used for averaging
            registry: Feature registry with name-to-index mapping
            config: Temporal configuration (TemporalFeatureConfig)
            compression: Compression algorithm ("gzip", "lzf", "none")
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for HDF5
            param_dim: Dimension of operator parameter space (0 = disabled)

        Raises:
            RuntimeError: If file not opened
            ValueError: If feature groups already exist and overwrite=False
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened. Use 'with writer.open_for_writing()'")

        features_group = cast(h5py.Group, self.file['features'])

        # Compression settings
        compression_kwargs = {}
        if compression != "none":
            compression_kwargs = {
                'compression': compression,
                'compression_opts': compression_opts if compression == "gzip" else None
            }

        # Estimate feature dimension (193D: spatial 24 + spectral 27 + cross_channel 12 + temporal 130)
        D_per_timestep = self._estimate_per_timestep_dim(registry, config)

        # Create TEMPORAL group (per-timestep features)
        if D_per_timestep > 0:
            if 'temporal' in features_group:
                if not self.overwrite:
                    raise ValueError(
                        "TEMPORAL features already exist. Set overwrite=True to replace."
                    )
                del features_group['temporal']

            temporal_group = features_group.create_group('temporal')
            temporal_group.attrs['version'] = "3.0.0"  # New per-timestep-only architecture
            temporal_group.attrs['feature_registry'] = registry.to_json()
            temporal_group.attrs['num_features'] = registry.num_features
            temporal_group.attrs['extraction_config'] = json.dumps({
                'per_channel': getattr(config, 'per_channel', True),
            })
            temporal_group.create_dataset(
                'features',
                shape=(num_samples, num_timesteps, D_per_timestep),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), num_timesteps, D_per_timestep),
                **compression_kwargs
            )
            self._feature_groups_created.add('temporal')

        # Operator parameters [N, D_param] - Sobol parameter vectors
        if param_dim > 0:
            if 'parameters' in features_group:
                if not self.overwrite:
                    raise ValueError(
                        "Parameter dataset already exists. Set overwrite=True to replace."
                    )
                del features_group['parameters']

            params_dataset = features_group.create_dataset(
                'parameters',
                shape=(num_samples, param_dim),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), param_dim),
                **compression_kwargs
            )
            params_dataset.attrs['version'] = "1.0.0"
            params_dataset.attrs['description'] = "Sobol parameter vectors in [0,1]^D unit hypercube"
            params_dataset.attrs['units'] = "unit_hypercube"
            params_dataset.attrs['param_dim'] = param_dim
            self._feature_groups_created.add('parameters')

        # Update family versions
        family_versions_str = str(features_group.attrs['family_versions'])
        family_versions = json.loads(family_versions_str)
        if D_per_timestep > 0:
            family_versions['temporal'] = "3.0.0"
        features_group.attrs['family_versions'] = json.dumps(family_versions)

    # Legacy alias for backward compatibility
    def create_summary_group(self, *args, **kwargs):
        """Legacy alias for create_temporal_group(). Deprecated."""
        return self.create_temporal_group(*args, **kwargs)

    def write_temporal_batch(
        self,
        batch_idx: int,
        per_timestep: np.ndarray,
    ) -> None:
        """
        Write a batch of per-timestep features to TEMPORAL group.

        Args:
            batch_idx: Starting index for this batch
            per_timestep: TEMPORAL features [B, T, 193]

        Raises:
            RuntimeError: If file not opened or groups not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'temporal' not in self._feature_groups_created:
            raise RuntimeError("TEMPORAL group not created. Call create_temporal_group() first")

        batch_size = per_timestep.shape[0]
        end_idx = batch_idx + batch_size

        # Write TEMPORAL features (per-timestep) to features/temporal
        temporal_group = cast(h5py.Group, self.file['features/temporal'])
        dataset = cast(h5py.Dataset, temporal_group['features'])
        dataset[batch_idx:end_idx] = per_timestep

    # Legacy alias for backward compatibility
    def write_summary_batch(self, batch_idx: int, per_timestep: Optional[np.ndarray] = None, **kwargs):
        """Legacy alias for write_temporal_batch(). Deprecated. Ignores all non-per_timestep args."""
        if per_timestep is None:
            raise ValueError("per_timestep features required")
        return self.write_temporal_batch(batch_idx, per_timestep)

    def write_parameters(
        self,
        batch_idx: int,
        parameters: np.ndarray,
    ) -> None:
        """
        Write operator parameter vectors to HDF5.

        Stores Sobol parameter vectors in [0,1]^D unit hypercube for reproducibility.
        These can be used to reconstruct operator configurations or analyze parameter
        space coverage.

        Args:
            batch_idx: Starting index for this batch
            parameters: Parameter vectors [B, D] in [0,1] range

        Raises:
            RuntimeError: If file not opened or parameters dataset not created
            ValueError: If parameter dimension mismatch
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'parameters' not in self._feature_groups_created:
            raise RuntimeError("Parameters dataset not created. Call create_summary_group() with param_dim > 0")

        features_group = cast(h5py.Group, self.file['features'])
        params_dataset = cast(h5py.Dataset, features_group['parameters'])

        batch_size = parameters.shape[0]
        end_idx = batch_idx + batch_size

        # Validate dimension
        expected_dim = params_dataset.shape[1]
        actual_dim = parameters.shape[1]
        if actual_dim != expected_dim:
            raise ValueError(
                f"Parameter dimension mismatch: expected {expected_dim}, got {actual_dim}"
            )

        params_dataset[batch_idx:end_idx] = parameters

    def write_operator_sensitivity_features(
        self,
        sample_idx: int,
        features: Dict[str, float]
    ) -> None:
        """
        DEPRECATED: Operator sensitivity features were part of the old trajectory-level architecture.

        Per-timestep-only architecture does not support trajectory-level features.
        This method is a no-op for backward compatibility.

        Args:
            sample_idx: Sample index (ignored)
            features: Feature dictionary (ignored)
        """
        pass  # No-op for backward compatibility

    def create_inputs_group(
        self,
        num_samples: int,
        num_realizations: int,
        grid_size: int,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: int = 100,
    ) -> None:
        """
        Create inputs group for storing raw initial conditions.

        Creates:
        - /inputs/fields - raw initial condition fields [N, M, H, W]

        Args:
            num_samples: Number of samples (N)
            num_realizations: Number of realizations (M)
            grid_size: Spatial grid size (H=W)
            compression: Compression algorithm
            compression_opts: Compression level
            chunk_size: Chunk size for HDF5

        Raises:
            RuntimeError: If file not opened
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        # Compression settings
        compression_kwargs = {}
        if compression != "none":
            compression_kwargs = {
                'compression': compression,
                'compression_opts': compression_opts if compression == "gzip" else None
            }

        # Create inputs group
        if 'inputs' not in self.file:
            inputs_group = self.file.create_group('inputs')
        else:
            inputs_group = cast(h5py.Group, self.file['inputs'])

        # Create fields dataset [N, M, H, W]
        if 'fields' in inputs_group:
            if self.overwrite:
                del inputs_group['fields']
            else:
                raise ValueError("inputs/fields already exists. Set overwrite=True to replace.")

        inputs_group.create_dataset(
            'fields',
            shape=(num_samples, num_realizations, grid_size, grid_size),
            dtype=np.float32,
            chunks=(min(chunk_size, num_samples), num_realizations, grid_size, grid_size),
            **compression_kwargs
        )

    def write_inputs_batch(
        self,
        batch_idx: int,
        fields: np.ndarray,
    ) -> None:
        """
        Write batch of initial condition fields.

        Args:
            batch_idx: Starting index for this batch
            fields: Initial condition fields [B, M, H, W]

        Raises:
            RuntimeError: If file not opened or inputs group not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'inputs' not in self.file or 'fields' not in self.file['inputs']:
            raise RuntimeError("inputs/fields not created. Call create_inputs_group() first")

        fields_dataset = cast(h5py.Dataset, self.file['inputs/fields'])
        batch_size = fields.shape[0]
        end_idx = batch_idx + batch_size

        if end_idx > fields_dataset.shape[0]:
            raise ValueError(
                f"Batch extends beyond dataset size: "
                f"end_idx={end_idx}, dataset_size={fields_dataset.shape[0]}"
            )

        fields_dataset[batch_idx:end_idx] = fields

    def create_noa_metadata_group(
        self,
        num_samples: int,
        compression: str = "gzip",
        compression_opts: int = 4,
    ) -> None:
        """
        Create metadata group for NOA-specific metadata.

        Creates:
        - /metadata/ic_types - Initial condition type strings
        - /metadata/noise_regimes - Noise regime classification
        - /metadata/grid_sizes - Grid size per sample
        - /metadata/evolution_policies - Evolution policy per sample

        Args:
            num_samples: Number of samples (N)
            compression: Compression algorithm
            compression_opts: Compression level

        Raises:
            RuntimeError: If file not opened
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        # Create metadata group if doesn't exist
        if 'metadata' not in self.file:
            metadata_group = self.file.create_group('metadata')
        else:
            metadata_group = cast(h5py.Group, self.file['metadata'])

        # String dtype for variable-length strings
        str_dtype = h5py.string_dtype(encoding='utf-8')

        # Compression settings
        compression_kwargs = {}
        if compression != "none":
            compression_kwargs = {
                'compression': compression,
                'compression_opts': compression_opts if compression == "gzip" else None
            }

        # Create datasets
        for key in ['ic_types', 'noise_regimes', 'evolution_policies']:
            if key in metadata_group:
                if self.overwrite:
                    del metadata_group[key]
                else:
                    raise ValueError(f"metadata/{key} already exists. Set overwrite=True to replace.")

            metadata_group.create_dataset(
                key,
                shape=(num_samples,),
                dtype=str_dtype,
                **compression_kwargs
            )

        # Grid sizes (int32)
        if 'grid_sizes' in metadata_group:
            if self.overwrite:
                del metadata_group['grid_sizes']
            else:
                raise ValueError("metadata/grid_sizes already exists. Set overwrite=True to replace.")

        metadata_group.create_dataset(
            'grid_sizes',
            shape=(num_samples,),
            dtype=np.int32,
            **compression_kwargs
        )

    def write_noa_metadata_batch(
        self,
        batch_idx: int,
        ic_types: list,
        noise_regimes: list,
        evolution_policies: list,
        grid_sizes: np.ndarray,
    ) -> None:
        """
        Write batch of NOA metadata.

        Args:
            batch_idx: Starting index for this batch
            ic_types: List of IC type strings
            noise_regimes: List of noise regime strings
            evolution_policies: List of evolution policy strings
            grid_sizes: Array of grid sizes [B]

        Raises:
            RuntimeError: If file not opened or metadata group not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'metadata' not in self.file:
            raise RuntimeError("metadata group not created. Call create_noa_metadata_group() first")

        metadata_group = cast(h5py.Group, self.file['metadata'])
        batch_size = len(ic_types)
        end_idx = batch_idx + batch_size

        # Write string arrays
        for key, values in [
            ('ic_types', ic_types),
            ('noise_regimes', noise_regimes),
            ('evolution_policies', evolution_policies),
        ]:
            if key not in metadata_group:
                raise RuntimeError(f"metadata/{key} not created. Call create_noa_metadata_group() first")
            dataset = cast(h5py.Dataset, metadata_group[key])
            dataset[batch_idx:end_idx] = values

        # Write grid sizes
        if 'grid_sizes' not in metadata_group:
            raise RuntimeError("metadata/grid_sizes not created. Call create_noa_metadata_group() first")
        dataset = cast(h5py.Dataset, metadata_group['grid_sizes'])
        dataset[batch_idx:end_idx] = grid_sizes

    def _estimate_per_timestep_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of per-timestep features.

        Per-timestep-only architecture: 193D total
        - Spatial: 24D
        - Spectral: 27D
        - Cross-channel: 12D
        - Enhanced temporal: 130D

        Args:
            registry: Feature registry
            config: Temporal configuration

        Returns:
            Estimated dimension (193D)
        """
        # Count features in all per-timestep categories
        per_timestep_categories = ['spatial', 'spectral', 'cross_channel', 'temporal']

        dim = 0
        for category in per_timestep_categories:
            dim += len(registry.get_features_by_category(category))

        return dim

    def _estimate_per_trajectory_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of per-trajectory features.

        DEPRECATED: Per-timestep-only architecture has no trajectory-level features.

        Returns:
            0 (no trajectory-level features)
        """
        return 0

    def _estimate_aggregated_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of aggregated features.

        DEPRECATED: Per-timestep-only architecture has no aggregated features.

        Returns:
            0 (no aggregated features)
        """
        return 0


class HDF5FeatureReader:
    """
    Reader for feature-augmented HDF5 datasets.

    Reads feature groups from datasets with /features/ structure.
    Per-timestep-only architecture:
    - TEMPORAL: Per-timestep features [N, T, 193] at /features/temporal

    Example:
        ```python
        with HDF5FeatureReader(Path("datasets/benchmark_10k.h5")) as reader:
            # Check which feature families exist
            families = reader.get_feature_families()

            # Read TEMPORAL features (per-timestep)
            temporal = reader.get_temporal_features()  # [N, T, 193]
            registry = reader.get_temporal_registry()
        ```
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize feature reader.

        Args:
            dataset_path: Path to HDF5 dataset
        """
        self.dataset_path = dataset_path
        self.file: Optional[h5py.File] = None

    def __enter__(self):
        """Open HDF5 file for reading."""
        self.file = h5py.File(self.dataset_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def has_features(self) -> bool:
        """
        Check if dataset has features.

        Returns:
            True if /features/ group exists
        """
        if self.file is None:
            raise RuntimeError("File not opened")
        return 'features' in self.file

    def get_feature_families(self) -> list:
        """
        Get list of available feature families.

        Returns:
            List of family names (e.g., ['temporal', 'summary'])
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        if not self.has_features():
            return []

        features_group = cast(h5py.Group, self.file['features'])
        return [key for key in features_group.keys() if isinstance(features_group[key], h5py.Group)]

    def has_temporal(self) -> bool:
        """Check if dataset has TEMPORAL features (per-timestep)."""
        return 'temporal' in self.get_feature_families()

    def has_summary(self) -> bool:
        """
        DEPRECATED: Check if dataset has SUMMARY features.

        Per-timestep-only architecture does not use summary features.
        Returns False for new datasets.
        """
        return 'summary' in self.get_feature_families()

    def get_temporal_features(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        Get TEMPORAL features (per-timestep).

        Args:
            idx: Optional slice or index

        Returns:
            Features [N, T, 193] or [T, 193] (if idx specified), None if not available
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        if not self.has_temporal():
            return None

        temporal_group = cast(h5py.Group, self.file['features/temporal'])
        dataset = cast(h5py.Dataset, temporal_group['features'])

        if idx is None:
            return dataset[:]
        return dataset[idx]

    def get_temporal_registry(self) -> Optional[FeatureRegistry]:
        """
        Get TEMPORAL feature registry.

        Returns:
            FeatureRegistry if TEMPORAL features exist, None otherwise
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        if not self.has_temporal():
            return None

        temporal_group = cast(h5py.Group, self.file['features/temporal'])
        registry_json_str = str(temporal_group.attrs['feature_registry'])

        return FeatureRegistry.from_json(registry_json_str, family_name='temporal')

    def get_summary_registry(self) -> Optional[FeatureRegistry]:
        """
        DEPRECATED: Get SUMMARY feature registry.

        Per-timestep-only architecture does not use summary features.
        Use get_temporal_registry() instead.
        """
        return self.get_temporal_registry()

    def get_summary_per_trajectory(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        DEPRECATED: Get per-trajectory SUMMARY features.

        Per-timestep-only architecture does not have trajectory-level features.
        Returns None.
        """
        return None

    def get_summary_aggregated(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        DEPRECATED: Get aggregated SUMMARY features.

        Per-timestep-only architecture does not have aggregated features.
        Returns None.
        """
        return None
