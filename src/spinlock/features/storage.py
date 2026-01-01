"""
HDF5 storage for extracted features.

Extends existing Spinlock datasets with feature groups following the established
HDF5 schema pattern. Supports multiple feature families with separate groups.

Schema (extends /features/ group in dataset):
    /features/
        @family_versions - {"summary": "1.0", ...}
        @extraction_timestamp - ISO timestamp
        @extraction_config - JSON config used
        /sdf/
            @version - "1.0"
            @feature_registry - JSON {name: index} mapping
            @num_features - Total feature dimension
            /per_timestep/
                features [N, T, D] - Per-timestep features
            /per_trajectory/
                features [N, M, D_traj] - Per-trajectory features
            /aggregated/
                features [N, D_final] - Final aggregated features
                /metadata/
                    extraction_time [N] - Extraction time per sample

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
    Supports chunked writes for memory efficiency.

    Example:
        ```python
        writer = HDF5FeatureWriter(
            dataset_path=Path("datasets/benchmark_10k.h5"),
            overwrite=False
        )

        with writer.open_for_writing():
            writer.create_sdf_group(
                num_samples=10000,
                num_timesteps=100,
                num_realizations=10,
                registry=registry,
                config=sdf_config
            )

            # Write batches
            for batch_idx in range(0, num_samples, batch_size):
                writer.write_sdf_batch(
                    batch_idx=batch_idx,
                    per_timestep=per_timestep_features,
                    per_trajectory=per_trajectory_features,
                    aggregated=aggregated_features
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

    def create_sdf_group(
        self,
        num_samples: int,
        num_timesteps: int,
        num_realizations: int,
        registry: FeatureRegistry,
        config: Any,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: int = 100
    ) -> None:
        """
        Create SDF feature group structure.

        Args:
            num_samples: Number of samples (N)
            num_timesteps: Number of timesteps (T)
            num_realizations: Number of realizations (M)
            registry: Feature registry with name-to-index mapping
            config: SDF configuration (SummaryConfig)
            compression: Compression algorithm ("gzip", "lzf", "none")
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for HDF5

        Raises:
            RuntimeError: If file not opened
            ValueError: If SDF group already exists and overwrite=False
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened. Use 'with writer.open_for_writing()'")

        features_group = cast(h5py.Group, self.file['features'])

        # Check if SDF group already exists
        if 'sdf' in features_group:
            if not self.overwrite:
                raise ValueError(
                    "SDF features already exist. Set overwrite=True to replace."
                )
            # Delete existing group
            del features_group['sdf']

        # Create SDF group
        sdf_group = features_group.create_group('sdf')
        sdf_group.attrs['version'] = "1.0.0"
        sdf_group.attrs['feature_registry'] = registry.to_json()
        sdf_group.attrs['num_features'] = registry.num_features
        sdf_group.attrs['extraction_config'] = json.dumps({
            'per_channel': config.per_channel,
            'temporal_aggregation': config.temporal_aggregation,
            'realization_aggregation': config.realization_aggregation,
        })

        # Estimate feature dimensions
        D_per_timestep = self._estimate_per_timestep_dim(registry, config)
        D_per_trajectory = self._estimate_per_trajectory_dim(registry, config)
        D_aggregated = self._estimate_aggregated_dim(registry, config)

        # Create subgroups with datasets
        compression_kwargs = {}
        if compression != "none":
            compression_kwargs = {
                'compression': compression,
                'compression_opts': compression_opts if compression == "gzip" else None
            }

        # Per-timestep features [N, T, D]
        if D_per_timestep > 0:
            per_timestep_group = sdf_group.create_group('per_timestep')
            per_timestep_group.create_dataset(
                'features',
                shape=(num_samples, num_timesteps, D_per_timestep),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), num_timesteps, D_per_timestep),
                **compression_kwargs
            )

        # Per-trajectory features [N, M, D_traj]
        if D_per_trajectory > 0:
            per_trajectory_group = sdf_group.create_group('per_trajectory')
            per_trajectory_group.create_dataset(
                'features',
                shape=(num_samples, num_realizations, D_per_trajectory),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), num_realizations, D_per_trajectory),
                **compression_kwargs
            )

        # Aggregated features [N, D_final]
        if D_aggregated > 0:
            aggregated_group = sdf_group.create_group('aggregated')
            aggregated_group.create_dataset(
                'features',
                shape=(num_samples, D_aggregated),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), D_aggregated),
                **compression_kwargs
            )

            # Metadata subgroup
            metadata_group = aggregated_group.create_group('metadata')
            metadata_group.create_dataset(
                'extraction_time',
                shape=(num_samples,),
                dtype=np.float64,
                chunks=(chunk_size,)
            )

        # Update family versions
        family_versions_str = str(features_group.attrs['family_versions'])
        family_versions = json.loads(family_versions_str)
        family_versions['sdf'] = "1.0.0"
        features_group.attrs['family_versions'] = json.dumps(family_versions)

        self._feature_groups_created.add('sdf')

    def write_sdf_batch(
        self,
        batch_idx: int,
        per_timestep: Optional[np.ndarray] = None,
        per_trajectory: Optional[np.ndarray] = None,
        aggregated: Optional[np.ndarray] = None,
        extraction_times: Optional[np.ndarray] = None
    ) -> None:
        """
        Write a batch of SDF features.

        Args:
            batch_idx: Starting index for this batch
            per_timestep: Per-timestep features [B, T, D] or None
            per_trajectory: Per-trajectory features [B, M, D_traj] or None
            aggregated: Aggregated features [B, D_final] or None
            extraction_times: Extraction time per sample [B] or None

        Raises:
            RuntimeError: If file not opened or SDF group not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'sdf' not in self._feature_groups_created:
            raise RuntimeError("SDF group not created. Call create_sdf_group() first")

        sdf_group = cast(h5py.Group, self.file['features/sdf'])

        # Determine batch size
        batch_size = None
        if per_timestep is not None:
            batch_size = per_timestep.shape[0]
        elif per_trajectory is not None:
            batch_size = per_trajectory.shape[0]
        elif aggregated is not None:
            batch_size = aggregated.shape[0]
        else:
            raise ValueError("At least one feature array must be provided")

        end_idx = batch_idx + batch_size

        # Write per-timestep features
        if per_timestep is not None and 'per_timestep' in sdf_group:
            dataset = cast(h5py.Dataset, sdf_group['per_timestep/features'])
            dataset[batch_idx:end_idx] = per_timestep

        # Write per-trajectory features
        if per_trajectory is not None and 'per_trajectory' in sdf_group:
            dataset = cast(h5py.Dataset, sdf_group['per_trajectory/features'])
            dataset[batch_idx:end_idx] = per_trajectory

        # Write aggregated features
        if aggregated is not None and 'aggregated' in sdf_group:
            dataset = cast(h5py.Dataset, sdf_group['aggregated/features'])
            dataset[batch_idx:end_idx] = aggregated

            # Write extraction times if provided
            if extraction_times is not None:
                time_dataset = cast(h5py.Dataset, sdf_group['aggregated/metadata/extraction_time'])
                time_dataset[batch_idx:end_idx] = extraction_times

    def write_operator_sensitivity_features(
        self,
        sample_idx: int,
        features: Dict[str, float]
    ) -> None:
        """
        Write operator sensitivity features for a single sample.

        These features are extracted during generation (inline) and stored
        separately from post-hoc extracted features. They're later merged
        into the per_trajectory feature array.

        Args:
            sample_idx: Sample index (0-based)
            features: Dictionary of feature_name -> scalar value

        Raises:
            RuntimeError: If file not opened or SDF group not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'sdf' not in self._feature_groups_created:
            raise RuntimeError("SDF group not created. Call create_sdf_group() first")

        sdf_group = cast(h5py.Group, self.file['features/sdf'])

        # Create operator_sensitivity subgroup if doesn't exist
        if 'operator_sensitivity_inline' not in sdf_group:
            ops_group = sdf_group.create_group('operator_sensitivity_inline')
            ops_group.attrs['description'] = (
                "Operator sensitivity features extracted during generation. "
                "Must be merged into per_trajectory features during post-hoc extraction."
            )
        else:
            ops_group = cast(h5py.Group, sdf_group['operator_sensitivity_inline'])

        # Store features as individual datasets (sparse storage)
        # Each feature is a 1D array [N] where N = num_samples
        for feature_name, value in features.items():
            if feature_name not in ops_group:
                # Create dataset on first write
                num_samples = self.file['inputs/fields'].shape[0]  # type: ignore
                ops_group.create_dataset(
                    feature_name,
                    shape=(num_samples,),
                    dtype=np.float32,
                    fillvalue=np.nan
                )

            # Write feature value
            dataset = cast(h5py.Dataset, ops_group[feature_name])
            dataset[sample_idx] = value

    def _estimate_per_timestep_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of per-timestep features.

        This is a rough estimate. Actual dimension determined by feature extractor.

        Args:
            registry: Feature registry
            config: SDF configuration

        Returns:
            Estimated dimension
        """
        # Count features in per-timestep categories (spatial, spectral, cross_channel, etc.)
        per_timestep_categories = ['spatial', 'spectral', 'cross_channel', 'distributional',
                                   'structural', 'physics', 'morphological', 'multiscale']

        dim = 0
        for category in per_timestep_categories:
            dim += len(registry.get_features_by_category(category))

        # Note: We do NOT multiply by temporal_aggregation here
        # The extractors return raw per-timestep features [N, T, D]
        # Temporal aggregation (if needed) happens in downstream analysis

        return dim

    def _estimate_per_trajectory_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of per-trajectory features.

        Args:
            registry: Feature registry
            config: SDF configuration

        Returns:
            Estimated dimension
        """
        # Temporal features are trajectory-level
        temporal_features = registry.get_features_by_category('temporal')

        # v2.0 trajectory-level features (causality, invariant_drift, operator_sensitivity)
        causality_features = registry.get_features_by_category('causality')
        invariant_drift_features = registry.get_features_by_category('invariant_drift')
        operator_sensitivity_features = registry.get_features_by_category('operator_sensitivity')

        return (len(temporal_features) + len(causality_features) +
                len(invariant_drift_features) + len(operator_sensitivity_features))

    def _estimate_aggregated_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of aggregated features.

        Args:
            registry: Feature registry
            config: SDF configuration

        Returns:
            Estimated dimension
        """
        # All features aggregated across realizations
        per_trajectory_dim = self._estimate_per_trajectory_dim(registry, config)

        # Multiply by realization aggregations (mean, std, cv)
        return per_trajectory_dim * len(config.realization_aggregation)


class HDF5FeatureReader:
    """
    Reader for feature-augmented HDF5 datasets.

    Reads feature groups from datasets with /features/ structure.

    Example:
        ```python
        with HDF5FeatureReader(Path("datasets/benchmark_10k.h5")) as reader:
            # Check which feature families exist
            families = reader.get_feature_families()

            # Read SDF features
            aggregated = reader.get_sdf_aggregated()  # [N, D_final]
            registry = reader.get_sdf_registry()
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
            List of family names (e.g., ['sdf'])
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        if not self.has_features():
            return []

        features_group = cast(h5py.Group, self.file['features'])
        return [key for key in features_group.keys() if isinstance(features_group[key], h5py.Group)]

    def has_summary(self) -> bool:
        """Check if dataset has SUMMARY features."""
        return 'summary' in self.get_feature_families()

    def get_summary_registry(self) -> Optional[FeatureRegistry]:
        """
        Get SUMMARY feature registry.

        Returns:
            FeatureRegistry if SUMMARY features exist, None otherwise
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        if not self.has_summary():
            return None

        summary_group = cast(h5py.Group, self.file['features/summary'])
        registry_json_str = str(summary_group.attrs['feature_registry'])

        return FeatureRegistry.from_json(registry_json_str, family_name='summary')

    def get_summary_per_timestep(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        Get per-timestep SUMMARY features.

        Args:
            idx: Optional slice or index

        Returns:
            Features [N, T, D] or [T, D] (if idx specified), None if not available
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        summary_group = cast(h5py.Group, self.file['features/summary']) if self.has_summary() else None
        if summary_group is None or 'per_timestep' not in summary_group:
            return None

        dataset = cast(h5py.Dataset, summary_group['per_timestep/features'])

        if idx is None:
            return dataset[:]
        return dataset[idx]

    def get_summary_per_trajectory(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        Get per-trajectory SUMMARY features.

        Args:
            idx: Optional slice or index

        Returns:
            Features [N, M, D_traj] or [M, D_traj] (if idx specified), None if not available
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        summary_group = cast(h5py.Group, self.file['features/summary']) if self.has_summary() else None
        if summary_group is None or 'per_trajectory' not in summary_group:
            return None

        dataset = cast(h5py.Dataset, summary_group['per_trajectory/features'])

        if idx is None:
            return dataset[:]
        return dataset[idx]

    def get_summary_aggregated(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        Get aggregated SUMMARY features.

        Args:
            idx: Optional slice or index

        Returns:
            Features [N, D_final] or [D_final] (if idx specified), None if not available
        """
        if self.file is None:
            raise RuntimeError("File not opened")

        summary_group = cast(h5py.Group, self.file['features/summary']) if self.has_summary() else None
        if summary_group is None or 'aggregated' not in summary_group:
            return None

        dataset = cast(h5py.Dataset, summary_group['aggregated/features'])

        if idx is None:
            return dataset[:]
        return dataset[idx]
