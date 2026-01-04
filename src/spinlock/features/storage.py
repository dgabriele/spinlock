"""
HDF5 storage for extracted features.

Extends existing Spinlock datasets with feature groups following the established
HDF5 schema pattern. Two feature families stored separately:

Schema (extends /features/ group in dataset):
    /features/
        @family_versions - {"temporal": "1.0", "summary": "1.0", ...}
        @extraction_timestamp - ISO timestamp
        @extraction_config - JSON config used
        /temporal/
            @version - "1.0"
            features [N, T, D] - Per-timestep time series features
        /summary/
            @version - "1.0"
            @feature_registry - JSON {name: index} mapping
            @num_features - Total feature dimension
            /per_trajectory/
                features [N, M, D_traj] - Per-trajectory features
            /aggregated/
                features [N, D_final] - Final aggregated features
                /metadata/
                    extraction_time [N] - Extraction time per sample

Feature Families:
    TEMPORAL: Per-timestep time series [N, T, D] - spatial, spectral, cross_channel
    SUMMARY: Aggregated scalars [N, D] - temporal dynamics, causality, invariant_drift

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
            writer.create_summary_group(
                num_samples=10000,
                num_timesteps=100,
                num_realizations=10,
                registry=registry,
                config=summary_config
            )

            # Write batches
            for batch_idx in range(0, num_samples, batch_size):
                writer.write_summary_batch(
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

    def create_summary_group(
        self,
        num_samples: int,
        num_timesteps: int,
        num_realizations: int,
        registry: FeatureRegistry,
        config: Any,
        compression: str = "gzip",
        compression_opts: int = 4,
        chunk_size: int = 100,
        temporal_enabled: bool = True,
        learned_dim: int = 0,
    ) -> None:
        """
        Create TEMPORAL and SUMMARY feature group structures.

        Creates:
        - /features/temporal/ - per-timestep features (if temporal_enabled)
        - /features/summary/ - per-trajectory, aggregated, and learned features

        Args:
            num_samples: Number of samples (N)
            num_timesteps: Number of timesteps (T)
            num_realizations: Number of realizations (M)
            registry: Feature registry with name-to-index mapping
            config: SUMMARY configuration (SummaryConfig)
            compression: Compression algorithm ("gzip", "lzf", "none")
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for HDF5
            temporal_enabled: Whether TEMPORAL (per-timestep) features are enabled
            learned_dim: Dimension of learned features (0 = disabled)

        Raises:
            RuntimeError: If file not opened
            ValueError: If feature groups already exist and overwrite=False
        """
        self._temporal_enabled = temporal_enabled
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

        # Estimate feature dimensions
        D_per_timestep = self._estimate_per_timestep_dim(registry, config)
        D_per_trajectory = self._estimate_per_trajectory_dim(registry, config)
        D_aggregated = self._estimate_aggregated_dim(registry, config)

        # Create TEMPORAL group (per-timestep features) if enabled
        if D_per_timestep > 0:
            if 'temporal' in features_group:
                if not self.overwrite:
                    raise ValueError(
                        "TEMPORAL features already exist. Set overwrite=True to replace."
                    )
                del features_group['temporal']

            temporal_group = features_group.create_group('temporal')
            temporal_group.attrs['version'] = "1.0.0"
            temporal_group.create_dataset(
                'features',
                shape=(num_samples, num_timesteps, D_per_timestep),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), num_timesteps, D_per_timestep),
                **compression_kwargs
            )
            self._feature_groups_created.add('temporal')

        # Create SUMMARY group (per-trajectory and aggregated features)
        if 'summary' in features_group:
            if not self.overwrite:
                raise ValueError(
                    "SUMMARY features already exist. Set overwrite=True to replace."
                )
            del features_group['summary']

        summary_group = features_group.create_group('summary')
        summary_group.attrs['version'] = "1.0.0"
        summary_group.attrs['feature_registry'] = registry.to_json()
        summary_group.attrs['num_features'] = registry.num_features
        summary_group.attrs['extraction_config'] = json.dumps({
            'per_channel': config.per_channel,
            'temporal_aggregation': config.temporal_aggregation,
            'realization_aggregation': config.realization_aggregation,
        })

        # Per-trajectory features [N, M, D_traj]
        if D_per_trajectory > 0:
            per_trajectory_group = summary_group.create_group('per_trajectory')
            per_trajectory_group.create_dataset(
                'features',
                shape=(num_samples, num_realizations, D_per_trajectory),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), num_realizations, D_per_trajectory),
                **compression_kwargs
            )

        # Aggregated features [N, D_final]
        if D_aggregated > 0:
            aggregated_group = summary_group.create_group('aggregated')
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

        # Learned features [N, D_learned] - extracted from U-AFNO latents
        if learned_dim > 0:
            learned_group = summary_group.create_group('learned')
            learned_group.attrs['version'] = "1.0.0"
            learned_group.attrs['description'] = "Learned features from U-AFNO intermediate representations"
            learned_group.create_dataset(
                'features',
                shape=(num_samples, learned_dim),
                dtype=np.float32,
                chunks=(min(chunk_size, num_samples), learned_dim),
                **compression_kwargs
            )
            self._feature_groups_created.add('learned')

        # Update family versions
        family_versions_str = str(features_group.attrs['family_versions'])
        family_versions = json.loads(family_versions_str)
        if D_per_timestep > 0:
            family_versions['temporal'] = "1.0.0"
        family_versions['summary'] = "1.0.0"
        features_group.attrs['family_versions'] = json.dumps(family_versions)

        self._feature_groups_created.add('summary')

    def write_summary_batch(
        self,
        batch_idx: int,
        per_timestep: Optional[np.ndarray] = None,
        per_trajectory: Optional[np.ndarray] = None,
        aggregated: Optional[np.ndarray] = None,
        extraction_times: Optional[np.ndarray] = None,
        learned: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write a batch of features to TEMPORAL and SUMMARY groups.

        Args:
            batch_idx: Starting index for this batch
            per_timestep: TEMPORAL features [B, T, D] or None (written to features/temporal)
            per_trajectory: SUMMARY per-trajectory features [B, M, D_traj] or None
            aggregated: SUMMARY aggregated features [B, D_final] or None
            extraction_times: Extraction time per sample [B] or None
            learned: Learned features from U-AFNO [B, D_learned] or None

        Raises:
            RuntimeError: If file not opened or groups not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'summary' not in self._feature_groups_created:
            raise RuntimeError("SUMMARY group not created. Call create_summary_group() first")

        # Determine batch size
        batch_size = None
        if per_timestep is not None:
            batch_size = per_timestep.shape[0]
        elif per_trajectory is not None:
            batch_size = per_trajectory.shape[0]
        elif aggregated is not None:
            batch_size = aggregated.shape[0]
        elif learned is not None:
            batch_size = learned.shape[0]
        else:
            raise ValueError("At least one feature array must be provided")

        end_idx = batch_idx + batch_size

        # Write TEMPORAL features (per-timestep) to features/temporal
        if per_timestep is not None and 'temporal' in self._feature_groups_created:
            temporal_group = cast(h5py.Group, self.file['features/temporal'])
            dataset = cast(h5py.Dataset, temporal_group['features'])
            dataset[batch_idx:end_idx] = per_timestep

        # Write SUMMARY features to features/summary
        summary_group = cast(h5py.Group, self.file['features/summary'])

        # Write per-trajectory features
        if per_trajectory is not None and 'per_trajectory' in summary_group:
            dataset = cast(h5py.Dataset, summary_group['per_trajectory/features'])
            dataset[batch_idx:end_idx] = per_trajectory

        # Write aggregated features
        if aggregated is not None and 'aggregated' in summary_group:
            dataset = cast(h5py.Dataset, summary_group['aggregated/features'])
            dataset[batch_idx:end_idx] = aggregated

            # Write extraction times if provided
            if extraction_times is not None:
                time_dataset = cast(h5py.Dataset, summary_group['aggregated/metadata/extraction_time'])
                time_dataset[batch_idx:end_idx] = extraction_times

        # Write learned features (from U-AFNO latents)
        if learned is not None and 'learned' in self._feature_groups_created:
            dataset = cast(h5py.Dataset, summary_group['learned/features'])
            expected_dim = dataset.shape[1]
            actual_dim = learned.shape[1] if len(learned.shape) > 1 else learned.shape[0]

            # Pad or truncate to match pre-allocated dimension
            if actual_dim < expected_dim:
                padding = np.zeros((learned.shape[0], expected_dim - actual_dim), dtype=learned.dtype)
                learned = np.concatenate([learned, padding], axis=1)
            elif actual_dim > expected_dim:
                learned = learned[:, :expected_dim]

            dataset[batch_idx:end_idx] = learned

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
            RuntimeError: If file not opened or SUMMARY group not created
        """
        if self.file is None:
            raise RuntimeError("HDF5 file not opened")

        if 'summary' not in self._feature_groups_created:
            raise RuntimeError("SUMMARY group not created. Call create_summary_group() first")

        summary_group = cast(h5py.Group, self.file['features/summary'])

        # Create operator_sensitivity subgroup if doesn't exist
        if 'operator_sensitivity_inline' not in summary_group:
            ops_group = summary_group.create_group('operator_sensitivity_inline')
            ops_group.attrs['description'] = (
                "Operator sensitivity features extracted during generation. "
                "Must be merged into per_trajectory features during post-hoc extraction."
            )
        else:
            ops_group = cast(h5py.Group, summary_group['operator_sensitivity_inline'])

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
            config: SUMMARY configuration

        Returns:
            Estimated dimension (0 if TEMPORAL extraction is disabled)
        """
        # Check if TEMPORAL (per-timestep) extraction is disabled
        if not self._temporal_enabled:
            return 0

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
        Estimate dimension of per-trajectory SUMMARY features.

        Args:
            registry: Feature registry
            config: SUMMARY configuration

        Returns:
            Estimated dimension
        """
        # Temporal features are trajectory-level
        temporal_features = registry.get_features_by_category('temporal')

        # v2.0 trajectory-level features (causality, invariant_drift)
        # Note: operator_sensitivity is only included when inline extraction is enabled
        # and features are provided via metadata. By default, we don't count them here.
        causality_features = registry.get_features_by_category('causality')
        invariant_drift_features = registry.get_features_by_category('invariant_drift')

        # Nonlinear features (if enabled)
        nonlinear_features = registry.get_features_by_category('nonlinear')

        return (len(temporal_features) + len(causality_features) +
                len(invariant_drift_features) + len(nonlinear_features))

    def _estimate_aggregated_dim(self, registry: FeatureRegistry, config: Any) -> int:
        """
        Estimate dimension of aggregated SUMMARY features.

        Args:
            registry: Feature registry
            config: SUMMARY configuration

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
    Two feature families:
    - TEMPORAL: Per-timestep time series [N, T, D] at /features/temporal
    - SUMMARY: Aggregated scalars [N, D] at /features/summary

    Example:
        ```python
        with HDF5FeatureReader(Path("datasets/benchmark_10k.h5")) as reader:
            # Check which feature families exist
            families = reader.get_feature_families()

            # Read TEMPORAL features (per-timestep)
            temporal = reader.get_temporal_features()  # [N, T, D]

            # Read SUMMARY features (aggregated)
            aggregated = reader.get_summary_aggregated()  # [N, D_final]
            registry = reader.get_summary_registry()
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
        """Check if dataset has SUMMARY features (aggregated)."""
        return 'summary' in self.get_feature_families()

    def get_temporal_features(self, idx: Optional[slice] = None) -> Optional[np.ndarray]:
        """
        Get TEMPORAL features (per-timestep time series).

        Args:
            idx: Optional slice or index

        Returns:
            Features [N, T, D] or [T, D] (if idx specified), None if not available
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
