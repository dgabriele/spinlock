"""
Data loading utilities for feature visualization.

Provides convenient wrappers around HDF5 readers for loading
operator features, parameters, and metadata.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from spinlock.features.storage import HDF5FeatureReader
from spinlock.features.registry import FeatureRegistry
from spinlock.dataset.storage import HDF5DatasetReader


def load_parameters(dataset_path: Path) -> np.ndarray:
    """
    Load operator parameters from dataset.

    Args:
        dataset_path: Path to HDF5 dataset

    Returns:
        Parameter matrix [N, P] where:
            N = number of operators
            P = parameter dimension

    Example:
        >>> params = load_parameters(Path("datasets/benchmark_10k.h5"))
        >>> print(params.shape)  # (10000, 14)
    """
    with HDF5DatasetReader(dataset_path) as reader:
        parameters = reader.get_parameters()
    return parameters


def load_per_timestep_features(dataset_path: Path) -> Tuple[np.ndarray, FeatureRegistry]:
    """
    Load per-timestep features and registry from dataset.

    Args:
        dataset_path: Path to HDF5 dataset with SDF features

    Returns:
        Tuple of (features, registry) where:
            features: [N, T, D] array of per-timestep features
            registry: FeatureRegistry mapping names to indices

    Raises:
        ValueError: If dataset does not contain SDF features

    Example:
        >>> features, registry = load_per_timestep_features(Path("datasets/benchmark_10k.h5"))
        >>> print(features.shape)  # (10000, 250, 96)
        >>> print(registry.num_features)  # 96
    """
    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            raise ValueError(f"Dataset {dataset_path} does not contain SDF features")

        features = reader.get_sdf_per_timestep()
        registry = reader.get_sdf_registry()

    if features is None:
        raise ValueError(f"Dataset {dataset_path} does not have per-timestep features")

    return features, registry


def load_operator_features(
    dataset_path: Path,
    operator_idx: int,
    registry: Optional[FeatureRegistry] = None
) -> Tuple[np.ndarray, FeatureRegistry]:
    """
    Load features for a single operator.

    Args:
        dataset_path: Path to HDF5 dataset
        operator_idx: Index of operator to load
        registry: Optional pre-loaded registry (avoids re-reading)

    Returns:
        Tuple of (features, registry) where:
            features: [T, D] array of per-timestep features for this operator
            registry: FeatureRegistry for feature metadata

    Example:
        >>> features, registry = load_operator_features(Path("datasets/benchmark_10k.h5"), 42)
        >>> print(features.shape)  # (250, 96)
    """
    with HDF5FeatureReader(dataset_path) as reader:
        if not reader.has_sdf():
            raise ValueError(f"Dataset {dataset_path} does not contain SDF features")

        # Load registry if not provided
        if registry is None:
            registry = reader.get_sdf_registry()

        # Load per-timestep features for this operator
        all_features = reader.get_sdf_per_timestep(idx=operator_idx)

        if all_features is None:
            raise ValueError(f"No per-timestep features found for operator {operator_idx}")

    return all_features, registry


def check_dataset_compatibility(dataset_path: Path) -> Dict[str, any]:
    """
    Check dataset compatibility and return metadata.

    Args:
        dataset_path: Path to HDF5 dataset

    Returns:
        Dictionary with dataset metadata:
            - has_sdf: bool
            - num_operators: int
            - num_timesteps: int | None
            - num_features: int | None
            - feature_categories: List[str] | None

    Example:
        >>> meta = check_dataset_compatibility(Path("datasets/benchmark_10k.h5"))
        >>> print(meta)
        {
            'has_sdf': True,
            'num_operators': 10000,
            'num_timesteps': 250,
            'num_features': 96,
            'feature_categories': ['spatial', 'spectral', ...]
        }
    """
    metadata = {
        'has_sdf': False,
        'num_operators': 0,
        'num_timesteps': None,
        'num_features': None,
        'feature_categories': None,
    }

    # Check features
    with HDF5FeatureReader(dataset_path) as reader:
        metadata['has_sdf'] = reader.has_sdf()

        if metadata['has_sdf']:
            registry = reader.get_sdf_registry()
            metadata['num_features'] = registry.num_features
            metadata['feature_categories'] = list(registry.categories)

            # Try to get per-timestep features to determine shape
            features = reader.get_sdf_per_timestep()
            if features is not None:
                metadata['num_operators'] = features.shape[0]
                metadata['num_timesteps'] = features.shape[1]

    # If no features, check parameters for num_operators
    if metadata['num_operators'] == 0:
        try:
            with HDF5DatasetReader(dataset_path) as reader:
                params = reader.get_parameters()
                metadata['num_operators'] = len(params)
        except Exception:
            pass

    return metadata
