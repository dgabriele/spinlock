"""Quick demo test to verify the grouping system works end-to-end."""

import numpy as np
import pytest

from spinlock.features.grouping import create_grouper, ClusteringParams, TemporalGroupingConfig


def test_quick_demo_no_gpu():
    """Quick demo that works without GPU (uses CPU gradient only)."""
    # Create synthetic temporal-like data
    np.random.seed(42)
    N, D = 200, 30  # 200 samples, 30 features

    # Create features with 3 correlated groups
    features = np.zeros((N, D), dtype=np.float32)

    # Group 1: features 0-9
    base1 = np.random.randn(N)
    for i in range(10):
        features[:, i] = base1 + np.random.randn(N) * 0.3

    # Group 2: features 10-19
    base2 = np.random.randn(N)
    for i in range(10, 20):
        features[:, i] = base2 + np.random.randn(N) * 0.3

    # Group 3: features 20-29
    base3 = np.random.randn(N)
    for i in range(20, 30):
        features[:, i] = base3 + np.random.randn(N) * 0.3

    feature_names = [f"temporal_{i}" for i in range(D)]

    # Create config with GPU disabled for clustering
    config = TemporalGroupingConfig()
    config.clustering = ClusteringParams(
        num_groups=3,  # Force 3 groups
        use_gpu=False,  # Disable GPU to avoid CuPy requirement
        distance_metric="euclidean",  # Use Euclidean (doesn't need GPU)
    )
    config.skip_gradient_refinement = True  # Skip gradient for speed

    # Create grouper
    grouper = create_grouper("temporal", config=config)

    # Group features
    result = grouper.group_features(features, feature_names)

    # Verify results
    assert result.num_groups == 3
    assert result.total_features == 30

    # Verify all features assigned
    all_indices = []
    for group in result.groups.values():
        all_indices.extend(group.feature_indices)
    assert len(all_indices) == 30
    assert set(all_indices) == set(range(30))

    # Test to_dict conversion
    group_dict = result.to_dict()
    assert len(group_dict) == 3
    assert all(isinstance(v, list) for v in group_dict.values())

    print("\n✅ Quick demo test passed!")
    print(f"Discovered {result.num_groups} groups:")
    for name, group in result.groups.items():
        print(f"  {name}: {group.size} features - {group.feature_indices[:5]}...")


if __name__ == "__main__":
    test_quick_demo_no_gpu()
