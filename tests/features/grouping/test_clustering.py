"""Tests for ClusteringEngine."""

import pytest
import numpy as np

from spinlock.features.grouping.clustering import ClusteringEngine
from spinlock.features.grouping.models import (
    ClusteringParams,
    LinkageMethod,
    DistanceMetric,
    KSelectionMethod,
)


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    # Create 100 samples with 10 features
    # Features 0-4 are correlated (group 1)
    # Features 5-9 are correlated (group 2)
    N, D = 100, 10
    features = np.random.randn(N, D)

    # Add correlation structure
    for i in range(1, 5):
        features[:, i] = features[:, 0] + np.random.randn(N) * 0.3
    for i in range(6, 10):
        features[:, i] = features[:, 5] + np.random.randn(N) * 0.3

    return features


@pytest.fixture
def feature_names():
    """Create feature names."""
    return [f"feat_{i}" for i in range(10)]


def test_clustering_engine_init():
    """Test ClusteringEngine initialization."""
    config = ClusteringParams(use_gpu=True)

    # This will raise if CuPy not available
    try:
        engine = ClusteringEngine(config)
        assert engine.config == config
        assert engine._gpu_available is True
    except ImportError:
        pytest.skip("CuPy not available")


def test_clustering_engine_no_gpu():
    """Test ClusteringEngine with GPU disabled."""
    config = ClusteringParams(use_gpu=False)

    # Should not raise, just set _gpu_available to False
    engine = ClusteringEngine(config)
    assert engine._gpu_available is False


def test_cluster_basic(sample_features, feature_names):
    """Test basic clustering."""
    config = ClusteringParams(
        num_groups=2,  # Manual K
        use_gpu=True,
    )

    try:
        engine = ClusteringEngine(config)
        groups = engine.cluster(sample_features, feature_names)

        # Should have 2 groups
        assert len(groups) == 2

        # All features should be assigned
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert len(all_indices) == 10
        assert set(all_indices) == set(range(10))
    except ImportError:
        pytest.skip("CuPy not available")


def test_cluster_auto_k(sample_features, feature_names):
    """Test automatic K selection."""
    config = ClusteringParams(
        min_groups=2,
        max_groups=5,
        k_selection_method=KSelectionMethod.SILHOUETTE,
        use_gpu=True,
    )

    try:
        engine = ClusteringEngine(config)
        groups = engine.cluster(sample_features, feature_names)

        # Should have between 2-5 groups
        assert 2 <= len(groups) <= 5

        # All features should be assigned
        all_indices = []
        for indices in groups.values():
            all_indices.extend(indices)
        assert len(all_indices) == 10
    except ImportError:
        pytest.skip("CuPy not available")


def test_euclidean_distance(sample_features):
    """Test Euclidean distance computation."""
    config = ClusteringParams(
        distance_metric=DistanceMetric.EUCLIDEAN,
        use_gpu=False,  # Euclidean doesn't need GPU
    )

    engine = ClusteringEngine(config)
    distance = engine._euclidean_distance(sample_features)

    # Should be 1D condensed distance matrix
    D = sample_features.shape[1]
    expected_len = D * (D - 1) // 2
    assert distance.shape == (expected_len,)
    assert np.all(distance >= 0)


def test_cosine_distance(sample_features):
    """Test cosine distance computation."""
    config = ClusteringParams(
        distance_metric=DistanceMetric.COSINE,
        use_gpu=False,  # Cosine doesn't need GPU
    )

    engine = ClusteringEngine(config)
    distance = engine._cosine_distance(sample_features)

    # Should be 1D condensed distance matrix
    D = sample_features.shape[1]
    expected_len = D * (D - 1) // 2
    assert distance.shape == (expected_len,)
    assert np.all(distance >= 0)


def test_correlation_distance_gpu(sample_features):
    """Test GPU correlation distance."""
    config = ClusteringParams(
        distance_metric=DistanceMetric.CORRELATION,
        use_gpu=True,
    )

    try:
        engine = ClusteringEngine(config)
        distance = engine._correlation_distance(sample_features)

        # Should be 1D condensed distance matrix
        D = sample_features.shape[1]
        expected_len = D * (D - 1) // 2
        assert distance.shape == (expected_len,)
        assert np.all(distance >= 0)
        assert np.all(distance <= 2)  # Max distance is 2 (1 - (-1))
    except ImportError:
        pytest.skip("CuPy not available")


def test_linkage_methods(sample_features):
    """Test different linkage methods."""
    config = ClusteringParams(use_gpu=True)

    try:
        engine = ClusteringEngine(config)
        distance = engine._correlation_distance(sample_features)

        for method in [LinkageMethod.WARD, LinkageMethod.AVERAGE,
                       LinkageMethod.COMPLETE, LinkageMethod.SINGLE]:
            config.linkage_method = method
            engine = ClusteringEngine(config)
            linkage = engine._compute_linkage(distance)

            # Linkage matrix should have correct shape
            D = sample_features.shape[1]
            assert linkage.shape == (D - 1, 4)
    except ImportError:
        pytest.skip("CuPy not available")
