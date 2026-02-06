"""Integration tests for feature grouping."""

import pytest
import numpy as np

from spinlock.features.grouping import (
    create_grouper,
    TemporalFeatureGrouper,
    InitialFeatureGrouper,
    TemporalGroupingConfig,
    InitialGroupingConfig,
)


@pytest.fixture
def temporal_features():
    """Create temporal-like features."""
    np.random.seed(42)
    # 200 samples, 30 features (typical for temporal)
    N, D = 200, 30
    features = np.random.randn(N, D).astype(np.float32)

    # Add correlation structure (3 groups)
    for i in range(1, 10):
        features[:, i] = features[:, 0] + np.random.randn(N) * 0.3
    for i in range(11, 20):
        features[:, i] = features[:, 10] + np.random.randn(N) * 0.3
    for i in range(21, 30):
        features[:, i] = features[:, 20] + np.random.randn(N) * 0.3

    return features


@pytest.fixture
def initial_features():
    """Create initial-like features."""
    np.random.seed(42)
    # 100 samples, 8 features (typical for initial)
    N, D = 100, 8
    features = np.random.randn(N, D).astype(np.float32)

    # Add correlation structure (2 groups)
    for i in range(1, 4):
        features[:, i] = features[:, 0] + np.random.randn(N) * 0.3
    for i in range(5, 8):
        features[:, i] = features[:, 4] + np.random.randn(N) * 0.3

    return features


def test_create_grouper_temporal():
    """Test factory creates correct temporal grouper."""
    # Skip if CuPy not available (GPU required by default)
    pytest.importorskip("cupy")

    grouper = create_grouper("temporal")
    assert isinstance(grouper, TemporalFeatureGrouper)
    assert isinstance(grouper.config, TemporalGroupingConfig)


def test_create_grouper_initial():
    """Test factory creates correct initial grouper."""
    # Skip if CuPy not available (GPU required by default)
    pytest.importorskip("cupy")

    grouper = create_grouper("initial")
    assert isinstance(grouper, InitialFeatureGrouper)
    assert isinstance(grouper.config, InitialGroupingConfig)


def test_create_grouper_invalid():
    """Test factory raises on invalid family."""
    with pytest.raises(ValueError):
        create_grouper("invalid")


def test_temporal_grouping_end_to_end(temporal_features):
    """Test end-to-end temporal feature grouping."""
    try:
        # Create grouper with custom config
        config = TemporalGroupingConfig(
            clustering__num_groups=3,  # Force 3 groups
            skip_gradient_refinement=True,  # Skip gradient for speed
        )
        # Manually create config with nested structure
        from spinlock.features.grouping.models import ClusteringParams
        config = TemporalGroupingConfig()
        config.clustering = ClusteringParams(
            num_groups=3,
            use_gpu=True,
        )
        config.skip_gradient_refinement = True

        grouper = create_grouper("temporal", config=config)

        feature_names = [f"temporal_{i}" for i in range(30)]
        result = grouper.group_features(temporal_features, feature_names)

        # Verify result structure
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
    except ImportError:
        pytest.skip("CuPy not available")


def test_initial_grouping_end_to_end(initial_features):
    """Test end-to-end initial feature grouping."""
    try:
        # Create grouper with custom config
        from spinlock.features.grouping.models import ClusteringParams
        config = InitialGroupingConfig()
        config.clustering = ClusteringParams(
            num_groups=2,
            use_gpu=True,
        )
        config.skip_gradient_refinement = True

        grouper = create_grouper("initial", config=config)

        feature_names = [f"initial_{i}" for i in range(8)]
        result = grouper.group_features(initial_features, feature_names)

        # Verify result structure
        assert result.num_groups == 2
        assert result.total_features == 8

        # Verify all features assigned
        all_indices = []
        for group in result.groups.values():
            all_indices.extend(group.feature_indices)
        assert len(all_indices) == 8
        assert set(all_indices) == set(range(8))
    except ImportError:
        pytest.skip("CuPy not available")


def test_temporal_with_gradient_refinement(temporal_features):
    """Test temporal grouping with gradient refinement."""
    try:
        # Create grouper with gradient enabled
        from spinlock.features.grouping.models import ClusteringParams, GradientParams
        config = TemporalGroupingConfig()
        config.clustering = ClusteringParams(
            num_groups=3,
            use_gpu=True,
        )
        config.gradient = GradientParams(
            num_epochs=50,  # Short for testing
            device="cpu",  # Use CPU for gradient
        )
        config.skip_gradient_refinement = False

        grouper = create_grouper("temporal", config=config)

        feature_names = [f"temporal_{i}" for i in range(30)]
        result = grouper.group_features(temporal_features, feature_names)

        # Verify result
        assert result.num_groups == 3
        assert result.total_features == 30
    except ImportError:
        pytest.skip("CuPy not available")


def test_preprocessing_methods(temporal_features):
    """Test different preprocessing methods."""
    try:
        from spinlock.features.grouping.models import PreprocessingParams, ClusteringParams

        for method in ["mad", "zscore", "minmax", "none"]:
            config = TemporalGroupingConfig()
            config.preprocessing = PreprocessingParams(method=method)
            config.clustering = ClusteringParams(num_groups=3, use_gpu=True)
            config.skip_gradient_refinement = True

            grouper = create_grouper("temporal", config=config)

            feature_names = [f"temporal_{i}" for i in range(30)]
            result = grouper.group_features(temporal_features, feature_names)

            # Should work with all methods
            assert result.num_groups == 3
    except ImportError:
        pytest.skip("CuPy not available")


def test_validation_min_samples():
    """Test validation for minimum samples."""
    # Skip if CuPy not available
    pytest.importorskip("cupy")

    grouper = create_grouper("temporal")

    # Too few samples
    features = np.random.randn(50, 30)  # Need 100 for temporal
    feature_names = [f"feat_{i}" for i in range(30)]

    with pytest.raises(ValueError, match="at least 100 samples"):
        grouper.group_features(features, feature_names)


def test_validation_min_features():
    """Test validation for minimum features."""
    # Skip if CuPy not available
    pytest.importorskip("cupy")

    grouper = create_grouper("temporal")

    # Too few features
    features = np.random.randn(200, 5)  # Need at least min_groups
    feature_names = [f"feat_{i}" for i in range(5)]

    with pytest.raises(ValueError, match="min_groups"):
        grouper.group_features(features, feature_names)
