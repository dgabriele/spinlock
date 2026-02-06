"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from spinlock.features.grouping.models import (
    ClusteringParams,
    GradientParams,
    PreprocessingParams,
    SplittingParams,
    GroupingConfig,
    TemporalGroupingConfig,
    InitialGroupingConfig,
    FeatureGroup,
    GroupingResult,
    LinkageMethod,
    DistanceMetric,
    KSelectionMethod,
)


def test_clustering_params_defaults():
    """Test ClusteringParams default values."""
    params = ClusteringParams()
    assert params.linkage_method == LinkageMethod.WARD
    assert params.distance_metric == DistanceMetric.CORRELATION
    assert params.k_selection_method == KSelectionMethod.SILHOUETTE
    assert params.min_groups == 2
    assert params.max_groups == 20
    assert params.use_gpu is True


def test_clustering_params_validation():
    """Test ClusteringParams validation."""
    # Valid
    params = ClusteringParams(min_groups=5, max_groups=15)
    assert params.min_groups == 5
    assert params.max_groups == 15

    # Invalid: min_groups < 2
    with pytest.raises(ValidationError):
        ClusteringParams(min_groups=1)

    # Invalid: max_groups < 2
    with pytest.raises(ValidationError):
        ClusteringParams(max_groups=1)


def test_gradient_params_defaults():
    """Test GradientParams default values."""
    params = GradientParams()
    assert params.num_epochs == 500
    assert params.learning_rate == 0.01
    assert params.temperature_start == 1.0
    assert params.temperature_end == 0.5
    assert params.orthogonality_weight == 1.0
    assert params.informativeness_weight == 1.0
    assert params.custom_loss_weight == 0.0


def test_gradient_params_validation():
    """Test GradientParams validation."""
    # Valid
    params = GradientParams(num_epochs=1000, learning_rate=0.001)
    assert params.num_epochs == 1000
    assert params.learning_rate == 0.001

    # Invalid: num_epochs < 1
    with pytest.raises(ValidationError):
        GradientParams(num_epochs=0)

    # Invalid: learning_rate <= 0
    with pytest.raises(ValidationError):
        GradientParams(learning_rate=0.0)

    # Invalid: orthogonality_target > 1
    with pytest.raises(ValidationError):
        GradientParams(orthogonality_target=1.5)


def test_preprocessing_params_defaults():
    """Test PreprocessingParams default values."""
    params = PreprocessingParams()
    assert params.method == "mad"
    assert params.mad_constant == 1.4826
    assert params.clip_outliers is False


def test_splitting_params_defaults():
    """Test SplittingParams default values."""
    params = SplittingParams()
    assert params.enabled is False
    assert params.max_group_size == 40
    assert params.max_recursion_depth == 3
    assert params.min_features_per_group == 3


def test_grouping_config_defaults():
    """Test GroupingConfig default values."""
    config = GroupingConfig()
    assert isinstance(config.preprocessing, PreprocessingParams)
    assert isinstance(config.clustering, ClusteringParams)
    assert isinstance(config.gradient, GradientParams)
    assert isinstance(config.splitting, SplittingParams)
    assert config.skip_gradient_refinement is False
    assert config.min_samples_required == 50
    assert config.min_features_required == 2


def test_temporal_grouping_config():
    """Test TemporalGroupingConfig defaults."""
    config = TemporalGroupingConfig()
    assert config.clustering.min_groups == 8
    assert config.clustering.max_groups == 20
    assert config.min_samples_required == 100


def test_initial_grouping_config():
    """Test InitialGroupingConfig defaults."""
    config = InitialGroupingConfig()
    assert config.clustering.min_groups == 2
    assert config.clustering.max_groups == 5
    assert config.min_samples_required == 50


def test_feature_group():
    """Test FeatureGroup model."""
    group = FeatureGroup(
        name="group_1",
        feature_indices=[0, 1, 2],
        feature_names=["feat_0", "feat_1", "feat_2"],
        size=3,
    )
    assert group.name == "group_1"
    assert group.indices == [0, 1, 2]
    assert group.size == 3


def test_grouping_result():
    """Test GroupingResult model."""
    groups = {
        "group_1": FeatureGroup(
            name="group_1",
            feature_indices=[0, 1],
            feature_names=["feat_0", "feat_1"],
            size=2,
        ),
        "group_2": FeatureGroup(
            name="group_2",
            feature_indices=[2, 3],
            feature_names=["feat_2", "feat_3"],
            size=2,
        ),
    }
    result = GroupingResult(
        groups=groups,
        num_groups=2,
        total_features=4,
        config=GroupingConfig(),
    )
    assert result.num_groups == 2
    assert result.total_features == 4

    # Test to_dict conversion
    group_dict = result.to_dict()
    assert group_dict == {"group_1": [0, 1], "group_2": [2, 3]}
