"""Tests for checkpoint feature metadata storage and loading."""

import pytest
import numpy as np
import torch
from pathlib import Path

from spinlock.tokens.checkpoint import (
    CleaningOperation,
    FeatureFamilyMetadata,
    CategoryMetadata,
    FeatureMetadata,
    TokenizerCheckpoint,
    save_checkpoint,
    load_checkpoint,
    inspect_checkpoint_features,
)
from spinlock.tokens.config import TokenizerConfig
from spinlock.encoding.feature_processor import FeatureProcessor


def test_cleaning_operation_model():
    """Test CleaningOperation Pydantic model."""
    op = CleaningOperation(
        operation_type="variance_filter",
        features_removed=[0, 2, 5],
        features_kept=[1, 3, 4],
        parameters={"variance_threshold": 1e-10},
        num_removed=3,
        num_kept=3,
    )

    assert op.operation_type == "variance_filter"
    assert len(op.features_removed) == 3
    assert len(op.features_kept) == 3
    assert op.num_removed == 3
    assert op.num_kept == 3


def test_feature_family_metadata_model():
    """Test FeatureFamilyMetadata Pydantic model."""
    family = FeatureFamilyMetadata(
        family_name="temporal",
        original_feature_count=100,
        cleaned_feature_count=78,
        original_feature_names=[f"feature_{i}" for i in range(100)],
        kept_feature_indices=list(range(78)),
        kept_feature_names=[f"feature_{i}" for i in range(78)],
        removed_feature_indices=list(range(78, 100)),
        removed_feature_names=[f"feature_{i}" for i in range(78, 100)],
        cleaning_operations=[],
    )

    assert family.family_name == "temporal"
    assert family.original_feature_count == 100
    assert family.cleaned_feature_count == 78
    assert len(family.removed_feature_indices) == 22


def test_category_metadata_model():
    """Test CategoryMetadata Pydantic model."""
    cat = CategoryMetadata(
        category_name="temporal_group_1_L0",
        feature_indices=[0, 1, 2, 3, 4],
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        original_indices=[0, 1, 2, 3, 4],
        num_features=5,
    )

    assert cat.category_name == "temporal_group_1_L0"
    assert cat.num_features == 5
    assert len(cat.feature_indices) == 5


def test_feature_metadata_model():
    """Test FeatureMetadata root model."""
    family = FeatureFamilyMetadata(
        family_name="temporal",
        original_feature_count=100,
        cleaned_feature_count=78,
        original_feature_names=[f"feature_{i}" for i in range(100)],
        kept_feature_indices=list(range(78)),
        kept_feature_names=[f"feature_{i}" for i in range(78)],
        removed_feature_indices=list(range(78, 100)),
        removed_feature_names=[f"feature_{i}" for i in range(78, 100)],
        cleaning_operations=[],
    )

    cat = CategoryMetadata(
        category_name="temporal_group_1_L0",
        feature_indices=[0, 1, 2, 3, 4],
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        original_indices=[0, 1, 2, 3, 4],
        num_features=5,
    )

    metadata = FeatureMetadata(
        families={"temporal": family},
        categories={"temporal_group_1_L0": cat},
        total_original_features=100,
        total_cleaned_features=78,
        total_features_removed=22,
        cleaning_config=None,
        grouping_config=None,
    )

    assert metadata.total_original_features == 100
    assert metadata.total_cleaned_features == 78
    assert metadata.total_features_removed == 22
    assert len(metadata.families) == 1
    assert len(metadata.categories) == 1


def test_feature_processor_returns_cleaning_report():
    """Test that FeatureProcessor.clean() returns cleaning report."""
    # Create synthetic features with known properties
    np.random.seed(42)
    N = 1000
    D = 50

    # Create features with:
    # - Some zero-variance features
    # - Some duplicated features
    # - Some features with NaNs
    # - Some features with outliers
    features = np.random.randn(N, D).astype(np.float32)

    # Add zero-variance features
    features[:, 0] = 1.0  # Constant
    features[:, 1] = 1.0  # Constant

    # Add duplicated features (with enough variance to pass variance filter)
    features[:, 5] = features[:, 3] + np.random.randn(N) * 0.01  # Slightly noisy duplicate

    # Don't add NaNs to features that will be used in deduplication
    # (NaN handling happens after deduplication in the pipeline)

    # Add outliers
    features[:5, 20] = 1000.0

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(D)]

    # Clean
    processor = FeatureProcessor(
        variance_threshold=1e-10,
        deduplicate_threshold=0.99,
        outlier_method="percentile",
        verbose=False,
    )

    cleaned, mask, cleaned_names, report = processor.clean(features, feature_names)

    # Verify report structure
    assert 'operations' in report
    assert 'original_feature_count' in report
    assert 'cleaned_feature_count' in report
    assert 'total_removed' in report

    assert report['original_feature_count'] == D
    assert report['cleaned_feature_count'] == cleaned.shape[1]
    assert report['total_removed'] == D - cleaned.shape[1]

    # Check operations
    assert len(report['operations']) == 4  # variance, dedup, nan, outlier

    # Verify variance filtering removed features 0 and 1
    variance_op = report['operations'][0]
    assert variance_op['operation_type'] == 'variance_filter'
    assert 0 in variance_op['features_removed']
    assert 1 in variance_op['features_removed']

    # Verify deduplication happened (may or may not remove feature 5 due to noise)
    dedup_op = report['operations'][1]
    assert dedup_op['operation_type'] == 'deduplication'


def test_checkpoint_saves_and_loads_feature_metadata(tmp_path):
    """Test that checkpoints can save and load feature_metadata."""
    # Create minimal config
    config = TokenizerConfig()

    # Create minimal model state dict
    model_state_dict = {"dummy": torch.tensor([1.0])}

    # Create feature metadata
    family = FeatureFamilyMetadata(
        family_name="temporal",
        original_feature_count=100,
        cleaned_feature_count=78,
        original_feature_names=[f"feature_{i}" for i in range(100)],
        kept_feature_indices=list(range(78)),
        kept_feature_names=[f"feature_{i}" for i in range(78)],
        removed_feature_indices=list(range(78, 100)),
        removed_feature_names=[f"feature_{i}" for i in range(78, 100)],
        cleaning_operations=[],
    )

    cat = CategoryMetadata(
        category_name="temporal_group_1_L0",
        feature_indices=[0, 1, 2, 3, 4],
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        original_indices=[0, 1, 2, 3, 4],
        num_features=5,
    )

    feature_metadata = FeatureMetadata(
        families={"temporal": family},
        categories={"temporal_group_1_L0": cat},
        total_original_features=100,
        total_cleaned_features=78,
        total_features_removed=22,
        cleaning_config=None,
        grouping_config=None,
    )

    # Create mock model with state_dict method
    class MockModel:
        def state_dict(self):
            return model_state_dict

    model = MockModel()

    # Save checkpoint with feature_metadata
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config=config,
        group_indices={"temporal_group_1_L0": [0, 1, 2, 3, 4]},
        feature_metadata=feature_metadata,
        epoch=10,
        val_loss=0.123,
    )

    # Load checkpoint
    loaded = load_checkpoint(checkpoint_path)

    # Verify feature_metadata was saved and loaded
    assert loaded.feature_metadata is not None
    assert loaded.feature_metadata.total_original_features == 100
    assert loaded.feature_metadata.total_cleaned_features == 78
    assert loaded.feature_metadata.total_features_removed == 22
    assert "temporal" in loaded.feature_metadata.families
    assert "temporal_group_1_L0" in loaded.feature_metadata.categories


def test_checkpoint_backward_compatibility(tmp_path):
    """Test that old checkpoints without feature_metadata load gracefully."""
    # Create minimal config
    config = TokenizerConfig()

    # Create minimal model state dict
    model_state_dict = {"dummy": torch.tensor([1.0])}

    class MockModel:
        def state_dict(self):
            return model_state_dict

    model = MockModel()

    # Save checkpoint WITHOUT feature_metadata (old format)
    checkpoint_path = tmp_path / "old_checkpoint.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        config=config,
        group_indices={"temporal_group_1_L0": [0, 1, 2, 3, 4]},
        feature_metadata=None,  # Explicitly None (old format)
        epoch=10,
        val_loss=0.123,
    )

    # Load checkpoint - should succeed with warning
    loaded = load_checkpoint(checkpoint_path)

    # Verify feature_metadata is None (backward compatibility)
    assert loaded.feature_metadata is None
    assert loaded.group_indices == {"temporal_group_1_L0": [0, 1, 2, 3, 4]}


def test_inspect_checkpoint_features():
    """Test inspect_checkpoint_features utility."""
    # Create checkpoint with feature_metadata
    config = TokenizerConfig()
    model_state_dict = {"dummy": torch.tensor([1.0])}

    family = FeatureFamilyMetadata(
        family_name="temporal",
        original_feature_count=100,
        cleaned_feature_count=78,
        original_feature_names=[f"feature_{i}" for i in range(100)],
        kept_feature_indices=list(range(78)),
        kept_feature_names=[f"feature_{i}" for i in range(78)],
        removed_feature_indices=list(range(78, 100)),
        removed_feature_names=[f"feature_{i}" for i in range(78, 100)],
        cleaning_operations=[],
    )

    feature_metadata = FeatureMetadata(
        families={"temporal": family},
        categories={},
        total_original_features=100,
        total_cleaned_features=78,
        total_features_removed=22,
        cleaning_config=None,
        grouping_config=None,
    )

    checkpoint = TokenizerCheckpoint(
        model_state_dict=model_state_dict,
        config=config,
        group_indices={},
        feature_metadata=feature_metadata,
    )

    # Generate summary
    summary = inspect_checkpoint_features(checkpoint)

    # Verify summary contains key information
    assert "FEATURE METADATA SUMMARY" in summary
    assert "Total Features:" in summary
    assert "Original: 100" in summary
    assert "After Cleaning: 78" in summary
    assert "Removed: 22" in summary
    assert "TEMPORAL:" in summary


def test_inspect_checkpoint_without_metadata():
    """Test inspect_checkpoint_features with old checkpoint."""
    config = TokenizerConfig()
    model_state_dict = {"dummy": torch.tensor([1.0])}

    checkpoint = TokenizerCheckpoint(
        model_state_dict=model_state_dict,
        config=config,
        group_indices={},
        feature_metadata=None,  # Old format
    )

    summary = inspect_checkpoint_features(checkpoint)
    assert "does not contain feature_metadata" in summary
