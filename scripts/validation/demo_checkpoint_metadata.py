#!/usr/bin/env python3
"""
Demo script to showcase checkpoint feature metadata functionality.

This script demonstrates the new v2.1 checkpoint format with feature metadata,
showing how it eliminates code duplication and provides transparency.
"""

import numpy as np
import torch
from pathlib import Path
import tempfile

from spinlock.tokens.checkpoint import (
    CleaningOperation,
    FeatureFamilyMetadata,
    CategoryMetadata,
    FeatureMetadata,
    TokenizerCheckpoint,
    save_checkpoint,
    load_checkpoint,
    inspect_checkpoint_features,
    export_feature_metadata_to_json,
)
from spinlock.tokens.config import TokenizerConfig
from spinlock.encoding.feature_processor import FeatureProcessor


def demo_feature_processor_cleaning_report():
    """Demo: FeatureProcessor returns detailed cleaning report."""
    print("=" * 80)
    print("DEMO 1: FeatureProcessor Cleaning Report")
    print("=" * 80)

    # Create synthetic features
    np.random.seed(42)
    N = 1000
    D = 50
    features = np.random.randn(N, D).astype(np.float32)

    # Add problematic features
    features[:, 0] = 1.0  # Zero variance
    features[:, 1] = 1.0  # Zero variance
    features[:, 5] = features[:, 3] + np.random.randn(N) * 0.01  # Duplicate
    features[:5, 20] = 1000.0  # Outliers

    # Clean with FeatureProcessor
    processor = FeatureProcessor(
        variance_threshold=1e-10,
        deduplicate_threshold=0.99,
        outlier_method="percentile",
        verbose=True,
    )

    feature_names = [f"feature_{i}" for i in range(D)]
    cleaned, mask, cleaned_names, report = processor.clean(features, feature_names)

    print(f"\n✓ Cleaning complete:")
    print(f"  Original features: {D}")
    print(f"  Cleaned features: {len(cleaned_names)}")
    print(f"  Removed features: {report['total_removed']}")
    print(f"\n✓ Cleaning report contains {len(report['operations'])} operations:")
    for i, op in enumerate(report['operations'], 1):
        print(f"  {i}. {op['operation_type']}: removed {op['num_removed']} features")

    return report


def demo_checkpoint_with_metadata():
    """Demo: Save and load checkpoint with feature metadata."""
    print("\n" + "=" * 80)
    print("DEMO 2: Checkpoint with Feature Metadata")
    print("=" * 80)

    # Create feature metadata
    operations = [
        CleaningOperation(
            operation_type="variance_filter",
            features_removed=[0, 1, 5, 10],
            features_kept=list(range(2, 100)),
            parameters={"variance_threshold": 1e-10},
            num_removed=4,
            num_kept=96,
        ),
    ]

    temporal_family = FeatureFamilyMetadata(
        family_name="temporal",
        original_feature_count=100,
        cleaned_feature_count=96,
        original_feature_names=[f"temporal_feature_{i}" for i in range(100)],
        kept_feature_indices=list(range(2, 100)) + [0, 1],  # Simulate removal of 0,1,5,10
        kept_feature_names=[f"temporal_feature_{i}" for i in range(2, 100)] + ["temporal_feature_0", "temporal_feature_1"],
        removed_feature_indices=[0, 1, 5, 10],
        removed_feature_names=[f"temporal_feature_{i}" for i in [0, 1, 5, 10]],
        cleaning_operations=operations,
    )

    categories = {
        "temporal_group_1_L0": CategoryMetadata(
            category_name="temporal_group_1_L0",
            feature_indices=list(range(10)),
            feature_names=[f"temporal_feature_{i}" for i in range(2, 12)],
            original_indices=list(range(2, 12)),
            num_features=10,
        ),
    }

    feature_metadata = FeatureMetadata(
        families={"temporal": temporal_family},
        categories=categories,
        total_original_features=100,
        total_cleaned_features=96,
        total_features_removed=4,
        cleaning_config={"variance_threshold": 1e-10},
        grouping_config={"method": "hierarchical"},
    )

    # Save checkpoint
    config = TokenizerConfig()
    model_state_dict = {"dummy": torch.tensor([1.0])}

    class MockModel:
        def state_dict(self):
            return model_state_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        save_checkpoint(
            path=checkpoint_path,
            model=MockModel(),
            config=config,
            group_indices={"temporal_group_1_L0": list(range(10))},
            feature_metadata=feature_metadata,
            epoch=10,
            val_loss=0.123,
        )

        print(f"\n✓ Checkpoint saved to: {checkpoint_path}")
        print(f"  Size: {checkpoint_path.stat().st_size / 1024:.2f} KB")

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path)

        print(f"\n✓ Checkpoint loaded successfully")
        print(f"  Contains feature_metadata: {loaded.feature_metadata is not None}")
        print(f"  Total features: {loaded.feature_metadata.total_cleaned_features}")
        print(f"  Families: {list(loaded.feature_metadata.families.keys())}")
        print(f"  Categories: {list(loaded.feature_metadata.categories.keys())}")

        return loaded


def demo_inspect_checkpoint(checkpoint):
    """Demo: Inspect checkpoint feature metadata."""
    print("\n" + "=" * 80)
    print("DEMO 3: Inspect Checkpoint Features")
    print("=" * 80)

    summary = inspect_checkpoint_features(checkpoint)
    print(summary)


def demo_export_metadata(checkpoint):
    """Demo: Export metadata to JSON."""
    print("\n" + "=" * 80)
    print("DEMO 4: Export Metadata to JSON")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metadata.json"
        export_feature_metadata_to_json(checkpoint, json_path)

        print(f"\n✓ Metadata exported to: {json_path}")
        print(f"  Size: {json_path.stat().st_size / 1024:.2f} KB")

        # Show sample content
        import json
        with open(json_path) as f:
            metadata = json.load(f)

        print(f"\n✓ JSON structure:")
        print(f"  Keys: {list(metadata.keys())}")
        print(f"  Families: {list(metadata['families'].keys())}")
        print(f"  Categories: {list(metadata['categories'].keys())}")


def demo_backward_compatibility():
    """Demo: Loading old checkpoints without metadata."""
    print("\n" + "=" * 80)
    print("DEMO 5: Backward Compatibility")
    print("=" * 80)

    config = TokenizerConfig()
    model_state_dict = {"dummy": torch.tensor([1.0])}

    class MockModel:
        def state_dict(self):
            return model_state_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "old_checkpoint.pt"

        # Save WITHOUT feature_metadata (v2.0 format)
        save_checkpoint(
            path=checkpoint_path,
            model=MockModel(),
            config=config,
            group_indices={"temporal_group_1_L0": list(range(10))},
            feature_metadata=None,  # Old format!
            epoch=10,
            val_loss=0.123,
        )

        print(f"\n✓ Old checkpoint (v2.0) saved to: {checkpoint_path}")

        # Load - should work with warning
        loaded = load_checkpoint(checkpoint_path)

        print(f"\n✓ Old checkpoint loaded successfully (backward compatible)")
        print(f"  Contains feature_metadata: {loaded.feature_metadata is not None}")
        print(f"  Fallback behavior: Will re-run FeatureProcessor during pretokenization")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("VQTokenizer Feature Metadata Checkpoint Storage - Demo")
    print("=" * 80)
    print("\nThis demo showcases the new v2.1 checkpoint format with feature metadata.")
    print("Feature metadata eliminates code duplication and provides transparency.\n")

    # Demo 1: FeatureProcessor cleaning report
    report = demo_feature_processor_cleaning_report()

    # Demo 2: Save/load checkpoint with metadata
    checkpoint = demo_checkpoint_with_metadata()

    # Demo 3: Inspect checkpoint
    demo_inspect_checkpoint(checkpoint)

    # Demo 4: Export metadata to JSON
    demo_export_metadata(checkpoint)

    # Demo 5: Backward compatibility
    demo_backward_compatibility()

    print("\n" + "=" * 80)
    print("All demos complete! ✓")
    print("=" * 80)
    print("\nKey Benefits:")
    print("  • Zero code duplication (FeatureProcessor logic in one place)")
    print("  • Perfect consistency (checkpoint is single source of truth)")
    print("  • Full transparency (inspect and export feature organization)")
    print("  • Backward compatible (old checkpoints still work)")
    print("\nFor more details, see: docs/checkpoint-feature-metadata-implementation.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
