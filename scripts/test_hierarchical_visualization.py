#!/usr/bin/env python3
"""Test script for hierarchical token pattern analysis visualization.

Usage:
    python scripts/test_hierarchical_visualization.py
"""

from pathlib import Path
from spinlock.visualization.vqvae.roundtrip_dashboard import (
    generate_hierarchical_pattern_analysis,
)


def main():
    # Configuration
    checkpoint_path = Path("checkpoints/production/10k_v9_intelligent_filtering")
    tokenized_dataset_path = Path("datasets/50k_baseline_tokenized.h5")
    output_dir = Path("visualizations/test_hierarchical_viz")

    print("=" * 80)
    print("Testing Hierarchical Token Pattern Analysis")
    print("=" * 80)

    # Verify inputs exist
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    if not tokenized_dataset_path.exists():
        print(f"Error: Tokenized dataset not found: {tokenized_dataset_path}")
        return 1

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {tokenized_dataset_path}")
    print(f"Output: {output_dir}")
    print()

    # Run analysis
    try:
        generate_hierarchical_pattern_analysis(
            checkpoint_path=checkpoint_path,
            tokenized_dataset_path=tokenized_dataset_path,
            output_dir=output_dir,
            family="temporal",
            max_rollouts=1000,  # Limit for faster testing
            similarity_metric="jaccard",
        )

        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        print(f"\nCheck outputs in: {output_dir}")

        # List generated files
        if output_dir.exists():
            files = sorted(output_dir.glob("*.png"))
            if files:
                print(f"\nGenerated {len(files)} visualization(s):")
                for f in files:
                    print(f"  - {f.name}")

        return 0

    except Exception as e:
        print(f"\nError during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
