#!/usr/bin/env python3
"""Extract INITIAL features from raw ICs for VQ-VAE training.

This script uses the unified InitialFeatureExtractionPipeline which handles
various input shapes and extractor types automatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.features.initial import extract_initial_features, ExtractorType


def main(dataset_path: str, use_statistical: bool = True) -> int:
    """
    Extract initial features from dataset.

    Args:
        dataset_path: Path to HDF5 dataset
        use_statistical: Use statistical features (recommended) vs manual features

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        extractor_type = ExtractorType.STATISTICAL if use_statistical else ExtractorType.MANUAL

        extract_initial_features(
            dataset_path=dataset_path,
            extractor_type=extractor_type,
            device='cpu',
            batch_size=100,
            include_spatial=False,  # Spatial features have collapsed variance
            overwrite=True,
            verbose=True,
        )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract initial features from dataset",
        epilog="""
Examples:
  # Extract statistical features (recommended, default)
  python extract_initial_manual_features.py datasets/qbm_50k.h5

  # Extract old manual/pattern features
  python extract_initial_manual_features.py datasets/data.h5 --manual

  # Use Python API directly
  python -c "from spinlock.features.initial import extract_initial_features; \\
             extract_initial_features('datasets/data.h5', extractor_type='statistical')"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", help="Path to HDF5 dataset")
    parser.add_argument(
        "--statistical",
        action="store_true",
        default=True,
        help="Use statistical features (distributional + energy, default)"
    )
    parser.add_argument(
        "--manual",
        dest="statistical",
        action="store_false",
        help="Use old manual/pattern features (legacy)"
    )
    args = parser.parse_args()

    sys.exit(main(args.dataset, use_statistical=args.statistical))
