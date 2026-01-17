#!/usr/bin/env python3
"""
Directly copy reference features from CNO dataset to MNO dataset.

Since MNO dataset was generated using generation_indices pointing into
the reference dataset, we can directly copy the corresponding features
without any matching logic.

Uses FeaturePreprocessor to clean reference features (remove operator_sensitivity)
before copying to match MNO dimension.
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.spinlock.features.preprocessing import FeaturePreprocessor


def main():
    parser = argparse.ArgumentParser(description="Copy reference features to MNO dataset")
    parser.add_argument(
        "--mno-dataset",
        type=str,
        required=True,
        help="Path to MNO feature dataset"
    )
    parser.add_argument(
        "--reference-dataset",
        type=str,
        required=True,
        help="Path to reference CNO dataset with features"
    )

    args = parser.parse_args()

    mno_path = Path(args.mno_dataset)
    ref_path = Path(args.reference_dataset)

    if not mno_path.exists():
        print(f"Error: MNO dataset not found: {mno_path}")
        return 1

    if not ref_path.exists():
        print(f"Error: Reference dataset not found: {ref_path}")
        return 1

    print("="*70)
    print("COPYING REFERENCE FEATURES")
    print("="*70)
    print(f"MNO dataset: {mno_path}")
    print(f"Reference dataset: {ref_path}")
    print()

    # Load generation indices from MNO dataset
    with h5py.File(mno_path, 'r') as f:
        if 'metadata/generation_indices' not in f:
            print("Error: MNO dataset missing generation_indices")
            return 1

        gen_indices = f['metadata/generation_indices'][:]
        print(f"Generation indices: {len(gen_indices)} samples")
        print(f"  Range: [{gen_indices[0]}, {gen_indices[-1]}]")
        print()

    # Create feature preprocessor from reference dataset to clean features
    print("Creating feature preprocessor from reference dataset...")
    preprocessor = FeaturePreprocessor.from_dataset(str(ref_path))
    info = preprocessor.get_info()
    if 'summary_aggregated' in info:
        print(f"  Summary features: {info['summary_aggregated']['total']} total, "
              f"{info['summary_aggregated']['valid']} valid, "
              f"{info['summary_aggregated']['nan']} NaN (operator_sensitivity)")
    print()

    # Copy and clean features from reference dataset
    with h5py.File(ref_path, 'r') as ref:
        if 'features/summary/aggregated/features' not in ref:
            print("Error: Reference dataset missing summary features")
            return 1

        ref_features = ref['features/summary/aggregated/features'][gen_indices]
        print(f"Reference features (raw): {ref_features.shape}")

        # Clean features using preprocessor (removes operator_sensitivity)
        import torch
        ref_features_tensor = torch.from_numpy(ref_features)
        cleaned_features = preprocessor.clean_features(ref_features_tensor, 'summary_aggregated')
        copied_features = cleaned_features.numpy()

        print(f"Reference features (cleaned): {copied_features.shape}")
        print(f"  Removed {ref_features.shape[1] - copied_features.shape[1]} operator_sensitivity features")
        print()

    # Write to MNO dataset
    with h5py.File(mno_path, 'r+') as mno:
        # Delete existing reference_features if present
        if 'features/reference_features' in mno:
            print("Deleting existing reference_features...")
            del mno['features/reference_features']

        # Create new dataset
        print("Writing reference_features to MNO dataset...")
        mno['features'].create_dataset(
            'reference_features',
            data=copied_features,
            compression='gzip',
            compression_opts=4,
        )
        mno['features/reference_features'].attrs['description'] = (
            'Reference CNO features at same parameter/IC points as MNO rollouts. '
            'Used for physical fidelity regularization during VQ-VAE training.'
        )

    print(f"✓ Successfully copied {len(gen_indices)} reference features")
    print()
    print("Done! VQ-VAE can now use reference regularization.")
    return 0


if __name__ == "__main__":
    exit(main())
