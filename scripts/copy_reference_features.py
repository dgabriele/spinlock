#!/usr/bin/env python3
"""
Directly copy reference features from CNO dataset to MNO dataset.

Since MNO dataset was generated using generation_indices pointing into
the reference dataset, we can directly copy the corresponding features
without any matching logic.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


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

    # Copy features from reference dataset
    with h5py.File(ref_path, 'r') as ref:
        if 'features/summary/aggregated/features' not in ref:
            print("Error: Reference dataset missing summary features")
            return 1

        ref_features = ref['features/summary/aggregated/features']
        print(f"Reference features shape: {ref_features.shape}")

        # Extract features at generation indices
        print(f"Copying features at indices {gen_indices[0]}-{gen_indices[-1]}...")
        copied_features = ref_features[gen_indices]
        print(f"Copied features shape: {copied_features.shape}")
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
