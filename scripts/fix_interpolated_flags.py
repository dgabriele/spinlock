#!/usr/bin/env python3
"""
Fix is_interpolated flags in MNO feature dataset.

MNO was trained on first 10K samples from 100K CNO dataset.
When generating 100K MNO features with sampling_mode="reference",
all samples were marked as is_interpolated=False.

This script corrects the flags:
- Samples 0-10K: is_interpolated=False (MNO training set)
- Samples 10K-100K: is_interpolated=True (MNO interpolating)
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fix is_interpolated flags in MNO dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to MNO feature dataset (e.g., datasets/mno_features_100k.h5)"
    )
    parser.add_argument(
        "--training-samples",
        type=int,
        default=10000,
        help="Number of samples MNO was trained on (default: 10000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying the file"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        return 1

    print(f"Processing: {dataset_path}")
    print(f"MNO training samples: {args.training_samples}")
    print()

    # Open dataset in read mode first to check current state
    with h5py.File(dataset_path, 'r') as f:
        if 'metadata/is_interpolated' not in f:
            print("Error: Dataset missing 'metadata/is_interpolated' field")
            return 1

        current_flags = f['metadata/is_interpolated'][:]
        total_samples = len(current_flags)

        print(f"Current state:")
        print(f"  Total samples: {total_samples}")
        print(f"  Exact (False): {(~current_flags).sum()}")
        print(f"  Interpolated (True): {current_flags.sum()}")
        print()

    # Create new flags
    new_flags = np.ones(total_samples, dtype=bool)
    new_flags[:args.training_samples] = False  # First N samples are exact

    print(f"Target state:")
    print(f"  Total samples: {total_samples}")
    print(f"  Exact (False): indices 0-{args.training_samples-1} = {(~new_flags).sum()}")
    print(f"  Interpolated (True): indices {args.training_samples}-{total_samples-1} = {new_flags.sum()}")
    print()

    if args.dry_run:
        print("DRY RUN - No changes made")
        return 0

    # Update the dataset
    print("Updating dataset...")
    with h5py.File(dataset_path, 'r+') as f:
        # Delete old dataset
        del f['metadata/is_interpolated']

        # Create new dataset with correct flags
        f['metadata'].create_dataset(
            'is_interpolated',
            data=new_flags,
            compression='gzip',
            compression_opts=4,
        )
        f['metadata/is_interpolated'].attrs['description'] = (
            'Boolean mask: True for interpolated points (MNO generalizing), '
            'False for exact training points (MNO learned during training). '
            f'First {args.training_samples} samples are exact, rest are interpolated.'
        )

    # Verify the update
    with h5py.File(dataset_path, 'r') as f:
        updated_flags = f['metadata/is_interpolated'][:]
        print(f"✓ Updated successfully:")
        print(f"  Exact (False): {(~updated_flags).sum()}")
        print(f"  Interpolated (True): {updated_flags.sum()}")

    print("\nDone! VQ-VAE training with reference regularization will now apply to samples 10K-100K.")
    return 0


if __name__ == "__main__":
    exit(main())
