#!/usr/bin/env python3
"""Add explicit format metadata to existing HDF5 datasets.

This script adds format specification metadata to datasets that were created
before the metadata standardization. It's safe to run multiple times.

Usage:
    poetry run python scripts/dataset/add_format_metadata.py datasets/qbm_50k.h5
"""

import argparse
import h5py
from pathlib import Path


def add_format_metadata(dataset_path: Path, verbose: bool = True):
    """Add explicit format metadata to HDF5 dataset.

    Args:
        dataset_path: Path to HDF5 dataset file
        verbose: Print detailed information
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with h5py.File(dataset_path, 'r+') as f:
        # Add metadata to /inputs/fields
        if 'inputs/fields' in f:
            inputs = f['inputs/fields']

            # Check current shape
            shape = inputs.shape
            if verbose:
                print(f"Processing: {dataset_path}")
                print(f"  Current shape: {shape}")

            # Validate 5D format [N, M, C, H, W]
            if len(shape) != 5:
                print(f"  ⚠️  Warning: Expected 5D tensor, got {len(shape)}D")
                print(f"  Skipping metadata addition")
                return

            # Add format specification
            inputs.attrs['format'] = 'NMCHW'  # Explicit format string
            inputs.attrs['num_realizations'] = int(shape[1])  # M dimension
            inputs.attrs['num_channels'] = int(shape[2])      # C dimension
            inputs.attrs['description'] = (
                f"Input fields in [N, M, C, H, W] format where "
                f"N=samples, M=realizations, C=channels, H=height, W=width"
            )

            if verbose:
                print(f"  ✓ Added metadata to inputs/fields")
                print(f"    Format: [N={shape[0]}, M={shape[1]}, "
                      f"C={shape[2]}, H={shape[3]}, W={shape[4]}]")
        else:
            print(f"  ⚠️  Warning: No inputs/fields found in dataset")

        # Add to /metadata group for global reference
        if 'metadata' not in f:
            f.create_group('metadata')

        f['metadata'].attrs['input_format'] = 'NMCHW'
        f['metadata'].attrs['realizations_dim'] = 1  # axis=1 is M
        f['metadata'].attrs['channels_dim'] = 2      # axis=2 is C

        if verbose:
            print(f"  ✓ Added metadata to /metadata group")
            print(f"\n✅ Metadata successfully added to {dataset_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Add format metadata to HDF5 datasets"
    )
    parser.add_argument(
        'dataset',
        type=Path,
        help="Path to HDF5 dataset file"
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    add_format_metadata(args.dataset, verbose=not args.quiet)


if __name__ == '__main__':
    main()
