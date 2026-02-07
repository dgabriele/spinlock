#!/usr/bin/env python3
"""Fix 50k_baseline.h5 metadata to match actual sample count."""

import h5py
from pathlib import Path

dataset_path = Path("datasets/50k_baseline.h5")

print(f"Fixing metadata in {dataset_path}")

with h5py.File(dataset_path, 'r+') as f:
    # Fix metadata group attributes
    if 'metadata' in f:
        metadata_group = f['metadata']

        print(f"\nCurrent metadata attributes:")
        print(f"  num_parameter_sets: {metadata_group.attrs.get('num_parameter_sets')}")

        # Update num_parameter_sets from 100000 to 50000
        metadata_group.attrs['num_parameter_sets'] = 50000

        print(f"\nUpdated metadata attributes:")
        print(f"  num_parameter_sets: {metadata_group.attrs.get('num_parameter_sets')}")
    else:
        print("No metadata group found")

print("\nDone!")
