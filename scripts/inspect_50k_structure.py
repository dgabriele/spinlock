#!/usr/bin/env python3
"""Inspect 50k_baseline.h5 structure."""

import h5py
import json
from pathlib import Path

dataset_path = Path("datasets/50k_baseline.h5")

print(f"Inspecting {dataset_path}\n")

with h5py.File(dataset_path, 'r') as f:
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            # Check for attributes
            if obj.attrs:
                for key, value in obj.attrs.items():
                    if isinstance(value, (str, bytes)):
                        try:
                            parsed = json.loads(value)
                            print(f"{indent}  @{key}: {type(parsed).__name__}")
                            if isinstance(parsed, dict) and 'n_samples' in parsed:
                                print(f"{indent}    n_samples: {parsed['n_samples']}")
                        except:
                            print(f"{indent}  @{key}: {value[:100] if len(str(value)) > 100 else value}")
                    else:
                        print(f"{indent}  @{key}: {value}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset): shape={obj.shape}, dtype={obj.dtype}")

    f.visititems(print_structure)
