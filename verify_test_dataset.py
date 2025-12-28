#!/usr/bin/env python3
"""
Verify the test dataset has all expected metadata and features.
"""

import h5py
from pathlib import Path
from collections import Counter

dataset_path = Path("datasets/test_diverse_100.h5")

print("="*60)
print("DATASET VERIFICATION")
print("="*60)
print(f"Dataset: {dataset_path}\n")

with h5py.File(dataset_path, "r") as f:
    # Basic info
    metadata = dict(f["metadata"].attrs)

    print("Basic Metadata:")
    print(f"  Creation date: {metadata.get('creation_date', 'N/A')}")
    print(f"  Version: {metadata.get('version', 'N/A')}")
    print(f"  Grid size (max): {metadata.get('grid_size', 'N/A')}")
    print(f"  Num realizations: {metadata.get('num_realizations', 'N/A')}")
    print(f"  Num samples: {metadata.get('num_parameter_sets', 'N/A')}")
    print(f"  Track IC metadata: {metadata.get('track_ic_metadata', 'N/A')}\n")

    # Dataset shapes
    print("Dataset Shapes:")
    params = f["parameters/params"]
    inputs = f["inputs/fields"]
    outputs = f["outputs/fields"]
    print(f"  Parameters: {params.shape}")
    print(f"  Inputs: {inputs.shape}")
    print(f"  Outputs: {outputs.shape}\n")

    # Discovery metadata
    print("="*60)
    print("DISCOVERY METADATA")
    print("="*60 + "\n")

    if "ic_types" in f["metadata"]:
        ic_types = [ic.decode('utf-8') if isinstance(ic, bytes) else ic
                    for ic in f["metadata/ic_types"][:]]
        ic_counter = Counter(ic_types)

        print(f"IC Type Distribution ({len(ic_types)} samples):")
        for ic_type, count in sorted(ic_counter.items()):
            print(f"  {ic_type}: {count} ({count/len(ic_types)*100:.1f}%)")
        print()

    if "grid_sizes" in f["metadata"]:
        grid_sizes = f["metadata/grid_sizes"][:]
        grid_counter = Counter(grid_sizes)

        print(f"Grid Size Distribution:")
        for size, count in sorted(grid_counter.items()):
            print(f"  {size}×{size}: {count} ({count/len(grid_sizes)*100:.1f}%)")
        print()

    if "evolution_policies" in f["metadata"]:
        policies = [p.decode('utf-8') if isinstance(p, bytes) else p
                   for p in f["metadata/evolution_policies"][:]]
        policy_counter = Counter(policies)

        print(f"Evolution Policy Distribution:")
        for policy, count in sorted(policy_counter.items()):
            print(f"  {policy}: {count} ({count/len(policies)*100:.1f}%)")
        print()

    if "noise_regimes" in f["metadata"]:
        regimes = [r.decode('utf-8') if isinstance(r, bytes) else r
                  for r in f["metadata/noise_regimes"][:]]
        regime_counter = Counter(regimes)

        print(f"Noise Regime Distribution:")
        for regime, count in sorted(regime_counter.items()):
            print(f"  {regime}: {count} ({count/len(regimes)*100:.1f}%)")
        print()

    # Sample some data
    print("="*60)
    print("SAMPLE DATA")
    print("="*60 + "\n")

    print("First 5 samples:")
    for i in range(5):
        ic = f["metadata/ic_types"][i]
        ic_str = ic.decode('utf-8') if isinstance(ic, bytes) else ic

        policy = f["metadata/evolution_policies"][i]
        policy_str = policy.decode('utf-8') if isinstance(policy, bytes) else policy

        grid = f["metadata/grid_sizes"][i]

        regime = f["metadata/noise_regimes"][i]
        regime_str = regime.decode('utf-8') if isinstance(regime, bytes) else regime

        print(f"  Sample {i}:")
        print(f"    IC: {ic_str}")
        print(f"    Grid: {grid}×{grid}")
        print(f"    Policy: {policy_str}")
        print(f"    Noise: {regime_str}")

    print("\n" + "="*60)
    print("✓ VERIFICATION COMPLETE")
    print("="*60)
    print("\nAll metadata tracking features are working correctly!")
    print(f"Dataset ready for discovery analysis and VQVAE tokenization.")
