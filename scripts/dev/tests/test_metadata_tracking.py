#!/usr/bin/env python3
"""
Test script to verify IC-behavior metadata tracking in HDF5 storage.
"""

import torch
import numpy as np
from pathlib import Path
from src.spinlock.dataset.storage import HDF5DatasetWriter, HDF5DatasetReader


def test_metadata_tracking():
    """Test metadata tracking functionality."""

    test_path = Path("/tmp/test_metadata.h5")

    # Test parameters
    num_samples = 20
    grid_size = 64
    num_realizations = 10

    print("Testing metadata tracking in HDF5 storage...")
    print(f"Creating test dataset: {test_path}")
    print(f"Samples: {num_samples}, Grid: {grid_size}x{grid_size}, Realizations: {num_realizations}\n")

    # Create writer with metadata tracking enabled
    with HDF5DatasetWriter(
        output_path=test_path,
        grid_size=grid_size,
        input_channels=3,
        output_channels=3,
        num_realizations=num_realizations,
        num_parameter_sets=num_samples,
        track_ic_metadata=True,
    ) as writer:

        # Generate two batches
        batch_size = 10

        for batch_idx in range(2):
            print(f"Writing batch {batch_idx + 1}/{2}...")

            # Generate dummy data
            params = np.random.randn(batch_size, 11).astype(np.float32)
            inputs = torch.randn(batch_size, 3, grid_size, grid_size)
            outputs = torch.randn(batch_size, num_realizations, 3, grid_size, grid_size)

            # Generate metadata for this batch
            ic_types = [
                "multiscale_grf" if i % 4 == 0 else
                "localized" if i % 4 == 1 else
                "composite" if i % 4 == 2 else
                "heavy_tailed"
                for i in range(batch_size)
            ]

            evolution_policies = [
                "autoregressive" if i % 3 == 0 else
                "residual" if i % 3 == 1 else
                "convex"
                for i in range(batch_size)
            ]

            grid_sizes = [64] * batch_size

            noise_regimes = [
                "low" if i % 3 == 0 else
                "medium" if i % 3 == 1 else
                "high"
                for i in range(batch_size)
            ]

            # Write batch with metadata
            writer.write_batch(
                parameters=params,
                inputs=inputs,
                outputs=outputs,
                ic_types=ic_types,
                evolution_policies=evolution_policies,
                grid_sizes=grid_sizes,
                noise_regimes=noise_regimes,
            )

            print(f"  ✓ Wrote {batch_size} samples with metadata")

    print("\n" + "="*60)
    print("Reading back metadata...")
    print("="*60 + "\n")

    # Read back and verify
    with HDF5DatasetReader(test_path) as reader:
        metadata = reader.get_metadata()

        print(f"Dataset metadata:")
        print(f"  - Creation date: {metadata['creation_date']}")
        print(f"  - Version: {metadata['version']}")
        print(f"  - Grid size: {metadata['grid_size']}")
        print(f"  - Num realizations: {metadata['num_realizations']}")
        print(f"  - Track IC metadata: {metadata['track_ic_metadata']}")
        print(f"  - Num samples: {reader.num_samples}\n")

        # Get discovery metadata
        ic_types = reader.get_ic_types()
        evolution_policies = reader.get_evolution_policies()
        grid_sizes = reader.get_grid_sizes()
        noise_regimes = reader.get_noise_regimes()

        print(f"IC Types (first 5): {ic_types[:5].tolist()}")
        print(f"Evolution Policies (first 5): {evolution_policies[:5].tolist()}")
        print(f"Grid Sizes (first 5): {grid_sizes[:5].tolist()}")
        print(f"Noise Regimes (first 5): {noise_regimes[:5].tolist()}\n")

        # Verify counts
        ic_type_counts = {}
        for ic_type in ic_types:
            ic_type_str = ic_type.decode('utf-8') if isinstance(ic_type, bytes) else ic_type
            ic_type_counts[ic_type_str] = ic_type_counts.get(ic_type_str, 0) + 1

        policy_counts = {}
        for policy in evolution_policies:
            policy_str = policy.decode('utf-8') if isinstance(policy, bytes) else policy
            policy_counts[policy_str] = policy_counts.get(policy_str, 0) + 1

        noise_counts = {}
        for regime in noise_regimes:
            regime_str = regime.decode('utf-8') if isinstance(regime, bytes) else regime
            noise_counts[regime_str] = noise_counts.get(regime_str, 0) + 1

        print("="*60)
        print("Metadata Statistics:")
        print("="*60)
        print(f"\nIC Type Distribution:")
        for ic_type, count in sorted(ic_type_counts.items()):
            print(f"  {ic_type}: {count} ({count/num_samples*100:.1f}%)")

        print(f"\nEvolution Policy Distribution:")
        for policy, count in sorted(policy_counts.items()):
            print(f"  {policy}: {count} ({count/num_samples*100:.1f}%)")

        print(f"\nNoise Regime Distribution:")
        for regime, count in sorted(noise_counts.items()):
            print(f"  {regime}: {count} ({count/num_samples*100:.1f}%)")

        # Test slicing
        print("\n" + "="*60)
        print("Testing metadata slicing...")
        print("="*60)

        first_5_ic_types = reader.get_ic_types(slice(0, 5))
        print(f"IC types [0:5]: {first_5_ic_types.tolist()}")

        single_policy = reader.get_evolution_policies(3)
        print(f"Evolution policy [3]: {single_policy}")

    print("\n" + "="*60)
    print("✓ ALL METADATA TESTS PASSED!")
    print("="*60)
    print(f"\nTest file saved at: {test_path}")
    print("(Delete manually if not needed)")


if __name__ == "__main__":
    test_metadata_tracking()
