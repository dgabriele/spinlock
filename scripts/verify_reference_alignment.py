#!/usr/bin/env python3
"""
Verify that MNO feature dataset aligns with reference dataset.

Usage:
    python scripts/verify_reference_alignment.py \
        --mno-dataset datasets/test_reference_10k.h5 \
        --reference-dataset datasets/local_100k_optimized.h5
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def verify_alignment(mno_dataset_path: Path, reference_dataset_path: Path) -> bool:
    """
    Verify that MNO dataset ICs and parameters match reference dataset.

    Args:
        mno_dataset_path: Path to generated MNO feature dataset
        reference_dataset_path: Path to reference CNO dataset

    Returns:
        True if datasets align, False otherwise
    """
    print("=" * 70)
    print("REFERENCE ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference dataset: {reference_dataset_path}")
    print()

    all_checks_passed = True

    with h5py.File(mno_dataset_path, 'r') as mno_f:
        with h5py.File(reference_dataset_path, 'r') as ref_f:
            # Get number of samples in MNO dataset
            # MNO feature datasets store parameters at /features/parameters
            n_mno_samples = mno_f['features/parameters'].shape[0]
            print(f"MNO dataset samples: {n_mno_samples}")
            print()

            # Check 1: Verify ICs match
            print("Check 1: Initial Conditions (ICs)")
            print("-" * 70)

            # MNO dataset stores ICs with M realizations: [N, M, H, W]
            # We compare the first realization (M=0) with reference first realization
            mno_ics = mno_f['inputs/fields'][:, 0, :, :]  # [N, H, W]
            ref_ics = ref_f['inputs/fields'][:n_mno_samples, 0, :, :]  # [N, H, W]

            if mno_ics.shape != ref_ics.shape:
                print(f"✗ IC shape mismatch!")
                print(f"  MNO shape: {mno_ics.shape}")
                print(f"  Reference shape: {ref_ics.shape}")
                all_checks_passed = False
            elif np.allclose(mno_ics, ref_ics, rtol=1e-5, atol=1e-8):
                print("✓ ICs match reference dataset")

                # Check a few random samples for visual verification
                sample_indices = np.random.choice(n_mno_samples, size=min(5, n_mno_samples), replace=False)
                print(f"  Spot-checked samples {sample_indices.tolist()}")
                for idx in sample_indices:
                    mse = np.mean((mno_ics[idx] - ref_ics[idx]) ** 2)
                    print(f"    Sample {idx}: MSE = {mse:.2e}")
            else:
                print("✗ ICs do NOT match reference dataset")

                # Calculate difference statistics
                diff = mno_ics - ref_ics
                print(f"  Max absolute difference: {np.abs(diff).max():.2e}")
                print(f"  Mean absolute difference: {np.abs(diff).mean():.2e}")
                print(f"  RMS difference: {np.sqrt(np.mean(diff ** 2)):.2e}")

                # Show which samples differ most
                per_sample_mse = np.mean((mno_ics - ref_ics) ** 2, axis=(1, 2))
                worst_indices = np.argsort(per_sample_mse)[-5:]
                print(f"  Worst 5 samples: {worst_indices.tolist()}")
                for idx in worst_indices:
                    print(f"    Sample {idx}: MSE = {per_sample_mse[idx]:.2e}")

                all_checks_passed = False

            print()

            # Check 2: Verify parameters match
            print("Check 2: Operator Parameters")
            print("-" * 70)

            # MNO feature datasets store parameters at /features/parameters
            # Reference datasets store parameters at /parameters/params
            mno_params = mno_f['features/parameters'][:]
            ref_params = ref_f['parameters/params'][:n_mno_samples]

            if mno_params.shape != ref_params.shape:
                print(f"✗ Parameter shape mismatch!")
                print(f"  MNO shape: {mno_params.shape}")
                print(f"  Reference shape: {ref_params.shape}")
                all_checks_passed = False
            elif np.allclose(mno_params, ref_params, rtol=1e-5, atol=1e-8):
                print("✓ Parameters match reference dataset")

                # Check parameter space dimensions
                n_params, param_dim = mno_params.shape
                print(f"  Shape: {n_params} samples × {param_dim} dimensions")

                # Show parameter ranges
                print(f"  Parameter ranges:")
                for dim in range(min(param_dim, 5)):  # Show first 5 dimensions
                    print(f"    Dim {dim}: [{mno_params[:, dim].min():.3f}, {mno_params[:, dim].max():.3f}]")
                if param_dim > 5:
                    print(f"    ... ({param_dim - 5} more dimensions)")
            else:
                print("✗ Parameters do NOT match reference dataset")

                # Calculate difference statistics
                diff = mno_params - ref_params
                print(f"  Max absolute difference: {np.abs(diff).max():.2e}")
                print(f"  Mean absolute difference: {np.abs(diff).mean():.2e}")

                # Show which dimensions differ most
                per_dim_mse = np.mean((mno_params - ref_params) ** 2, axis=0)
                print(f"  Per-dimension MSE: {per_dim_mse}")

                all_checks_passed = False

            print()

            # Check 3: Verify metadata
            print("Check 3: Metadata")
            print("-" * 70)

            if 'metadata/generation_indices' in mno_f:
                gen_indices = mno_f['metadata/generation_indices'][:]
                expected_indices = np.arange(n_mno_samples)

                if np.array_equal(gen_indices, expected_indices):
                    print("✓ Generation indices are sequential [0, 1, 2, ..., N-1]")
                else:
                    print("⚠ Generation indices are not sequential")
                    print(f"  First 10: {gen_indices[:10].tolist()}")
                    print(f"  Last 10: {gen_indices[-10:].tolist()}")
            else:
                print("⚠ No generation_indices metadata found")

            if 'metadata/is_interpolated' in mno_f:
                is_interp = mno_f['metadata/is_interpolated'][:]
                n_interp = is_interp.sum()
                n_exact = (~is_interp).sum()

                print(f"  Interpolation mask:")
                print(f"    Exact reference points: {n_exact}")
                print(f"    Interpolated points: {n_interp}")

                if n_interp == 0:
                    print("  ✓ All samples are exact reference points (expected for reference mode)")
                else:
                    print(f"  ⚠ Contains {n_interp} interpolated points (unexpected for reference mode)")
            else:
                print("⚠ No is_interpolated metadata found")

            print()

    # Final verdict
    print("=" * 70)
    if all_checks_passed:
        print("✓ VERIFICATION PASSED: MNO dataset aligns with reference dataset")
    else:
        print("✗ VERIFICATION FAILED: MNO dataset does NOT align with reference dataset")
    print("=" * 70)

    return all_checks_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify MNO feature dataset alignment with reference dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify 10K test dataset
  python scripts/verify_reference_alignment.py \\
      --mno-dataset datasets/test_reference_10k.h5 \\
      --reference-dataset datasets/local_100k_optimized.h5

  # Verify 100K production dataset
  python scripts/verify_reference_alignment.py \\
      --mno-dataset datasets/mno_features_100k.h5 \\
      --reference-dataset datasets/local_100k_optimized.h5
"""
    )

    parser.add_argument(
        "--mno-dataset",
        type=Path,
        required=True,
        help="Path to generated MNO feature dataset (HDF5)",
    )

    parser.add_argument(
        "--reference-dataset",
        type=Path,
        required=True,
        help="Path to reference CNO dataset (HDF5)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.mno_dataset.exists():
        print(f"Error: MNO dataset not found: {args.mno_dataset}")
        return 1

    if not args.reference_dataset.exists():
        print(f"Error: Reference dataset not found: {args.reference_dataset}")
        return 1

    # Run verification
    success = verify_alignment(args.mno_dataset, args.reference_dataset)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
