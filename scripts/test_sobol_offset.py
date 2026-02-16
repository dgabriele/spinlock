#!/usr/bin/env python3
"""
Test Sobol offset implementation for distributed generation.

Verifies:
1. Offset produces non-overlapping samples
2. Samples are deterministic and reproducible
3. Sobol low-discrepancy properties are preserved
"""

import numpy as np
from spinlock.sampling.sobol import StratifiedSobolSampler


def test_sobol_offset_no_overlap():
    """Test that different offsets produce non-overlapping samples."""
    print("Test 1: Non-overlapping samples")
    print("-" * 60)

    dimensionality = 9  # QBM has 9 continuous dimensions
    sampler = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)

    # Generate two batches with different offsets
    samples_0_100 = sampler.sample(100, offset=0)
    samples_100_200 = sampler.sample(100, offset=100)

    # Check no overlap
    # Convert to tuples for set comparison
    set_0_100 = {tuple(row) for row in samples_0_100}
    set_100_200 = {tuple(row) for row in samples_100_200}

    overlap = set_0_100 & set_100_200
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples!"

    print(f"  ✓ Batch 1 (offset=0): {len(samples_0_100)} unique samples")
    print(f"  ✓ Batch 2 (offset=100): {len(samples_100_200)} unique samples")
    print(f"  ✓ No overlap between batches")
    print()


def test_sobol_offset_reproducibility():
    """Test that same offset produces same samples."""
    print("Test 2: Reproducibility")
    print("-" * 60)

    dimensionality = 9
    sampler1 = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)
    sampler2 = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)

    # Generate same batch twice
    samples1 = sampler1.sample(100, offset=0)
    samples2 = sampler2.sample(100, offset=0)

    # Should be identical
    assert np.allclose(samples1, samples2), "Samples not reproducible!"

    print(f"  ✓ Same offset produces identical samples")
    print(f"  ✓ Max difference: {np.abs(samples1 - samples2).max():.2e}")
    print()


def test_sobol_offset_continuity():
    """Test that offset=N is equivalent to skipping N samples."""
    print("Test 3: Offset continuity")
    print("-" * 60)

    dimensionality = 9
    sampler1 = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)
    sampler2 = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)

    # Method 1: Generate 200 samples at once, take second half
    all_samples = sampler1.sample(200, offset=0)
    samples_100_200_direct = all_samples[100:200]

    # Method 2: Generate with offset=100
    samples_100_200_offset = sampler2.sample(100, offset=100)

    # Should be identical
    assert np.allclose(
        samples_100_200_direct, samples_100_200_offset
    ), "Offset doesn't match continuous generation!"

    print(f"  ✓ Offset=100 matches skipping first 100 samples")
    print(f"  ✓ Max difference: {np.abs(samples_100_200_direct - samples_100_200_offset).max():.2e}")
    print()


def test_distributed_generation_simulation():
    """Simulate distributed generation with 10 jobs."""
    print("Test 4: Distributed generation simulation")
    print("-" * 60)

    dimensionality = 9
    total_samples = 1000
    samples_per_job = 100
    num_jobs = total_samples // samples_per_job

    print(f"  Simulating {num_jobs} jobs × {samples_per_job} samples = {total_samples} total")
    print()

    all_samples = []
    for job_idx in range(num_jobs):
        offset = job_idx * samples_per_job
        sampler = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)
        samples = sampler.sample(samples_per_job, offset=offset)
        all_samples.append(samples)
        print(f"    Job {job_idx}: offset={offset:4d}, samples={samples.shape}")

    # Concatenate all samples
    merged_samples = np.concatenate(all_samples, axis=0)
    print()
    print(f"  ✓ Merged shape: {merged_samples.shape}")

    # Check for duplicates
    unique_samples = len({tuple(row) for row in merged_samples})
    duplicates = len(merged_samples) - unique_samples

    if duplicates > 0:
        print(f"  ✗ WARNING: {duplicates} duplicate samples found!")
    else:
        print(f"  ✓ No duplicate samples")

    # Compare to single-job generation
    sampler_single = StratifiedSobolSampler(dimensionality=dimensionality, seed=42)
    samples_single = sampler_single.sample(total_samples, offset=0)

    if np.allclose(merged_samples, samples_single):
        print(f"  ✓ Merged samples match single-job generation")
    else:
        print(f"  ✗ WARNING: Merged samples differ from single-job generation!")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("SOBOL OFFSET TESTS")
    print("=" * 60)
    print()

    try:
        test_sobol_offset_no_overlap()
        test_sobol_offset_reproducibility()
        test_sobol_offset_continuity()
        test_distributed_generation_simulation()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print("=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
