#!/usr/bin/env python3
"""
Validate Tier 4 research frontiers initial condition generators.

Tests all 6 Tier 4 IC types to ensure they generate correctly:
- coherent_state
- relativistic_wave_packet
- mutual_information
- regulatory_network
- dla_cluster
- error_correcting_code
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.dataset.generators import InputFieldGenerator


def validate_ic_type(
    generator: InputFieldGenerator,
    ic_type: str,
    expected_shape: tuple,
    **kwargs
):
    """Validate a single IC type."""
    print(f"\n{'='*60}")
    print(f"Testing IC type: {ic_type}")
    print(f"{'='*60}")

    try:
        # Generate batch
        batch = generator.generate_batch(
            batch_size=4,
            field_type=ic_type,
            **kwargs
        )

        # Check shape
        assert batch.shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {batch.shape}"

        # Check dtype
        assert batch.dtype == torch.float32, \
            f"Dtype mismatch: expected float32, got {batch.dtype}"

        # Check device
        assert batch.device.type == "cuda", \
            f"Device mismatch: expected cuda, got {batch.device.type}"

        # Check for NaN/Inf
        assert not torch.isnan(batch).any(), "Found NaN values"
        assert not torch.isinf(batch).any(), "Found Inf values"

        # Basic statistics
        print(f"✓ Shape: {batch.shape}")
        print(f"✓ Dtype: {batch.dtype}")
        print(f"✓ Device: {batch.device}")
        print(f"✓ Range: [{batch.min():.3f}, {batch.max():.3f}]")
        print(f"✓ Mean: {batch.mean():.3f}")
        print(f"✓ Std: {batch.std():.3f}")
        print(f"✓ PASS: {ic_type}")

        return True

    except Exception as e:
        print(f"✗ FAIL: {ic_type}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run validation for all Tier 4 IC types."""
    print("="*60)
    print("TIER 4 RESEARCH FRONTIERS IC VALIDATION")
    print("="*60)

    # Initialize generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    generator = InputFieldGenerator(
        grid_size=64,
        num_channels=3,
        device=device
    )

    results = {}

    # Test 1: Coherent State
    results["coherent_state"] = validate_ic_type(
        generator,
        "coherent_state",
        expected_shape=(4, 3, 64, 64),
        alpha=1.0,
        sigma=10.0,
        oscillation_frequency=0.2,
        num_modes=1
    )

    # Test 2: Relativistic Wave Packet
    results["relativistic_wave_packet"] = validate_ic_type(
        generator,
        "relativistic_wave_packet",
        expected_shape=(4, 3, 64, 64),
        velocity=0.5,
        sigma=10.0,
        wavenumber_range=(0.1, 0.5)
    )

    # Test 3: Mutual Information
    results["mutual_information"] = validate_ic_type(
        generator,
        "mutual_information",
        expected_shape=(4, 3, 64, 64),
        correlation_pattern="local",
        latent_length_scale=0.3,
        local_noise_scale=0.5,
        num_channels_correlated=2
    )

    # Test 4: Regulatory Network
    results["regulatory_network"] = validate_ic_type(
        generator,
        "regulatory_network",
        expected_shape=(4, 3, 64, 64),
        network_topology="scale_free",
        network_size=5,
        connection_probability=0.3,
        spatial_decay=0.2
    )

    # Test 5: DLA Cluster (reduced cluster size for testing speed)
    print("\n⚠ WARNING: DLA cluster generation is computationally expensive, using small cluster size for testing")
    results["dla_cluster"] = validate_ic_type(
        generator,
        "dla_cluster",
        expected_shape=(4, 3, 64, 64),
        cluster_size=50,  # Small for testing
        seed_type="point",
        stickiness=1.0,
        field_type_dla="distance_transform"
    )

    # Test 6: Error-Correcting Code
    results["error_correcting_code"] = validate_ic_type(
        generator,
        "error_correcting_code",
        expected_shape=(4, 3, 64, 64),
        code_type="repetition",
        code_rate=0.5,
        block_size=8,
        error_rate=0.1
    )

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for ic_type, pass_status in results.items():
        status = "✓ PASS" if pass_status else "✗ FAIL"
        print(f"{status}: {ic_type}")

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\n✗ VALIDATION FAILED")
        sys.exit(1)
    else:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
