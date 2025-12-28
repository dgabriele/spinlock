#!/usr/bin/env python3
"""
Validate Tier 2 domain-specific initial condition generators.

Tests all 5 Tier 2 IC types to ensure they generate correctly:
- light_cone
- critical_fluctuation
- phase_boundary
- bz_reaction
- shannon_entropy
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
    """Run validation for all Tier 2 IC types."""
    print("="*60)
    print("TIER 2 DOMAIN-SPECIFIC IC VALIDATION")
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

    # Test 1: Light Cone
    results["light_cone"] = validate_ic_type(
        generator,
        "light_cone",
        expected_shape=(4, 3, 64, 64),
        cone_radius=20.0,
        smoothing=2.0,
        interior_length_scale=5.0,
        num_cones=1
    )

    # Test 2: Critical Fluctuation
    results["critical_fluctuation"] = validate_ic_type(
        generator,
        "critical_fluctuation",
        expected_shape=(4, 3, 64, 64),
        correlation_length=15.0,
        eta=0.04
    )

    # Test 3: Phase Boundary
    results["phase_boundary"] = validate_ic_type(
        generator,
        "phase_boundary",
        expected_shape=(4, 3, 64, 64),
        interface_width=3.0,
        fluctuation_amplitude=0.1
    )

    # Test 4: BZ Reaction
    results["bz_reaction"] = validate_ic_type(
        generator,
        "bz_reaction",
        expected_shape=(4, 3, 64, 64),
        pattern_type="spiral",
        num_spirals=1,
        wavelength=16.0,
        phase_offset=0.0
    )

    # Test 5: Shannon Entropy
    results["shannon_entropy"] = validate_ic_type(
        generator,
        "shannon_entropy",
        expected_shape=(4, 3, 64, 64),
        entropy_pattern="gradient",
        entropy_range=(0.1, 1.0),
        patch_size=8
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
