#!/usr/bin/env python3
"""
Validate Tier 1 domain-specific initial condition generators.

Tests all 5 Tier 1 IC types to ensure they generate correctly:
- quantum_wave_packet
- turing_pattern
- thermal_gradient
- morphogen_gradient
- reaction_front
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
    """Run validation for all Tier 1 IC types."""
    print("="*60)
    print("TIER 1 DOMAIN-SPECIFIC IC VALIDATION")
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

    # Test 1: Quantum Wave Packet
    results["quantum_wave_packet"] = validate_ic_type(
        generator,
        "quantum_wave_packet",
        expected_shape=(4, 3, 64, 64),
        sigma=10.0,
        momentum_range=(0.1, 0.5),
        num_packets=1
    )

    # Test 2: Turing Pattern
    results["turing_pattern"] = validate_ic_type(
        generator,
        "turing_pattern",
        expected_shape=(4, 3, 64, 64),
        pattern_type="spots",
        wavelength=16.0,
        perturbation_amplitude=0.1
    )

    # Test 3: Thermal Gradient
    results["thermal_gradient"] = validate_ic_type(
        generator,
        "thermal_gradient",
        expected_shape=(4, 3, 64, 64),
        gradient_direction="x",
        temperature_range=(0.0, 1.0),
        beta=1.0,
        thermal_noise_amplitude=0.1,
        thermal_length_scale=5.0
    )

    # Test 4: Morphogen Gradient
    results["morphogen_gradient"] = validate_ic_type(
        generator,
        "morphogen_gradient",
        expected_shape=(4, 3, 64, 64),
        decay_length=20.0,
        num_sources=1,
        noise_level=0.05,
        gradient_type="exponential"
    )

    # Test 5: Reaction Front
    results["reaction_front"] = validate_ic_type(
        generator,
        "reaction_front",
        expected_shape=(4, 3, 64, 64),
        front_shape="planar",
        front_width=3.0,
        num_fronts=1
    )

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    for ic_type, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
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
