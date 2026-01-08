#!/usr/bin/env python3
"""
Validate Tier 3 advanced domain-specific initial condition generators.

Tests all 5 Tier 3 IC types to ensure they generate correctly:
- interference_pattern
- cell_population
- chromatin_domain
- shock_front
- gene_expression
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
    """Run validation for all Tier 3 IC types."""
    print("="*60)
    print("TIER 3 ADVANCED DOMAIN-SPECIFIC IC VALIDATION")
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

    # Test 1: Interference Pattern
    results["interference_pattern"] = validate_ic_type(
        generator,
        "interference_pattern",
        expected_shape=(4, 3, 64, 64),
        num_sources=2,
        wavelength=16.0,
        coherence_length=100.0,
        source_spacing=40.0
    )

    # Test 2: Cell Population
    results["cell_population"] = validate_ic_type(
        generator,
        "cell_population",
        expected_shape=(4, 3, 64, 64),
        num_cells=50,
        cell_radius=3.0,
        clustering_strength=0.5,
        kernel_width=5.0
    )

    # Test 3: Chromatin Domain
    results["chromatin_domain"] = validate_ic_type(
        generator,
        "chromatin_domain",
        expected_shape=(4, 3, 64, 64),
        num_domains=5,
        domain_size_range=(10, 30),
        boundary_sharpness=3.0,
        compartment_type="TAD"
    )

    # Test 4: Shock Front
    results["shock_front"] = validate_ic_type(
        generator,
        "shock_front",
        expected_shape=(4, 3, 64, 64),
        shock_position=None,
        shock_angle=None,
        shock_width=2.0,
        amplitude_ratio=2.0,
        fluctuation_amplitude=0.05
    )

    # Test 5: Gene Expression
    results["gene_expression"] = validate_ic_type(
        generator,
        "gene_expression",
        expected_shape=(4, 3, 64, 64),
        num_genes=3,
        expression_patterns=["gradient", "spots", "stripes"],
        correlation_strength=0.3,
        expression_range=(0.0, 1.0)
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
