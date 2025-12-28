#!/usr/bin/env python3
"""
Quick test script to verify diverse IC types in InputFieldGenerator.
"""

import torch
from src.spinlock.dataset.generators import InputFieldGenerator


def test_ic_types():
    """Test all new IC types for correct shape and basic properties."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = InputFieldGenerator(
        grid_size=64,
        num_channels=3,
        device=device
    )

    batch_size = 4

    print("Testing diverse IC types...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Expected shape: [{batch_size}, 3, 64, 64]\n")

    # Test 1: Multi-scale GRF
    print("1. Testing multiscale_grf...")
    fields_ms = generator.generate_batch(
        batch_size=batch_size,
        field_type="multiscale_grf"
    )
    print(f"   Shape: {fields_ms.shape}")
    print(f"   Mean: {fields_ms.mean().item():.4f}")
    print(f"   Std: {fields_ms.std().item():.4f}")
    print(f"   Min/Max: [{fields_ms.min().item():.4f}, {fields_ms.max().item():.4f}]")
    assert fields_ms.shape == (batch_size, 3, 64, 64), "Shape mismatch!"
    print("   ✓ PASSED\n")

    # Test 2: Localized features
    print("2. Testing localized...")
    fields_loc = generator.generate_batch(
        batch_size=batch_size,
        field_type="localized",
        num_blobs=5
    )
    print(f"   Shape: {fields_loc.shape}")
    print(f"   Mean: {fields_loc.mean().item():.4f}")
    print(f"   Std: {fields_loc.std().item():.4f}")
    print(f"   Min/Max: [{fields_loc.min().item():.4f}, {fields_loc.max().item():.4f}]")
    assert fields_loc.shape == (batch_size, 3, 64, 64), "Shape mismatch!"
    print("   ✓ PASSED\n")

    # Test 3: Composite fields
    print("3. Testing composite (waves)...")
    fields_comp = generator.generate_batch(
        batch_size=batch_size,
        field_type="composite",
        pattern="waves"
    )
    print(f"   Shape: {fields_comp.shape}")
    print(f"   Mean: {fields_comp.mean().item():.4f}")
    print(f"   Std: {fields_comp.std().item():.4f}")
    print(f"   Min/Max: [{fields_comp.min().item():.4f}, {fields_comp.max().item():.4f}]")
    assert fields_comp.shape == (batch_size, 3, 64, 64), "Shape mismatch!"
    print("   ✓ PASSED\n")

    # Test 4: Heavy-tailed distributions
    print("4. Testing heavy_tailed...")
    fields_ht = generator.generate_batch(
        batch_size=batch_size,
        field_type="heavy_tailed",
        alpha=1.5
    )
    print(f"   Shape: {fields_ht.shape}")
    print(f"   Mean: {fields_ht.mean().item():.4f}")
    print(f"   Std: {fields_ht.std().item():.4f}")
    print(f"   Min/Max: [{fields_ht.min().item():.4f}, {fields_ht.max().item():.4f}]")
    assert fields_ht.shape == (batch_size, 3, 64, 64), "Shape mismatch!"
    print("   ✓ PASSED\n")

    # Test 5: Verify original GRF still works
    print("5. Testing original gaussian_random_field...")
    fields_grf = generator.generate_batch(
        batch_size=batch_size,
        field_type="gaussian_random_field"
    )
    print(f"   Shape: {fields_grf.shape}")
    print(f"   Mean: {fields_grf.mean().item():.4f}")
    print(f"   Std: {fields_grf.std().item():.4f}")
    assert fields_grf.shape == (batch_size, 3, 64, 64), "Shape mismatch!"
    print("   ✓ PASSED\n")

    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("All IC types generate correct tensor shapes.")
    print("="*60)


if __name__ == "__main__":
    test_ic_types()
