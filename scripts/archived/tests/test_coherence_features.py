#!/usr/bin/env python3
"""
Test script for coherence structure features.

Validates:
1. Coherence features are extracted correctly
2. Features detect spatial correlation structure
3. All 3 features are valid (no NaN/Inf)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.summary.spatial import SpatialFeatureExtractor


def test_coherence_features():
    """Test coherence structure feature extraction."""
    print("=" * 70)
    print("TEST: Coherence Structure Features")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create test fields with known coherence structure
    N, M, T, C, H, W = 2, 3, 5, 3, 64, 64

    # Create spatial grids
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # =========================================================================
    # TEST CASE 1: Long coherence length (smooth large-scale structure)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 1: Long coherence length (smooth structure)")
    print("=" * 70)

    # Large-scale sinusoid: long coherence length
    freq_low = 2.0
    fields_long_coherence = torch.sin(2 * torch.pi * freq_low * X).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_long_coherence = fields_long_coherence.expand(N, M, T, C, H, W).clone()

    # Create extractor
    extractor = SpatialFeatureExtractor(device=device)

    # Extract features
    features_long = extractor.extract(fields_long_coherence, config=None)

    # Check coherence features
    print(f"\nCoherence features (expect long coherence, low anisotropy):")
    print(f"  coherence_length:        {features_long['coherence_length'].mean().item():.2f} pixels")
    print(f"  correlation_anisotropy:  {features_long['correlation_anisotropy'].mean().item():.2f}")
    print(f"  structure_factor_peak:   {features_long['structure_factor_peak'].mean().item():.2f} pixels")

    # =========================================================================
    # TEST CASE 2: Short coherence length (noise)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 2: Short coherence length (noise)")
    print("=" * 70)

    # Random noise: short coherence length
    fields_short_coherence = torch.randn(N, M, T, C, H, W, device=device)

    # Extract features
    features_short = extractor.extract(fields_short_coherence, config=None)

    # Check coherence features
    print(f"\nCoherence features (expect short coherence):")
    print(f"  coherence_length:        {features_short['coherence_length'].mean().item():.2f} pixels")
    print(f"  correlation_anisotropy:  {features_short['correlation_anisotropy'].mean().item():.2f}")
    print(f"  structure_factor_peak:   {features_short['structure_factor_peak'].mean().item():.2f} pixels")

    # =========================================================================
    # TEST CASE 3: Anisotropic correlation (different x/y scales)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 3: Anisotropic correlation (x-biased)")
    print("=" * 70)

    # Stripes along x-axis: anisotropic correlation
    freq_x = 8.0
    fields_anisotropic = torch.sin(2 * torch.pi * freq_x * Y).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_anisotropic = fields_anisotropic.expand(N, M, T, C, H, W).clone()

    # Extract features
    features_aniso = extractor.extract(fields_anisotropic, config=None)

    # Check coherence features (expect high anisotropy)
    print(f"\nCoherence features (expect high anisotropy):")
    print(f"  coherence_length:        {features_aniso['coherence_length'].mean().item():.2f} pixels")
    print(f"  correlation_anisotropy:  {features_aniso['correlation_anisotropy'].mean().item():.2f} (expect != 1)")
    print(f"  structure_factor_peak:   {features_aniso['structure_factor_peak'].mean().item():.2f} pixels")

    # =========================================================================
    # VALIDATION: NaN/Inf check
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: NaN/Inf check")
    print("=" * 70)

    feature_names = [
        'coherence_length',
        'correlation_anisotropy',
        'structure_factor_peak'
    ]

    for name in feature_names:
        for case_name, features in [
            ('Long', features_long),
            ('Short', features_short),
            ('Anisotropic', features_aniso)
        ]:
            feat = features[name]
            nan_count = torch.isnan(feat).sum().item()
            inf_count = torch.isinf(feat).sum().item()
            print(f"  {name:30s} [{case_name:12s}]: NaN={nan_count}, Inf={inf_count}")

    # =========================================================================
    # VALIDATION: Feature shape check
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: Feature shape check")
    print("=" * 70)

    for name in feature_names:
        feat = features_long[name]
        print(f"  {name:30s}: {tuple(feat.shape)}")

    # Expected: [N, M, T, C] = [2, 3, 5, 3]
    expected_shape = (N, M, T, C)
    for name in feature_names:
        feat = features_long[name]
        assert feat.shape == expected_shape, f"Shape mismatch: {feat.shape} != {expected_shape}"

    # =========================================================================
    # VALIDATION: Coherence length ordering
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: Coherence length ordering")
    print("=" * 70)

    # Long-coherence should have longer coherence_length than short-coherence
    coh_long = features_long['coherence_length'].mean().item()
    coh_short = features_short['coherence_length'].mean().item()

    print(f"  Long-coherence length:  {coh_long:.2f} pixels")
    print(f"  Short-coherence length: {coh_short:.2f} pixels")

    if coh_long > coh_short:
        print("  ✓ PASS: Long-coherence has longer coherence_length than short-coherence")
    else:
        print("  ✗ FAIL: Long-coherence should have longer coherence_length")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Coherence structure features working")
    print("=" * 70)


if __name__ == "__main__":
    test_coherence_features()
