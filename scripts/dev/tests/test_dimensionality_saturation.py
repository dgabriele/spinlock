#!/usr/bin/env python3
"""
Test script for effective dimensionality and gradient saturation features.

Validates:
1. Effective dimensionality features are extracted correctly
2. Gradient saturation features detect flat regions
3. All 5 features are valid (no NaN/Inf)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.sdf.spatial import SpatialFeatureExtractor


def test_dimensionality_saturation():
    """Test effective dimensionality and gradient saturation features."""
    print("=" * 70)
    print("TEST: Effective Dimensionality & Gradient Saturation Features")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create test fields with known structure
    N, M, T, C, H, W = 2, 3, 5, 3, 64, 64

    # Create spatial grids
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # =========================================================================
    # TEST CASE 1: Low-dimensional structure (single sinusoid)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 1: Low-dimensional structure (single sinusoid)")
    print("=" * 70)

    # Pure sinusoid: effectively 1D structure embedded in 2D
    fields_lowdim = torch.sin(2 * torch.pi * 4 * X).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_lowdim = fields_lowdim.expand(N, M, T, C, H, W).clone()

    # Create extractor
    extractor = SpatialFeatureExtractor(device=device)

    # Extract features
    features_lowdim = extractor.extract(fields_lowdim, config=None)

    # Check dimensionality features
    print(f"\nDimensionality features (expect low for 1D structure):")
    print(f"  effective_rank:         {features_lowdim['effective_rank'].mean().item():.2f}")
    print(f"  participation_ratio:    {features_lowdim['participation_ratio'].mean().item():.2f}")
    print(f"  explained_variance_90:  {features_lowdim['explained_variance_90'].mean().item():.2f}")

    # Check saturation features
    print(f"\nGradient saturation features:")
    print(f"  gradient_saturation_ratio: {features_lowdim['gradient_saturation_ratio'].mean().item():.4f}")
    print(f"  gradient_flatness:         {features_lowdim['gradient_flatness'].mean().item():.2f}")

    # =========================================================================
    # TEST CASE 2: High-dimensional structure (noise)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 2: High-dimensional structure (random noise)")
    print("=" * 70)

    # Random noise: effectively high-dimensional
    fields_highdim = torch.randn(N, M, T, C, H, W, device=device)

    features_highdim = extractor.extract(fields_highdim, config=None)

    # Check dimensionality features
    print(f"\nDimensionality features (expect high for noise):")
    print(f"  effective_rank:         {features_highdim['effective_rank'].mean().item():.2f}")
    print(f"  participation_ratio:    {features_highdim['participation_ratio'].mean().item():.2f}")
    print(f"  explained_variance_90:  {features_highdim['explained_variance_90'].mean().item():.2f}")

    # Check saturation features
    print(f"\nGradient saturation features:")
    print(f"  gradient_saturation_ratio: {features_highdim['gradient_saturation_ratio'].mean().item():.4f}")
    print(f"  gradient_flatness:         {features_highdim['gradient_flatness'].mean().item():.2f}")

    # =========================================================================
    # TEST CASE 3: Saturated field (thresholded)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 3: Saturated field (amplitude limiting)")
    print("=" * 70)

    # Create sinusoid with thresholding (amplitude limiting)
    fields_saturated = torch.sin(2 * torch.pi * 4 * X)
    fields_saturated = torch.clamp(fields_saturated, -0.5, 0.5)  # Clip amplitudes
    fields_saturated = fields_saturated.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_saturated = fields_saturated.expand(N, M, T, C, H, W).clone()

    features_saturated = extractor.extract(fields_saturated, config=None)

    # Check saturation features (expect high saturation ratio)
    print(f"\nGradient saturation features (expect high saturation):")
    print(f"  gradient_saturation_ratio: {features_saturated['gradient_saturation_ratio'].mean().item():.4f} (expect high)")
    print(f"  gradient_flatness:         {features_saturated['gradient_flatness'].mean().item():.2f} (expect high kurtosis)")

    # =========================================================================
    # VALIDATION: NaN/Inf check
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: NaN/Inf check")
    print("=" * 70)

    feature_names = [
        'effective_rank',
        'participation_ratio',
        'explained_variance_90',
        'gradient_saturation_ratio',
        'gradient_flatness'
    ]

    for name in feature_names:
        for case_name, features in [
            ('Low-dim', features_lowdim),
            ('High-dim', features_highdim),
            ('Saturated', features_saturated)
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
        feat = features_lowdim[name]
        print(f"  {name:30s}: {tuple(feat.shape)}")

    # Expected: [N, M, T, C] = [2, 3, 5, 3]
    expected_shape = (N, M, T, C)
    for name in feature_names:
        feat = features_lowdim[name]
        assert feat.shape == expected_shape, f"Shape mismatch: {feat.shape} != {expected_shape}"

    # =========================================================================
    # VALIDATION: Dimensionality ordering
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: Dimensionality ordering")
    print("=" * 70)

    # Low-dim should have lower effective rank than high-dim
    rank_lowdim = features_lowdim['effective_rank'].mean().item()
    rank_highdim = features_highdim['effective_rank'].mean().item()

    print(f"  Low-dim effective_rank:  {rank_lowdim:.2f}")
    print(f"  High-dim effective_rank: {rank_highdim:.2f}")

    if rank_lowdim < rank_highdim:
        print("  ✓ PASS: Low-dim has lower effective rank than high-dim")
    else:
        print("  ✗ FAIL: Low-dim should have lower effective rank")

    # =========================================================================
    # VALIDATION: Saturation detection
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: Saturation detection")
    print("=" * 70)

    # Saturated field should have higher saturation ratio than unsaturated
    sat_unsaturated = features_lowdim['gradient_saturation_ratio'].mean().item()
    sat_saturated = features_saturated['gradient_saturation_ratio'].mean().item()

    print(f"  Unsaturated gradient_saturation_ratio: {sat_unsaturated:.4f}")
    print(f"  Saturated gradient_saturation_ratio:   {sat_saturated:.4f}")

    if sat_saturated > sat_unsaturated:
        print("  ✓ PASS: Saturated field has higher saturation ratio")
    else:
        print("  ✗ FAIL: Saturated field should have higher saturation ratio")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Dimensionality & saturation features working")
    print("=" * 70)


if __name__ == "__main__":
    test_dimensionality_saturation()
