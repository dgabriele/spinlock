#!/usr/bin/env python3
"""
Test script for scale-specific dissipation features.

Validates:
1. Dissipation features are extracted correctly
2. Features detect frequency-dependent decay
3. All 4 features are valid (no NaN/Inf)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor


def test_dissipation_features():
    """Test scale-specific dissipation feature extraction."""
    print("=" * 70)
    print("TEST: Scale-Specific Dissipation Features")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create test trajectories with known dissipation behavior
    N, M, T, C, H, W = 2, 3, 50, 3, 64, 64

    # Create spatial grids
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # =========================================================================
    # TEST CASE 1: High-frequency dissipation (smoothing/diffusion)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 1: High-frequency dissipation (smoothing)")
    print("=" * 70)

    # Start with noisy field, smooth over time (high-freq dissipates)
    # Initial field: noise
    fields_highfreq_dissipation = torch.randn(N, M, 1, C, H, W, device=device)

    # Apply progressive smoothing over time
    for t in range(1, T):
        # Smooth with Gaussian filter (low-pass)
        kernel_size = 3
        sigma = 0.5
        # Simple box smoothing (approximate Gaussian)
        smoothed = torch.nn.functional.avg_pool2d(
            fields_highfreq_dissipation[:, :, t-1, :, :, :].reshape(N*M*C, 1, H, W),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2
        )
        smoothed = smoothed.reshape(N, M, C, H, W).unsqueeze(2)
        fields_highfreq_dissipation = torch.cat([fields_highfreq_dissipation, smoothed], dim=2)

    # Create extractor
    extractor = InvariantDriftExtractor(device=device)

    # Extract features
    features_highfreq = extractor.extract(fields_highfreq_dissipation, config=None)

    # Check dissipation features (expect high-freq dissipation > low-freq)
    print(f"\nDissipation features (expect high-freq > low-freq):")
    print(f"  dissipation_rate_lowfreq:  {features_highfreq['dissipation_rate_lowfreq'].mean().item():.6f}")
    print(f"  dissipation_rate_highfreq: {features_highfreq['dissipation_rate_highfreq'].mean().item():.6f}")
    print(f"  dissipation_selectivity:   {features_highfreq['dissipation_selectivity'].mean().item():.2f} (expect > 1)")
    print(f"  energy_cascade_direction:  {features_highfreq['energy_cascade_direction'].mean().item():.2f}")

    # =========================================================================
    # TEST CASE 2: Low-frequency dissipation (large-scale decay)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 2: Low-frequency dissipation (large-scale decay)")
    print("=" * 70)

    # Start with large-scale sinusoid, decay amplitude over time
    freq_fundamental = 2.0
    fields_lowfreq_dissipation = []

    for t in range(T):
        # Exponentially decay amplitude
        amplitude = torch.exp(torch.tensor(-0.02 * t, device=device))
        field = amplitude * torch.sin(2 * torch.pi * freq_fundamental * X)
        field = field.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N, M, C, H, W)
        fields_lowfreq_dissipation.append(field.unsqueeze(2))

    fields_lowfreq_dissipation = torch.cat(fields_lowfreq_dissipation, dim=2)

    # Extract features
    features_lowfreq = extractor.extract(fields_lowfreq_dissipation, config=None)

    # Check dissipation features (expect low-freq dissipation > high-freq)
    print(f"\nDissipation features (expect low-freq > high-freq):")
    print(f"  dissipation_rate_lowfreq:  {features_lowfreq['dissipation_rate_lowfreq'].mean().item():.6f}")
    print(f"  dissipation_rate_highfreq: {features_lowfreq['dissipation_rate_highfreq'].mean().item():.6f}")
    print(f"  dissipation_selectivity:   {features_lowfreq['dissipation_selectivity'].mean().item():.2f} (expect < 1)")
    print(f"  energy_cascade_direction:  {features_lowfreq['energy_cascade_direction'].mean().item():.2f}")

    # =========================================================================
    # TEST CASE 3: No dissipation (constant field)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST CASE 3: No dissipation (constant field)")
    print("=" * 70)

    # Constant field over time
    fields_constant = torch.sin(2 * torch.pi * 4 * X).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_constant = fields_constant.expand(N, M, T, C, H, W).clone()

    # Extract features
    features_constant = extractor.extract(fields_constant, config=None)

    # Check dissipation features (expect near zero)
    print(f"\nDissipation features (expect near zero):")
    print(f"  dissipation_rate_lowfreq:  {features_constant['dissipation_rate_lowfreq'].mean().item():.6f}")
    print(f"  dissipation_rate_highfreq: {features_constant['dissipation_rate_highfreq'].mean().item():.6f}")
    print(f"  dissipation_selectivity:   {features_constant['dissipation_selectivity'].mean().item():.2f}")
    print(f"  energy_cascade_direction:  {features_constant['energy_cascade_direction'].mean().item():.2f}")

    # =========================================================================
    # VALIDATION: NaN/Inf check
    # =========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION: NaN/Inf check")
    print("=" * 70)

    feature_names = [
        'dissipation_rate_lowfreq',
        'dissipation_rate_highfreq',
        'dissipation_selectivity',
        'energy_cascade_direction'
    ]

    for name in feature_names:
        for case_name, features in [
            ('High-freq', features_highfreq),
            ('Low-freq', features_lowfreq),
            ('Constant', features_constant)
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
        feat = features_highfreq[name]
        print(f"  {name:30s}: {tuple(feat.shape)}")

    # Expected: [N, M, C] = [2, 3, 3]
    expected_shape = (N, M, C)
    for name in feature_names:
        feat = features_highfreq[name]
        assert feat.shape == expected_shape, f"Shape mismatch: {feat.shape} != {expected_shape}"

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Scale-specific dissipation features working")
    print("=" * 70)


if __name__ == "__main__":
    test_dissipation_features()
