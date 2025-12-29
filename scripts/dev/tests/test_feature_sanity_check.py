#!/usr/bin/env python3
"""
Sanity check for feature values - verify they're meaningful, not garbage.

Tests that features respond correctly to physical changes:
1. Harmonic content increases with nonlinearity
2. Dimensionality decreases with structure
3. Saturation increases with clipping
4. Dissipation rates reflect actual energy decay
5. Coherence length reflects correlation scales
6. Features are in reasonable value ranges

This validates that features are physically meaningful, not just numerically valid.
"""

import torch
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.sdf.spatial import SpatialFeatureExtractor
from spinlock.features.sdf.spectral import SpectralFeatureExtractor
from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor


def test_harmonic_response_to_nonlinearity():
    """Test that harmonic features correctly detect increasing nonlinearity."""
    print("=" * 70)
    print("SANITY CHECK 1: Harmonic Content Response to Nonlinearity")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create fields with increasing nonlinearity
    H, W = 64, 64
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test cases with increasing nonlinearity
    cases = [
        ("Pure sine (linear)", lambda: torch.sin(2 * torch.pi * 4 * X)),
        ("Weak nonlinear (0.1×2f)", lambda: torch.sin(2 * torch.pi * 4 * X) + 0.1 * torch.sin(2 * torch.pi * 8 * X)),
        ("Medium nonlinear (0.3×2f)", lambda: torch.sin(2 * torch.pi * 4 * X) + 0.3 * torch.sin(2 * torch.pi * 8 * X)),
        ("Strong nonlinear (0.5×2f)", lambda: torch.sin(2 * torch.pi * 4 * X) + 0.5 * torch.sin(2 * torch.pi * 8 * X)),
    ]

    extractor = SpectralFeatureExtractor(device=device)

    thd_values = []
    purity_values = []

    print(f"\n{'Case':<30s} {'THD':>10s} {'Purity':>10s} {'2f Ratio':>10s}")
    print("-" * 70)

    for name, field_fn in cases:
        field = field_fn().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 1, 1, H, W)
        features = extractor.extract(field, config=None)

        thd = features['total_harmonic_distortion'].mean().item()
        purity = features['fundamental_purity'].mean().item()
        h2f = features['harmonic_ratio_2f'].mean().item()

        thd_values.append(thd)
        purity_values.append(purity)

        print(f"{name:<30s} {thd:10.6f} {purity:10.6f} {h2f:10.6f}")

    # Validate monotonic behavior
    print("\nValidation:")
    thd_increasing = all(thd_values[i] <= thd_values[i+1] for i in range(len(thd_values)-1))
    purity_decreasing = all(purity_values[i] >= purity_values[i+1] for i in range(len(purity_values)-1))

    if thd_increasing:
        print("  ✓ THD increases with nonlinearity")
    else:
        print(f"  ✗ THD not monotonic: {thd_values}")

    if purity_decreasing:
        print("  ✓ Purity decreases with nonlinearity")
    else:
        print(f"  ✗ Purity not monotonic: {purity_values}")

    # Check value ranges
    if all(0 <= thd <= 1 for thd in thd_values):
        print("  ✓ THD in reasonable range [0, 1]")
    else:
        print(f"  ✗ THD out of range: {thd_values}")

    if all(0 <= p <= 1 for p in purity_values):
        print("  ✓ Purity in reasonable range [0, 1]")
    else:
        print(f"  ✗ Purity out of range: {purity_values}")

    return thd_increasing and purity_decreasing


def test_dimensionality_response_to_structure():
    """Test that dimensionality features correctly detect structure."""
    print("\n" + "=" * 70)
    print("SANITY CHECK 2: Dimensionality Response to Structure")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W = 64, 64
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test cases with decreasing structure (increasing dimensionality)
    cases = [
        ("1D structure (pure X)", lambda: torch.sin(2 * torch.pi * 4 * X)),
        ("2D structure (X+Y)", lambda: torch.sin(2 * torch.pi * 4 * X) + torch.sin(2 * torch.pi * 4 * Y)),
        ("Mixed (signal + noise 0.1)", lambda: torch.sin(2 * torch.pi * 4 * X) + 0.1 * torch.randn(H, W, device=device)),
        ("Mixed (signal + noise 0.5)", lambda: torch.sin(2 * torch.pi * 4 * X) + 0.5 * torch.randn(H, W, device=device)),
        ("Pure noise", lambda: torch.randn(H, W, device=device)),
    ]

    extractor = SpatialFeatureExtractor(device=device)

    rank_values = []

    print(f"\n{'Case':<30s} {'Eff. Rank':>12s} {'Part. Ratio':>12s} {'EV90':>8s}")
    print("-" * 70)

    for name, field_fn in cases:
        field = field_fn().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 1, 1, H, W)
        features = extractor.extract(field, config=None)

        rank = features['effective_rank'].mean().item()
        part_ratio = features['participation_ratio'].mean().item()
        ev90 = features['explained_variance_90'].mean().item()

        rank_values.append(rank)

        print(f"{name:<30s} {rank:12.2f} {part_ratio:12.2f} {ev90:8.1f}")

    # Validate monotonic increase
    print("\nValidation:")
    rank_increasing = all(rank_values[i] <= rank_values[i+1] for i in range(len(rank_values)-1))

    if rank_increasing:
        print("  ✓ Effective rank increases with complexity (1D → noise)")
    else:
        print(f"  ✗ Rank not monotonic: {rank_values}")

    # Check specific values
    if rank_values[0] < 5:  # Pure 1D should be low rank
        print(f"  ✓ 1D structure has low rank ({rank_values[0]:.2f})")
    else:
        print(f"  ✗ 1D structure has unexpectedly high rank: {rank_values[0]:.2f}")

    if rank_values[-1] > 10:  # Noise should be high rank
        print(f"  ✓ Noise has high rank ({rank_values[-1]:.2f})")
    else:
        print(f"  ✗ Noise has unexpectedly low rank: {rank_values[-1]:.2f}")

    return rank_increasing


def test_saturation_response_to_clipping():
    """Test that saturation features correctly detect clipping."""
    print("\n" + "=" * 70)
    print("SANITY CHECK 3: Saturation Response to Clipping")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W = 64, 64
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test cases with increasing clipping
    base_field = torch.sin(2 * torch.pi * 4 * X)

    cases = [
        ("No clipping", lambda: base_field),
        ("Mild clip (±0.8)", lambda: torch.clamp(base_field, -0.8, 0.8)),
        ("Medium clip (±0.5)", lambda: torch.clamp(base_field, -0.5, 0.5)),
        ("Strong clip (±0.3)", lambda: torch.clamp(base_field, -0.3, 0.3)),
    ]

    extractor = SpatialFeatureExtractor(device=device)

    sat_values = []

    print(f"\n{'Case':<30s} {'Saturation Ratio':>18s} {'Flatness':>12s}")
    print("-" * 70)

    for name, field_fn in cases:
        field = field_fn().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 1, 1, H, W)
        features = extractor.extract(field, config=None)

        sat = features['gradient_saturation_ratio'].mean().item()
        flatness = features['gradient_flatness'].mean().item()

        sat_values.append(sat)

        print(f"{name:<30s} {sat:18.6f} {flatness:12.2f}")

    # Validate monotonic increase
    print("\nValidation:")
    sat_increasing = all(sat_values[i] <= sat_values[i+1] for i in range(len(sat_values)-1))

    if sat_increasing:
        print("  ✓ Saturation ratio increases with clipping")
    else:
        print(f"  ✗ Saturation not monotonic: {sat_values}")

    # Check value ranges
    if all(0 <= s <= 1 for s in sat_values):
        print("  ✓ Saturation ratio in valid range [0, 1]")
    else:
        print(f"  ✗ Saturation out of range: {sat_values}")

    # Check that clipping significantly increases saturation
    if sat_values[-1] > 2 * sat_values[0]:
        print(f"  ✓ Strong clipping increases saturation ({sat_values[-1]:.3f} vs {sat_values[0]:.3f})")
    else:
        print(f"  ✗ Clipping effect too weak: {sat_values[-1]:.3f} vs {sat_values[0]:.3f}")

    return sat_increasing


def test_dissipation_response_to_decay():
    """Test that dissipation features correctly measure energy decay."""
    print("\n" + "=" * 70)
    print("SANITY CHECK 4: Dissipation Response to Energy Decay")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W = 64, 64
    N, M, T, C = 1, 1, 50, 1

    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test cases with different decay rates
    cases = [
        ("No decay", 0.00),
        ("Slow decay", 0.01),
        ("Medium decay", 0.02),
        ("Fast decay", 0.05),
    ]

    extractor = InvariantDriftExtractor(device=device)

    low_rates = []
    high_rates = []

    print(f"\n{'Case':<20s} {'Low-freq Rate':>15s} {'High-freq Rate':>15s} {'Selectivity':>12s}")
    print("-" * 70)

    for name, decay_rate in cases:
        # Create exponentially decaying sinusoid
        fields = []
        for t in range(T):
            amplitude = torch.exp(torch.tensor(-decay_rate * t, device=device))
            field = amplitude * torch.sin(2 * torch.pi * 4 * X)
            fields.append(field.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0))

        trajectory = torch.cat(fields, dim=2).expand(N, M, T, C, H, W)
        features = extractor.extract(trajectory, config=None)

        low_rate = features['dissipation_rate_lowfreq'].mean().item()
        high_rate = features['dissipation_rate_highfreq'].mean().item()
        selectivity = features['dissipation_selectivity'].mean().item()

        low_rates.append(low_rate)
        high_rates.append(high_rate)

        print(f"{name:<20s} {low_rate:15.6f} {high_rate:15.6f} {selectivity:12.2f}")

    # Validate monotonic increase in dissipation rates
    print("\nValidation:")
    low_increasing = all(low_rates[i] <= low_rates[i+1] for i in range(len(low_rates)-1))

    if low_increasing:
        print("  ✓ Dissipation rate increases with decay rate")
    else:
        print(f"  ✗ Dissipation not monotonic: {low_rates}")

    # Check that measured rates match expected
    expected_rate = cases[-1][1]  # Fast decay rate
    measured_rate = low_rates[-1]

    # Should be within 50% of expected (rough check)
    if 0.5 * expected_rate < measured_rate < 2.0 * expected_rate:
        print(f"  ✓ Measured rate ({measured_rate:.4f}) matches expected ({expected_rate:.4f})")
    else:
        print(f"  ⚠ Measured rate ({measured_rate:.4f}) differs from expected ({expected_rate:.4f})")

    # Check no decay case
    if low_rates[0] < 0.005:  # Should be near zero
        print(f"  ✓ No-decay case has low dissipation ({low_rates[0]:.6f})")
    else:
        print(f"  ✗ No-decay case has unexpected dissipation: {low_rates[0]:.6f}")

    return low_increasing


def test_coherence_response_to_scale():
    """Test that coherence length correctly measures correlation scale."""
    print("\n" + "=" * 70)
    print("SANITY CHECK 5: Coherence Length Response to Structure Scale")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    H, W = 64, 64
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test cases with different spatial scales
    cases = [
        ("High freq (f=16)", 16.0, "short"),
        ("Medium freq (f=8)", 8.0, "medium"),
        ("Low freq (f=4)", 4.0, "long"),
        ("Very low freq (f=2)", 2.0, "very long"),
    ]

    extractor = SpatialFeatureExtractor(device=device)

    coherence_values = []

    print(f"\n{'Case':<25s} {'Wavelength':>12s} {'Coh. Length':>12s} {'Expected':>12s}")
    print("-" * 70)

    for name, freq, expected_desc in cases:
        field = torch.sin(2 * torch.pi * freq * X).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 1, 1, H, W)
        features = extractor.extract(field, config=None)

        coh_len = features['coherence_length'].mean().item()
        wavelength = W / freq  # Approx wavelength in pixels

        coherence_values.append(coh_len)

        print(f"{name:<25s} {wavelength:12.1f} {coh_len:12.2f} {expected_desc:>12s}")

    # Validate monotonic decrease (higher freq → shorter coherence)
    print("\nValidation:")
    coh_decreasing = all(coherence_values[i] >= coherence_values[i+1] for i in range(len(coherence_values)-1))

    if coh_decreasing:
        print("  ✓ Coherence length decreases with frequency (larger scale → longer coherence)")
    else:
        print(f"  ✗ Coherence not monotonic: {coherence_values}")

    # Check reasonable range
    if all(0 < c < W for c in coherence_values):
        print(f"  ✓ Coherence lengths in reasonable range (0, {W})")
    else:
        print(f"  ✗ Coherence out of range: {coherence_values}")

    return coh_decreasing


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 70)
    print("FEATURE SANITY CHECK: Validate Features Are Meaningful")
    print("=" * 70)
    print("\nThis test validates that features respond correctly to physical changes")
    print("and are not just numerically valid garbage values.\n")

    results = []

    # Run all sanity checks
    results.append(("Harmonic content", test_harmonic_response_to_nonlinearity()))
    results.append(("Dimensionality", test_dimensionality_response_to_structure()))
    results.append(("Saturation", test_saturation_response_to_clipping()))
    results.append(("Dissipation", test_dissipation_response_to_decay()))
    results.append(("Coherence", test_coherence_response_to_scale()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL SANITY CHECKS PASSED")
        print("  Features are physically meaningful and respond correctly to changes")
    else:
        print("✗ SOME SANITY CHECKS FAILED")
        print("  Some features may be producing garbage values")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
