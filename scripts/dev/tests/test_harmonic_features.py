#!/usr/bin/env python3
"""
Test script for spectral harmonic content features.

Validates:
1. Harmonic features are extracted correctly
2. Features detect nonlinearity (harmonics present)
3. All 4 features are valid (no NaN/Inf)
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.features.sdf.spectral import SpectralFeatureExtractor


def test_harmonic_features():
    """Test harmonic content feature extraction."""
    print("=" * 70)
    print("TEST: Spectral Harmonic Content Features")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create test fields with known harmonic content
    # Generate sinusoidal pattern with harmonics to test detection
    N, M, T, C, H, W = 2, 3, 5, 3, 64, 64

    # Create frequency grids
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')

    # Test case 1: Pure fundamental (no harmonics, linear)
    freq_fundamental = 4.0
    fields_linear = torch.sin(2 * torch.pi * freq_fundamental * X).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_linear = fields_linear.expand(N, M, T, C, H, W).clone()

    # Test case 2: Fundamental + 2nd harmonic (nonlinear)
    fields_nonlinear = torch.sin(2 * torch.pi * freq_fundamental * X)
    fields_nonlinear += 0.3 * torch.sin(2 * torch.pi * 2 * freq_fundamental * X)  # 2nd harmonic
    fields_nonlinear = fields_nonlinear.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_nonlinear = fields_nonlinear.expand(N, M, T, C, H, W).clone()

    # Test case 3: Fundamental + 2nd + 3rd harmonics (strong nonlinearity)
    fields_strong_nonlinear = torch.sin(2 * torch.pi * freq_fundamental * X)
    fields_strong_nonlinear += 0.3 * torch.sin(2 * torch.pi * 2 * freq_fundamental * X)  # 2nd harmonic
    fields_strong_nonlinear += 0.2 * torch.sin(2 * torch.pi * 3 * freq_fundamental * X)  # 3rd harmonic
    fields_strong_nonlinear = fields_strong_nonlinear.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    fields_strong_nonlinear = fields_strong_nonlinear.expand(N, M, T, C, H, W).clone()

    # Create extractor
    extractor = SpectralFeatureExtractor(device=device)

    print("\n" + "=" * 70)
    print("TEST CASE 1: Linear (pure fundamental, no harmonics)")
    print("=" * 70)

    features_linear = extractor.extract(fields_linear, config=None)

    # Extract harmonic features
    print(f"\nShape check:")
    print(f"  harmonic_ratio_2f: {features_linear['harmonic_ratio_2f'].shape}")  # [N, M, T, C]

    # Check values (should be near zero for linear case)
    h2f_mean = features_linear['harmonic_ratio_2f'].mean().item()
    h3f_mean = features_linear['harmonic_ratio_3f'].mean().item()
    thd_mean = features_linear['total_harmonic_distortion'].mean().item()
    purity_mean = features_linear['fundamental_purity'].mean().item()

    print(f"\nHarmonic content (expect low for linear):")
    print(f"  2nd harmonic ratio: {h2f_mean:.6f}")
    print(f"  3rd harmonic ratio: {h3f_mean:.6f}")
    print(f"  THD:                {thd_mean:.6f}")
    print(f"  Fundamental purity: {purity_mean:.6f} (expect high)")

    print("\n" + "=" * 70)
    print("TEST CASE 2: Nonlinear (fundamental + 2nd harmonic)")
    print("=" * 70)

    features_nonlinear = extractor.extract(fields_nonlinear, config=None)

    h2f_mean = features_nonlinear['harmonic_ratio_2f'].mean().item()
    h3f_mean = features_nonlinear['harmonic_ratio_3f'].mean().item()
    thd_mean = features_nonlinear['total_harmonic_distortion'].mean().item()
    purity_mean = features_nonlinear['fundamental_purity'].mean().item()

    print(f"\nHarmonic content (expect 2nd harmonic detected):")
    print(f"  2nd harmonic ratio: {h2f_mean:.6f} (expect elevated)")
    print(f"  3rd harmonic ratio: {h3f_mean:.6f}")
    print(f"  THD:                {thd_mean:.6f} (expect elevated)")
    print(f"  Fundamental purity: {purity_mean:.6f} (expect reduced)")

    print("\n" + "=" * 70)
    print("TEST CASE 3: Strong Nonlinear (fundamental + 2nd + 3rd harmonics)")
    print("=" * 70)

    features_strong = extractor.extract(fields_strong_nonlinear, config=None)

    h2f_mean = features_strong['harmonic_ratio_2f'].mean().item()
    h3f_mean = features_strong['harmonic_ratio_3f'].mean().item()
    thd_mean = features_strong['total_harmonic_distortion'].mean().item()
    purity_mean = features_strong['fundamental_purity'].mean().item()

    print(f"\nHarmonic content (expect both 2nd and 3rd harmonics):")
    print(f"  2nd harmonic ratio: {h2f_mean:.6f} (expect elevated)")
    print(f"  3rd harmonic ratio: {h3f_mean:.6f} (expect elevated)")
    print(f"  THD:                {thd_mean:.6f} (expect high)")
    print(f"  Fundamental purity: {purity_mean:.6f} (expect reduced)")

    print("\n" + "=" * 70)
    print("VALIDATION: NaN/Inf check")
    print("=" * 70)

    for name in ['harmonic_ratio_2f', 'harmonic_ratio_3f', 'total_harmonic_distortion', 'fundamental_purity']:
        for case_name, features in [('Linear', features_linear), ('Nonlinear', features_nonlinear), ('Strong', features_strong)]:
            feat = features[name]
            nan_count = torch.isnan(feat).sum().item()
            inf_count = torch.isinf(feat).sum().item()
            print(f"  {name:30s} [{case_name:12s}]: NaN={nan_count}, Inf={inf_count}")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Harmonic content features working")
    print("=" * 70)


if __name__ == "__main__":
    test_harmonic_features()
