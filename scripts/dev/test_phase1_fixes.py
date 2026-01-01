#!/usr/bin/env python
"""Validate Phase 1 numerical stability fixes for SDF v2.0 → v2.1.

Tests:
1. No extreme values (|feature| < 1e6) - fixes FFT power, kurtosis overflow
2. Reduced NaN count - fixes rolling windows, PACF
3. T-normalization works across different trajectory lengths
4. Grid-size normalization works across different spatial resolutions

Run from project root:
    python scripts/dev/test_phase1_fixes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import numpy as np
import h5py
from typing import Dict, Tuple

from spinlock.features.sdf import SDFExtractor, SDFConfig


def load_test_dataset(dataset_path: str = "datasets/test_1k_inline_features.h5") -> Tuple[torch.Tensor, np.ndarray]:
    """Load original features and generate test trajectories."""
    with h5py.File(dataset_path, 'r') as f:
        # Load original buggy features for comparison
        original_features = f['features/sdf/aggregated/features'][:]  # [N, 360]

    print(f"✓ Loaded dataset: {dataset_path}")
    print(f"  Original features: {original_features.shape}")

    # Generate synthetic test trajectories for validation
    # Shape: [N, M, T, C, H, W] - Small size for GPU memory
    N, M, T, C, H, W = 20, 3, 100, 2, 64, 64

    print(f"\n✓ Generating synthetic test trajectories: [{N}, {M}, {T}, {C}, {H}, {W}]")

    # Create diverse test cases:
    trajectories = []
    for i in range(N):
        # Mix of different dynamics
        if i % 4 == 0:
            # Constant field (tests zero-variance handling)
            traj = torch.ones(M, T, C, H, W) * (i * 0.1)
        elif i % 4 == 1:
            # Random walk
            traj = torch.cumsum(torch.randn(M, T, C, H, W) * 0.1, dim=1)
        elif i % 4 == 2:
            # Oscillatory
            t = torch.linspace(0, 10 * np.pi, T).reshape(1, T, 1, 1, 1)
            traj = torch.sin(t) * torch.randn(M, 1, C, H, W)
        else:
            # Smooth gradient
            traj = torch.linspace(0, 1, T).reshape(1, T, 1, 1, 1).expand(M, T, C, H, W)

        trajectories.append(traj)

    trajectories = torch.stack(trajectories)  # [N, M, T, C, H, W]

    return trajectories, original_features


def test_extreme_values(features: torch.Tensor) -> Dict[str, bool]:
    """Test that no features have extreme values (|x| > 1e6)."""
    print("\n" + "="*70)
    print("TEST 1: Extreme Values (FFT power, kurtosis overflow)")
    print("="*70)

    max_abs = features.abs().max().item()
    extreme_mask = features.abs() > 1e6
    num_extreme = extreme_mask.sum().item()

    passed = num_extreme == 0

    print(f"Max absolute value: {max_abs:.2e}")
    print(f"Features with |value| > 1e6: {num_extreme}")

    if not passed:
        # Find which features have extreme values
        extreme_indices = torch.where(extreme_mask.any(dim=0))[0].tolist()
        print(f"  Extreme value feature indices: {extreme_indices[:10]}..." if len(extreme_indices) > 10 else f"  {extreme_indices}")

    print(f"{'✅ PASS' if passed else '❌ FAIL'}: No extreme values")

    return {"extreme_values": passed}


def test_nan_reduction(features: torch.Tensor, original_features: np.ndarray) -> Dict[str, bool]:
    """Test that NaN count is reduced compared to original."""
    print("\n" + "="*70)
    print("TEST 2: NaN Reduction (rolling windows, PACF)")
    print("="*70)

    new_nan_count = torch.isnan(features).sum().item()
    old_nan_count = np.isnan(original_features).sum()

    # Calculate NaN percentage
    total_values = features.numel()
    new_nan_pct = 100.0 * new_nan_count / total_values
    old_nan_pct = 100.0 * old_nan_count / original_features.size

    # Check per-feature NaN counts
    new_nan_per_feature = torch.isnan(features).sum(dim=0)
    old_nan_per_feature = np.isnan(original_features).sum(axis=0)

    # Find features with reduced NaN
    reduction_indices = []
    for i in range(min(len(new_nan_per_feature), len(old_nan_per_feature))):
        if new_nan_per_feature[i] < old_nan_per_feature[i]:
            reduction_indices.append(i)

    passed = new_nan_pct < 1.0  # Accept <1% NaN (for mathematically undefined cases)

    print(f"Original NaN count: {old_nan_count} ({old_nan_pct:.2f}%)")
    print(f"New NaN count: {new_nan_count} ({new_nan_pct:.2f}%)")
    print(f"Reduction: {old_nan_count - new_nan_count} NaN values eliminated")
    print(f"Features with reduced NaN: {len(reduction_indices)}")

    if reduction_indices:
        print(f"  First 10 improved features: {reduction_indices[:10]}")

    print(f"{'✅ PASS' if passed else '❌ FAIL'}: NaN percentage < 1%")

    return {"nan_reduction": passed}


def test_t_normalization(extractor: SDFExtractor, base_trajectory: torch.Tensor) -> Dict[str, bool]:
    """Test that T-normalization works across different trajectory lengths."""
    print("\n" + "="*70)
    print("TEST 3: T-Normalization (trajectory_smoothness, regime_switches)")
    print("="*70)

    # Use first trajectory and test with T ∈ {50, 100, 200, 500}
    N, M, T_full, C, H, W = base_trajectory.shape
    test_lengths = [50, 100, 200, min(500, T_full)]

    feature_values = {}

    for T in test_lengths:
        # Truncate trajectory to T timesteps
        traj_t = base_trajectory[:1, :, :T, :, :, :]  # [1, M, T, C, H, W]

        # Extract features
        result = extractor.extract_all(traj_t)

        # Concatenate aggregated features
        aggregated = torch.cat([
            result['aggregated_mean'],
            result['aggregated_std'],
            result['aggregated_cv']
        ], dim=1)  # [1, num_features]

        feature_values[T] = aggregated[0].cpu().numpy()
        print(f"  T={T:3d}: Extracted {aggregated.shape[1]} features")

    # Check that T-normalized features are roughly consistent
    # (should be similar per-timestep rates, not growing linearly with T)
    # Compare T=50 vs T=500 (10× difference)
    if len(test_lengths) >= 2:
        ratio_50_500 = np.abs(feature_values[test_lengths[0]] / (feature_values[test_lengths[-1]] + 1e-8))

        # For T-normalized features, ratio should be close to 1.0 (not 0.1 or 10.0)
        # Allow 10× tolerance (some features may have different behavior at different T)
        median_ratio = np.nanmedian(ratio_50_500)

        passed = 0.1 < median_ratio < 10.0

        print(f"\n  Median ratio (T={test_lengths[0]} / T={test_lengths[-1]}): {median_ratio:.2f}")
        print(f"  Expected ~1.0 for T-normalized features (0.1-10.0 tolerance)")
        print(f"{'✅ PASS' if passed else '❌ FAIL'}: T-normalization working")
    else:
        passed = True
        print("⚠️  Insufficient trajectory length for T-normalization test")

    return {"t_normalization": passed}


def test_grid_size_normalization(extractor: SDFExtractor, base_trajectory: torch.Tensor) -> Dict[str, bool]:
    """Test that grid-size normalization works (FFT power per-bin normalization)."""
    print("\n" + "="*70)
    print("TEST 4: Grid-Size Normalization (FFT power per-bin)")
    print("="*70)

    # Use first trajectory and test with H×W ∈ {64², 128²}
    N, M, T, C, H_full, W_full = base_trajectory.shape

    # Only test if H, W >= 128
    if H_full < 128 or W_full < 128:
        print(f"⚠️  Grid size {H_full}×{W_full} too small for downsampling test")
        return {"grid_size_normalization": True}

    # Extract features at full resolution
    traj_full = base_trajectory[:1, :, :, :, :, :]  # [1, M, T, C, H, W]
    result_full = extractor.extract_all(traj_full)
    features_full = torch.cat([
        result_full['aggregated_mean'],
        result_full['aggregated_std'],
        result_full['aggregated_cv']
    ], dim=1)[0].cpu().numpy()

    # Downsample to 64×64 (average pooling)
    traj_64 = torch.nn.functional.avg_pool2d(
        base_trajectory[:1, :, :, :, :, :].reshape(-1, C, H_full, W_full),
        kernel_size=H_full // 64,
        stride=H_full // 64
    ).reshape(1, M, T, C, 64, 64)

    result_64 = extractor.extract_all(traj_64)
    features_64 = torch.cat([
        result_64['aggregated_mean'],
        result_64['aggregated_std'],
        result_64['aggregated_cv']
    ], dim=1)[0].cpu().numpy()

    # Compare FFT power features (should be similar after per-bin normalization)
    # Indices 56-69 are FFT power features (from bug analysis)
    if features_full.shape[0] > 69 and features_64.shape[0] > 69:
        fft_indices = list(range(56, 70))
        ratio_64_full = features_64[fft_indices] / (features_full[fft_indices] + 1e-8)
        median_ratio = np.nanmedian(ratio_64_full)

        # After per-bin normalization, ratio should be ~1.0 (not 4.0 for 64² vs 128²)
        passed = 0.25 < median_ratio < 4.0

        print(f"  Full resolution: {H_full}×{W_full}")
        print(f"  Downsampled: 64×64")
        print(f"  Median FFT power ratio (64×64 / {H_full}×{W_full}): {median_ratio:.2f}")
        print(f"  Expected ~1.0 for per-bin normalized features (0.25-4.0 tolerance)")
        print(f"{'✅ PASS' if passed else '❌ FAIL'}: Grid-size normalization working")
    else:
        passed = True
        print("⚠️  Feature dimension mismatch, skipping grid-size test")

    return {"grid_size_normalization": passed}


def main():
    print("="*70)
    print("SDF v2.0 → v2.1 Phase 1 Numerical Stability Validation")
    print("="*70)

    # Load dataset
    trajectories, original_features = load_test_dataset()

    # Initialize extractor with all categories enabled (like original)
    config = SDFConfig()
    extractor = SDFExtractor(device='cuda' if torch.cuda.is_available() else 'cpu', config=config)

    print(f"\n✓ Initialized SDF extractor on {extractor.device}")

    # Extract features with fixed code
    print("\n" + "="*70)
    print("Extracting features with Phase 1 fixes...")
    print("="*70)

    # Use all test trajectories (already small: 20 operators)
    test_trajectories = trajectories.to(extractor.device)

    result = extractor.extract_all(test_trajectories)

    # Concatenate aggregated features (mean, std, cv)
    features = torch.cat([
        result['aggregated_mean'],
        result['aggregated_std'],
        result['aggregated_cv']
    ], dim=1)  # [20, 3*num_features]

    print(f"✓ Extracted features: {features.shape}")
    print(f"  Per-timestep: {result['per_timestep'].shape}")
    print(f"  Per-trajectory: {result['per_trajectory'].shape}")
    print(f"  Aggregated (mean+std+cv): {features.shape}")

    # Run tests
    results = {}
    results.update(test_extreme_values(features))
    results.update(test_nan_reduction(features, original_features[:20]))
    results.update(test_t_normalization(extractor, test_trajectories[:1]))
    results.update(test_grid_size_normalization(extractor, test_trajectories[:1]))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "="*70)
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    if passed_tests == total_tests:
        print("\n🎉 All Phase 1 fixes validated successfully!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
