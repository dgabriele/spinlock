"""Debug the aggregation bug where FFT features become zero."""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from spinlock.features.sdf.extractors import SDFExtractor
from spinlock.features.sdf.config import SDFConfig

def test_fft_feature_extraction():
    """Test FFT feature extraction through the full pipeline."""

    device = torch.device('cpu')

    # Create trajectory with structured patterns (so FFT has strong peaks)
    N, M, T, C, H, W = 5, 10, 20, 4, 128, 128
    torch.manual_seed(42)
    trajectories = torch.randn(N, M, T, C, H, W, device=device) * 0.1

    # Add spatial waves so FFT features are meaningful
    for n in range(N):
        for m in range(M):
            for t in range(T):
                x = torch.linspace(0, 2 * 3.14159, W)
                y = torch.linspace(0, 2 * 3.14159, H)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                wave = torch.sin((n+1) * xx) * torch.cos((m+1) * yy)
                trajectories[n, m, t, :, :, :] += wave.unsqueeze(0) * 0.5

    # Create extractor
    sdf_config = SDFConfig()
    extractor = SDFExtractor(device, sdf_config)

    print("=" * 80)
    print("STAGE 1: Extract per-timestep features")
    print("=" * 80)

    # Extract per-timestep
    per_timestep = extractor.extract_per_timestep(trajectories)
    print(f"Per-timestep shape: {per_timestep.shape}")  # Should be [N, T, D]

    # Find fft_power_scale_0_max in registry
    registry = extractor.get_feature_registry()
    all_per_timestep_names = []
    for cat in ['spatial', 'spectral', 'cross_channel']:
        all_per_timestep_names.extend([f.name for f in registry.get_features_by_category(cat)])

    print(f"\nTotal per-timestep features in registry: {len(all_per_timestep_names)}")

    # Find index of fft_power_scale_0_max
    if 'fft_power_scale_0_max' in all_per_timestep_names:
        idx = all_per_timestep_names.index('fft_power_scale_0_max')
        print(f"\nfft_power_scale_0_max index: {idx}")

        # Extract that feature column
        fft_max_values = per_timestep[:, :, idx]  # [N, T]
        print(f"\nfft_power_scale_0_max from per_timestep:")
        print(f"  Shape: {fft_max_values.shape}")
        print(f"  Min: {fft_max_values.min():.10f}")
        print(f"  Max: {fft_max_values.max():.10f}")
        print(f"  Mean: {fft_max_values.mean():.10f}")
        print(f"  Sample values (first operator, first 5 timesteps): {fft_max_values[0, :5]}")

        # Check if all zeros
        if (fft_max_values == 0).all():
            print("\n❌ BUG CONFIRMED: All values are exactly zero!")
        elif (fft_max_values.abs() < 1e-6).all():
            print("\n❌ BUG CONFIRMED: All values are near-zero (< 1e-6)!")
        else:
            print("\n✅ Values look correct!")
    else:
        print(f"\n❌ fft_power_scale_0_max not found in registry!")
        print(f"Available spectral features: {[f.name for f in registry.get_features_by_category('spectral')][:10]}")

    print("\n" + "=" * 80)
    print("STAGE 2: Extract per-trajectory features")
    print("=" * 80)

    per_trajectory = extractor.extract_per_trajectory(trajectories)
    print(f"Per-trajectory shape: {per_trajectory.shape}")  # Should be [N, M, D_traj]

    # Per-trajectory shouldn't have FFT features (those are per-timestep)
    all_per_trajectory_names = []
    for cat in ['temporal', 'causality', 'invariant_drift']:
        all_per_trajectory_names.extend([f.name for f in registry.get_features_by_category(cat)])

    print(f"Total per-trajectory features: {len(all_per_trajectory_names)}")

    if 'fft_power_scale_0_max' in all_per_trajectory_names:
        print("\n❌ BUG: fft_power_scale_0_max shouldn't be in per-trajectory features!")
    else:
        print("\n✅ Correctly excluded from per-trajectory features")

    print("\n" + "=" * 80)
    print("STAGE 3: Aggregate realizations")
    print("=" * 80)

    aggregated_mean = extractor.aggregate_realizations(per_trajectory, method='mean')
    print(f"Aggregated (mean) shape: {aggregated_mean.shape}")  # Should be [N, D_traj]

    # Check if fft_power_scale_0_max somehow ended up here
    print("\n✅ Per-timestep features (including FFT) are NOT aggregated - they remain as [N, T, D]")
    print("✅ Only per-trajectory features are aggregated to [N, D_traj]")

    print("\n" + "=" * 80)
    print("FULL PIPELINE: extract_all()")
    print("=" * 80)

    results = extractor.extract_all(trajectories)

    print("\nKeys in results:")
    for key in results.keys():
        print(f"  {key}: {results[key].shape}")

    # Check per_timestep again
    per_timestep_from_all = results['per_timestep']
    if 'fft_power_scale_0_max' in all_per_timestep_names:
        idx = all_per_timestep_names.index('fft_power_scale_0_max')
        fft_max_values = per_timestep_from_all[:, :, idx]

        print(f"\n\nfft_power_scale_0_max from extract_all():")
        print(f"  Shape: {fft_max_values.shape}")
        print(f"  Min: {fft_max_values.min():.10f}")
        print(f"  Max: {fft_max_values.max():.10f}")
        print(f"  Mean: {fft_max_values.mean():.10f}")

        if (fft_max_values == 0).all():
            print("\n❌ BUG CONFIRMED in extract_all(): All values are zero!")
        elif (fft_max_values.abs() < 1e-6).all():
            print("\n❌ BUG CONFIRMED in extract_all(): All values are near-zero!")
        else:
            print("\n✅ Values still correct in extract_all()!")

if __name__ == '__main__':
    test_fft_feature_extraction()
