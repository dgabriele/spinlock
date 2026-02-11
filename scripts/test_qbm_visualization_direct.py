"""Direct test of QBM visualization components (bypassing CLI)."""
import sys
sys.path.insert(0, 'src')

import h5py
import numpy as np
import torch
from pathlib import Path

from spinlock.visualization.quantum.data_loader import QBMDatasetLoader
from spinlock.visualization.quantum.renderer import QBMRenderer
from spinlock.visualization.quantum.wigner_renderer import WignerRenderer
from spinlock.visualization.quantum.aggregates import QuantumObservableOverlay
from spinlock.visualization.exporters.video import create_video_exporter_with_gpu_fallback


def create_mock_dataset(path: Path):
    """Create a minimal mock QBM dataset."""
    N, M, T, H, W = 10, 3, 20, 32, 32

    with h5py.File(path, 'w') as f:
        # Create Gaussian wavepacket-like data for more realistic rendering
        X, Y = np.meshgrid(np.linspace(-2, 2, H), np.linspace(-2, 2, W))

        outputs = np.zeros((N, M, T, 2, H, W), dtype=np.float32)
        for n in range(N):
            for m in range(M):
                for t in range(T):
                    # Time-evolving Gaussian
                    sigma = 0.5 + 0.3 * t / T
                    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

                    # Add some phase
                    phase = 2 * np.pi * (t / T) * (X + Y)

                    outputs[n, m, t, 0] = gaussian * np.cos(phase)  # Real part
                    outputs[n, m, t, 1] = gaussian * np.sin(phase)  # Imag part

        f.create_dataset('outputs', data=outputs)

        # Inputs
        f.create_dataset('inputs', data=outputs[:, :, 0])  # t=0

        # Temporal features with quantum observables
        temporal = np.random.randn(N, T, 188).astype(np.float32)

        # Set last 10 features to realistic quantum values
        temporal[:, :, 178] = 0.5 + 0.3 * np.random.randn(N, T)  # Purity [0.2, 0.8]
        temporal[:, :, 179] = 0.5 + 0.3 * np.random.randn(N, T)  # Entropy
        temporal[:, :, 180] = 0.5 + 0.2 * np.random.randn(N, T)  # Coherence mean
        temporal[:, :, 181] = 0.7 + 0.2 * np.random.randn(N, T)  # Coherence max

        f.create_dataset('features/temporal', data=temporal)

        # Parameters
        f.create_dataset('parameters/gamma', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/nbar', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/alpha_init', data=np.random.rand(N).astype(np.float32))

    print(f"✓ Created mock dataset: {path}")


def test_probability_rendering():
    """Test probability density rendering."""
    print("\n=== Test 1: Probability Density Rendering ===")

    dataset_path = Path("/tmp/test_qbm.h5")
    output_path = Path("/tmp/test_qbm_probability.mp4")

    create_mock_dataset(dataset_path)

    device = torch.device('cpu')

    with QBMDatasetLoader(dataset_path) as loader:
        # Sample 2 rollouts
        indices = torch.tensor([0, 1])

        # Load data
        rollouts = loader.load_rollouts(indices, output_format="real_imag")  # [2, 3, 20, 2, 32, 32]
        quantum_features = loader.load_quantum_features(indices)  # [2, 20, 10]

        print(f"  Rollouts shape: {rollouts.shape}")
        print(f"  Quantum features shape: {quantum_features.shape}")

        # Setup renderer
        renderer = QBMRenderer(colormap='viridis', normalize_mode='global', device=device)
        global_stats = renderer.compute_global_stats(rollouts)

        print(f"  Global stats: vmin={global_stats['vmin']:.4f}, vmax={global_stats['vmax']:.4f}")

        # Setup overlay
        overlay = QuantumObservableOverlay(observable='purity', display_mode='text')

        # Render frames
        N, M, T = rollouts.shape[0], rollouts.shape[1], rollouts.shape[2]
        all_frames = []

        for t in range(0, T, 2):  # Stride 2
            frame_data = rollouts[:, :, t].to(device)  # [2, 3, 2, 32, 32]
            frame_data_flat = frame_data.reshape(N * M, *frame_data.shape[2:])  # [6, 2, 32, 32]

            rendered = renderer.render(frame_data_flat, global_stats=global_stats)  # [6, 32, 32, 3]

            # Add overlay
            obs_values = overlay.extract_observable(quantum_features, timestep=t)  # [2]
            obs_values = obs_values.repeat_interleave(M)  # [6]
            rendered = overlay.render_overlay(rendered, obs_values)

            # Stack to grid [2, 3, 32, 32, 3] -> [64, 96, 3]
            H, W = rendered.shape[1], rendered.shape[2]
            rendered_grid = rendered.reshape(N, M, H, W, 3)
            grid_frame = torch.cat([
                torch.cat(list(rendered_grid[n]), dim=1)  # Concat across width (realizations)
                for n in range(N)
            ], dim=0)  # Concat across height (rollouts)

            all_frames.append(grid_frame)

            print(f"  Rendered frame {t // 2 + 1}/{T // 2}: {grid_frame.shape}")

        # Stack and export
        all_frames = torch.stack(all_frames, dim=0)  # [T/2, H_grid, W_grid, 3]
        all_frames = all_frames.permute(0, 3, 1, 2)  # [T/2, 3, H_grid, W_grid]

        print(f"  Final video tensor: {all_frames.shape}")

        exporter = create_video_exporter_with_gpu_fallback(fps=10, try_gpu=False, verbose=True)
        exporter.export(all_frames, output_path=output_path)

        print(f"✓ Video saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Cleanup
    dataset_path.unlink()
    output_path.unlink()

    return True


def test_wigner_rendering():
    """Test Wigner function rendering."""
    print("\n=== Test 2: Wigner Phase-Space Rendering ===")

    dataset_path = Path("/tmp/test_qbm.h5")
    output_path = Path("/tmp/test_qbm_wigner.mp4")

    create_mock_dataset(dataset_path)

    device = torch.device('cpu')

    with QBMDatasetLoader(dataset_path) as loader:
        indices = torch.tensor([0, 1])
        rollouts = loader.load_rollouts(indices, output_format="real_imag")

        # Setup Wigner renderer
        renderer = WignerRenderer(colormap='RdBu_r', device=device)
        global_stats = renderer.compute_global_stats(rollouts)

        print(f"  Wigner stats: vmin={global_stats['vmin']:.4f}, vmax={global_stats['vmax']:.4f}")

        # Render single frame as test
        frame_data = rollouts[:, :, 0].reshape(-1, 2, 32, 32).to(device)
        rendered = renderer.render(frame_data, global_stats=global_stats)

        print(f"  Rendered Wigner frame: {rendered.shape}")
        print(f"✓ Wigner rendering successful")

    # Cleanup
    dataset_path.unlink()

    return True


if __name__ == "__main__":
    print("Testing QBM Visualization Components")
    print("=" * 60)

    try:
        test_probability_rendering()
        test_wigner_rendering()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
