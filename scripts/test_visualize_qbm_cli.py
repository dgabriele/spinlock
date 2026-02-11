"""Manual test for visualize-qbm CLI command."""
import sys
import h5py
import numpy as np
from pathlib import Path
from argparse import Namespace

# Direct import to avoid CLI __init__ import issues
sys.path.insert(0, 'src')

# Import directly from module file, not through __init__
import importlib.util
spec = importlib.util.spec_from_file_location("visualize_qbm", "src/spinlock/cli/visualize_qbm.py")
visualize_qbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_qbm)
VisualizeQBMCommand = visualize_qbm.VisualizeQBMCommand


def create_mock_dataset(path: Path):
    """Create a minimal mock QBM dataset."""
    N, M, T, H, W = 10, 3, 20, 32, 32

    with h5py.File(path, 'w') as f:
        # Outputs: [N, M, T, 2, H, W]
        f.create_dataset('outputs', data=np.random.randn(N, M, T, 2, H, W).astype(np.float32))

        # Inputs: [N, M, 2, H, W]
        f.create_dataset('inputs', data=np.random.randn(N, M, 2, H, W).astype(np.float32))

        # Temporal features: [N, T, 188]
        f.create_dataset('features/temporal', data=np.random.randn(N, T, 188).astype(np.float32))

        # Parameters
        f.create_dataset('parameters/gamma', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/nbar', data=np.random.rand(N).astype(np.float32))
        f.create_dataset('parameters/alpha_init', data=np.random.rand(N).astype(np.float32))

    print(f"Created mock dataset: {path}")


def test_probability_mode():
    """Test probability density rendering."""
    dataset_path = Path("/tmp/test_qbm.h5")
    output_path = Path("/tmp/test_qbm_probability.mp4")

    print("\n=== Test 1: Probability Mode ===")
    create_mock_dataset(dataset_path)

    args = Namespace(
        dataset=dataset_path,
        output=output_path,
        n_rollouts=2,
        sampling_method='random',
        renderer='probability',
        overlay_observable='purity',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=True,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    if exit_code == 0:
        print(f"✓ Success! Video saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"✗ Failed with exit code {exit_code}")

    # Cleanup
    dataset_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    return exit_code == 0


def test_wigner_mode():
    """Test Wigner phase-space rendering."""
    dataset_path = Path("/tmp/test_qbm.h5")
    output_path = Path("/tmp/test_qbm_wigner.mp4")

    print("\n=== Test 2: Wigner Mode ===")
    create_mock_dataset(dataset_path)

    args = Namespace(
        dataset=dataset_path,
        output=output_path,
        n_rollouts=2,
        sampling_method='random',
        renderer='wigner',
        overlay_observable='entropy',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=True,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    if exit_code == 0:
        print(f"✓ Success! Video saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"✗ Failed with exit code {exit_code}")

    # Cleanup
    dataset_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    return exit_code == 0


def test_both_mode():
    """Test side-by-side rendering."""
    dataset_path = Path("/tmp/test_qbm.h5")
    output_path = Path("/tmp/test_qbm_both.mp4")

    print("\n=== Test 3: Both Mode (Side-by-Side) ===")
    create_mock_dataset(dataset_path)

    args = Namespace(
        dataset=dataset_path,
        output=output_path,
        n_rollouts=2,
        sampling_method='random',
        renderer='both',
        overlay_observable='coherence_mean',
        colormap='viridis',
        wigner_colormap='RdBu_r',
        fps=10,
        stride=2,
        device='cpu',
        seed=42,
        verbose=True,
    )

    command = VisualizeQBMCommand()
    exit_code = command.execute(args)

    if exit_code == 0:
        print(f"✓ Success! Video saved to {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"✗ Failed with exit code {exit_code}")

    # Cleanup
    dataset_path.unlink(missing_ok=True)
    output_path.unlink(missing_ok=True)

    return exit_code == 0


if __name__ == "__main__":
    print("Testing visualize-qbm CLI command")
    print("=" * 50)

    results = []
    results.append(("Probability Mode", test_probability_mode()))
    results.append(("Wigner Mode", test_wigner_mode()))
    results.append(("Both Mode", test_both_mode()))

    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    sys.exit(0 if all_passed else 1)
