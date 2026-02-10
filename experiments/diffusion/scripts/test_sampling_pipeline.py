#!/usr/bin/env python3
"""Test sampling pipeline end-to-end.

Run with: poetry run python experiments/diffusion/scripts/test_sampling_pipeline.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from sampling import DiffusionSampler, SampleWriter, SampleVisualizer


def test_token_sampling():
    """Test token sampling with mock models."""
    print("Testing token sampling...")

    # Mock diffusion and denoiser
    class MockDiffusion:
        def eval(self):
            pass

        def sample(self, batch_size, denoising_network, device):
            return {
                'temporal_group_1_L0': torch.randint(0, 28, (batch_size,)),
                'initial_group_1_L0': torch.randint(0, 28, (batch_size,)),
                'theta_group_1_L0': torch.randint(0, 28, (batch_size,)),
            }

    class MockDenoiser:
        def eval(self):
            pass

    diffusion = MockDiffusion()
    denoiser = MockDenoiser()

    sampler = DiffusionSampler(diffusion, denoiser, device='cpu')

    # Test unconditional sampling
    tokens = sampler.sample_unconditional(num_samples=10, batch_size=5)

    assert len(tokens) == 10, f"Expected 10 samples, got {len(tokens)}"
    assert all(len(t) == 3 for t in tokens), "Each token dict should have 3 keys"

    print("✓ Token sampling test passed")
    return tokens


def test_output_writer(tokens):
    """Test output writer."""
    print("Testing output writer...")

    writer = SampleWriter(Path("/tmp/test_samples"))

    # Mock data
    theta = np.random.rand(10, 14)
    u0 = np.random.rand(10, 3, 64, 64)

    metadata = {
        "num_samples": 10,
        "mode": "test",
        "timestamp": "2026-02-10"
    }

    writer.write_samples(
        tokens=tokens,
        theta=theta,
        u0=u0,
        trajectories=None,
        metadata=metadata
    )

    # Check files exist
    assert (Path("/tmp/test_samples") / "samples.h5").exists()
    assert (Path("/tmp/test_samples") / "metadata.json").exists()

    print("✓ Output writer test passed")


def test_visualizer():
    """Test visualizer."""
    print("Testing visualizer...")

    visualizer = SampleVisualizer()

    # Mock data
    trajectory = np.random.rand(1, 256, 3, 64, 64)
    theta = np.random.rand(14)
    theta_samples = np.random.rand(100, 14)

    # Test trajectory plot
    visualizer.plot_trajectory(
        trajectory, theta, Path("/tmp/test_samples/test_trajectory.png")
    )

    # Test parameter distribution
    visualizer.plot_parameter_distribution(
        theta_samples, Path("/tmp/test_samples/test_params.png")
    )

    # Test diversity comparison
    trajectories = [np.random.rand(1, 256, 3, 64, 64) for _ in range(5)]
    visualizer.plot_diversity_comparison(
        trajectories, Path("/tmp/test_samples/test_diversity.png")
    )

    print("✓ Visualizer test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Sampling Pipeline")
    print("=" * 60)

    try:
        tokens = test_token_sampling()
        test_output_writer(tokens)
        test_visualizer()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
