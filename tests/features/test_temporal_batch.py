"""Test batch-parallel temporal feature extraction.

Validates numerical equivalence between sequential and batch-parallel implementations.
"""

import pytest
import torch
import numpy as np

from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator
from spinlock.features.temporal.temporal import TemporalFeatureExtractor
from spinlock.features.temporal.temporal_batch import BatchParallelTemporalExtractor


@pytest.fixture
def device():
    """Get CUDA device if available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_trajectory(device):
    """Create a test trajectory."""
    # [N=2, T=16, C=3, H=32, W=32]
    torch.manual_seed(42)
    return torch.randn(2, 16, 3, 32, 32, device=device)


class TestBatchParallelTemporalExtractor:
    """Test batch-parallel temporal feature extractor."""

    def test_instantaneous_features(self, device, test_trajectory):
        """Test instantaneous feature extraction matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential (ground truth)
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor._extract_instantaneous(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor._extract_instantaneous_batch(test_trajectory)

        # Compare
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-5, atol=1e-6,
            msg="Batch instantaneous features don't match sequential"
        )

    def test_local_temporal_features(self, device, test_trajectory):
        """Test local temporal feature extraction matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor._extract_local_temporal(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor._extract_local_temporal_batch(test_trajectory)

        # Compare (allow slightly larger tolerance for accumulation)
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-4, atol=1e-5,
            msg="Batch local temporal features don't match sequential"
        )

    def test_stability_features(self, device, test_trajectory):
        """Test stability feature extraction matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor._extract_local_stability(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor._extract_local_stability_batch(test_trajectory)

        # Compare
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-4, atol=1e-5,
            msg="Batch stability features don't match sequential"
        )

    def test_phase_space_features(self, device, test_trajectory):
        """Test phase space feature extraction matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor._extract_phase_space(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor._extract_phase_space_batch(test_trajectory)

        # Compare
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-4, atol=1e-5,
            msg="Batch phase space features don't match sequential"
        )

    def test_multiscale_features(self, device, test_trajectory):
        """Test multiscale feature extraction matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor._extract_multiscale(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor._extract_multiscale_batch(test_trajectory)

        # Compare
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-4, atol=1e-5,
            msg="Batch multiscale features don't match sequential"
        )

    def test_full_extraction_equivalence(self, device, test_trajectory):
        """Test full extraction pipeline matches sequential."""
        N, T, C, H, W = test_trajectory.shape

        # Sequential extractor
        seq_extractor = TemporalFeatureExtractor(device=device)

        # Batch extractor
        batch_extractor = BatchParallelTemporalExtractor(device=device)

        # Extract with sequential
        seq_features = []
        seq_extractor.reset()
        for t in range(T):
            u_t = test_trajectory[:, t]
            feat = seq_extractor.extract(u_t)
            seq_features.append(feat)
        seq_result = torch.stack(seq_features, dim=1)  # [N, T, D]

        # Extract with batch
        batch_result = batch_extractor.extract_batch(test_trajectory)

        # Compare dimensions
        assert batch_result.shape == seq_result.shape, \
            f"Shape mismatch: batch {batch_result.shape} vs seq {seq_result.shape}"

        # Compare values
        torch.testing.assert_close(
            batch_result, seq_result,
            rtol=1e-4, atol=1e-5,
            msg="Batch full extraction doesn't match sequential"
        )


class TestTemporalFeatureOrchestrator:
    """Test orchestrator with batch mode."""

    def test_orchestrator_batch_mode(self, device):
        """Test orchestrator uses batch mode correctly."""
        # Create test trajectory: [N=2, M=3, T=16, C=3, H=32, W=32]
        torch.manual_seed(42)
        trajectories = torch.randn(2, 3, 16, 3, 32, 32, device=device)

        # Sequential mode
        orchestrator_seq = TemporalFeatureOrchestrator(device=device, use_batch_mode=False)
        features_seq = orchestrator_seq.extract_per_timestep(trajectories)

        # Batch mode
        orchestrator_batch = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)
        features_batch = orchestrator_batch.extract_per_timestep(trajectories)

        # Compare
        assert features_batch.shape == features_seq.shape, \
            f"Shape mismatch: batch {features_batch.shape} vs seq {features_seq.shape}"

        # Compare values (temporal features should match)
        # Note: Allow slightly larger tolerance due to accumulation
        torch.testing.assert_close(
            features_batch, features_seq,
            rtol=1e-3, atol=1e-4,
            msg="Orchestrator batch mode doesn't match sequential"
        )

    def test_orchestrator_dimensions(self, device):
        """Test orchestrator produces correct dimensions."""
        # Create test trajectory
        N, M, T, C, H, W = 2, 3, 256, 3, 64, 64
        trajectories = torch.randn(N, M, T, C, H, W, device=device)

        # Extract with batch mode
        orchestrator = TemporalFeatureOrchestrator(device=device, use_batch_mode=True)
        features = orchestrator.extract_per_timestep(trajectories)

        # Check shape
        assert features.shape[0] == N, f"Batch size mismatch: {features.shape[0]} != {N}"
        assert features.shape[1] == T, f"Timesteps mismatch: {features.shape[1]} != {T}"
        assert features.shape[2] > 0, "Feature dimension is zero"

        print(f"Feature shape: {features.shape}")
        print(f"Expected: [N={N}, T={T}, D=~345]")


class TestPerformance:
    """Performance comparison tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_faster_than_sequential(self, device):
        """Test that batch mode is faster than sequential."""
        import time

        # Create larger trajectory for timing
        N, T, C, H, W = 4, 256, 3, 64, 64
        torch.manual_seed(42)
        trajectory = torch.randn(N, T, C, H, W, device=device)

        # Warm up GPU
        batch_extractor = BatchParallelTemporalExtractor(device=device)
        _ = batch_extractor.extract_batch(trajectory)
        torch.cuda.synchronize()

        # Time batch extraction
        start = time.time()
        for _ in range(3):
            _ = batch_extractor.extract_batch(trajectory)
        torch.cuda.synchronize()
        batch_time = (time.time() - start) / 3

        # Time sequential extraction
        seq_extractor = TemporalFeatureExtractor(device=device)
        start = time.time()
        for _ in range(3):
            seq_features = []
            seq_extractor.reset()
            for t in range(T):
                u_t = trajectory[:, t]
                feat = seq_extractor.extract(u_t)
                seq_features.append(feat)
            _ = torch.stack(seq_features, dim=1)
        torch.cuda.synchronize()
        seq_time = (time.time() - start) / 3

        print(f"\nPerformance comparison (N={N}, T={T}):")
        print(f"  Sequential: {seq_time:.3f}s")
        print(f"  Batch:      {batch_time:.3f}s")
        print(f"  Speedup:    {seq_time/batch_time:.2f}x")

        # Batch should be faster
        assert batch_time < seq_time, \
            f"Batch mode ({batch_time:.3f}s) not faster than sequential ({seq_time:.3f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
