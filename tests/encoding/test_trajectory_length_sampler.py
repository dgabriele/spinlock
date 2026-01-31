"""
Unit tests for TrajectoryLengthSampler and LengthCurriculumScheduler.
"""

import pytest
import torch
import numpy as np

from spinlock.encoding.trajectory_length_sampler import (
    TrajectoryLengthSampler,
    LengthCurriculumScheduler,
)


class TestTrajectoryLengthSampler:
    """Tests for TrajectoryLengthSampler."""

    def test_uniform_sampling(self):
        """Test uniform sampling strategy."""
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)

        lengths = sampler.sample_lengths(batch_size=1000)

        assert lengths.shape == (1000,)
        assert (lengths >= 8).all()
        assert (lengths <= 500).all()

        # Check distribution is roughly uniform
        mean = lengths.float().mean().item()
        assert 200 < mean < 300  # Should be around 254

    def test_geometric_sampling(self):
        """Test geometric sampling with decay."""
        sampler = TrajectoryLengthSampler(
            min_timesteps=8,
            max_timesteps=500,
            strategy="geometric",
            geometric_decay=0.02,
        )

        lengths = sampler.sample_lengths(batch_size=1000)

        assert lengths.shape == (1000,)
        assert (lengths >= 8).all()
        assert (lengths <= 500).all()

        # With positive decay, should favor shorter trajectories
        mean = lengths.float().mean().item()
        assert mean < 254  # Should be less than uniform mean

    def test_fixed_bins_sampling(self):
        """Test fixed bins sampling strategy."""
        bins = [100, 200, 300, 400, 500]
        sampler = TrajectoryLengthSampler(
            min_timesteps=100,
            max_timesteps=500,
            strategy="fixed_bins",
            length_bins=bins,
        )

        lengths = sampler.sample_lengths(batch_size=1000)

        assert lengths.shape == (1000,)

        # All lengths should be in bins
        for length in lengths:
            assert length.item() in bins

    def test_create_mask(self):
        """Test mask creation from lengths."""
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)

        lengths = torch.tensor([10, 20, 15])
        mask = sampler.create_mask(lengths, max_T=25)

        assert mask.shape == (3, 25)

        # Check first sample (length=10)
        assert mask[0, :10].all()
        assert not mask[0, 10:].any()

        # Check second sample (length=20)
        assert mask[1, :20].all()
        assert not mask[1, 20:].any()

        # Check third sample (length=15)
        assert mask[2, :15].all()
        assert not mask[2, 15:].any()

    def test_create_mask_auto_max_T(self):
        """Test mask creation with automatic max_T."""
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)

        lengths = torch.tensor([10, 20, 15])
        mask = sampler.create_mask(lengths)  # No max_T specified

        # Should auto-detect max_T = 20
        assert mask.shape == (3, 20)

        assert mask[0, :10].all()
        assert not mask[0, 10:].any()
        assert mask[1, :20].all()
        assert mask[2, :15].all()
        assert not mask[2, 15:].any()

    def test_validation_min_timesteps(self):
        """Test validation of min_timesteps."""
        with pytest.raises(ValueError, match="min_timesteps must be >= 1"):
            TrajectoryLengthSampler(min_timesteps=0, max_timesteps=500)

        with pytest.raises(ValueError, match="min_timesteps must be >= 1"):
            TrajectoryLengthSampler(min_timesteps=-5, max_timesteps=500)

    def test_validation_max_timesteps(self):
        """Test validation of max_timesteps."""
        with pytest.raises(ValueError, match="must be < max_timesteps"):
            TrajectoryLengthSampler(min_timesteps=500, max_timesteps=500)

        with pytest.raises(ValueError, match="must be < max_timesteps"):
            TrajectoryLengthSampler(min_timesteps=600, max_timesteps=500)

    def test_validation_geometric_decay(self):
        """Test validation of geometric_decay."""
        with pytest.raises(ValueError, match="geometric_decay must be > 0"):
            TrajectoryLengthSampler(
                min_timesteps=8,
                max_timesteps=500,
                strategy="geometric",
                geometric_decay=0,
            )

        with pytest.raises(ValueError, match="geometric_decay must be > 0"):
            TrajectoryLengthSampler(
                min_timesteps=8,
                max_timesteps=500,
                strategy="geometric",
                geometric_decay=-0.01,
            )

    def test_validation_fixed_bins(self):
        """Test validation of fixed_bins."""
        with pytest.raises(ValueError, match="requires 'length_bins'"):
            TrajectoryLengthSampler(
                min_timesteps=100,
                max_timesteps=500,
                strategy="fixed_bins",
            )

        with pytest.raises(ValueError, match="cannot be empty"):
            TrajectoryLengthSampler(
                min_timesteps=100,
                max_timesteps=500,
                strategy="fixed_bins",
                length_bins=[],
            )

        with pytest.raises(ValueError, match="outside valid range"):
            TrajectoryLengthSampler(
                min_timesteps=100,
                max_timesteps=500,
                strategy="fixed_bins",
                length_bins=[50, 100, 200],  # 50 < min_timesteps
            )

    def test_get_stats(self):
        """Test distribution statistics."""
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)

        stats = sampler.get_stats(num_samples=10000)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats

        # Check ranges
        assert stats["min"] >= 8
        assert stats["max"] <= 500
        assert 200 < stats["mean"] < 300  # Roughly uniform

    def test_repr(self):
        """Test string representation."""
        sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)
        repr_str = repr(sampler)

        assert "TrajectoryLengthSampler" in repr_str
        assert "min_T=8" in repr_str
        assert "max_T=500" in repr_str
        assert "strategy='uniform'" in repr_str


class TestLengthCurriculumScheduler:
    """Tests for LengthCurriculumScheduler."""

    def test_linear_schedule(self):
        """Test linear curriculum schedule."""
        scheduler = LengthCurriculumScheduler(
            start_min=400, end_min=8, total_epochs=200, schedule="linear"
        )

        # Epoch 0: start
        assert scheduler.get_min_timesteps(0) == 400

        # Epoch 100: halfway
        mid = scheduler.get_min_timesteps(100)
        assert 200 <= mid <= 210  # Should be around 204

        # Epoch 200: end
        assert scheduler.get_min_timesteps(200) == 8

        # Beyond end: stay at end
        assert scheduler.get_min_timesteps(300) == 8

    def test_cosine_schedule(self):
        """Test cosine curriculum schedule."""
        scheduler = LengthCurriculumScheduler(
            start_min=400, end_min=8, total_epochs=200, schedule="cosine"
        )

        # Epoch 0: start
        assert scheduler.get_min_timesteps(0) == 400

        # Cosine should have slower initial decay
        quarter = scheduler.get_min_timesteps(50)
        assert quarter > 300  # Still close to start

        # Epoch 200: end
        assert scheduler.get_min_timesteps(200) == 8

    def test_exponential_schedule(self):
        """Test exponential curriculum schedule."""
        scheduler = LengthCurriculumScheduler(
            start_min=400, end_min=8, total_epochs=200, schedule="exponential"
        )

        # Epoch 0: start
        assert scheduler.get_min_timesteps(0) == 400

        # Exponential should have faster initial decay
        quarter = scheduler.get_min_timesteps(50)
        assert quarter < 300  # Significant decay already

        # Epoch 200: end
        assert scheduler.get_min_timesteps(200) == 8

    def test_validation_start_end(self):
        """Test validation of start and end values."""
        with pytest.raises(ValueError, match="must be > end_min"):
            LengthCurriculumScheduler(start_min=8, end_min=400, total_epochs=200)

        with pytest.raises(ValueError, match="must be > end_min"):
            LengthCurriculumScheduler(start_min=100, end_min=100, total_epochs=200)

    def test_validation_total_epochs(self):
        """Test validation of total_epochs."""
        with pytest.raises(ValueError, match="must be > 0"):
            LengthCurriculumScheduler(start_min=400, end_min=8, total_epochs=0)

        with pytest.raises(ValueError, match="must be > 0"):
            LengthCurriculumScheduler(start_min=400, end_min=8, total_epochs=-10)

    def test_repr(self):
        """Test string representation."""
        scheduler = LengthCurriculumScheduler(
            start_min=400, end_min=8, total_epochs=200
        )
        repr_str = repr(scheduler)

        assert "LengthCurriculumScheduler" in repr_str
        assert "400" in repr_str
        assert "8" in repr_str
        assert "200" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
