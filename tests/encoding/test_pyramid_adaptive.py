"""
Unit tests for adaptive pyramid level selection in TemporalPyramid.
"""

import pytest
import torch

from spinlock.encoding.temporal_pyramid import TemporalPyramid


class TestAdaptivePyramidLevels:
    """Tests for dynamic pyramid level selection."""

    def test_get_valid_levels_all_valid(self):
        """Test when all pyramid levels are valid."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True, min_pyramid_length=1)

        # T=500: All levels valid
        assert pyramid.get_valid_levels(500) == [1, 2, 4, 8]

        # T=8: All levels valid (minimum for all)
        assert pyramid.get_valid_levels(8) == [1, 2, 4, 8]

    def test_get_valid_levels_skip_some(self):
        """Test when some pyramid levels must be skipped."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True, min_pyramid_length=1)

        # T=4: Skip 8× (would give T=0.5)
        assert pyramid.get_valid_levels(4) == [1, 2, 4]

        # T=2: Skip 4× and 8×
        assert pyramid.get_valid_levels(2) == [1, 2]

        # T=1: Only full resolution
        assert pyramid.get_valid_levels(1) == [1]

    def test_get_valid_levels_min_pyramid_length(self):
        """Test with min_pyramid_length > 1."""
        pyramid = TemporalPyramid([1, 2, 4], adaptive=True, min_pyramid_length=2)

        # T=8: All valid (8/4=2)
        assert pyramid.get_valid_levels(8) == [1, 2, 4]

        # T=4: Skip 4× (4/4=1 < 2)
        assert pyramid.get_valid_levels(4) == [1, 2]

        # T=2: Skip 2× and 4×
        assert pyramid.get_valid_levels(2) == [1]

    def test_get_valid_levels_non_adaptive(self):
        """Test that non-adaptive mode returns all levels."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=False)

        # Should always return all levels regardless of T
        assert pyramid.get_valid_levels(4) == [1, 2, 4, 8]
        assert pyramid.get_valid_levels(2) == [1, 2, 4, 8]
        assert pyramid.get_valid_levels(1) == [1, 2, 4, 8]

    def test_forward_with_adaptive_long_trajectory(self):
        """Test forward pass with long trajectory (all levels valid)."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True)

        x = torch.randn(16, 256, 64)  # [B, T, D]
        lengths = torch.full((16,), 256)

        levels, _, valid_factors = pyramid(x, lengths=lengths)

        # All levels should be valid
        assert len(levels) == 4
        assert valid_factors == [1, 2, 4, 8]

        # Check shapes
        assert levels[0].shape == (16, 256, 64)  # 1×
        assert levels[1].shape == (16, 128, 64)  # 2×
        assert levels[2].shape == (16, 64, 64)   # 4×
        assert levels[3].shape == (16, 32, 64)   # 8×

    def test_forward_with_adaptive_short_trajectory(self):
        """Test forward pass with short trajectory (some levels skipped)."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=True)

        x = torch.randn(16, 4, 64)  # [B, T=4, D]
        lengths = torch.full((16,), 4)

        levels, _, valid_factors = pyramid(x, lengths=lengths)

        # Should skip 8× level
        assert len(levels) == 3
        assert valid_factors == [1, 2, 4]

        # Check shapes
        assert levels[0].shape == (16, 4, 64)  # 1×
        assert levels[1].shape == (16, 2, 64)  # 2×
        assert levels[2].shape == (16, 1, 64)  # 4×

    def test_forward_with_mask(self):
        """Test forward pass with mask propagation."""
        pyramid = TemporalPyramid([1, 2, 4], adaptive=True, mask_downsample_method="ceil")

        x = torch.randn(16, 8, 64)
        mask = torch.ones(16, 8, dtype=torch.bool)
        mask[:, 6:] = False  # Mask last 2 timesteps

        levels, level_masks, valid_factors = pyramid(x, mask=mask)

        assert len(levels) == 3
        assert len(level_masks) == 3
        assert valid_factors == [1, 2, 4]

        # Check mask shapes
        assert level_masks[0].shape == (16, 8)  # 1×
        assert level_masks[1].shape == (16, 4)  # 2×
        assert level_masks[2].shape == (16, 2)  # 4×

        # Check mask values (ceil method)
        # Original: [T,T,T,T,T,T,F,F]
        # Level 0 (1×): [T,T,T,T,T,T,F,F]
        assert level_masks[0][:, :6].all()
        assert not level_masks[0][:, 6:].any()

        # Level 1 (2×): [T,T,T,T] (each covers 2 timesteps)
        # Last position covers timesteps [6,7], should be F with ceil
        # Actually with ceil, if ANY source is valid, target is valid
        # Position 3 covers [6,7] which are both False, so should be False
        assert level_masks[1][:, :3].all()

    def test_mask_downsample_ceil_vs_floor(self):
        """Test difference between ceil and floor mask downsampling."""
        x = torch.randn(4, 8, 64)
        mask = torch.tensor([
            [True, True, True, True, False, False, False, False],
            [True, True, True, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
        ], dtype=torch.bool)

        # Test ceil method (conservative)
        pyramid_ceil = TemporalPyramid([1, 2], adaptive=True, mask_downsample_method="ceil")
        _, masks_ceil, _ = pyramid_ceil(x, mask=mask)

        # Test floor method (aggressive)
        pyramid_floor = TemporalPyramid([1, 2], adaptive=True, mask_downsample_method="floor")
        _, masks_floor, _ = pyramid_floor(x, mask=mask)

        # Level 1 (2×): Each position covers 2 source timesteps
        # Ceil: valid if ANY source valid
        # Floor: valid if ALL sources valid

        # Sample 0: [TT|TT|FF|FF]
        # Ceil 2×: [T, T, F, F] (both sources valid for pos 0,1)
        # Floor 2×: [T, T, F, F] (same as ceil for this case)

        # Sample 1: [TT|TF|FF|FF]
        # Ceil 2×: [T, T, F, F] (position 1 has one T, one F → T with ceil)
        # Floor 2×: [T, F, F, F] (position 1 has one T, one F → F with floor)

        assert masks_ceil[1][1, 1] == True   # Ceil: ANY valid
        assert masks_floor[1][1, 1] == False  # Floor: ALL valid

    def test_backward_compatibility(self):
        """Test backward compatibility without adaptive mode."""
        pyramid = TemporalPyramid([1, 2, 4, 8], adaptive=False)

        x = torch.randn(16, 256, 64)

        # Old signature: just input
        levels = pyramid(x)[0]  # Returns (levels, None, factors)

        assert len(levels) == 4

    def test_validation_mask_downsample_method(self):
        """Test validation of mask_downsample_method."""
        with pytest.raises(ValueError, match="must be 'ceil' or 'floor'"):
            TemporalPyramid([1, 2, 4], mask_downsample_method="invalid")

    def test_validation_min_pyramid_length(self):
        """Test validation of min_pyramid_length."""
        with pytest.raises(ValueError, match="must be >= 1"):
            TemporalPyramid([1, 2, 4], min_pyramid_length=0)

        with pytest.raises(ValueError, match="must be >= 1"):
            TemporalPyramid([1, 2, 4], min_pyramid_length=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
