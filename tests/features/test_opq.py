"""Tests for OPQ (Optimized Product Quantization) rotation."""

import numpy as np
import pytest

from spinlock.features.grouping.opq import compute_opq_rotation, OPQGrouper
from spinlock.features.grouping.models import GroupingConfig, ClusteringParams


class TestComputeOPQRotation:
    """Tests for the core OPQ algorithm."""

    def test_rotation_is_orthogonal(self):
        """R @ R.T should be approximately identity."""
        rng = np.random.RandomState(42)
        D, M, d_sub = 60, 10, 6
        X = rng.randn(500, D)

        R, mean = compute_opq_rotation(X, M=M, d_sub=d_sub, n_iter=5, n_codes=16)

        # Check orthogonality
        eye = R @ R.T
        np.testing.assert_allclose(eye, np.eye(D), atol=1e-6)

    def test_rotation_shape(self):
        """R should be [D, D], mean should be [D]."""
        rng = np.random.RandomState(42)
        D, M, d_sub = 30, 5, 6
        X = rng.randn(200, D)

        R, mean = compute_opq_rotation(X, M=M, d_sub=d_sub, n_iter=3, n_codes=8)

        assert R.shape == (D, D)
        assert mean.shape == (D,)

    def test_per_group_variance_equalization(self):
        """OPQ should reduce per-group variance imbalance."""
        rng = np.random.RandomState(42)
        D, M, d_sub = 60, 10, 6

        # Create data with highly imbalanced variance across groups
        # Group 0 has 100x variance vs group 9
        scales = np.linspace(10.0, 0.1, D)
        X = rng.randn(1000, D) * scales[None, :]

        # Measure pre-rotation imbalance
        group_vars_before = [
            X[:, g * d_sub:(g + 1) * d_sub].var()
            for g in range(M)
        ]
        ratio_before = max(group_vars_before) / max(min(group_vars_before), 1e-12)

        # Apply OPQ
        R, mean = compute_opq_rotation(
            X, M=M, d_sub=d_sub, n_iter=20, n_codes=32, random_state=42,
        )
        X_rot = (X - mean) @ R.T

        # Measure post-rotation imbalance
        group_vars_after = [
            X_rot[:, g * d_sub:(g + 1) * d_sub].var()
            for g in range(M)
        ]
        ratio_after = max(group_vars_after) / max(min(group_vars_after), 1e-12)

        # OPQ should significantly reduce the variance ratio
        assert ratio_after < ratio_before, (
            f"OPQ failed to reduce imbalance: {ratio_before:.1f} → {ratio_after:.1f}"
        )
        # With 20 iterations, should get well under 10x (typically < 3x)
        assert ratio_after < 10.0, (
            f"Post-OPQ variance ratio {ratio_after:.1f} still too high"
        )

    def test_deterministic_with_seed(self):
        """Same seed should produce identical rotations."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 30)

        R1, mean1 = compute_opq_rotation(X, M=5, d_sub=6, n_iter=3, random_state=7)
        R2, mean2 = compute_opq_rotation(X, M=5, d_sub=6, n_iter=3, random_state=7)

        np.testing.assert_array_equal(R1, R2)
        np.testing.assert_array_equal(mean1, mean2)

    def test_assertion_on_dimension_mismatch(self):
        """Should raise if D != M * d_sub."""
        X = np.random.randn(100, 31)
        with pytest.raises(AssertionError):
            compute_opq_rotation(X, M=5, d_sub=6)


class TestOPQGrouper:
    """Tests for the OPQGrouper wrapper."""

    def test_grouper_returns_correct_structure(self):
        """GroupingResult should have M groups with d_sub features each."""
        rng = np.random.RandomState(42)
        D, M, d_sub = 30, 5, 6
        X = rng.randn(200, D)
        names = [f"feat_{i}" for i in range(D)]

        config = GroupingConfig(
            method="opq",
            clustering=ClusteringParams(num_groups=M),
            random_seed=42,
        )
        grouper = OPQGrouper(config)
        result = grouper.group_features(X, names, num_groups=M)

        assert result.num_groups == M
        assert result.total_features == D
        assert result.linear_transform is not None
        assert len(result.groups) == M

        # Each group should have d_sub features
        for group in result.groups.values():
            assert group.size == d_sub

    def test_grouper_linear_transform(self):
        """LinearTransform should apply correctly."""
        rng = np.random.RandomState(42)
        D, M = 30, 5
        X = rng.randn(200, D)
        names = [f"feat_{i}" for i in range(D)]

        config = GroupingConfig(
            method="opq",
            clustering=ClusteringParams(num_groups=M),
            random_seed=42,
        )
        grouper = OPQGrouper(config)
        result = grouper.group_features(X, names, num_groups=M)

        # Apply transform
        X_rot = result.linear_transform.apply(X)
        assert X_rot.shape == X.shape

        # Transform should be orthogonal
        R = result.linear_transform.components
        eye = R @ R.T
        np.testing.assert_allclose(eye, np.eye(D), atol=1e-6)

    def test_grouper_dimension_error(self):
        """Should raise if D not divisible by num_groups."""
        X = np.random.randn(100, 31)
        names = [f"feat_{i}" for i in range(31)]

        config = GroupingConfig(method="opq", random_seed=42)
        grouper = OPQGrouper(config)

        with pytest.raises(ValueError, match="not divisible"):
            grouper.group_features(X, names, num_groups=5)
