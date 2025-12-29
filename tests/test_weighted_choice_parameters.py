"""Tests for weighted choice parameter sampling.

Verifies that:
1. Weighted choice parameters sample according to specified proportions
2. Backward compatibility: omitting weights gives uniform distribution
3. Validation catches errors (length mismatch, negative weights, sum != 1.0)
"""

import numpy as np
import pytest

from spinlock.config.schema import ChoiceParameter
from spinlock.operators.builder import OperatorBuilder


class TestWeightedChoiceSchema:
    """Test schema validation for weighted choice parameters."""

    def test_weights_optional(self):
        """Test that weights field is optional."""
        param = ChoiceParameter(choices=["a", "b", "c"])
        assert param.weights is None

    def test_weights_provided(self):
        """Test that weights can be provided."""
        param = ChoiceParameter(
            choices=["a", "b", "c"],
            weights=[0.5, 0.3, 0.2]
        )
        assert param.weights == [0.5, 0.3, 0.2]

    def test_weights_must_match_choices_length(self):
        """Test that weights length must match choices length."""
        with pytest.raises(ValueError, match="weights length.*must match"):
            ChoiceParameter(
                choices=["a", "b"],
                weights=[0.5, 0.3, 0.2]  # 3 weights for 2 choices
            )

    def test_weights_must_be_non_negative(self):
        """Test that all weights must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            ChoiceParameter(
                choices=["a", "b"],
                weights=[0.6, -0.4]  # Negative weight
            )

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ChoiceParameter(
                choices=["a", "b"],
                weights=[0.3, 0.5]  # Sum = 0.8, not 1.0
            )

    def test_weights_sum_tolerance(self):
        """Test that small floating point errors in sum are tolerated."""
        # This should pass (1e-6 tolerance)
        param = ChoiceParameter(
            choices=["a", "b"],
            weights=[0.6, 0.4 + 1e-7]  # Sum = 1.0 + 1e-7 (within tolerance)
        )
        assert param.weights == [0.6, 0.4 + 1e-7]


class TestWeightedChoiceSampling:
    """Test weighted choice parameter sampling behavior."""

    def test_uniform_sampling_when_no_weights(self):
        """Test that omitting weights gives uniform distribution."""
        builder = OperatorBuilder()

        spec = {
            "policy": {
                "type": "choice",
                "choices": ["a", "b", "c"]
                # weights omitted → uniform
            }
        }

        # Sample 9000 times with evenly spaced u values
        samples = []
        for i in range(9000):
            u = np.array([i / 9000.0])
            params = builder.map_parameters(u, spec)
            samples.append(params["policy"])

        # Each choice should be ~33.3% (within ±1%)
        for choice in ["a", "b", "c"]:
            count = samples.count(choice)
            expected = 3000
            # Allow ±1% tolerance (30 samples)
            assert abs(count - expected) <= 30, \
                f"Expected ~{expected} {choice}, got {count}"

    def test_weighted_sampling_two_choices(self):
        """Test weighted sampling with two choices (75/25 split)."""
        builder = OperatorBuilder()

        spec = {
            "update_policy": {
                "type": "choice",
                "choices": ["residual", "convex"],
                "weights": [0.75, 0.25]
            }
        }

        # Sample 10000 times
        samples = []
        for i in range(10000):
            u = np.array([i / 10000.0])
            params = builder.map_parameters(u, spec)
            samples.append(params["update_policy"])

        # Check proportions (±1% tolerance)
        residual_count = samples.count("residual")
        convex_count = samples.count("convex")

        assert 7400 <= residual_count <= 7600, \
            f"Expected ~7500 residual (75%), got {residual_count}"
        assert 2400 <= convex_count <= 2600, \
            f"Expected ~2500 convex (25%), got {convex_count}"

    def test_weighted_sampling_three_choices(self):
        """Test weighted sampling with three choices (50/30/20 split)."""
        builder = OperatorBuilder()

        spec = {
            "activation": {
                "type": "choice",
                "choices": ["relu", "gelu", "silu"],
                "weights": [0.5, 0.3, 0.2]
            }
        }

        # Sample 10000 times
        samples = []
        for i in range(10000):
            u = np.array([i / 10000.0])
            params = builder.map_parameters(u, spec)
            samples.append(params["activation"])

        # Check proportions (±1% tolerance)
        relu_count = samples.count("relu")
        gelu_count = samples.count("gelu")
        silu_count = samples.count("silu")

        assert 4900 <= relu_count <= 5100, \
            f"Expected ~5000 relu (50%), got {relu_count}"
        assert 2900 <= gelu_count <= 3100, \
            f"Expected ~3000 gelu (30%), got {gelu_count}"
        assert 1900 <= silu_count <= 2100, \
            f"Expected ~2000 silu (20%), got {silu_count}"

    def test_weighted_sampling_boundary_values(self):
        """Test weighted sampling behavior at u=0 and u≈1."""
        builder = OperatorBuilder()

        spec = {
            "policy": {
                "type": "choice",
                "choices": ["a", "b", "c"],
                "weights": [0.3, 0.5, 0.2]
            }
        }

        # u=0 should map to first choice
        params_0 = builder.map_parameters(np.array([0.0]), spec)
        assert params_0["policy"] == "a"

        # u just below first boundary (0.3)
        params_below = builder.map_parameters(np.array([0.29]), spec)
        assert params_below["policy"] == "a"

        # u at first boundary
        params_at = builder.map_parameters(np.array([0.3]), spec)
        assert params_at["policy"] == "b"

        # u in middle of second range
        params_mid = builder.map_parameters(np.array([0.5]), spec)
        assert params_mid["policy"] == "b"

        # u at second boundary (0.8)
        params_at2 = builder.map_parameters(np.array([0.8]), spec)
        assert params_at2["policy"] == "c"

        # u near 1.0 should map to last choice
        params_near_1 = builder.map_parameters(np.array([0.9999]), spec)
        assert params_near_1["policy"] == "c"

        # u=1.0 should be safely clamped to last choice
        params_1 = builder.map_parameters(np.array([1.0]), spec)
        assert params_1["policy"] == "c"

    def test_weighted_sampling_extreme_weights(self):
        """Test weighted sampling with extreme weight distributions."""
        builder = OperatorBuilder()

        # 95% first choice, 5% second choice
        spec = {
            "policy": {
                "type": "choice",
                "choices": ["dominant", "rare"],
                "weights": [0.95, 0.05]
            }
        }

        # Sample 10000 times
        samples = []
        for i in range(10000):
            u = np.array([i / 10000.0])
            params = builder.map_parameters(u, spec)
            samples.append(params["policy"])

        dominant_count = samples.count("dominant")
        rare_count = samples.count("rare")

        assert 9400 <= dominant_count <= 9600, \
            f"Expected ~9500 dominant (95%), got {dominant_count}"
        assert 400 <= rare_count <= 600, \
            f"Expected ~500 rare (5%), got {rare_count}"

    def test_multiple_weighted_parameters_independent(self):
        """Test that multiple weighted parameters are sampled independently."""
        builder = OperatorBuilder()

        spec = {
            "policy": {
                "type": "choice",
                "choices": ["a", "b"],
                "weights": [0.7, 0.3]
            },
            "activation": {
                "type": "choice",
                "choices": ["x", "y"],
                "weights": [0.4, 0.6]
            }
        }

        # Sample with different u values for each parameter
        u_values = np.array([0.2, 0.5])  # [policy, activation]
        params = builder.map_parameters(u_values, spec)

        # u=0.2 with weights [0.7, 0.3] → choice "a" (< 0.7 boundary)
        assert params["policy"] == "a"

        # u=0.5 with weights [0.4, 0.6] → choice "y" (≥ 0.4 boundary)
        assert params["activation"] == "y"


class TestBackwardCompatibility:
    """Test backward compatibility with existing configs."""

    def test_existing_configs_unchanged(self):
        """Test that existing configs without weights still work."""
        builder = OperatorBuilder()

        # Typical existing config
        spec = {
            "update_policy": {
                "type": "choice",
                "choices": ["autoregressive", "residual", "convex"]
            },
            "activation": {
                "type": "choice",
                "choices": ["relu", "gelu"]
            }
        }

        # Should sample without errors
        u_values = np.array([0.5, 0.7])
        params = builder.map_parameters(u_values, spec)

        assert params["update_policy"] in ["autoregressive", "residual", "convex"]
        assert params["activation"] in ["relu", "gelu"]

    def test_mixed_weighted_and_unweighted(self):
        """Test that weighted and unweighted parameters can coexist."""
        builder = OperatorBuilder()

        spec = {
            "policy": {
                "type": "choice",
                "choices": ["a", "b"],
                "weights": [0.8, 0.2]  # Weighted
            },
            "activation": {
                "type": "choice",
                "choices": ["x", "y"]
                # No weights → uniform
            }
        }

        # Sample many times
        samples = []
        for i in range(10000):
            u_values = np.array([i / 10000.0, (i * 7) % 10000 / 10000.0])
            params = builder.map_parameters(u_values, spec)
            samples.append(params)

        # Check weighted parameter
        policy_a_count = sum(1 for s in samples if s["policy"] == "a")
        assert 7900 <= policy_a_count <= 8100, \
            f"Expected ~8000 policy=a (80%), got {policy_a_count}"

        # Check unweighted parameter (should be ~50/50)
        activation_x_count = sum(1 for s in samples if s["activation"] == "x")
        assert 4900 <= activation_x_count <= 5100, \
            f"Expected ~5000 activation=x (50%), got {activation_x_count}"
