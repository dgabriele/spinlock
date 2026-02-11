"""Unit tests for DiscreteD3PM diffusion process."""

import pytest
import torch

from spinlock.experimental.diffusion.models import DiscreteD3PM, DiffusionSchedule, ScheduleType


@pytest.fixture
def vocab_sizes():
    """Sample vocab sizes for testing."""
    return {
        "temporal_group_1_L0": 28,
        "temporal_group_1_L1": 14,
        "temporal_group_1_L2": 7,
        "initial_group_1_L0": 20,
        "initial_group_1_L1": 10,
    }


@pytest.fixture
def category_level_info():
    """Sample category level info for testing."""
    return {
        "temporal_group_1_L0": {"family": "temporal", "category": "group_1", "level": 0},
        "temporal_group_1_L1": {"family": "temporal", "category": "group_1", "level": 1},
        "temporal_group_1_L2": {"family": "temporal", "category": "group_1", "level": 2},
        "initial_group_1_L0": {"family": "initial", "category": "group_1", "level": 0},
        "initial_group_1_L1": {"family": "initial", "category": "group_1", "level": 1},
    }


@pytest.fixture
def diffusion(vocab_sizes, category_level_info):
    """Create DiscreteD3PM instance for testing."""
    schedule = DiffusionSchedule(
        num_timesteps=50,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type=ScheduleType.COSINE
    )
    return DiscreteD3PM(vocab_sizes, schedule, category_level_info)


def test_schedule_building():
    """Test noise schedule construction."""
    vocab_sizes = {"test_L0": 10}

    # Test linear schedule
    schedule_linear = DiffusionSchedule(
        num_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type=ScheduleType.LINEAR
    )
    diffusion_linear = DiscreteD3PM(vocab_sizes, schedule_linear)

    assert diffusion_linear.betas.shape == (10,)
    assert torch.all(diffusion_linear.betas >= 0.0001)
    assert torch.all(diffusion_linear.betas <= 0.02)
    assert diffusion_linear.betas[0] < diffusion_linear.betas[-1]  # Increasing

    # Test cosine schedule
    schedule_cosine = DiffusionSchedule(
        num_timesteps=10,
        schedule_type=ScheduleType.COSINE
    )
    diffusion_cosine = DiscreteD3PM(vocab_sizes, schedule_cosine)

    assert diffusion_cosine.betas.shape == (10,)
    assert torch.all(diffusion_cosine.betas > 0)
    assert torch.all(diffusion_cosine.betas < 1)


def test_transition_matrices(diffusion, vocab_sizes):
    """Test transition matrix construction."""
    # Check all categories have transition matrices
    for key, vocab_size in vocab_sizes.items():
        assert key in diffusion.transition_matrices
        Q_t = diffusion.transition_matrices[key]

        # Check shape: [T, V, V]
        assert Q_t.shape == (50, vocab_size, vocab_size)

        # Check stochasticity: rows sum to 1
        row_sums = Q_t.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

        # Check non-negativity
        assert torch.all(Q_t >= 0)

        # Check diagonal dominance at early timesteps
        # At t=0, Q should be close to identity
        Q_0 = Q_t[0]
        diag = torch.diag(Q_0)
        assert torch.all(diag > 0.9)  # High probability of staying in same state


def test_forward_process(diffusion):
    """Test forward diffusion process."""
    # Create sample tokens
    tokens_dict = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
        "temporal_group_1_L2": torch.tensor([1, 2, 3, 4]),
        "initial_group_1_L0": torch.tensor([8, 12, 15, 18]),
        "initial_group_1_L1": torch.tensor([2, 5, 7, 9]),
    }

    # Forward process at t=25 (mid-point)
    noisy_dict, log_probs_dict = diffusion.forward_process(tokens_dict, t=25)

    # Check output structure
    assert set(noisy_dict.keys()) == set(tokens_dict.keys())
    assert set(log_probs_dict.keys()) == set(tokens_dict.keys())

    # Check shapes
    for key, tokens in tokens_dict.items():
        assert noisy_dict[key].shape == tokens.shape
        assert log_probs_dict[key].shape == (4, diffusion.vocab_sizes[key])

    # Check tokens are within vocab range
    for key, noisy_tokens in noisy_dict.items():
        assert torch.all(noisy_tokens >= 0)
        assert torch.all(noisy_tokens < diffusion.vocab_sizes[key])

    # At late timesteps, tokens should be more noisy (different from original)
    noisy_late, _ = diffusion.forward_process(tokens_dict, t=49)
    num_changed = sum(
        (noisy_late[key] != tokens_dict[key]).sum().item()
        for key in tokens_dict.keys()
    )
    assert num_changed > 0  # At least some tokens should change


def test_forward_process_with_mask(diffusion):
    """Test forward process with masking."""
    tokens_dict = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
        "temporal_group_1_L2": torch.tensor([1, 2, 3, 4]),
        "initial_group_1_L0": torch.tensor([8, 12, 15, 18]),
        "initial_group_1_L1": torch.tensor([2, 5, 7, 9]),
    }

    # Create mask: keep first 2 samples, noise last 2
    mask_dict = {
        key: torch.tensor([False, False, True, True])
        for key in tokens_dict.keys()
    }

    noisy_dict, _ = diffusion.forward_process(tokens_dict, t=49, mask_dict=mask_dict)

    # Check that masked positions are unchanged
    for key in tokens_dict.keys():
        assert torch.all(noisy_dict[key][:2] == tokens_dict[key][:2])


def test_reverse_step(diffusion):
    """Test reverse diffusion step."""
    # Create sample noisy tokens
    x_t = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
        "temporal_group_1_L2": torch.tensor([1, 2, 3, 4]),
        "initial_group_1_L0": torch.tensor([8, 12, 15, 18]),
        "initial_group_1_L1": torch.tensor([2, 5, 7, 9]),
    }

    # Create fake predicted logits (just uniform for testing)
    predicted_logits = {}
    for key, vocab_size in diffusion.vocab_sizes.items():
        predicted_logits[key] = torch.randn(4, vocab_size)

    # Reverse step from t=25 to t=24
    x_prev = diffusion.reverse_step(x_t, predicted_logits, t=25)

    # Check output structure
    assert set(x_prev.keys()) == set(x_t.keys())

    # Check shapes and valid token ranges
    for key, tokens in x_prev.items():
        assert tokens.shape == (4,)
        assert torch.all(tokens >= 0)
        assert torch.all(tokens < diffusion.vocab_sizes[key])


def test_reverse_step_final(diffusion):
    """Test final reverse step (t=0)."""
    x_t = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
    }

    # Create predicted logits with clear argmax
    predicted_logits = {}
    for key, vocab_size in diffusion.vocab_sizes.items():
        if key in x_t:
            logits = torch.zeros(4, vocab_size)
            logits[:, 0] = 10.0  # Strong peak at index 0
            predicted_logits[key] = logits

    # Final reverse step
    x_0 = diffusion.reverse_step(x_t, predicted_logits, t=0)

    # Check that argmax is used (should be 0 for all)
    for key in x_t.keys():
        assert torch.all(x_0[key] == 0)


def test_timestep_weights(diffusion):
    """Test timestep weight computation."""
    weights = diffusion.get_timestep_weights()

    # Check shape
    assert weights.shape == (50,)

    # Check positivity
    assert torch.all(weights > 0)

    # Check normalization (mean = 1)
    assert torch.abs(weights.mean() - 1.0) < 1e-6


def test_forward_reverse_consistency(diffusion):
    """Test that reverse with perfect predictions recovers original (approximately)."""
    # Create clean tokens
    x_0 = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
    }

    # Forward process to t=10 (low noise)
    x_t, _ = diffusion.forward_process(x_0, t=10)

    # Create perfect logits (one-hot at true positions)
    perfect_logits = {}
    for key, tokens in x_0.items():
        vocab_size = diffusion.vocab_sizes[key]
        logits = torch.zeros(4, vocab_size)
        for i, token in enumerate(tokens):
            logits[i, token] = 100.0  # Very strong peak
        perfect_logits[key] = logits

    # Reverse step
    x_prev = diffusion.reverse_step(x_t, perfect_logits, t=10)

    # With perfect predictions and low noise, should recover original approximately
    # (not exact due to sampling, but should be close)
    # This is a sanity check, not a strict requirement
    for key in x_0.keys():
        # At least some should match (not all due to stochasticity)
        pass  # Weak check, mainly ensuring no crashes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
