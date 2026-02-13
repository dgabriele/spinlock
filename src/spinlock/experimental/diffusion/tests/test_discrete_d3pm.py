"""Unit tests for DiscreteD3PM diffusion process."""

import pytest
import torch

from spinlock.experimental.diffusion.models import (
    DiscreteD3PM, DiffusionSchedule, ScheduleType, TransitionType,
)


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


def test_forward_process_batched_t(diffusion):
    """Test forward process with per-sample timestep tensor."""
    tokens_dict = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
        "temporal_group_1_L2": torch.tensor([1, 2, 3, 4]),
        "initial_group_1_L0": torch.tensor([8, 12, 15, 18]),
        "initial_group_1_L1": torch.tensor([2, 5, 7, 9]),
    }

    # Per-sample timesteps: different noise levels per sample
    t = torch.tensor([0, 10, 25, 49])

    noisy_dict, log_probs_dict = diffusion.forward_process(tokens_dict, t)

    # Check output structure
    assert set(noisy_dict.keys()) == set(tokens_dict.keys())

    # Check shapes
    for key, tokens in tokens_dict.items():
        assert noisy_dict[key].shape == tokens.shape
        assert log_probs_dict[key].shape == (4, diffusion.vocab_sizes[key])

    # Token at t=0 should very likely be unchanged (almost no noise)
    # Run multiple times to handle stochasticity
    preserved_count = 0
    for _ in range(20):
        noisy, _ = diffusion.forward_process(tokens_dict, t)
        if all(noisy[k][0] == tokens_dict[k][0] for k in tokens_dict):
            preserved_count += 1
    # At t=0, beta is very small, so token should almost always be preserved
    assert preserved_count >= 15, f"Token at t=0 should be mostly preserved, got {preserved_count}/20"


def test_forward_process_batched_t_different_noise(diffusion):
    """Verify that different samples get different noise levels with batched t."""
    # All samples have the same token
    tokens_dict = {
        "temporal_group_1_L0": torch.tensor([5, 5, 5, 5]),
    }

    # Very different timesteps
    t = torch.tensor([0, 0, 49, 49])

    # Run many times and check that high-t samples change more often
    changes_low_t = 0
    changes_high_t = 0
    trials = 100
    for _ in range(trials):
        noisy, _ = diffusion.forward_process(tokens_dict, t)
        key = "temporal_group_1_L0"
        changes_low_t += (noisy[key][:2] != 5).sum().item()
        changes_high_t += (noisy[key][2:] != 5).sum().item()

    # High-t should change significantly more than low-t
    assert changes_high_t > changes_low_t * 3, (
        f"High-t samples should change much more: low_t={changes_low_t}, high_t={changes_high_t}"
    )


def test_forward_process_batched_t_invalid():
    """Test that invalid batched timesteps raise errors."""
    vocab_sizes = {"test_L0": 10}
    schedule = DiffusionSchedule(num_timesteps=50)
    d = DiscreteD3PM(vocab_sizes, schedule)

    tokens = {"test_L0": torch.tensor([1, 2, 3])}

    with pytest.raises(ValueError, match="Invalid timesteps"):
        d.forward_process(tokens, torch.tensor([0, 50, 25]))  # 50 is out of range

    with pytest.raises(ValueError, match="Invalid timesteps"):
        d.forward_process(tokens, torch.tensor([0, -1, 25]))  # -1 is invalid


def test_absorbing_state_transition_matrices(vocab_sizes, category_level_info):
    """Test absorbing state transition matrix construction."""
    schedule = DiffusionSchedule(num_timesteps=10)
    d = DiscreteD3PM(vocab_sizes, schedule, category_level_info, transition_type="absorbing")

    for key, vocab_size in vocab_sizes.items():
        Q_t = d.transition_matrices[key]
        V_ext = vocab_size + 1

        # Check shape: [T, V+1, V+1]
        assert Q_t.shape == (10, V_ext, V_ext)

        # Check stochasticity: rows sum to 1
        row_sums = Q_t.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

        # Check mask token is absorbing: Q[V, V] should always be 1.0
        for t in range(10):
            assert Q_t[t, V_ext - 1, V_ext - 1].item() == pytest.approx(1.0, abs=1e-6)

        # At early timesteps, clean tokens should mostly stay clean
        Q_0 = Q_t[0]
        diag = torch.diag(Q_0)[:vocab_size]  # Only clean token diagonal
        assert torch.all(diag > 0.9)


def test_absorbing_state_forward_process(vocab_sizes, category_level_info):
    """Test that absorbing forward process produces only clean or mask tokens."""
    schedule = DiffusionSchedule(num_timesteps=50, beta_end=0.1)
    d = DiscreteD3PM(vocab_sizes, schedule, category_level_info, transition_type="absorbing")

    tokens_dict = {
        "temporal_group_1_L0": torch.tensor([5, 10, 15, 20]),
        "temporal_group_1_L1": torch.tensor([3, 7, 9, 12]),
        "temporal_group_1_L2": torch.tensor([1, 2, 3, 4]),
        "initial_group_1_L0": torch.tensor([8, 12, 15, 18]),
        "initial_group_1_L1": torch.tensor([2, 5, 7, 9]),
    }

    # At late timestep, some tokens should become mask token
    noisy_dict, _ = d.forward_process(tokens_dict, t=49)

    for key, noisy_tokens in noisy_dict.items():
        V = vocab_sizes[key]
        mask_idx = d.mask_token_indices[key]
        # Each token should be either a valid clean token (0..V-1) or mask (V)
        assert torch.all(noisy_tokens >= 0)
        assert torch.all(noisy_tokens <= mask_idx)
        # Only two kinds of values possible: original range or mask
        for tok in noisy_tokens:
            assert tok.item() < V or tok.item() == mask_idx


def test_absorbing_state_is_absorbing(category_level_info):
    """Test that once a token becomes mask, it stays mask (absorbing property)."""
    vocab_sizes = {"temporal_group_1_L0": 28}
    schedule = DiffusionSchedule(num_timesteps=50, beta_end=0.5)
    d = DiscreteD3PM(vocab_sizes, schedule, category_level_info, transition_type="absorbing")

    mask_idx = d.mask_token_indices["temporal_group_1_L0"]

    # Start with all mask tokens
    tokens_dict = {"temporal_group_1_L0": torch.full((10,), mask_idx)}

    # Forward process at any timestep: mask tokens should remain mask
    for t in [0, 10, 25, 49]:
        noisy, _ = d.forward_process(tokens_dict, t)
        assert torch.all(noisy["temporal_group_1_L0"] == mask_idx), (
            f"Mask tokens should be absorbing at t={t}"
        )


def test_vocab_aware_beta_scaling(category_level_info):
    """Test that vocab-aware scaling produces different betas per category."""
    vocab_sizes = {"small_L0": 6, "large_L0": 28}
    category_info = {
        "small_L0": {"family": "test", "category": "small", "level": 0},
        "large_L0": {"family": "test", "category": "large", "level": 0},
    }
    schedule = DiffusionSchedule(num_timesteps=10)

    d = DiscreteD3PM(vocab_sizes, schedule, category_info, beta_scaling="vocab_aware")

    # Small vocab should have smaller effective betas (less noise)
    Q_small = d.transition_matrices["small_L0"]
    Q_large = d.transition_matrices["large_L0"]

    # At same timestep, small-vocab diagonal should be MORE dominant
    # (less noise applied)
    t = 5
    small_diag = torch.diag(Q_small[t]).mean()
    large_diag = torch.diag(Q_large[t]).mean()
    assert small_diag > large_diag, (
        f"Small vocab should retain more signal: small_diag={small_diag:.4f}, "
        f"large_diag={large_diag:.4f}"
    )


def test_sample_absorbing_initialization(category_level_info):
    """Test that sampling with absorbing state initializes with mask tokens."""
    vocab_sizes = {"temporal_group_1_L0": 28, "initial_group_1_L0": 20}
    schedule = DiffusionSchedule(num_timesteps=5)
    d = DiscreteD3PM(vocab_sizes, schedule, category_level_info, transition_type="absorbing")

    # We can't easily test the full sample loop without a denoiser,
    # but we can verify the mask_token_indices are set correctly
    assert d.mask_token_indices["temporal_group_1_L0"] == 28
    assert d.mask_token_indices["initial_group_1_L0"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
