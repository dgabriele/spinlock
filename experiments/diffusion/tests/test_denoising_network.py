"""Unit tests for DenoisingNetwork."""

import pytest
import torch

from experiments.diffusion.models import DenoisingNetwork


@pytest.fixture
def vocab_sizes():
    """Sample vocab sizes with variable sizes per level."""
    return {
        "temporal_group_1_L0": 28,
        "temporal_group_1_L1": 14,
        "temporal_group_1_L2": 7,
        "initial_group_1_L0": 20,
        "initial_group_1_L1": 10,
    }


@pytest.fixture
def category_level_info():
    """Sample category level info."""
    return {
        "temporal_group_1_L0": {"family": "temporal", "category": "group_1", "level": 0},
        "temporal_group_1_L1": {"family": "temporal", "category": "group_1", "level": 1},
        "temporal_group_1_L2": {"family": "temporal", "category": "group_1", "level": 2},
        "initial_group_1_L0": {"family": "initial", "category": "group_1", "level": 0},
        "initial_group_1_L1": {"family": "initial", "category": "group_1", "level": 1},
    }


@pytest.fixture
def denoiser(vocab_sizes, category_level_info):
    """Create DenoisingNetwork instance."""
    return DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_hierarchical_guidance=True,
    )


def test_initialization(denoiser, vocab_sizes):
    """Test denoiser initialization."""
    assert denoiser.hidden_dim == 128
    assert denoiser.num_tokens == 5  # 5 category-levels
    assert len(denoiser.sorted_keys) == 5

    # Check embeddings created for each category-level
    for key, vocab_size in vocab_sizes.items():
        assert key in denoiser.token_embeddings
        assert denoiser.token_embeddings[key].num_embeddings == vocab_size
        assert denoiser.token_embeddings[key].embedding_dim == 128

    # Check output heads
    for key, vocab_size in vocab_sizes.items():
        assert key in denoiser.output_heads
        assert denoiser.output_heads[key].out_features == vocab_size


def test_forward_pass(denoiser, vocab_sizes):
    """Test forward pass through denoiser."""
    batch_size = 4

    # Create sample tokens
    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),
    }

    # Create timesteps
    timesteps = torch.randint(0, 50, (batch_size,))

    # Forward pass
    logits_dict = denoiser(tokens_dict, timesteps)

    # Check output structure
    assert set(logits_dict.keys()) == set(vocab_sizes.keys())

    # Check shapes and valid logit ranges
    for key, vocab_size in vocab_sizes.items():
        logits = logits_dict[key]
        assert logits.shape == (batch_size, vocab_size)
        assert torch.all(torch.isfinite(logits))


def test_variable_vocab_sizes(denoiser):
    """Test handling of variable vocab sizes per category-level."""
    batch_size = 4

    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),  # 28 tokens
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),  # 14 tokens
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),   # 7 tokens
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),   # 20 tokens
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),   # 10 tokens
    }
    timesteps = torch.randint(0, 50, (batch_size,))

    logits_dict = denoiser(tokens_dict, timesteps)

    # Verify correct vocab sizes in output
    assert logits_dict["temporal_group_1_L0"].shape == (batch_size, 28)
    assert logits_dict["temporal_group_1_L1"].shape == (batch_size, 14)
    assert logits_dict["temporal_group_1_L2"].shape == (batch_size, 7)
    assert logits_dict["initial_group_1_L0"].shape == (batch_size, 20)
    assert logits_dict["initial_group_1_L1"].shape == (batch_size, 10)


def test_hierarchical_guidance(vocab_sizes, category_level_info):
    """Test hierarchical guidance from L0 tokens."""
    # Create denoiser with guidance enabled
    denoiser_with = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=True,
    )

    # Create denoiser without guidance
    denoiser_without = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=False,
    )

    batch_size = 4
    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),
    }
    timesteps = torch.randint(0, 50, (batch_size,))

    # Both should run without errors
    logits_with = denoiser_with(tokens_dict, timesteps)
    logits_without = denoiser_without(tokens_dict, timesteps)

    # Outputs should differ (guidance changes predictions)
    for key in vocab_sizes.keys():
        assert not torch.allclose(logits_with[key], logits_without[key], atol=1e-5)


def test_conditioning_with_observed_dict(denoiser):
    """Test conditioning on observed tokens."""
    batch_size = 4

    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),
    }
    timesteps = torch.randint(0, 50, (batch_size,))

    # Create observed mask (half observed, half masked)
    observed_dict = {
        key: torch.tensor([True, True, False, False])
        for key in tokens_dict.keys()
    }

    # Forward pass with conditioning
    logits_dict = denoiser(tokens_dict, timesteps, observed_dict=observed_dict)

    # Should still produce valid outputs
    for key, vocab_size in denoiser.vocab_sizes.items():
        assert logits_dict[key].shape == (batch_size, vocab_size)
        assert torch.all(torch.isfinite(logits_dict[key]))


def test_time_embedding_effect():
    """Test that different timesteps produce different outputs."""
    vocab_sizes = {"test_L0": 20, "test_L1": 10}
    category_level_info = {
        "test_L0": {"family": "test", "category": "test", "level": 0},
        "test_L1": {"family": "test", "category": "test", "level": 1},
    }

    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
    )

    tokens_dict = {
        "test_L0": torch.randint(0, 20, (2,)),
        "test_L1": torch.randint(0, 10, (2,)),
    }

    # Same tokens, different timesteps
    t_early = torch.tensor([5, 5])
    t_late = torch.tensor([45, 45])

    logits_early = denoiser(tokens_dict, t_early)
    logits_late = denoiser(tokens_dict, t_late)

    # Outputs should differ based on timestep
    for key in vocab_sizes.keys():
        assert not torch.allclose(logits_early[key], logits_late[key], atol=1e-5)


def test_gradient_flow(denoiser):
    """Test that gradients flow through the network."""
    batch_size = 4

    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),
    }
    timesteps = torch.randint(0, 50, (batch_size,))

    # Forward pass
    logits_dict = denoiser(tokens_dict, timesteps)

    # Compute dummy loss
    loss = sum(logits.sum() for logits in logits_dict.values())

    # Backward pass
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in denoiser.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.any(param.grad != 0), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
