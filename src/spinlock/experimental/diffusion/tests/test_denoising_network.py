"""Unit tests for DenoisingNetwork."""

import pytest
import torch

from spinlock.experimental.diffusion.models import DenoisingNetwork


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


def test_per_category_guidance(vocab_sizes, category_level_info):
    """Test per-category hierarchical guidance mode."""
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=True,
        guidance_mode="per_category",
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

    # Should run without errors
    logits = denoiser(tokens_dict, timesteps)
    for key, vocab_size in vocab_sizes.items():
        assert logits[key].shape == (batch_size, vocab_size)
        assert torch.all(torch.isfinite(logits[key]))


def test_per_category_guidance_parent_mask(vocab_sizes, category_level_info):
    """Test that parent_mask correctly maps positions to their L0 parents."""
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=True,
        guidance_mode="per_category",
    )

    # Check parent_mask structure
    mask = denoiser.parent_mask  # [N, N]
    sorted_keys = denoiser.sorted_keys

    for i, key_i in enumerate(sorted_keys):
        info_i = category_level_info[key_i]
        for j, key_j in enumerate(sorted_keys):
            info_j = category_level_info[key_j]

            if info_j['level'] == 0 and (
                info_i['family'] == info_j['family']
                and info_i['category'] == info_j['category']
            ):
                # Same family+category L0 → should have nonzero weight
                assert mask[i, j].item() > 0, (
                    f"{key_i} should attend to its L0 parent {key_j}"
                )
            else:
                # Different family/category or not L0 → should be zero
                assert mask[i, j].item() == 0, (
                    f"{key_i} should NOT attend to {key_j}"
                )


def test_per_category_vs_global_guidance_differ(vocab_sizes, category_level_info):
    """Test that per-category and global guidance produce different outputs."""
    torch.manual_seed(42)
    denoiser_global = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=True,
        guidance_mode="global",
    )

    torch.manual_seed(42)
    denoiser_per_cat = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        use_hierarchical_guidance=True,
        guidance_mode="per_category",
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

    logits_global = denoiser_global(tokens_dict, timesteps)
    logits_per_cat = denoiser_per_cat(tokens_dict, timesteps)

    # They share the same weights but different guidance paths → different outputs
    any_differ = any(
        not torch.allclose(logits_global[key], logits_per_cat[key], atol=1e-5)
        for key in vocab_sizes
    )
    assert any_differ, "Per-category and global guidance should produce different outputs"


def test_eval_mode_drops_attention_mask(denoiser):
    """Test that eval mode drops the src_key_padding_mask."""
    batch_size = 4
    tokens_dict = {
        "temporal_group_1_L0": torch.randint(0, 28, (batch_size,)),
        "temporal_group_1_L1": torch.randint(0, 14, (batch_size,)),
        "temporal_group_1_L2": torch.randint(0, 7, (batch_size,)),
        "initial_group_1_L0": torch.randint(0, 20, (batch_size,)),
        "initial_group_1_L1": torch.randint(0, 10, (batch_size,)),
    }
    timesteps = torch.randint(0, 50, (batch_size,))

    observed_dict = {
        key: torch.tensor([True, True, False, False])
        for key in tokens_dict
    }

    # Train mode: with mask (should work)
    denoiser.train()
    logits_train = denoiser(tokens_dict, timesteps, observed_dict=observed_dict)

    # Eval mode: should drop mask (full attention)
    denoiser.eval()
    logits_eval = denoiser(tokens_dict, timesteps, observed_dict=observed_dict)

    # Both should produce valid outputs
    for key in tokens_dict:
        assert torch.all(torch.isfinite(logits_train[key]))
        assert torch.all(torch.isfinite(logits_eval[key]))

    # Outputs should differ since masking changes attention patterns
    any_differ = any(
        not torch.allclose(logits_train[key], logits_eval[key], atol=1e-5)
        for key in tokens_dict
    )
    assert any_differ, "Train and eval mode should produce different outputs with observed_dict"


def test_absorbing_state_embeddings(vocab_sizes, category_level_info):
    """Test that absorbing mode creates V+1 embeddings but V output heads."""
    denoiser = DenoisingNetwork(
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        transition_type="absorbing",
    )

    for key, vocab_size in vocab_sizes.items():
        # Embeddings should have V+1 entries (for mask token)
        assert denoiser.token_embeddings[key].num_embeddings == vocab_size + 1

        # Output heads should still predict V clean tokens only
        assert denoiser.output_heads[key].out_features == vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
