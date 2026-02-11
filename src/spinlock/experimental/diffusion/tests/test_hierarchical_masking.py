"""Unit tests for HierarchicalMaskGenerator."""

import pytest
import torch

from spinlock.experimental.diffusion.data import HierarchicalMaskGenerator, MaskingStrategy


@pytest.fixture
def vocab_sizes():
    """Sample vocab sizes for testing."""
    return {
        "temporal_group_1_L0": 28,
        "temporal_group_1_L1": 14,
        "temporal_group_1_L2": 7,
        "temporal_group_2_L0": 28,
        "temporal_group_2_L1": 14,
        "initial_group_1_L0": 20,
        "initial_group_1_L1": 10,
        "initial_group_1_L2": 5,
    }


@pytest.fixture
def category_level_info():
    """Sample category level info for testing."""
    return {
        "temporal_group_1_L0": {"family": "temporal", "category": "group_1", "level": 0},
        "temporal_group_1_L1": {"family": "temporal", "category": "group_1", "level": 1},
        "temporal_group_1_L2": {"family": "temporal", "category": "group_1", "level": 2},
        "temporal_group_2_L0": {"family": "temporal", "category": "group_2", "level": 0},
        "temporal_group_2_L1": {"family": "temporal", "category": "group_2", "level": 1},
        "initial_group_1_L0": {"family": "initial", "category": "group_1", "level": 0},
        "initial_group_1_L1": {"family": "initial", "category": "group_1", "level": 1},
        "initial_group_1_L2": {"family": "initial", "category": "group_1", "level": 2},
    }


def test_random_masking(vocab_sizes, category_level_info):
    """Test random masking strategy."""
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=0.5,
        seed=42,
    )

    batch_size = 32
    observed_dict, target_dict = mask_gen.generate_dict_mask(batch_size)

    # Check dict structure preserved
    assert set(observed_dict.keys()) == set(vocab_sizes.keys())
    assert set(target_dict.keys()) == set(vocab_sizes.keys())

    # Check shapes
    for key in vocab_sizes.keys():
        assert observed_dict[key].shape == (batch_size,)
        assert target_dict[key].shape == (batch_size,)
        assert observed_dict[key].dtype == torch.bool
        assert target_dict[key].dtype == torch.bool

    # Check complement relationship
    for key in vocab_sizes.keys():
        assert torch.all(observed_dict[key] == ~target_dict[key])

    # Check masking probability (approximately 50% masked per category-level)
    total_observed = sum(v.sum().item() for v in observed_dict.values())
    total_tokens = batch_size * len(vocab_sizes)
    observed_ratio = total_observed / total_tokens

    # Allow some variance due to randomness
    assert 0.4 < observed_ratio < 0.6, f"Observed ratio: {observed_ratio}"


def test_coarse_only_masking(vocab_sizes, category_level_info):
    """Test coarse-only masking strategy."""
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.COARSE_ONLY,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
    )

    batch_size = 16
    observed_dict, target_dict = mask_gen.generate_dict_mask(batch_size)

    # Check that only L0 is observed
    for key, info in category_level_info.items():
        if info['level'] == 0:
            # L0 should be fully observed
            assert torch.all(observed_dict[key])
            assert torch.all(~target_dict[key])
        else:
            # L1, L2 should be fully masked (target)
            assert torch.all(~observed_dict[key])
            assert torch.all(target_dict[key])


def test_hierarchical_masking(vocab_sizes, category_level_info):
    """Test hierarchical masking strategy."""
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.HIERARCHICAL,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
    )

    batch_size = 16
    observed_dict, target_dict = mask_gen.generate_dict_mask(batch_size)

    # Check that L0 and L1 are observed, L2 is masked
    for key, info in category_level_info.items():
        if info['level'] <= 1:
            # L0, L1 should be fully observed
            assert torch.all(observed_dict[key])
            assert torch.all(~target_dict[key])
        else:
            # L2 should be fully masked (target)
            assert torch.all(~observed_dict[key])
            assert torch.all(target_dict[key])


def test_family_selective_masking(vocab_sizes, category_level_info):
    """Test family-selective masking strategy."""
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.FAMILY_SELECTIVE,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        seed=42,
    )

    batch_size = 16
    observed_dict, target_dict = mask_gen.generate_dict_mask(batch_size)

    # Check that one family is fully masked, others fully observed
    temporal_keys = [k for k, info in category_level_info.items() if info['family'] == 'temporal']
    initial_keys = [k for k, info in category_level_info.items() if info['family'] == 'initial']

    # Either temporal or initial should be fully masked
    temporal_observed = all(torch.all(observed_dict[k]) for k in temporal_keys)
    initial_observed = all(torch.all(observed_dict[k]) for k in initial_keys)

    temporal_masked = all(torch.all(~observed_dict[k]) for k in temporal_keys)
    initial_masked = all(torch.all(~observed_dict[k]) for k in initial_keys)

    # Exactly one family should be observed, one masked
    assert (temporal_observed and initial_masked) or (initial_observed and temporal_masked)


def test_extract_categories(vocab_sizes, category_level_info):
    """Test category extraction from info dict."""
    mask_gen = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
    )

    # Check families extracted correctly
    assert "temporal" in mask_gen.families
    assert "initial" in mask_gen.families

    # Check that keys are grouped by family
    for family, keys in mask_gen.families.items():
        for key in keys:
            assert category_level_info[key]['family'] == family

    # Check levels extracted correctly
    assert 0 in mask_gen.categories_by_level
    assert 1 in mask_gen.categories_by_level
    assert 2 in mask_gen.categories_by_level

    # Check that keys are grouped by level
    for level, keys in mask_gen.categories_by_level.items():
        for key in keys:
            assert category_level_info[key]['level'] == level


def test_reproducibility_with_seed():
    """Test that same seed produces same masks."""
    vocab_sizes = {"test_L0": 20, "test_L1": 10}
    category_level_info = {
        "test_L0": {"family": "test", "category": "test", "level": 0},
        "test_L1": {"family": "test", "category": "test", "level": 1},
    }

    mask_gen1 = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=0.5,
        seed=123,
    )

    mask_gen2 = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=0.5,
        seed=123,
    )

    batch_size = 16
    obs1, tgt1 = mask_gen1.generate_dict_mask(batch_size)
    obs2, tgt2 = mask_gen2.generate_dict_mask(batch_size)

    # Should produce identical masks
    for key in vocab_sizes.keys():
        assert torch.all(obs1[key] == obs2[key])
        assert torch.all(tgt1[key] == tgt2[key])


def test_different_mask_probabilities():
    """Test different masking probabilities."""
    vocab_sizes = {"test_L0": 20}
    category_level_info = {
        "test_L0": {"family": "test", "category": "test", "level": 0},
    }

    # High masking probability (90% masked)
    mask_gen_high = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=0.9,
        seed=42,
    )

    # Low masking probability (10% masked)
    mask_gen_low = HierarchicalMaskGenerator(
        strategy=MaskingStrategy.RANDOM,
        vocab_sizes=vocab_sizes,
        category_level_info=category_level_info,
        mask_probability=0.1,
        seed=43,
    )

    batch_size = 100
    obs_high, _ = mask_gen_high.generate_dict_mask(batch_size)
    obs_low, _ = mask_gen_low.generate_dict_mask(batch_size)

    # High prob should have fewer observed tokens
    obs_high_ratio = obs_high["test_L0"].float().mean().item()
    obs_low_ratio = obs_low["test_L0"].float().mean().item()

    assert obs_high_ratio < 0.2  # ~10% observed (90% masked)
    assert obs_low_ratio > 0.8  # ~90% observed (10% masked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
