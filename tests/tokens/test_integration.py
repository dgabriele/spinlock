"""Integration tests for V2 tokens package."""

import pytest
import torch

from spinlock.tokens import (
    TokenizerConfig,
    VQTokenizer,
    JointHierarchicalVQVAE,
)


@pytest.fixture
def simple_config():
    """Create simple config for testing with mean encoder (no temporal_input_dim issue)."""
    from spinlock.tokens.config import TemporalEncoderConfig, EncoderConfig
    return TokenizerConfig(
        encoder=EncoderConfig(
            temporal=TemporalEncoderConfig(variant="mean"),
            embedding_dim=32,
            hidden_dim=64,
        ),
    )


@pytest.fixture
def group_indices():
    """Create simple group indices."""
    return {
        "temporal_group_1": [0, 1, 2],
        "temporal_group_2": [3, 4],
    }


def test_joint_vqvae_creation(simple_config, group_indices):
    """Test JointHierarchicalVQVAE can be created."""
    model = JointHierarchicalVQVAE(
        simple_config, group_indices, temporal_input_dim=5,
    )

    assert model is not None
    assert len(model.projectors) == 2
    expected_quantizers = 2 * simple_config.hierarchy.num_levels
    assert len(model.quantizers) == expected_quantizers


def test_joint_vqvae_forward_temporal_only(simple_config):
    """Test forward pass with temporal features only."""
    group_indices = {
        "temporal_group_1": [0, 1, 2],
    }

    model = JointHierarchicalVQVAE(
        simple_config, group_indices, temporal_input_dim=3,
    )

    temporal = torch.randn(4, 64, 3)  # [B, T, D_t]
    outputs = model(temporal_features=temporal)

    assert "reconstructed" in outputs
    assert "vq_loss" in outputs
    assert outputs["reconstructed"].shape[0] == 4


def test_vq_tokenizer_initialization(simple_config, group_indices):
    """Test VQTokenizer can be initialized."""
    tokenizer = VQTokenizer(simple_config, group_indices=group_indices)

    assert tokenizer.config == simple_config
    assert tokenizer.group_indices == group_indices


def test_vq_tokenizer_from_checkpoint_not_implemented():
    """Test VQTokenizer.from_checkpoint with missing file."""
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        VQTokenizer.from_checkpoint(Path("nonexistent.pt"))


def test_tokenizer_config_export_import(simple_config):
    """Test config can be exported and imported."""
    # Export
    config_dict = simple_config.model_dump()

    # Import
    config2 = TokenizerConfig(**config_dict)

    assert config2.encoder.embedding_dim == simple_config.encoder.embedding_dim
    assert config2.quantizer.num_embeddings == simple_config.quantizer.num_embeddings
