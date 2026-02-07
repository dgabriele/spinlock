"""Tests for V2 tokens checkpoint utilities."""

import pytest
import torch
import tempfile
from pathlib import Path

from spinlock.tokens.config import TokenizerConfig
from spinlock.tokens.checkpoint import save_checkpoint, load_checkpoint, verify_pretrained_cnn
from spinlock.tokens.model import JointHierarchicalVQVAE


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_config():
    """Create simple tokenizer config for testing."""
    from spinlock.tokens.config import EncoderConfig, QuantizerConfig

    config = TokenizerConfig()
    config.encoder = EncoderConfig(embedding_dim=32)
    config.quantizer = QuantizerConfig(num_embeddings=64, embedding_dim=32)
    return config


@pytest.fixture
def simple_model(simple_config):
    """Create simple model for testing."""
    group_indices = {
        "temporal_group_1": [0, 1, 2],
        "initial_group_1": [3, 4],
    }
    return JointHierarchicalVQVAE(simple_config, group_indices)


def test_save_checkpoint(temp_dir, simple_model, simple_config):
    """Test checkpoint saving."""
    checkpoint_path = temp_dir / "test_checkpoint.pt"

    group_indices = {
        "temporal_group_1": [0, 1, 2],
        "initial_group_1": [3, 4],
    }

    save_checkpoint(
        path=checkpoint_path,
        model=simple_model,
        config=simple_config,
        group_indices=group_indices,
        normalization_stats=None,
        optimizer=None,
        epoch=10,
        val_loss=0.5,
        metadata={"test": "data"},
    )

    assert checkpoint_path.exists()

    # Load and verify
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    assert 'model_state_dict' in checkpoint
    assert 'config' in checkpoint
    assert 'group_indices' in checkpoint
    assert 'version' in checkpoint
    assert checkpoint['version'] == 'v2'
    assert checkpoint['epoch'] == 10
    assert checkpoint['val_loss'] == 0.5


def test_load_checkpoint(temp_dir, simple_model, simple_config):
    """Test checkpoint loading."""
    checkpoint_path = temp_dir / "test_checkpoint.pt"

    group_indices = {
        "temporal_group_1": [0, 1, 2],
        "initial_group_1": [3, 4],
    }

    # Save
    save_checkpoint(
        path=checkpoint_path,
        model=simple_model,
        config=simple_config,
        group_indices=group_indices,
        epoch=5,
    )

    # Load
    loaded = load_checkpoint(checkpoint_path)

    assert 'model_state_dict' in loaded
    assert 'config' in loaded
    assert isinstance(loaded['config'], TokenizerConfig)
    assert loaded['config'].encoder.embedding_dim == 32
    assert loaded['group_indices'] == group_indices
    assert loaded['epoch'] == 5


def test_load_checkpoint_version_check(temp_dir):
    """Test checkpoint version validation."""
    checkpoint_path = temp_dir / "v1_checkpoint.pt"

    # Create fake V1 checkpoint
    checkpoint = {
        'model_state_dict': {},
        'config': {},
        'version': 'v1',
    }
    torch.save(checkpoint, checkpoint_path)

    # Should raise error
    with pytest.raises(ValueError, match="version v1 not supported"):
        load_checkpoint(checkpoint_path)


def test_verify_pretrained_cnn_not_found(temp_dir):
    """Test verify_pretrained_cnn with missing file."""
    fake_path = temp_dir / "nonexistent.pt"

    with pytest.raises(FileNotFoundError, match="not found"):
        verify_pretrained_cnn(fake_path, embedding_dim=256)


def test_verify_pretrained_cnn_success(temp_dir):
    """Test verify_pretrained_cnn with valid checkpoint."""
    checkpoint_path = temp_dir / "cnn_pretrained.pt"

    # Create valid CNN checkpoint
    checkpoint = {
        'encoder_state_dict': {},
        'embedding_dim': 256,
        'epoch': 50,
        'val_loss': 0.1,
    }
    torch.save(checkpoint, checkpoint_path)

    # Verify
    metadata = verify_pretrained_cnn(checkpoint_path, embedding_dim=256)

    assert metadata['embedding_dim'] == 256
    assert metadata['epoch'] == 50
    assert metadata['val_loss'] == 0.1


def test_verify_pretrained_cnn_dimension_mismatch(temp_dir):
    """Test verify_pretrained_cnn with dimension mismatch."""
    checkpoint_path = temp_dir / "cnn_pretrained.pt"

    # Create checkpoint with different embedding dim
    checkpoint = {
        'encoder_state_dict': {},
        'embedding_dim': 128,
    }
    torch.save(checkpoint, checkpoint_path)

    # Should raise error
    with pytest.raises(ValueError, match="dimension mismatch"):
        verify_pretrained_cnn(checkpoint_path, embedding_dim=256)
