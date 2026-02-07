"""Tests for CNN pretraining module."""

import pytest
import torch
import tempfile
from pathlib import Path

from spinlock.tokens.pretraining import CNNPretrainer
from spinlock.tokens.config import PretrainingConfig


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_config():
    """Create simple pretraining config."""
    return PretrainingConfig(
        embedding_dim=64,
        in_channels=1,  # Match test data
        num_epochs=2,  # Fast for testing
        batch_size=4,
        verbose=False,
    )


def test_cnn_pretrainer_initialization(simple_config):
    """Test CNNPretrainer can be initialized."""
    pretrainer = CNNPretrainer(simple_config)

    assert pretrainer.config == simple_config
    assert pretrainer.encoder is not None
    assert pretrainer.decoder is not None
    assert pretrainer.optimizer is not None


def test_cnn_decoder():
    """Test CNNDecoder architecture."""
    from spinlock.tokens.pretraining.cnn_pretrainer import CNNDecoder

    decoder = CNNDecoder(embedding_dim=64, out_channels=1)

    # Forward pass
    x = torch.randn(4, 64)
    out = decoder(x)

    assert out.shape == (4, 1, 128, 128)


def test_cnn_pretrainer_train(temp_dir, simple_config):
    """Test CNNPretrainer training loop."""
    pretrainer = CNNPretrainer(simple_config)

    # Create fake initial conditions
    ics = torch.randn(32, 1, 128, 128)

    output_path = temp_dir / "cnn_pretrained.pt"

    # Train
    history = pretrainer.train(ics, output_path)

    # Check outputs
    assert 'train_losses' in history
    assert 'val_losses' in history
    assert 'best_epoch' in history
    assert 'best_val_loss' in history

    assert len(history['train_losses']) == simple_config.num_epochs
    assert output_path.exists()

    # Check checkpoint
    checkpoint = torch.load(output_path, map_location='cpu')

    assert 'encoder_state_dict' in checkpoint
    assert 'embedding_dim' in checkpoint
    assert checkpoint['embedding_dim'] == 64


def test_cnn_pretrainer_train_with_validation_split(temp_dir, simple_config):
    """Test CNNPretrainer with explicit validation split."""
    pretrainer = CNNPretrainer(simple_config)

    # Create fake data
    train_ics = torch.randn(24, 1, 128, 128)
    val_ics = torch.randn(8, 1, 128, 128)

    output_path = temp_dir / "cnn_pretrained.pt"

    # Train
    history = pretrainer.train(
        train_ics,
        output_path,
        val_initial_conditions=val_ics,
    )

    assert len(history['train_losses']) == simple_config.num_epochs
    assert len(history['val_losses']) == simple_config.num_epochs
