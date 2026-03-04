"""Tests for V2 tokens configuration models."""

import pytest
from pathlib import Path

from spinlock.tokens.config import (
    TokenizerConfig,
    EncoderConfig,
    InitialEncoderConfig,
    TemporalEncoderConfig,
    QuantizerConfig,
    HierarchyConfig,
    LossConfig,
    TrainingConfig,
    NormalizationConfig,
    PretrainingConfig,
)


def test_tokenizer_config_defaults():
    """Test TokenizerConfig creates with all defaults."""
    config = TokenizerConfig()

    assert config.encoder is not None
    assert config.quantizer is not None
    assert config.hierarchy is not None
    assert config.training is not None
    assert config.loss is not None
    assert config.normalization is not None
    assert config.grouping is None
    assert config.verbose is True


def test_encoder_config_defaults():
    """Test EncoderConfig defaults."""
    config = EncoderConfig()

    assert config.initial.variant == "hybrid"
    assert config.temporal.variant == "pyramid"
    assert config.embedding_dim == 64
    assert config.hidden_dim == 128
    assert config.dropout == 0.1


def test_initial_encoder_config():
    """Test InitialEncoderConfig."""
    config = InitialEncoderConfig(
        variant="cnn",
        cnn_embedding_dim=256,
        in_channels=3,
    )

    assert config.variant == "cnn"
    assert config.cnn_embedding_dim == 256
    assert config.in_channels == 3
    assert config.pretrained_cnn_path is None


def test_temporal_encoder_config():
    """Test TemporalEncoderConfig."""
    config = TemporalEncoderConfig(
        variant="pyramid",
        level_dims=[32, 64, 96, 128],
        variable_length=True,
    )

    assert config.variant == "pyramid"
    assert config.level_dims == [32, 64, 96, 128]
    assert config.variable_length is True
    assert config.adaptive_pyramid is True


def test_quantizer_config():
    """Test QuantizerConfig."""
    config = QuantizerConfig(
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25,
    )

    assert config.num_embeddings == 512
    assert config.embedding_dim == 64
    assert config.commitment_cost == 0.25
    assert config.use_ema is True


def test_hierarchy_config():
    """Test HierarchyConfig."""
    config = HierarchyConfig(
        num_levels=3,
        level_ratios=[1.5, 0.75, 0.4],
    )

    assert config.num_levels == 3
    assert config.level_ratios == [1.5, 0.75, 0.4]


def test_hierarchy_config_defaults():
    """Test HierarchyConfig defaults (adaptive formula)."""
    config = HierarchyConfig()
    assert config.level_ratios is None
    assert config.num_levels == 3


def test_hierarchy_config_validation():
    """Test HierarchyConfig validates level_ratios."""
    # Valid ratios (non-increasing)
    config = HierarchyConfig(level_ratios=[1.5, 0.75, 0.4])
    assert config.level_ratios == [1.5, 0.75, 0.4]

    # Invalid: negative ratio
    with pytest.raises(ValueError, match="must be positive"):
        HierarchyConfig(level_ratios=[-0.5, 0.75, 0.4])

    # Invalid: non-decreasing (L1 > L0)
    with pytest.raises(ValueError, match="non-increasing"):
        HierarchyConfig(level_ratios=[0.5, 1.0, 0.4])


def test_loss_config():
    """Test LossConfig."""
    config = LossConfig(
        reconstruction_weight=1.0,
        orthogonality_weight=0.1,
        informativeness_weight=0.1,
    )

    assert config.reconstruction_weight == 1.0
    assert config.orthogonality_weight == 0.1
    assert config.informativeness_weight == 0.1
    assert config.topographic_weight == 0.0


def test_training_config():
    """Test TrainingConfig."""
    config = TrainingConfig(
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        device="auto",
    )

    assert config.num_epochs == 100
    assert config.batch_size == 256
    assert config.learning_rate == 1e-3
    assert config.device == "auto"
    assert config.optimizer == "adam"


def test_normalization_config():
    """Test NormalizationConfig."""
    config = NormalizationConfig(method="per_category")

    assert config.method == "per_category"
    assert config.clip_std_multiplier is None


def test_pretraining_config():
    """Test PretrainingConfig."""
    config = PretrainingConfig(
        embedding_dim=256,
        num_epochs=50,
        batch_size=128,
    )

    assert config.embedding_dim == 256
    assert config.num_epochs == 50
    assert config.batch_size == 128
    assert config.optimizer == "adam"


def test_tokenizer_config_serialization():
    """Test TokenizerConfig can be serialized/deserialized."""
    config = TokenizerConfig()

    # Serialize
    config_dict = config.model_dump()

    assert isinstance(config_dict, dict)
    assert "encoder" in config_dict
    assert "quantizer" in config_dict

    # Deserialize
    config2 = TokenizerConfig(**config_dict)

    assert config2.encoder.embedding_dim == config.encoder.embedding_dim
    assert config2.quantizer.num_embeddings == config.quantizer.num_embeddings
