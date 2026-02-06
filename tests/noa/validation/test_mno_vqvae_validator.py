"""Tests for MNO-VQ-VAE validator."""

import pytest
import torch
from pathlib import Path

from spinlock.noa.validation.mno_vqvae_validator import MNOVQVAEValidator
from spinlock.noa.validation.config import ValidationConfig
from spinlock.noa.validation.metrics import ValidationMetrics


def test_validation_config_defaults():
    """Test validation config has sensible defaults."""
    config = ValidationConfig()

    assert config.num_samples == 100
    assert config.batch_size == 8
    assert config.max_reconstruction_ratio == 2.0
    assert config.device == "cuda"
    assert config.rollout_steps == 256


def test_validation_config_custom():
    """Test validation config accepts custom values."""
    config = ValidationConfig(
        num_samples=50,
        batch_size=4,
        max_reconstruction_ratio=1.5,
        device="cpu",
        rollout_steps=128
    )

    assert config.num_samples == 50
    assert config.batch_size == 4
    assert config.max_reconstruction_ratio == 1.5
    assert config.device == "cpu"
    assert config.rollout_steps == 128


def test_validation_metrics():
    """Test metrics computation."""
    # Mock features
    features_original = torch.randn(10, 100)
    features_reconstructed = features_original + 0.01 * torch.randn(10, 100)
    tokens = torch.randint(0, 50, (10, 20))

    metrics = ValidationMetrics.compute(
        features_original=features_original,
        features_reconstructed=features_reconstructed,
        tokens=tokens,
        cno_baseline_error=0.027,
        config=ValidationConfig()
    )

    assert 'reconstruction_mse' in metrics
    assert 'reconstruction_mae' in metrics
    assert 'reconstruction_ratio' in metrics
    assert 'relative_l2' in metrics
    assert 'mean_correlation' in metrics
    assert 'median_correlation' in metrics
    assert 'unique_tokens' in metrics
    assert 'token_entropy' in metrics

    assert metrics['reconstruction_mse'] > 0
    assert metrics['reconstruction_mae'] > 0
    assert metrics['relative_l2'] > 0
    assert metrics['reconstruction_ratio'] > 0
    assert 0 <= metrics['mean_correlation'] <= 1
    assert 0 <= metrics['median_correlation'] <= 1
    assert metrics['unique_tokens'] <= 50
    assert metrics['token_entropy'] > 0


def test_token_entropy():
    """Test token entropy computation."""
    # Uniform distribution should have high entropy
    uniform_tokens = torch.randint(0, 50, (1000, 20))
    entropy_uniform = ValidationMetrics._compute_token_entropy(uniform_tokens)

    # Single token should have zero entropy
    single_token = torch.zeros(1000, 20, dtype=torch.long)
    entropy_single = ValidationMetrics._compute_token_entropy(single_token)

    assert entropy_uniform > entropy_single
    assert entropy_single == 0.0


@pytest.mark.skipif(
    not Path("checkpoints/mno/50k_baseline/meta_operator_best.pt").exists()
    or not Path("checkpoints/vqvae/50k_baseline/best_model.pt").exists(),
    reason="Checkpoints not available"
)
def test_validator_initialization():
    """Test validator can initialize with valid checkpoints."""
    mno_ckpt = Path("checkpoints/mno/50k_baseline/meta_operator_best.pt")
    vqvae_ckpt = Path("checkpoints/vqvae/50k_baseline/best_model.pt")

    validator = MNOVQVAEValidator(
        mno_checkpoint=mno_ckpt,
        vqvae_checkpoint=vqvae_ckpt,
        device="cpu"
    )

    assert validator.mno is not None
    assert validator.vqvae is not None
    assert validator.feature_pipeline is not None
    assert validator.cno_reconstruction_error > 0


@pytest.mark.skipif(
    not Path("checkpoints/mno/50k_baseline/meta_operator_best.pt").exists()
    or not Path("checkpoints/vqvae/50k_baseline/best_model.pt").exists()
    or not Path("datasets/cno_50k_v3_1.h5").exists(),
    reason="Checkpoints or dataset not available"
)
def test_validator_end_to_end():
    """Test validator runs end-to-end with small sample."""
    mno_ckpt = Path("checkpoints/mno/50k_baseline/meta_operator_best.pt")
    vqvae_ckpt = Path("checkpoints/vqvae/50k_baseline/best_model.pt")
    dataset = Path("datasets/cno_50k_v3_1.h5")

    config = ValidationConfig(
        num_samples=2,
        batch_size=1,
        device="cpu",
        rollout_steps=32  # Short rollout for testing
    )

    validator = MNOVQVAEValidator(
        mno_checkpoint=mno_ckpt,
        vqvae_checkpoint=vqvae_ckpt,
        config=config,
        device="cpu"
    )

    result = validator.validate(
        dataset_path=dataset,
        num_samples=2,
        batch_size=1
    )

    assert result.num_samples == 2
    assert result.mno_reconstruction_error > 0
    assert result.cno_reconstruction_error > 0
    assert result.reconstruction_ratio > 0
    assert isinstance(result.pass_threshold, bool)
