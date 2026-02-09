"""Integration tests for VQ-VAE model with theta support."""

import torch
import pytest
import yaml
from pathlib import Path

from spinlock.tokens.config import TokenizerConfig
from spinlock.tokens.model import JointHierarchicalVQVAE


@pytest.fixture
def theta_config():
    """Load config with theta support."""
    config_path = Path("configs/tokenizer_with_theta.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return TokenizerConfig(**config_dict)


@pytest.fixture
def group_indices_with_theta():
    """Create simple group indices including theta."""
    return {
        "theta_group_1": list(range(14)),  # All 14 parameters in one group
    }


def test_model_initialization_with_theta(theta_config, group_indices_with_theta):
    """Test that model initializes with theta encoder."""
    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices_with_theta,
        temporal_input_dim=None,
        initial_input_dim=None,
    )

    # Check theta encoder exists
    assert hasattr(model, 'theta_encoder'), "Model missing theta_encoder"
    assert model.theta_dim == 32, f"Expected theta_dim=32, got {model.theta_dim}"

    # Check families parsed correctly
    assert "theta" in model.families, "Theta not in parsed families"


def test_model_forward_pass_theta_only(theta_config, group_indices_with_theta):
    """Test forward pass with only theta features."""
    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices_with_theta,
        temporal_input_dim=None,
        initial_input_dim=None,
    )

    batch_size = 4
    theta = torch.randn(batch_size, 14)

    # Forward pass
    outputs = model(theta_features=theta)

    # Check outputs
    assert "reconstructed" in outputs
    assert "quantized" in outputs
    assert "vq_loss" in outputs
    assert "perplexity" in outputs
    assert "encodings" in outputs
    assert "token_indices" in outputs

    # Check shapes
    assert outputs["reconstructed"].shape[0] == batch_size
    assert outputs["quantized"].shape[0] == batch_size

    # Check we have tokens for theta at all levels
    num_levels = theta_config.hierarchy.num_levels
    for level in range(num_levels):
        key = f"theta_group_1_L{level}"
        assert key in outputs["encodings"], f"Missing encodings for {key}"
        assert key in outputs["token_indices"], f"Missing token indices for {key}"


def test_model_encode_theta(theta_config, group_indices_with_theta):
    """Test encode method produces discrete tokens."""
    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices_with_theta,
    )
    model.eval()

    batch_size = 2
    theta = torch.randn(batch_size, 14)

    # Encode to tokens
    tokens = model.encode(theta_features=theta)

    # Check token structure
    num_levels = theta_config.hierarchy.num_levels
    assert len(tokens) == num_levels, f"Expected {num_levels} token levels, got {len(tokens)}"

    for level in range(num_levels):
        key = f"theta_group_1_L{level}"
        assert key in tokens, f"Missing tokens for {key}"
        assert tokens[key].shape == (batch_size,), f"Wrong token shape for {key}"
        assert tokens[key].dtype == torch.long, f"Tokens should be long dtype, got {tokens[key].dtype}"


def test_model_gradient_flow_theta(theta_config, group_indices_with_theta):
    """Test gradients flow through theta encoder."""
    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices_with_theta,
    )

    theta = torch.randn(2, 14, requires_grad=True)

    # Forward pass
    outputs = model(theta_features=theta)

    # Compute loss and backward
    loss = outputs["vq_loss"] + outputs["reconstructed"].sum()
    loss.backward()

    # Check gradients exist
    assert theta.grad is not None, "No gradient for input theta"
    assert model.theta_encoder.layer1.weight.grad is not None, "No gradient in theta_encoder"


def test_model_mixed_families(theta_config):
    """Test model with temporal + theta families."""
    group_indices = {
        "temporal_group_1": list(range(64)),  # 64 temporal features
        "theta_group_1": list(range(14)),     # 14 theta features
    }

    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices,
        temporal_input_dim=64,
        initial_input_dim=None,
    )

    batch_size = 3
    T = 32  # timesteps
    temporal = torch.randn(batch_size, T, 64)
    theta = torch.randn(batch_size, 14)

    # Forward pass with both families
    outputs = model(
        temporal_features=temporal,
        theta_features=theta,
    )

    # Check outputs include both families
    num_levels = theta_config.hierarchy.num_levels
    for level in range(num_levels):
        assert f"temporal_group_1_L{level}" in outputs["encodings"]
        assert f"theta_group_1_L{level}" in outputs["encodings"]


def test_model_missing_theta_raises_error(theta_config, group_indices_with_theta):
    """Test that missing theta features raises error when required."""
    model = JointHierarchicalVQVAE(
        config=theta_config,
        group_indices=group_indices_with_theta,
    )

    # Try forward pass without theta (should fail at batch size check)
    with pytest.raises(ValueError, match="At least one feature input must be provided"):
        model(theta_features=None)


def test_config_validation_theta():
    """Test theta encoder config validation."""
    from spinlock.tokens.config import ThetaEncoderConfig

    # Valid config
    config = ThetaEncoderConfig(
        variant="mlp",
        param_dim=14,
        hidden_dim=64,
        output_dim=32,
        dropout=0.1,
    )
    assert config.param_dim == 14
    assert config.dropout == 0.1

    # Invalid dropout (>1.0)
    with pytest.raises(Exception):  # Pydantic validation error
        ThetaEncoderConfig(dropout=1.5)

    # Invalid param_dim (<1)
    with pytest.raises(Exception):
        ThetaEncoderConfig(param_dim=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
