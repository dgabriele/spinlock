"""Unit tests for ThetaMLPEncoder."""

import torch
import pytest
from spinlock.tokens.encoders.theta import ThetaMLPEncoder


def test_theta_encoder_forward_pass():
    """Test basic forward pass through ThetaMLPEncoder."""
    encoder = ThetaMLPEncoder(
        param_dim=14,
        hidden_dim=64,
        output_dim=32,
        dropout=0.1,
        use_layer_norm=True,
    )

    # Create batch of parameter vectors
    batch_size = 4
    theta = torch.randn(batch_size, 14)

    # Forward pass
    output = encoder(theta)

    # Check output shape
    assert output.shape == (batch_size, 32), f"Expected shape (4, 32), got {output.shape}"


def test_theta_encoder_gradient_flow():
    """Test that gradients flow through encoder."""
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=64, output_dim=32)

    theta = torch.randn(4, 14, requires_grad=True)
    output = encoder(theta)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    assert encoder.layer1.weight.grad is not None, "No gradient for layer1.weight"
    assert encoder.layer2.weight.grad is not None, "No gradient for layer2.weight"
    assert theta.grad is not None, "No gradient for input"


def test_theta_encoder_without_layer_norm():
    """Test encoder without layer normalization."""
    encoder = ThetaMLPEncoder(
        param_dim=14,
        hidden_dim=64,
        output_dim=32,
        use_layer_norm=False,
    )

    theta = torch.randn(2, 14)
    output = encoder(theta)

    assert output.shape == (2, 32)
    assert isinstance(encoder.norm1, torch.nn.Identity)
    assert isinstance(encoder.norm2, torch.nn.Identity)


def test_theta_encoder_different_dims():
    """Test encoder with different dimensions."""
    # Test with larger hidden dimension
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=128, output_dim=64)
    theta = torch.randn(3, 14)
    output = encoder(theta)
    assert output.shape == (3, 64)

    # Test with smaller output dimension
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=32, output_dim=16)
    theta = torch.randn(3, 14)
    output = encoder(theta)
    assert output.shape == (3, 16)


def test_theta_encoder_dropout_disabled_eval():
    """Test that dropout is disabled in eval mode."""
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=64, output_dim=32, dropout=0.5)
    encoder.eval()

    theta = torch.randn(1, 14)

    # Run multiple times to check consistency (dropout should be off)
    output1 = encoder(theta)
    output2 = encoder(theta)

    torch.testing.assert_close(output1, output2, msg="Outputs differ in eval mode (dropout not disabled?)")


def test_theta_encoder_repr():
    """Test string representation."""
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=64, output_dim=32)
    repr_str = repr(encoder)

    assert "ThetaMLPEncoder" in repr_str
    assert "param_dim=14" in repr_str
    assert "hidden_dim=64" in repr_str
    assert "output_dim=32" in repr_str


def test_theta_encoder_serialization():
    """Test save/load checkpoint."""
    encoder = ThetaMLPEncoder(param_dim=14, hidden_dim=64, output_dim=32)

    # Save state dict
    state_dict = encoder.state_dict()

    # Create new encoder and load
    encoder2 = ThetaMLPEncoder(param_dim=14, hidden_dim=64, output_dim=32)
    encoder2.load_state_dict(state_dict)

    # Test they produce same output
    theta = torch.randn(2, 14)
    encoder.eval()
    encoder2.eval()

    output1 = encoder(theta)
    output2 = encoder2(theta)

    torch.testing.assert_close(output1, output2, msg="Outputs differ after loading checkpoint")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
