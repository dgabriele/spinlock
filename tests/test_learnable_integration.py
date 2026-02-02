"""Integration test for learnable categorical VQ-VAE."""

import pytest
import torch
import numpy as np
from spinlock.encoding import (
    CategoricalVQVAEConfig,
    CategoricalHierarchicalVQVAE,
    LearnableAssignmentConfig
)
from spinlock.encoding.learnable_assignment import SoftAssignmentMatrix


def test_learnable_vqvae_forward():
    """Test full forward pass with learnable assignments."""
    batch_size = 4
    input_dim = 20
    num_categories = 3

    # Create group indices (static for now, will be replaced by learned assignments)
    group_indices = {
        "cat_0": list(range(0, 7)),
        "cat_1": list(range(7, 14)),
        "cat_2": list(range(14, 20))
    }

    # Create configs
    vqvae_config = CategoricalVQVAEConfig(
        input_dim=input_dim,
        group_indices=group_indices,
        group_embedding_dim=16,
        group_hidden_dim=32,
        levels={
            "cat_0": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_1": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_2": [{"latent_dim": 8, "num_tokens": 64}]
        },
        commitment_cost=0.25
    )

    # Create assignment matrix
    assignment_matrix = SoftAssignmentMatrix(input_dim, num_categories)

    # Create unified model with assignment matrix
    model = CategoricalHierarchicalVQVAE(
        config=vqvae_config,
        assignment_matrix=assignment_matrix
    )

    # Forward pass
    x = torch.randn(batch_size, input_dim)
    outputs = model(x, temperature=1.0)

    # Check outputs (unified model format)
    assert "reconstruction" in outputs
    assert "tokens" in outputs
    assert "vq_losses" in outputs

    # Check shapes
    assert outputs["reconstruction"]["features"].shape == (batch_size, input_dim)
    assert outputs["tokens"].shape == (batch_size, num_categories)  # N×L with L=1


def test_learnable_vqvae_backward():
    """Test gradients flow through model."""
    batch_size = 4
    input_dim = 20
    num_categories = 3

    group_indices = {
        "cat_0": list(range(0, 7)),
        "cat_1": list(range(7, 14)),
        "cat_2": list(range(14, 20))
    }

    vqvae_config = CategoricalVQVAEConfig(
        input_dim=input_dim,
        group_indices=group_indices,
        group_embedding_dim=16,
        group_hidden_dim=32,
        levels={
            "cat_0": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_1": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_2": [{"latent_dim": 8, "num_tokens": 64}]
        }
    )

    assignment_matrix = SoftAssignmentMatrix(input_dim, num_categories)
    model = CategoricalHierarchicalVQVAE(
        config=vqvae_config,
        assignment_matrix=assignment_matrix
    )

    # Forward + backward
    x = torch.randn(batch_size, input_dim)
    outputs = model(x, temperature=1.0)

    # Compute total loss
    recon_loss = ((outputs["reconstruction"]["features"] - x) ** 2).mean()
    vq_loss = sum(outputs["vq_losses"])

    # Compute assignment losses
    from spinlock.encoding.training.assignment_losses import (
        soft_orthogonality_loss,
        soft_balance_loss
    )
    soft_assign = model.assignment_matrix(1.0)
    ortho_loss = soft_orthogonality_loss(x, soft_assign)
    balance_loss = soft_balance_loss(soft_assign)

    total_loss = recon_loss + vq_loss + ortho_loss + balance_loss

    # Backward
    total_loss.backward()

    # Check assignment matrix gradients exist
    assert model.assignment_matrix.logits.grad is not None
    assert not torch.isnan(model.assignment_matrix.logits.grad).any()


def test_static_and_learnable_same_output_format():
    """Test that static and learnable modes produce same output format."""
    input_dim = 20
    num_categories = 3

    group_indices = {
        "cat_0": list(range(0, 7)),
        "cat_1": list(range(7, 14)),
        "cat_2": list(range(14, 20))
    }

    vqvae_config = CategoricalVQVAEConfig(
        input_dim=input_dim,
        group_indices=group_indices,
        group_embedding_dim=16,
        group_hidden_dim=32,
        levels={
            "cat_0": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_1": [{"latent_dim": 8, "num_tokens": 64}],
            "cat_2": [{"latent_dim": 8, "num_tokens": 64}]
        }
    )

    # Static model
    static_model = CategoricalHierarchicalVQVAE(config=vqvae_config)

    # Learnable model
    assignment_matrix = SoftAssignmentMatrix(input_dim, num_categories)
    learnable_model = CategoricalHierarchicalVQVAE(
        config=vqvae_config,
        assignment_matrix=assignment_matrix
    )

    # Forward pass
    x = torch.randn(4, input_dim)
    static_outputs = static_model(x)
    learnable_outputs = learnable_model(x, temperature=1.0)

    # Check both have same keys
    assert set(static_outputs.keys()) == set(learnable_outputs.keys())
    assert "reconstruction" in static_outputs
    assert "tokens" in static_outputs
    assert "vq_losses" in static_outputs

    # Check shapes match
    assert static_outputs["reconstruction"]["features"].shape == learnable_outputs["reconstruction"]["features"].shape
    assert static_outputs["tokens"].shape == learnable_outputs["tokens"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
