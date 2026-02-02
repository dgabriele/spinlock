"""Tests for compilation wrapper with variable-length encoding."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from spinlock.encoding.training.compilation_wrapper import (
    CompilableVQVAECore,
    DynamicEncodingWrapper,
)
from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig


@pytest.fixture
def simple_config():
    """Create a simple VQ-VAE config for testing."""
    return CategoricalVQVAEConfig(
        input_dim=100,
        group_indices={
            "category_1": list(range(50)),
            "category_2": list(range(50, 100)),
        },
        group_embedding_dim=32,
        group_hidden_dim=64,
        decoder_hidden_dim=128,
        levels={
            "category_1": [{"num_tokens": 16, "latent_dim": 16}],
            "category_2": [{"num_tokens": 16, "latent_dim": 16}],
        },
        dropout=0.0,
    )


@pytest.fixture
def simple_model(simple_config):
    """Create a simple VQ-VAE model for testing."""
    model = CategoricalHierarchicalVQVAE(simple_config)
    model.eval()
    return model


@pytest.fixture
def simple_temporal_encoder():
    """Create a simple temporal encoder for testing."""
    class SimpleTemporalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 50)  # Encode 10D temporal to 50D

        def forward(self, x, mask=None, lengths=None):
            # x: [B, T, D]
            if mask is not None:
                # Variable-length: aggregate over valid timesteps
                x_masked = x * mask.unsqueeze(-1).float()
                x_agg = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                encoded = self.encoder(x_agg)
                return encoded, {"mask": mask}
            else:
                # Fixed-length: just aggregate
                x_agg = x.mean(dim=1)
                encoded = self.encoder(x_agg)
                return encoded

    encoder = SimpleTemporalEncoder()
    encoder.eval()
    return encoder


def test_compilable_core_matches_original(simple_model):
    """Test that CompilableVQVAECore produces identical outputs to original model."""
    # Create wrapped core
    core = CompilableVQVAECore(simple_model)
    core.eval()

    # Create test input
    batch_size = 8
    features = torch.randn(batch_size, 100)

    # Forward through both
    with torch.no_grad():
        output_original = simple_model(features)
        output_wrapped = core(features)

    # Check that outputs match
    assert output_original.keys() == output_wrapped.keys()

    # Check reconstruction
    recon_orig = output_original["reconstruction"]["features"]
    recon_wrapped = output_wrapped["reconstruction"]["features"]
    assert torch.allclose(recon_orig, recon_wrapped, rtol=1e-5, atol=1e-6)

    # Check quantized (output is a list of tensors, not a dict)
    quantized_orig = output_original["quantized"]
    quantized_wrapped = output_wrapped["quantized"]

    if isinstance(quantized_orig, dict):
        # Dict format: {category: [level0, level1, ...]}
        for cat, levels in quantized_orig.items():
            for level_idx, orig_level in enumerate(levels):
                wrapped_level = quantized_wrapped[cat][level_idx]
                assert torch.allclose(orig_level, wrapped_level, rtol=1e-5, atol=1e-6)
    else:
        # List format: [tensor0, tensor1, ...]
        for orig_level, wrapped_level in zip(quantized_orig, quantized_wrapped):
            assert torch.allclose(orig_level, wrapped_level, rtol=1e-5, atol=1e-6)


def test_dynamic_wrapper_variable_length(simple_model, simple_temporal_encoder):
    """Test DynamicEncodingWrapper with variable-length encoding."""
    # Create wrapper (without compilation for deterministic testing)
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=simple_temporal_encoder,
        temporal_feature_mask=None,
        compile_core=False,  # Disable compilation for testing
    )
    wrapper.eval()

    # Create test input with variable lengths
    batch_size = 8
    max_time = 16
    temporal_dim = 10

    # Raw temporal features [B, T, D]
    features = torch.randn(batch_size, max_time, temporal_dim)

    # Create variable-length mask
    lengths = torch.randint(4, max_time + 1, (batch_size,))
    mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    # Create encoded initial features
    encoded_initial = torch.randn(batch_size, 50)

    # Forward through wrapper
    with torch.no_grad():
        outputs = wrapper(features, mask, lengths, encoded_initial)

    # Check outputs
    assert "reconstruction" in outputs
    assert "quantized" in outputs
    assert outputs["reconstruction"]["features"].shape[0] == batch_size
    assert outputs["reconstruction"]["features"].shape[1] == 100  # Full input_dim


def test_wrapper_fallback_on_compilation_failure(simple_model):
    """Test that wrapper gracefully falls back to eager mode on compilation failure."""
    # Create wrapper with compilation enabled
    # Note: compilation might succeed or fail depending on environment
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=None,
        temporal_feature_mask=None,
        compile_core=True,
    )
    wrapper.eval()

    # Test that forward still works
    batch_size = 8
    features = torch.randn(batch_size, 100)

    with torch.no_grad():
        outputs = wrapper(features)

    assert "reconstruction" in outputs
    assert outputs["reconstruction"]["features"].shape[0] == batch_size


def test_wrapper_attribute_delegation(simple_model):
    """Test that wrapper correctly delegates attributes to original model."""
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=None,
        temporal_feature_mask=None,
        compile_core=False,
    )

    # Test config access
    assert hasattr(wrapper, "config")
    assert wrapper.config.input_dim == 100

    # Test parameter access
    original_params = list(simple_model.parameters())
    wrapper_params = list(wrapper.parameters())
    assert len(original_params) == len(wrapper_params)

    # Test state_dict
    original_state = simple_model.state_dict()
    wrapper_state = wrapper.state_dict()
    assert original_state.keys() == wrapper_state.keys()


def test_wrapper_training_mode(simple_model):
    """Test that wrapper correctly propagates training mode."""
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=None,
        temporal_feature_mask=None,
        compile_core=False,
    )

    # Test train mode
    wrapper.train()
    assert simple_model.training

    # Test eval mode
    wrapper.eval()
    assert not simple_model.training


def test_encode_features_with_masking(simple_model, simple_temporal_encoder):
    """Test encode_features method with variable-length masking."""
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=simple_temporal_encoder,
        temporal_feature_mask=None,
        compile_core=False,
    )
    wrapper.eval()

    # Create test input
    batch_size = 8
    max_time = 16
    temporal_dim = 10

    features = torch.randn(batch_size, max_time, temporal_dim)
    lengths = torch.randint(4, max_time + 1, (batch_size,))
    mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True

    encoded_initial = torch.randn(batch_size, 50)

    # Encode features
    with torch.no_grad():
        encoded = wrapper.encode_features(features, mask, lengths, encoded_initial)

    # Check output shape (50D initial + 50D temporal = 100D)
    assert encoded.shape == (batch_size, 100)


def test_encode_features_without_temporal_encoder(simple_model):
    """Test encode_features when no temporal encoder (returns features as-is)."""
    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=None,
        temporal_feature_mask=None,
        compile_core=False,
    )
    wrapper.eval()

    # Create test input (already encoded)
    batch_size = 8
    features = torch.randn(batch_size, 100)

    # Encode features
    with torch.no_grad():
        encoded = wrapper.encode_features(features)

    # Should return features unchanged
    assert torch.equal(encoded, features)


def test_feature_cleaning_mask(simple_model, simple_temporal_encoder):
    """Test that temporal feature mask is applied correctly."""
    # Create mask that keeps only first 25 temporal features
    temporal_mask = torch.zeros(50, dtype=torch.bool)
    temporal_mask[:25] = True

    wrapper = DynamicEncodingWrapper(
        model=simple_model,
        temporal_encoder=simple_temporal_encoder,
        temporal_feature_mask=temporal_mask,
        compile_core=False,
    )
    wrapper.eval()

    # Create test input
    batch_size = 8
    max_time = 16
    temporal_dim = 10

    features = torch.randn(batch_size, max_time, temporal_dim)
    encoded_initial = torch.randn(batch_size, 50)

    # Encode features
    with torch.no_grad():
        encoded = wrapper.encode_features(features, None, None, encoded_initial)

    # Check output shape (50D initial + 25D temporal = 75D)
    assert encoded.shape == (batch_size, 75)
