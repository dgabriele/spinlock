"""Tests for V2 tokens encoder re-exports."""

import pytest
import torch

from spinlock.v2.tokens.encoders import (
    PyramidTemporalEncoder,
    TemporalMeanEncoder,
    TemporalCNNEncoder,
    InitialCNNEncoder,
    InitialHybridEncoder,
)


def test_pyramid_temporal_encoder_import():
    """Test PyramidTemporalEncoder can be imported and instantiated."""
    encoder = PyramidTemporalEncoder(
        input_dim=64,
        level_dims=[32, 64, 96, 128],
    )

    assert encoder.output_dim == 320
    assert len(encoder.output_dims_per_level) == 4


def test_temporal_mean_encoder_import():
    """Test TemporalMeanEncoder can be imported and instantiated."""
    encoder = TemporalMeanEncoder(input_dim=64)

    assert encoder.output_dim == 64


def test_temporal_cnn_encoder_import():
    """Test TemporalCNNEncoder can be imported and instantiated."""
    encoder = TemporalCNNEncoder(input_dim=56, embedding_dim=64)

    assert encoder.output_dim == 64


def test_initial_cnn_encoder_import():
    """Test InitialCNNEncoder can be imported and instantiated."""
    encoder = InitialCNNEncoder(embedding_dim=28)

    assert encoder.output_dim == 28


def test_initial_hybrid_encoder_import():
    """Test InitialHybridEncoder can be imported and instantiated."""
    encoder = InitialHybridEncoder(
        manual_dim=14,
        cnn_embedding_dim=28,
    )

    assert encoder.output_dim == 42  # 14 + 28


def test_pyramid_temporal_encoder_forward():
    """Test PyramidTemporalEncoder forward pass."""
    encoder = PyramidTemporalEncoder(
        input_dim=64,
        level_dims=[32, 64],
        downsample_factors=[1, 2],
    )

    x = torch.randn(4, 128, 64)  # [B, T, D]
    out = encoder(x)

    assert out.shape == (4, 96)  # 32 + 64


def test_temporal_mean_encoder_forward():
    """Test TemporalMeanEncoder forward pass."""
    encoder = TemporalMeanEncoder(input_dim=64)

    x = torch.randn(4, 128, 64)  # [B, T, D]
    out = encoder(x)

    assert out.shape == (4, 64)


def test_initial_cnn_encoder_forward():
    """Test InitialCNNEncoder forward pass."""
    encoder = InitialCNNEncoder(embedding_dim=28)

    x = torch.randn(4, 1, 128, 128)  # [B, C, H, W]
    out = encoder(x)

    assert out.shape == (4, 28)


def test_initial_hybrid_encoder_forward():
    """Test InitialHybridEncoder forward pass."""
    encoder = InitialHybridEncoder(
        manual_dim=14,
        cnn_embedding_dim=28,
    )

    manual = torch.randn(4, 14)  # [B, D_manual]
    raw = torch.randn(4, 1, 128, 128)  # [B, C, H, W]

    out = encoder(manual, raw)

    assert out.shape == (4, 42)  # 14 + 28
