"""Unit tests for TokenEmbedding module."""

import torch
import pytest
from spinlock.noa.token_embedding import TokenEmbedding


def test_token_embedding_basic():
    """Test basic token embedding functionality."""
    # Create embedding for 21 tokens with varying codebook sizes
    codebook_sizes = [16, 32, 32, 64, 64, 64, 128] * 3  # 21 tokens total

    embedding = TokenEmbedding(
        num_tokens=21,
        codebook_sizes=codebook_sizes,
        embed_dim=32,
        projection_dim=64,
    )

    # Test forward pass
    batch_size = 4
    tokens = torch.randint(0, 16, (batch_size, 21))  # Random token indices
    output = embedding(tokens)

    # Verify output shape
    assert output.shape == (batch_size, 64), f"Expected shape (4, 64), got {output.shape}"

    # Verify no NaNs or infs
    assert not torch.isnan(output).any(), "Output contains NaNs"
    assert not torch.isinf(output).any(), "Output contains infs"

    print("✓ Basic token embedding test passed")


def test_token_embedding_different_tokens():
    """Test that different tokens produce different embeddings."""
    codebook_sizes = [16] * 21

    embedding = TokenEmbedding(
        num_tokens=21,
        codebook_sizes=codebook_sizes,
        embed_dim=32,
        projection_dim=64,
    )

    # Create two different token sets
    tokens1 = torch.zeros(1, 21, dtype=torch.long)  # All zeros
    tokens2 = torch.ones(1, 21, dtype=torch.long)   # All ones

    output1 = embedding(tokens1)
    output2 = embedding(tokens2)

    # Verify they produce different embeddings
    assert not torch.allclose(output1, output2), "Different tokens should produce different embeddings"

    print("✓ Token differentiation test passed")


def test_token_embedding_batch_independence():
    """Test that batch samples are processed independently."""
    codebook_sizes = [32] * 21

    embedding = TokenEmbedding(
        num_tokens=21,
        codebook_sizes=codebook_sizes,
        embed_dim=32,
        projection_dim=64,
    )

    # Create batch with identical samples
    tokens_single = torch.randint(0, 32, (1, 21))
    tokens_batch = tokens_single.repeat(4, 1)  # [4, 21] with identical rows

    output_single = embedding(tokens_single)
    output_batch = embedding(tokens_batch)

    # Verify all batch outputs are identical to single output
    for i in range(4):
        assert torch.allclose(output_batch[i], output_single[0]), \
            f"Batch sample {i} differs from single sample"

    print("✓ Batch independence test passed")


if __name__ == "__main__":
    test_token_embedding_basic()
    test_token_embedding_different_tokens()
    test_token_embedding_batch_independence()
    print("\n✓ All token embedding tests passed!")
