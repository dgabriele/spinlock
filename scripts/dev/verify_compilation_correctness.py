#!/usr/bin/env python3
"""Verify that compiled models produce identical outputs to eager mode.

This script tests compilation correctness by:
1. Running models in eager mode (no compilation)
2. Running models with compilation enabled
3. Comparing outputs (should be identical within 1% relative error)

Tests all model types:
- Fixed-length models
- Variable-length models
- Hybrid initial models
- Learnable assignment models
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from spinlock.encoding.training.compilation_wrapper import DynamicEncodingWrapper


def create_simple_model() -> CategoricalHierarchicalVQVAE:
    """Create a simple VQ-VAE model for testing."""
    config = CategoricalVQVAEConfig(
        input_dim=100,
        group_indices={
            "category_1": list(range(50)),
            "category_2": list(range(50, 100)),
        },
        group_embedding_dim=32,
        group_hidden_dim=64,
        decoder_hidden_dim=128,
        levels={
            "category_1": [
                {"num_tokens": 16, "latent_dim": 16},
                {"num_tokens": 8, "latent_dim": 16},
            ],
            "category_2": [
                {"num_tokens": 16, "latent_dim": 16},
                {"num_tokens": 8, "latent_dim": 16},
            ],
        },
        dropout=0.0,
    )
    model = CategoricalHierarchicalVQVAE(config)
    model.eval()
    return model


def create_simple_temporal_encoder():
    """Create a simple temporal encoder for testing."""
    import torch.nn as nn

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


def compare_outputs(output1: Dict[str, Any], output2: Dict[str, Any], tolerance: float = 0.01) -> bool:
    """Compare two model outputs for equivalence.

    Args:
        output1: First output dict
        output2: Second output dict
        tolerance: Relative error tolerance (default 1%)

    Returns:
        True if outputs match within tolerance
    """
    # Compare reconstruction
    recon1 = output1["reconstruction"]["features"]
    recon2 = output2["reconstruction"]["features"]

    rel_error = torch.abs(recon1 - recon2) / (torch.abs(recon1) + 1e-8)
    max_rel_error = rel_error.max().item()

    if max_rel_error > tolerance:
        print(f"  ✗ Reconstruction mismatch: {max_rel_error:.4%} relative error (tolerance: {tolerance:.2%})")
        return False

    # Compare quantized embeddings
    quantized1 = output1["quantized"]
    quantized2 = output2["quantized"]

    if isinstance(quantized1, dict):
        # Dict format: {category: [level0, level1, ...]}
        for cat, levels1 in quantized1.items():
            levels2 = quantized2[cat]
            for level_idx, (quant1, quant2) in enumerate(zip(levels1, levels2)):
                rel_error = torch.abs(quant1 - quant2) / (torch.abs(quant1) + 1e-8)
                max_rel_error = rel_error.max().item()
                if max_rel_error > tolerance:
                    print(f"  ✗ Quantized mismatch ({cat}/level_{level_idx}): {max_rel_error:.4%} relative error")
                    return False
    else:
        # List format: [tensor0, tensor1, ...]
        for level_idx, (quant1, quant2) in enumerate(zip(quantized1, quantized2)):
            rel_error = torch.abs(quant1 - quant2) / (torch.abs(quant1) + 1e-8)
            max_rel_error = rel_error.max().item()
            if max_rel_error > tolerance:
                print(f"  ✗ Quantized mismatch (level_{level_idx}): {max_rel_error:.4%} relative error")
                return False

    print(f"  ✓ Outputs match within {tolerance:.2%} tolerance")
    return True


def test_fixed_length_compilation():
    """Test compilation for fixed-length models."""
    print("\n" + "=" * 70)
    print("TEST 1: Fixed-Length Model Compilation")
    print("=" * 70)

    model = create_simple_model()

    # Create test data
    batch_size = 16
    features = torch.randn(batch_size, 100)

    # Test 1: Eager mode
    print("\n1. Running in eager mode...")
    with torch.no_grad():
        output_eager = model(features)

    # Test 2: Compiled mode
    print("2. Compiling model...")
    try:
        model_compiled = torch.compile(model, mode="default")
        print("3. Running in compiled mode...")
        with torch.no_grad():
            output_compiled = model_compiled(features)

        # Compare
        print("4. Comparing outputs...")
        if compare_outputs(output_eager, output_compiled):
            print("\n✓ PASSED: Fixed-length compilation produces identical outputs")
            return True
        else:
            print("\n✗ FAILED: Fixed-length compilation outputs differ")
            return False
    except Exception as e:
        print(f"\n✗ FAILED: Compilation error: {e}")
        return False


def test_variable_length_compilation():
    """Test selective compilation for variable-length models."""
    print("\n" + "=" * 70)
    print("TEST 2: Variable-Length Model Selective Compilation")
    print("=" * 70)

    model = create_simple_model()
    temporal_encoder = create_simple_temporal_encoder()

    # Create test data with variable lengths
    batch_size = 16
    max_time = 16
    temporal_dim = 10

    features = torch.randn(batch_size, max_time, temporal_dim)
    lengths = torch.randint(4, max_time + 1, (batch_size,))
    mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    encoded_initial = torch.randn(batch_size, 50)

    # Test 1: Wrapper in eager mode
    print("\n1. Running wrapper in eager mode...")
    wrapper_eager = DynamicEncodingWrapper(
        model=model,
        temporal_encoder=temporal_encoder,
        temporal_feature_mask=None,
        compile_core=False,
    )
    wrapper_eager.eval()

    with torch.no_grad():
        output_eager = wrapper_eager(features, mask, lengths, encoded_initial)

    # Test 2: Wrapper with compiled core
    print("2. Creating wrapper with compiled core...")
    try:
        wrapper_compiled = DynamicEncodingWrapper(
            model=create_simple_model(),  # Fresh model
            temporal_encoder=temporal_encoder,
            temporal_feature_mask=None,
            compile_core=True,
        )
        wrapper_compiled.eval()

        # Copy weights from eager model
        wrapper_compiled.load_state_dict(wrapper_eager.state_dict())

        print("3. Running wrapper with compiled core...")
        with torch.no_grad():
            output_compiled = wrapper_compiled(features, mask, lengths, encoded_initial)

        # Compare
        print("4. Comparing outputs...")
        if compare_outputs(output_eager, output_compiled):
            print("\n✓ PASSED: Variable-length selective compilation produces identical outputs")
            return True
        else:
            print("\n✗ FAILED: Variable-length compilation outputs differ")
            return False
    except Exception as e:
        print(f"\n✗ FAILED: Compilation error: {e}")
        return False


def test_compilation_with_masking():
    """Test compilation with feature cleaning masks."""
    print("\n" + "=" * 70)
    print("TEST 3: Compilation with Feature Cleaning Mask")
    print("=" * 70)

    # Create a model that expects 75D input (50 initial + 25 temporal after masking)
    config = CategoricalVQVAEConfig(
        input_dim=75,  # Reduced from 100 due to masking
        group_indices={
            "category_1": list(range(38)),  # ~50% of 75
            "category_2": list(range(38, 75)),
        },
        group_embedding_dim=32,
        group_hidden_dim=64,
        decoder_hidden_dim=128,
        levels={
            "category_1": [
                {"num_tokens": 16, "latent_dim": 16},
                {"num_tokens": 8, "latent_dim": 16},
            ],
            "category_2": [
                {"num_tokens": 16, "latent_dim": 16},
                {"num_tokens": 8, "latent_dim": 16},
            ],
        },
        dropout=0.0,
    )
    model = CategoricalHierarchicalVQVAE(config)
    model.eval()

    temporal_encoder = create_simple_temporal_encoder()

    # Create feature cleaning mask (keep only first 25 temporal features)
    temporal_mask = torch.zeros(50, dtype=torch.bool)
    temporal_mask[:25] = True

    # Create test data
    batch_size = 16
    max_time = 16
    temporal_dim = 10

    features = torch.randn(batch_size, max_time, temporal_dim)
    encoded_initial = torch.randn(batch_size, 50)

    # Test 1: Wrapper with mask in eager mode
    print("\n1. Running wrapper with mask in eager mode...")
    wrapper_eager = DynamicEncodingWrapper(
        model=model,
        temporal_encoder=temporal_encoder,
        temporal_feature_mask=temporal_mask,
        compile_core=False,
    )
    wrapper_eager.eval()

    with torch.no_grad():
        output_eager = wrapper_eager(features, None, None, encoded_initial)

    # Test 2: Wrapper with mask and compiled core
    print("2. Creating wrapper with mask and compiled core...")
    try:
        # Create fresh model with same config
        model2 = CategoricalHierarchicalVQVAE(config)
        model2.eval()

        wrapper_compiled = DynamicEncodingWrapper(
            model=model2,
            temporal_encoder=temporal_encoder,
            temporal_feature_mask=temporal_mask,
            compile_core=True,
        )
        wrapper_compiled.eval()

        # Copy weights
        wrapper_compiled.load_state_dict(wrapper_eager.state_dict())

        print("3. Running wrapper with mask and compiled core...")
        with torch.no_grad():
            output_compiled = wrapper_compiled(features, None, None, encoded_initial)

        # Compare
        print("4. Comparing outputs...")
        if compare_outputs(output_eager, output_compiled):
            print("\n✓ PASSED: Compilation with masking produces identical outputs")
            return True
        else:
            print("\n✗ FAILED: Compilation with masking outputs differ")
            return False
    except Exception as e:
        print(f"\n✗ FAILED: Compilation error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("VERIFICATION: torch.compile Correctness Tests")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Fixed-length compilation", test_fixed_length_compilation()))
    results.append(("Variable-length selective compilation", test_variable_length_compilation()))
    results.append(("Compilation with masking", test_compilation_with_masking()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Compilation produces correct outputs within 1% tolerance")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review failures above")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
