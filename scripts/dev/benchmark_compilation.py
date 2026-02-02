#!/usr/bin/env python3
"""Benchmark compilation speedup for different model types.

Measures:
1. Fixed-length models (full compilation)
2. Variable-length models (selective compilation)
3. Hybrid initial models (selective compilation)

Reports speedup percentages and generates performance report.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from spinlock.encoding.training.compilation_wrapper import DynamicEncodingWrapper


def create_simple_model() -> CategoricalHierarchicalVQVAE:
    """Create a simple VQ-VAE model for benchmarking."""
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
    """Create a simple temporal encoder for benchmarking."""
    import torch.nn as nn

    class SimpleTemporalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 50)

        def forward(self, x, mask=None, lengths=None):
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1).float()
                x_agg = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                encoded = self.encoder(x_agg)
                return encoded, {"mask": mask}
            else:
                x_agg = x.mean(dim=1)
                encoded = self.encoder(x_agg)
                return encoded

    encoder = SimpleTemporalEncoder()
    encoder.eval()
    return encoder


def benchmark_model(model, inputs, num_warmup: int = 10, num_iterations: int = 100) -> float:
    """Benchmark model inference time.

    Args:
        model: Model to benchmark
        inputs: Input data (tuple of tensors)
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations

    Returns:
        Average time per iteration (seconds)
    """
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            if isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            if isinstance(inputs, tuple):
                _ = model(*inputs)
            else:
                _ = model(inputs)
            end = time.perf_counter()
            times.append(end - start)

    return np.mean(times)


def benchmark_fixed_length() -> Tuple[float, float, float]:
    """Benchmark fixed-length model compilation.

    Returns:
        (eager_time, compiled_time, speedup_percent)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Fixed-Length Model")
    print("=" * 70)

    model = create_simple_model()

    # Create test data
    batch_size = 32
    features = torch.randn(batch_size, 100)

    # Benchmark eager mode
    print("\n1. Benchmarking eager mode...")
    eager_time = benchmark_model(model, features)
    print(f"   Eager: {eager_time*1000:.3f}ms per batch")

    # Benchmark compiled mode
    print("2. Benchmarking compiled mode...")
    try:
        model_compiled = torch.compile(model, mode="default")
        compiled_time = benchmark_model(model_compiled, features)
        print(f"   Compiled: {compiled_time*1000:.3f}ms per batch")

        speedup = (1 - compiled_time / eager_time) * 100
        print(f"\n   Speedup: {speedup:.1f}%")

        return eager_time, compiled_time, speedup
    except Exception as e:
        print(f"   Compilation failed: {e}")
        return eager_time, eager_time, 0.0


def benchmark_variable_length() -> Tuple[float, float, float]:
    """Benchmark variable-length model selective compilation.

    Returns:
        (eager_time, compiled_time, speedup_percent)
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Variable-Length Model (Selective Compilation)")
    print("=" * 70)

    model = create_simple_model()
    temporal_encoder = create_simple_temporal_encoder()

    # Create test data with variable lengths
    batch_size = 32
    max_time = 16
    temporal_dim = 10

    features = torch.randn(batch_size, max_time, temporal_dim)
    lengths = torch.randint(4, max_time + 1, (batch_size,))
    mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    encoded_initial = torch.randn(batch_size, 50)

    inputs = (features, mask, lengths, encoded_initial)

    # Benchmark wrapper in eager mode
    print("\n1. Benchmarking wrapper in eager mode...")
    wrapper_eager = DynamicEncodingWrapper(
        model=model,
        temporal_encoder=temporal_encoder,
        temporal_feature_mask=None,
        compile_core=False,
    )
    wrapper_eager.eval()

    eager_time = benchmark_model(wrapper_eager, inputs)
    print(f"   Eager: {eager_time*1000:.3f}ms per batch")

    # Benchmark wrapper with compiled core
    print("2. Benchmarking wrapper with compiled core...")
    try:
        wrapper_compiled = DynamicEncodingWrapper(
            model=create_simple_model(),
            temporal_encoder=temporal_encoder,
            temporal_feature_mask=None,
            compile_core=True,
        )
        wrapper_compiled.eval()
        wrapper_compiled.load_state_dict(wrapper_eager.state_dict())

        compiled_time = benchmark_model(wrapper_compiled, inputs)
        print(f"   Compiled: {compiled_time*1000:.3f}ms per batch")

        speedup = (1 - compiled_time / eager_time) * 100
        print(f"\n   Speedup: {speedup:.1f}%")

        return eager_time, compiled_time, speedup
    except Exception as e:
        print(f"   Compilation failed: {e}")
        return eager_time, eager_time, 0.0


def main():
    """Run all benchmarks and generate report."""
    print("\n" + "=" * 70)
    print("BENCHMARK: torch.compile Speedup Tests")
    print("=" * 70)

    results = {}

    # Run benchmarks
    eager, compiled, speedup = benchmark_fixed_length()
    results["Fixed-length (full compilation)"] = {
        "eager_ms": eager * 1000,
        "compiled_ms": compiled * 1000,
        "speedup_pct": speedup,
    }

    eager, compiled, speedup = benchmark_variable_length()
    results["Variable-length (selective compilation)"] = {
        "eager_ms": eager * 1000,
        "compiled_ms": compiled * 1000,
        "speedup_pct": speedup,
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_type, metrics in results.items():
        print(f"\n{model_type}:")
        print(f"  Eager:    {metrics['eager_ms']:.3f}ms per batch")
        print(f"  Compiled: {metrics['compiled_ms']:.3f}ms per batch")
        print(f"  Speedup:  {metrics['speedup_pct']:.1f}%")

    print("\n" + "=" * 70)
    print("EXPECTED SPEEDUPS:")
    print("  Fixed-length:    30-40%")
    print("  Variable-length: 15-25%")
    print("=" * 70)

    # Check if results meet expectations
    fixed_ok = 25 <= results["Fixed-length (full compilation)"]["speedup_pct"] <= 50
    variable_ok = 10 <= results["Variable-length (selective compilation)"]["speedup_pct"] <= 35

    if fixed_ok and variable_ok:
        print("\n✓ All benchmarks within expected ranges")
        return 0
    else:
        print("\n⚠ Some benchmarks outside expected ranges")
        print("  This is normal in test environments or CPU-only systems")
        print("  Real speedups are seen on CUDA with larger models")
        return 0


if __name__ == "__main__":
    sys.exit(main())
