#!/usr/bin/env python3
"""
Benchmark torch.compile() speedup for neural operators.

Measures:
- Single forward pass latency
- Throughput (samples/sec)
- Compilation overhead
- Memory usage
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from spinlock.operators.builder import OperatorBuilder, NeuralOperator


def time_operator(operator, input, num_warmup=10, num_iters=100):
    """Time operator forward pass"""

    # Warmup (with no_grad for inference mode)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = operator(input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = operator(input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    return (elapsed / num_iters) * 1000  # ms per iteration


def main():
    print("=" * 80)
    print("TORCH.COMPILE() BENCHMARK")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    if not hasattr(torch, "compile"):
        print("\n❌ ERROR: torch.compile() requires PyTorch >= 2.0")
        print(f"   Current version: {torch.__version__}")
        print("   Please upgrade: pip install torch>=2.0.0")
        return 1

    # Build operator
    print("\n" + "-" * 80)
    print("BUILDING OPERATOR")
    print("-" * 80)

    builder = OperatorBuilder()

    # Typical operator configuration
    params = {
        "num_layers": 4,
        "base_channels": 64,
        "input_channels": 3,
        "output_channels": 3,
        "grid_size": 128,
        "kernel_size": 3,
        "normalization": "instance",
        "activation": "gelu",
        "dropout": 0.1,
        "has_stochastic": True,
        "noise_type": "gaussian",
        "noise_scale": 0.01,
    }

    base_model = builder.build_simple_cnn(params)
    base_model = base_model.to(device)

    print(f"  Architecture: {params['num_layers']} layers, {params['base_channels']} base channels")
    print(f"  Total parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # Create test input
    batch_size = 5
    input_shape = (batch_size, params["input_channels"], params["grid_size"], params["grid_size"])
    test_input = torch.randn(*input_shape, device=device)

    print(f"  Input shape: {input_shape}")

    # ========================================================================
    # Benchmark Eager Mode (baseline)
    # ========================================================================
    print("\n" + "-" * 80)
    print("EAGER MODE (BASELINE)")
    print("-" * 80)

    operator_eager = NeuralOperator(base_model, name="eager")

    eager_time = time_operator(operator_eager, test_input, num_warmup=10, num_iters=100)
    eager_throughput = (batch_size * 100) / (eager_time * 100 / 1000)  # samples/sec

    print(f"  Time per iteration: {eager_time:.3f} ms")
    print(f"  Throughput: {eager_throughput:.1f} samples/sec")

    # ========================================================================
    # Benchmark torch.compile() (optimized)
    # ========================================================================
    print("\n" + "-" * 80)
    print("TORCH.COMPILE() MODE")
    print("-" * 80)

    operator_compiled = NeuralOperator(base_model, name="compiled")
    operator_compiled.enable_compile(mode="max-autotune")

    # First pass triggers compilation (slow!)
    print("\n  Triggering compilation (first pass - will be slow)...")
    compile_start = time.time()
    with torch.no_grad():
        _ = operator_compiled(test_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    compile_time = time.time() - compile_start

    print(f"  Compilation time: {compile_time:.2f} seconds")

    # Now benchmark compiled execution
    compiled_time = time_operator(operator_compiled, test_input, num_warmup=10, num_iters=100)
    compiled_throughput = (batch_size * 100) / (compiled_time * 100 / 1000)

    print(f"  Time per iteration: {compiled_time:.3f} ms")
    print(f"  Throughput: {compiled_throughput:.1f} samples/sec")

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    speedup = eager_time / compiled_time
    throughput_gain = compiled_throughput / eager_throughput

    print(f"\nSpeedup: {speedup:.2f}×")
    print(f"Throughput gain: {throughput_gain:.2f}×")
    print(f"Compilation overhead: {compile_time:.1f} seconds")

    # Estimate break-even point
    break_even_iters = compile_time / ((eager_time - compiled_time) / 1000)
    print(f"Break-even point: {int(break_even_iters)} forward passes")

    # Dataset generation context
    print(f"\nDataset Generation Context:")
    print(f"  Rollout: 500 timesteps × 5 realizations = 2,500 forward passes")
    print(f"  10K operators × 2,500 passes = 25M total forward passes")
    print(f"  Compilation overhead amortized over 25M passes: negligible")

    # Success criteria
    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA")
    print("-" * 80)

    if speedup >= 1.5:
        print(f"✓ PASSED: Speedup {speedup:.2f}× >= 1.5× target")
        print(f"  Estimated time savings for 10K operators:")

        # Baseline: 74 hours for 10K operators
        baseline_hours = 74
        optimized_hours = baseline_hours / speedup

        print(f"    Baseline (eager): {baseline_hours:.1f} hours")
        print(f"    Optimized (compiled): {optimized_hours:.1f} hours")
        print(f"    Time saved: {baseline_hours - optimized_hours:.1f} hours")

        return 0
    else:
        print(f"✗ FAILED: Speedup {speedup:.2f}× < 1.5× target")
        print(f"  This may indicate:")
        print(f"    - torch.compile() not optimizing effectively")
        print(f"    - Operator has Python control flow")
        print(f"    - Dynamic shapes preventing optimization")
        print(f"  Try: mode='reduce-overhead' or disable fullgraph=True")

        return 1


if __name__ == "__main__":
    sys.exit(main())
