#!/usr/bin/env python3
"""
Benchmark mixed precision (FP16/BF16) speedup for temporal rollouts.

Measures:
- Throughput (timesteps/sec)
- Memory usage
- Numerical accuracy vs FP32
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from spinlock.operators.builder import OperatorBuilder, NeuralOperator
from spinlock.rollout.engine import OperatorRollout


def benchmark_precision(precision, operator, initial_condition, num_timesteps=500, num_realizations=5):
    """Benchmark a specific precision mode"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    engine = OperatorRollout(
        policy_type="convex",
        alpha=0.5,
        num_timesteps=num_timesteps,
        device=torch.device(device),
        precision=precision,
        compute_metrics=False  # Disable metrics for pure speed test
    )

    # Warmup (trigger compilation if using torch.compile)
    _ = engine.evolve_operator(
        operator=operator,
        initial_condition=initial_condition,
        num_realizations=num_realizations,
        base_seed=42
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark
    start = time.perf_counter()
    trajectories, _, _ = engine.evolve_operator(
        operator=operator,
        initial_condition=initial_condition,
        num_realizations=num_realizations,
        base_seed=42
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_allocated = 0

    return {
        'time': elapsed,
        'throughput': (num_timesteps * num_realizations) / elapsed,
        'memory_gb': mem_allocated,
        'trajectories': trajectories
    }


def main():
    print("=" * 80)
    print("MIXED PRECISION BENCHMARK")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        capability = torch.cuda.get_device_capability()
        print(f"Compute capability: sm_{capability[0]}{capability[1]}")

    print(f"PyTorch version: {torch.__version__}")

    # Build operator
    print("\n" + "-" * 80)
    print("BUILDING OPERATOR")
    print("-" * 80)

    builder = OperatorBuilder()

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
    operator = NeuralOperator(base_model, name="test_operator")

    print(f"  Layers: {params['num_layers']}, Base channels: {params['base_channels']}")
    print(f"  Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

    # Create initial condition
    initial_condition = torch.randn(3, 128, 128)

    # Rollout configuration
    num_timesteps = 500
    num_realizations = 5

    print(f"  Rollout: {num_timesteps} timesteps × {num_realizations} realizations")
    print(f"  Total forward passes: {num_timesteps * num_realizations:,}")

    # ========================================================================
    # Benchmark FP32 (baseline)
    # ========================================================================
    print("\n" + "-" * 80)
    print("FLOAT32 (BASELINE)")
    print("-" * 80)

    results_fp32 = benchmark_precision(
        "float32", operator, initial_condition, num_timesteps, num_realizations
    )

    print(f"  Time: {results_fp32['time']:.2f} seconds")
    print(f"  Throughput: {results_fp32['throughput']:.1f} timesteps/sec")
    print(f"  Memory: {results_fp32['memory_gb']:.2f} GB")

    # ========================================================================
    # Benchmark FP16
    # ========================================================================
    print("\n" + "-" * 80)
    print("FLOAT16 (MIXED PRECISION)")
    print("-" * 80)

    results_fp16 = benchmark_precision(
        "float16", operator, initial_condition, num_timesteps, num_realizations
    )

    print(f"  Time: {results_fp16['time']:.2f} seconds")
    print(f"  Throughput: {results_fp16['throughput']:.1f} timesteps/sec")
    print(f"  Memory: {results_fp16['memory_gb']:.2f} GB")

    speedup_fp16 = results_fp32['time'] / results_fp16['time']
    memory_reduction_fp16 = results_fp32['memory_gb'] / results_fp16['memory_gb']

    print(f"  Speedup: {speedup_fp16:.2f}×")
    print(f"  Memory reduction: {memory_reduction_fp16:.2f}×")

    # ========================================================================
    # Benchmark BF16 (if supported)
    # ========================================================================
    if device == 'cuda':
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # Ampere or newer
            print("\n" + "-" * 80)
            print("BFLOAT16 (MIXED PRECISION)")
            print("-" * 80)

            results_bf16 = benchmark_precision(
                "bfloat16", operator, initial_condition, num_timesteps, num_realizations
            )

            print(f"  Time: {results_bf16['time']:.2f} seconds")
            print(f"  Throughput: {results_bf16['throughput']:.1f} timesteps/sec")
            print(f"  Memory: {results_bf16['memory_gb']:.2f} GB")

            speedup_bf16 = results_fp32['time'] / results_bf16['time']
            memory_reduction_bf16 = results_fp32['memory_gb'] / results_bf16['memory_gb']

            print(f"  Speedup: {speedup_bf16:.2f}×")
            print(f"  Memory reduction: {memory_reduction_bf16:.2f}×")
        else:
            results_bf16 = None
            print("\n" + "-" * 80)
            print("BFLOAT16: NOT SUPPORTED (requires Ampere or newer)")
            print("-" * 80)
    else:
        results_bf16 = None

    # ========================================================================
    # Numerical Accuracy
    # ========================================================================
    print("\n" + "-" * 80)
    print("NUMERICAL ACCURACY")
    print("-" * 80)

    # Compare final states
    fp32_final = results_fp32['trajectories'][:, -1]  # [M, C, H, W]
    fp16_final = results_fp16['trajectories'][:, -1]

    abs_error = (fp16_final.float() - fp32_final.float()).abs()
    rel_error = abs_error / (fp32_final.float().abs() + 1e-8)

    print(f"\nFLOAT16 vs FLOAT32:")
    print(f"  Max absolute error: {abs_error.max().item():.2e}")
    print(f"  Mean absolute error: {abs_error.mean().item():.2e}")
    print(f"  Max relative error: {rel_error.max().item():.2e}")
    print(f"  Mean relative error: {rel_error.mean().item():.2e}")

    if results_bf16 is not None:
        bf16_final = results_bf16['trajectories'][:, -1]

        abs_error_bf16 = (bf16_final.float() - fp32_final.float()).abs()
        rel_error_bf16 = abs_error_bf16 / (fp32_final.float().abs() + 1e-8)

        print(f"\nBFLOAT16 vs FLOAT32:")
        print(f"  Max absolute error: {abs_error_bf16.max().item():.2e}")
        print(f"  Mean absolute error: {abs_error_bf16.mean().item():.2e}")
        print(f"  Max relative error: {rel_error_bf16.max().item():.2e}")
        print(f"  Mean relative error: {rel_error_bf16.mean().item():.2e}")

    # ========================================================================
    # Dataset Generation Projection
    # ========================================================================
    print("\n" + "=" * 80)
    print("DATASET GENERATION PROJECTION (10K OPERATORS)")
    print("=" * 80)

    # Each operator: 500 timesteps × 5 realizations
    operators_per_sec_fp32 = 1.0 / results_fp32['time']
    operators_per_sec_fp16 = 1.0 / results_fp16['time']

    total_operators = 10000
    time_fp32_hours = (total_operators / operators_per_sec_fp32) / 3600
    time_fp16_hours = (total_operators / operators_per_sec_fp16) / 3600

    print(f"\nFLOAT32:")
    print(f"  Time for 10K operators: {time_fp32_hours:.1f} hours")
    print(f"  Throughput: {operators_per_sec_fp32:.2f} operators/sec")

    print(f"\nFLOAT16:")
    print(f"  Time for 10K operators: {time_fp16_hours:.1f} hours")
    print(f"  Throughput: {operators_per_sec_fp16:.2f} operators/sec")
    print(f"  Time saved: {time_fp32_hours - time_fp16_hours:.1f} hours")

    if results_bf16 is not None:
        operators_per_sec_bf16 = 1.0 / results_bf16['time']
        time_bf16_hours = (total_operators / operators_per_sec_bf16) / 3600

        print(f"\nBFLOAT16:")
        print(f"  Time for 10K operators: {time_bf16_hours:.1f} hours")
        print(f"  Throughput: {operators_per_sec_bf16:.2f} operators/sec")
        print(f"  Time saved: {time_fp32_hours - time_bf16_hours:.1f} hours")

    # ========================================================================
    # Success Criteria
    # ========================================================================
    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA")
    print("-" * 80)

    target_speedup = 1.5
    if speedup_fp16 >= target_speedup:
        print(f"\n✓ PASSED: FP16 speedup {speedup_fp16:.2f}× >= {target_speedup:.2f}× target")
        return 0
    else:
        print(f"\n✗ FAILED: FP16 speedup {speedup_fp16:.2f}× < {target_speedup:.2f}× target")
        return 1


if __name__ == "__main__":
    sys.exit(main())
