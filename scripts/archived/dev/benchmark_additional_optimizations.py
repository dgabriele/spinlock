#!/usr/bin/env python3
"""
Benchmark additional optimizations for 2× speedup target.

Tests:
1. Baseline: FP16 + cuDNN benchmark, batch_size=2
2. Batch size 3 (memory permitting)
3. torch.compile with reduce-overhead mode (better for stochastic ops)
4. Combined optimizations

Goal: Achieve ~2× additional speedup on top of existing 1.89× FP16 speedup
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


def benchmark_configuration(
    batch_size: int,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    num_timesteps: int = 500,
    num_realizations: int = 5,
    num_warmup: int = 2,
    num_runs: int = 3
):
    """Benchmark a specific configuration"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build operator
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

    # Enable torch.compile if requested
    if use_compile:
        operator.enable_compile(mode=compile_mode)

    # Create initial conditions (batched)
    initial_condition = torch.randn(3, 128, 128, device=device)

    # Create rollout engine (FP16 + cuDNN benchmark enabled by default)
    engine = OperatorRollout(
        policy_type="convex",
        alpha=0.5,
        num_timesteps=num_timesteps,
        device=torch.device(device),
        precision="float16",  # Already using FP16
        compute_metrics=False
    )

    # Warmup runs
    print(f"  Warming up ({num_warmup} runs)...", end="", flush=True)
    for _ in range(num_warmup):
        try:
            _ = engine.evolve_operator(
                operator=operator,
                initial_condition=initial_condition,
                num_realizations=num_realizations * batch_size,  # Effective batch size
                base_seed=42
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n  OOM ERROR with batch_size={batch_size}")
                return None
            raise
    print(" done")

    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...", end="", flush=True)
    times = []
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        start = time.perf_counter()
        trajectories, _, _ = engine.evolve_operator(
            operator=operator,
            initial_condition=initial_condition,
            num_realizations=num_realizations * batch_size,
            base_seed=42
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    print(" done")

    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    else:
        mem_allocated = 0

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return {
        'time': avg_time,
        'std': std_time,
        'throughput': (num_timesteps * num_realizations * batch_size) / avg_time,
        'memory_gb': mem_allocated,
        'times': times
    }


def main():
    print("=" * 80)
    print("ADDITIONAL OPTIMIZATIONS BENCHMARK")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        capability = torch.cuda.get_device_capability()
        print(f"Compute capability: sm_{capability[0]}{capability[1]}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU memory: {total_mem:.1f} GB")

    print(f"PyTorch version: {torch.__version__}")

    results = {}

    # ========================================================================
    # Configuration 1: Baseline (batch_size=2, no compile)
    # ========================================================================
    print("\n" + "-" * 80)
    print("CONFIG 1: BASELINE (batch_size=2, FP16 + cuDNN benchmark)")
    print("-" * 80)

    results['baseline'] = benchmark_configuration(batch_size=2, use_compile=False)

    if results['baseline']:
        print(f"  Time: {results['baseline']['time']:.2f} ± {results['baseline']['std']:.2f} seconds")
        print(f"  Throughput: {results['baseline']['throughput']:.1f} timesteps/sec")
        print(f"  Memory: {results['baseline']['memory_gb']:.2f} GB")

    # ========================================================================
    # Configuration 2: Batch size 3
    # ========================================================================
    print("\n" + "-" * 80)
    print("CONFIG 2: BATCH_SIZE=3 (no compile)")
    print("-" * 80)

    results['batch3'] = benchmark_configuration(batch_size=3, use_compile=False)

    if results['batch3']:
        print(f"  Time: {results['batch3']['time']:.2f} ± {results['batch3']['std']:.2f} seconds")
        print(f"  Throughput: {results['batch3']['throughput']:.1f} timesteps/sec")
        print(f"  Memory: {results['batch3']['memory_gb']:.2f} GB")

        speedup = results['baseline']['time'] / results['batch3']['time']
        print(f"  Speedup vs baseline: {speedup:.2f}×")

    # ========================================================================
    # Configuration 3: torch.compile with reduce-overhead
    # ========================================================================
    print("\n" + "-" * 80)
    print("CONFIG 3: torch.compile(mode='reduce-overhead') + batch_size=2")
    print("-" * 80)
    print("  (Compiling... this takes 15-20 seconds)")

    results['compile'] = benchmark_configuration(
        batch_size=2,
        use_compile=True,
        compile_mode="reduce-overhead"
    )

    if results['compile']:
        print(f"  Time: {results['compile']['time']:.2f} ± {results['compile']['std']:.2f} seconds")
        print(f"  Throughput: {results['compile']['throughput']:.1f} timesteps/sec")
        print(f"  Memory: {results['compile']['memory_gb']:.2f} GB")

        speedup = results['baseline']['time'] / results['compile']['time']
        print(f"  Speedup vs baseline: {speedup:.2f}×")

    # ========================================================================
    # Configuration 4: Combined (batch_size=3 + compile)
    # ========================================================================
    if results['batch3'] is not None:  # Only if batch_size=3 works
        print("\n" + "-" * 80)
        print("CONFIG 4: COMBINED (batch_size=3 + torch.compile)")
        print("-" * 80)
        print("  (Compiling... this takes 15-20 seconds)")

        results['combined'] = benchmark_configuration(
            batch_size=3,
            use_compile=True,
            compile_mode="reduce-overhead"
        )

        if results['combined']:
            print(f"  Time: {results['combined']['time']:.2f} ± {results['combined']['std']:.2f} seconds")
            print(f"  Throughput: {results['combined']['throughput']:.1f} timesteps/sec")
            print(f"  Memory: {results['combined']['memory_gb']:.2f} GB")

            speedup = results['baseline']['time'] / results['combined']['time']
            print(f"  Speedup vs baseline: {speedup:.2f}×")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_time = results['baseline']['time']

    print(f"\nBaseline (batch_size=2, FP16 + cuDNN): {baseline_time:.2f}s (1.0×)")

    if results.get('batch3'):
        speedup = baseline_time / results['batch3']['time']
        print(f"+ Batch size 3:                         {results['batch3']['time']:.2f}s ({speedup:.2f}×)")

    if results.get('compile'):
        speedup = baseline_time / results['compile']['time']
        print(f"+ torch.compile (reduce-overhead):      {results['compile']['time']:.2f}s ({speedup:.2f}×)")

    if results.get('combined'):
        speedup = baseline_time / results['combined']['time']
        print(f"+ COMBINED (batch_size=3 + compile):    {results['combined']['time']:.2f}s ({speedup:.2f}×)")

        if speedup >= 2.0:
            print(f"\n✓ SUCCESS: {speedup:.2f}× speedup >= 2.0× target!")
        else:
            print(f"\n⚠ PARTIAL: {speedup:.2f}× speedup < 2.0× target")
            print(f"  Gap: {2.0 / speedup:.2f}× additional speedup needed")

    # Dataset generation projection
    print("\n" + "=" * 80)
    print("DATASET GENERATION PROJECTION (10K OPERATORS)")
    print("=" * 80)

    baseline_hours = (10000 * baseline_time) / 3600
    print(f"\nBaseline: {baseline_hours:.1f} hours")

    if results.get('combined'):
        combined_hours = (10000 * results['combined']['time']) / 3600
        time_saved = baseline_hours - combined_hours
        print(f"Combined: {combined_hours:.1f} hours (saves {time_saved:.1f} hours)")
    elif results.get('batch3'):
        batch3_hours = (10000 * results['batch3']['time']) / 3600
        time_saved = baseline_hours - batch3_hours
        print(f"Batch size 3: {batch3_hours:.1f} hours (saves {time_saved:.1f} hours)")


if __name__ == "__main__":
    main()
