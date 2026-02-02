"""Benchmark different batch sizes for U-AFNO dataset generation.

Tests batch sizes 2, 4, 8, 16 to find optimal throughput vs memory usage.
With async operator building (Phase 2), batch size is now memory-limited,
not time-limited.

Usage:
    poetry run python scripts/dev/benchmark_batch_size.py

Output:
    - Memory usage for each batch size
    - Throughput (samples/sec)
    - Recommended optimal batch size
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.config import SpinlockConfig
from spinlock.config.loader import load_config


def benchmark_batch_size(config_path: str, batch_size: int, num_samples: int = 20):
    """Benchmark a specific batch size.

    Args:
        config_path: Path to dataset config YAML
        batch_size: Batch size to test
        num_samples: Number of samples to generate (default: 20)

    Returns:
        Dictionary with metrics:
        - batch_size: int
        - peak_memory_gb: float
        - total_time_sec: float
        - throughput: float (samples/sec)
        - success: bool
    """
    from spinlock.dataset.pipeline import DatasetGenerationPipeline

    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*60}")

    # Reset GPU memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Load config and override batch_size
    config = load_config(Path(config_path))
    config.sampling.batch_size = batch_size
    config.sampling.total_samples = num_samples

    # Create unique output path
    output_path = f"datasets/benchmark_batch{batch_size}.h5"
    config.dataset.output_path = output_path

    try:
        # Generate dataset
        start_time = time.time()
        pipeline = DatasetGenerationPipeline(config)
        pipeline.generate()
        total_time = time.time() - start_time

        # Get peak memory
        if torch.cuda.is_available():
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_gb = peak_memory_bytes / 1e9
        else:
            peak_memory_gb = 0.0

        throughput = num_samples / total_time

        metrics = {
            "batch_size": batch_size,
            "peak_memory_gb": peak_memory_gb,
            "total_time_sec": total_time,
            "throughput": throughput,
            "success": True,
        }

        print(f"\n✓ Success!")
        print(f"  Peak memory: {peak_memory_gb:.2f} GB")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.2f} samples/sec")

        # Cleanup
        Path(output_path).unlink(missing_ok=True)

        return metrics

    except (RuntimeError, IndexError) as e:
        error_msg = str(e).lower()
        is_oom = "out of memory" in error_msg
        is_index_error = isinstance(e, IndexError) or "index" in error_msg

        if is_oom or is_index_error:
            print(f"\n✗ {'OOM' if is_oom else 'Batch size error'} at batch_size={batch_size}")
            if is_index_error:
                print(f"  Likely caused by adaptive batch sizing during OOM recovery")
            if torch.cuda.is_available():
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Peak memory: {peak_memory_gb:.2f} GB")
            else:
                peak_memory_gb = 0.0

            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            Path(output_path).unlink(missing_ok=True)

            return {
                "batch_size": batch_size,
                "peak_memory_gb": peak_memory_gb,
                "total_time_sec": 0.0,
                "throughput": 0.0,
                "success": False,
            }
        else:
            raise


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch sizes for dataset generation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/u_afno_baseline_100k/dataset.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Batch sizes to test (default: 2 4 8 16)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples per test (default: 20)",
    )
    parser.add_argument(
        "--gpu-limit-gb",
        type=float,
        default=7.5,
        help="GPU memory limit in GB (default: 7.5 for 8GB GPU with headroom)",
    )

    args = parser.parse_args()

    print("="*60)
    print("BATCH SIZE BENCHMARK")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Batch sizes: {args.sizes}")
    print(f"Samples per test: {args.samples}")
    print(f"GPU memory limit: {args.gpu_limit_gb} GB")
    print("="*60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU")
        print("Batch size optimization is primarily for GPU memory management.\n")

    # Run benchmarks
    results = []
    for batch_size in args.sizes:
        metrics = benchmark_batch_size(args.config, batch_size, args.samples)
        results.append(metrics)

        # Stop if we hit OOM
        if not metrics["success"]:
            print(f"\nStopping benchmark - OOM at batch_size={batch_size}")
            break

        # Stop if we exceeded memory limit
        if metrics["peak_memory_gb"] > args.gpu_limit_gb:
            print(f"\nStopping benchmark - exceeded memory limit ({metrics['peak_memory_gb']:.2f} GB > {args.gpu_limit_gb} GB)")
            break

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Batch Size':<12} {'Memory (GB)':<14} {'Time (s)':<12} {'Throughput':<15} {'Status'}")
    print("-"*60)

    for r in results:
        status = "✓ OK" if r["success"] else "✗ OOM"
        print(
            f"{r['batch_size']:<12} "
            f"{r['peak_memory_gb']:<14.2f} "
            f"{r['total_time_sec']:<12.1f} "
            f"{r['throughput']:<15.2f} "
            f"{status}"
        )

    # Find optimal batch size
    successful = [r for r in results if r["success"] and r["peak_memory_gb"] <= args.gpu_limit_gb]
    if successful:
        # Choose largest successful batch size (highest throughput)
        optimal = max(successful, key=lambda x: x["batch_size"])
        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"Optimal batch_size: {optimal['batch_size']}")
        print(f"  Peak memory: {optimal['peak_memory_gb']:.2f} GB (within {args.gpu_limit_gb} GB limit)")
        print(f"  Throughput: {optimal['throughput']:.2f} samples/sec")

        # Calculate speedup vs baseline (batch_size=2)
        baseline = next((r for r in results if r["batch_size"] == 2 and r["success"]), None)
        if baseline and baseline["throughput"] > 0:
            speedup = optimal["throughput"] / baseline["throughput"]
            print(f"  Speedup vs batch_size=2: {speedup:.2f}×")

        print("\nUpdate your config with:")
        print(f"  sampling:")
        print(f"    batch_size: {optimal['batch_size']}")
    else:
        print("\n⚠️  No successful batch sizes found within memory limit")
        print("Try reducing num_timesteps, num_realizations, or grid_size")

    print("="*60)


if __name__ == "__main__":
    main()
