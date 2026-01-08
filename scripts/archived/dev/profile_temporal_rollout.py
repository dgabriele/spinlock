#!/usr/bin/env python3
"""
Profile temporal rollout to identify performance bottlenecks.

Usage:
    poetry run python scripts/dev/profile_temporal_rollout.py [--samples N]
"""

import cProfile
import pstats
import io
import argparse
from pathlib import Path
from spinlock.config import load_config
from spinlock.dataset.pipeline import DatasetGenerationPipeline


def profile_generation(num_samples: int = 50):
    """Profile dataset generation with temporal rollout."""

    # Load config and modify for small test
    config_path = Path("configs/experiments/vqvae_baseline_10k_temporal/dataset.yaml")
    config = load_config(config_path)

    # Override for small test
    config.sampling.total_samples = num_samples
    config.sampling.batch_size = min(10, num_samples)  # Small batch for profiling
    config.dataset.output_path = Path(f"datasets/profile_test_{num_samples}.h5")
    config.dataset.storage.chunk_size = min(config.dataset.storage.chunk_size, num_samples)

    print("=" * 80)
    print("PROFILING TEMPORAL ROLLOUT")
    print("=" * 80)
    print(f"Samples: {num_samples}")
    print(f"Batch size: {config.sampling.batch_size}")
    print(f"Timesteps: {config.simulation.num_timesteps}")
    print(f"Realizations: {config.simulation.num_realizations}")
    print("=" * 80)
    print()

    # Create profiler
    profiler = cProfile.Profile()

    # Run generation with profiling
    pipeline = DatasetGenerationPipeline(config)

    print("Starting profiled generation...")
    profiler.enable()
    pipeline.generate()
    profiler.disable()

    # Analyze results
    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    # Sort by cumulative time
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')

    print("\nTop 50 functions by cumulative time:")
    print("-" * 80)
    stats.print_stats(50)
    print(s.getvalue())

    # Sort by time per call
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('tottime')

    print("\nTop 30 functions by total time (excluding subcalls):")
    print("-" * 80)
    stats.print_stats(30)
    print(s.getvalue())

    # Save detailed stats to file
    output_file = f"profiling_results_{num_samples}_samples.txt"
    with open(output_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        f.write("=" * 80 + "\n")
        f.write("PROFILING RESULTS - SORTED BY CUMULATIVE TIME\n")
        f.write("=" * 80 + "\n\n")
        stats.print_stats()

        stats.sort_stats('tottime')
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("PROFILING RESULTS - SORTED BY TOTAL TIME\n")
        f.write("=" * 80 + "\n\n")
        stats.print_stats()

    print(f"\n✓ Detailed profiling results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_time = sum(stats.total_tt for stats in [pstats.Stats(profiler)])
    print(f"Total time: {total_time:.2f}s")
    print(f"Time per sample: {total_time / num_samples:.2f}s")
    print(f"Projected 10K time: {(total_time / num_samples * 10000) / 3600:.2f}h")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile temporal rollout generation")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)"
    )

    args = parser.parse_args()
    profile_generation(args.samples)
