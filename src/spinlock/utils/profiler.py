"""Performance profiling utilities for dataset generation.

Provides clean, structured profiling without scattered timing code.
"""

import time
import torch
from typing import Dict, Optional
from contextlib import contextmanager


class PerformanceProfiler:
    """Context-manager-based profiler for tracking operation timings.

    Usage:
        profiler = PerformanceProfiler()

        with profiler.profile("operation_name"):
            # ... code to profile
            pass

        profiler.print_summary()
    """

    def __init__(self, enable_cuda_sync: bool = True):
        """Initialize profiler.

        Args:
            enable_cuda_sync: Whether to sync CUDA before timing (more accurate but slower)
        """
        self.timings: Dict[str, list] = {}
        self.enable_cuda_sync = enable_cuda_sync

    @contextmanager
    def profile(self, operation: str):
        """Profile a code block.

        Args:
            operation: Name of the operation being profiled
        """
        # Sync CUDA for accurate timing
        if self.enable_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        try:
            yield
        finally:
            # Sync again before measuring end time
            if self.enable_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            elapsed = end_time - start_time

            if operation not in self.timings:
                self.timings[operation] = []
            self.timings[operation].append(elapsed)

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for a specific operation.

        Args:
            operation: Name of the operation

        Returns:
            Dict with min, max, mean, total times, or None if no data
        """
        if operation not in self.timings or not self.timings[operation]:
            return None

        times = self.timings[operation]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }

    def print_summary(self):
        """Print summary of all profiled operations."""
        if not self.timings:
            print("No profiling data collected")
            return

        print("\n" + "="*70)
        print("Performance Profile Summary")
        print("="*70)

        # Calculate total time across all operations
        all_times = [t for times in self.timings.values() for t in times]
        total_time = sum(all_times)

        # Sort by total time (descending)
        sorted_ops = sorted(
            self.timings.items(),
            key=lambda x: sum(x[1]),
            reverse=True
        )

        for operation, times in sorted_ops:
            stats = self.get_stats(operation)
            if stats is None:
                continue

            percent = (stats['total'] / total_time * 100) if total_time > 0 else 0

            print(f"\n{operation}:")
            print(f"  Count:   {stats['count']}")
            print(f"  Total:   {stats['total']:.3f}s ({percent:.1f}%)")
            print(f"  Mean:    {stats['mean']:.3f}s")
            print(f"  Min:     {stats['min']:.3f}s")
            print(f"  Max:     {stats['max']:.3f}s")

        print("\n" + "="*70)
        print(f"Total profiled time: {total_time:.3f}s")
        print("="*70 + "\n")

    def reset(self):
        """Clear all profiling data."""
        self.timings.clear()
