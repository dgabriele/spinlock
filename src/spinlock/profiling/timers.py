"""GPU-aware timing utilities for feature profiling."""

import time
import threading
from typing import Dict, List
from dataclasses import dataclass, field
from collections import defaultdict

import torch


@dataclass
class TimingRecord:
    """Single timing measurement."""

    name: str
    elapsed_ms: float
    category: str
    batch_idx: int
    num_samples: int  # Batch size for normalization


class CUDATimer:
    """
    GPU-aware timer using CUDA events for accurate GPU timing.

    Uses torch.cuda.Event when running on GPU, falls back to time.perf_counter() for CPU.
    Properly synchronizes to ensure accurate measurements of GPU operations.
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()

        if self.use_cuda and self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None

    def __enter__(self):
        if not self.enabled:
            return self

        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            self.start_event.record(torch.cuda.current_stream(self.device))
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        if not self.enabled:
            return

        if self.use_cuda:
            self.end_event.record(torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if not self.enabled:
            return 0.0

        if self.use_cuda:
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self.start_time) * 1000.0


class TimingAccumulator:
    """
    Thread-safe accumulator for timing measurements.

    Collects timing records from multiple batches and provides aggregate statistics.
    """

    def __init__(self):
        self.records: List[TimingRecord] = []
        self.lock = threading.Lock()

    def add(
        self, name: str, elapsed_ms: float, category: str, batch_idx: int, num_samples: int
    ):
        """Add timing record (thread-safe)."""
        with self.lock:
            self.records.append(
                TimingRecord(
                    name=name,
                    elapsed_ms=elapsed_ms,
                    category=category,
                    batch_idx=batch_idx,
                    num_samples=num_samples,
                )
            )

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate statistics per feature/category.

        Returns:
            Dict mapping feature/category name to stats dict containing:
                - total_ms: Total time across all batches
                - count: Number of measurements
                - mean_ms: Average time per measurement
                - min_ms: Minimum time
                - max_ms: Maximum time
                - total_samples: Total samples processed
                - ms_per_sample: Time per sample
                - category: Category name
        """
        stats = defaultdict(
            lambda: {
                "total_ms": 0.0,
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": float("inf"),
                "max_ms": 0.0,
                "total_samples": 0,
            }
        )

        with self.lock:
            for record in self.records:
                s = stats[record.name]
                s["total_ms"] += record.elapsed_ms
                s["count"] += 1
                s["min_ms"] = min(s["min_ms"], record.elapsed_ms)
                s["max_ms"] = max(s["max_ms"], record.elapsed_ms)
                s["total_samples"] += record.num_samples
                s["category"] = record.category

        # Compute means
        for name, s in stats.items():
            if s["count"] > 0:
                s["mean_ms"] = s["total_ms"] / s["count"]
            if s["total_samples"] > 0:
                s["ms_per_sample"] = s["total_ms"] / s["total_samples"]
            else:
                s["ms_per_sample"] = 0.0

        return dict(stats)

    def clear(self):
        """Clear all timing records."""
        with self.lock:
            self.records.clear()

    def __len__(self) -> int:
        """Get number of timing records."""
        with self.lock:
            return len(self.records)
