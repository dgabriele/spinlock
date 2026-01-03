"""Feature profiling context for dataset generation."""

import torch
from contextlib import nullcontext
from typing import Optional

from .timers import CUDATimer, TimingAccumulator
from .report import ProfilingReport


class FeatureProfilingContext:
    """
    Context manager for feature extraction profiling.

    Provides GPU-aware timing for feature categories and individual features,
    with minimal overhead when disabled.

    Usage:
        profiler = FeatureProfilingContext(
            device=device,
            level='category',  # or 'feature' for fine-grained
            enabled=True
        )

        with profiler.time_category('spatial', batch_idx=0, num_samples=32):
            spatial_features = spatial_extractor.extract(...)

        # After generation completes
        report = profiler.generate_report()
        report.print_summary()
        report.save_json('profiling_results.json')
    """

    def __init__(
        self, device: torch.device, level: str = "category", enabled: bool = True
    ):
        """
        Initialize profiling context.

        Args:
            device: torch.device for GPU/CPU timing
            level: 'category' for per-category timing or 'feature' for per-feature
            enabled: If False, all timing operations are no-ops (zero overhead)
        """
        self.device = device
        self.level = level
        self.enabled = enabled
        self.accumulator = TimingAccumulator()
        self._batch_idx = 0

    def time_category(
        self, category: str, batch_idx: int, num_samples: int
    ) -> "CategoryTimingContext":
        """
        Time an entire feature category (per-timestep or per-trajectory).

        Args:
            category: Category name (e.g., 'spatial', 'temporal', 'spectral')
            batch_idx: Current batch index
            num_samples: Number of samples in batch

        Returns:
            Context manager for category timing
        """
        return CategoryTimingContext(
            category=category,
            batch_idx=batch_idx,
            num_samples=num_samples,
            device=self.device,
            accumulator=self.accumulator,
            enabled=self.enabled,
        )

    def time_feature(
        self, feature_name: str, category: str, batch_idx: int, num_samples: int
    ) -> "FeatureTimingContext":
        """
        Time a single feature extraction (fine-grained profiling).

        Only records timing if level='feature', otherwise is a no-op.

        Args:
            feature_name: Feature name (e.g., 'spatial_mean', 'fft_power_scale_0')
            category: Category this feature belongs to
            batch_idx: Current batch index
            num_samples: Number of samples in batch

        Returns:
            Context manager for feature timing
        """
        return FeatureTimingContext(
            feature_name=feature_name,
            category=category,
            batch_idx=batch_idx,
            num_samples=num_samples,
            device=self.device,
            accumulator=self.accumulator,
            enabled=self.enabled and self.level == "feature",
        )

    def generate_report(self) -> ProfilingReport:
        """
        Generate profiling report from collected data.

        Returns:
            ProfilingReport with analysis and formatting methods
        """
        stats = self.accumulator.get_stats()
        return ProfilingReport(stats, device=self.device)

    def clear(self):
        """Clear all collected timing data."""
        self.accumulator.clear()


class CategoryTimingContext:
    """Context manager for category-level timing."""

    def __init__(
        self,
        category: str,
        batch_idx: int,
        num_samples: int,
        device: torch.device,
        accumulator: TimingAccumulator,
        enabled: bool,
    ):
        self.category = category
        self.batch_idx = batch_idx
        self.num_samples = num_samples
        self.timer = CUDATimer(device, enabled)
        self.accumulator = accumulator
        self.enabled = enabled

    def __enter__(self):
        self.timer.__enter__()
        return self

    def __exit__(self, *args):
        self.timer.__exit__(*args)
        if self.enabled:
            elapsed = self.timer.elapsed_ms()
            self.accumulator.add(
                name=self.category,
                elapsed_ms=elapsed,
                category=self.category,
                batch_idx=self.batch_idx,
                num_samples=self.num_samples,
            )


class FeatureTimingContext:
    """Context manager for feature-level timing (fine-grained)."""

    def __init__(
        self,
        feature_name: str,
        category: str,
        batch_idx: int,
        num_samples: int,
        device: torch.device,
        accumulator: TimingAccumulator,
        enabled: bool,
    ):
        self.feature_name = feature_name
        self.category = category
        self.batch_idx = batch_idx
        self.num_samples = num_samples
        self.timer = CUDATimer(device, enabled)
        self.accumulator = accumulator
        self.enabled = enabled

    def __enter__(self):
        self.timer.__enter__()
        return self

    def __exit__(self, *args):
        self.timer.__exit__(*args)
        if self.enabled:
            elapsed = self.timer.elapsed_ms()
            self.accumulator.add(
                name=f"{self.category}.{self.feature_name}",
                elapsed_ms=elapsed,
                category=self.category,
                batch_idx=self.batch_idx,
                num_samples=self.num_samples,
            )
