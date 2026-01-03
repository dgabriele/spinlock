"""
Feature profiling utilities for Neural Operator dataset generation.

This package provides GPU-aware profiling tools to measure runtime of individual
features during feature extraction, helping identify bottlenecks for optimization.
"""

from .context import FeatureProfilingContext
from .report import ProfilingReport
from .timers import CUDATimer, TimingAccumulator, TimingRecord

__all__ = [
    "FeatureProfilingContext",
    "ProfilingReport",
    "CUDATimer",
    "TimingAccumulator",
    "TimingRecord",
]
