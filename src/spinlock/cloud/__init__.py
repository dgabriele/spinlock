"""
Cloud provider abstractions for dataset generation.

This module provides pluggable backends for:
- Storage: Local HDF5, S3, GCS
- Execution: Local GPU, Lambda Labs, RunPod, AWS
- Orchestration: Local sequential, Cloud jobs, Multi-node

Design principles:
- DRY: Reuse local pipeline logic with minimal changes
- Modular: Add new providers with ~50 lines
- Backward compatible: Defaults to local execution
"""

from spinlock.cloud.storage import StorageBackend, LocalHDF5Backend, S3Backend
from spinlock.cloud.execution import ExecutionBackend, LocalExecutionBackend
from spinlock.cloud.orchestration import OrchestrationBackend, LocalSequentialOrchestrator

__all__ = [
    # Storage
    "StorageBackend",
    "LocalHDF5Backend",
    "S3Backend",
    # Execution
    "ExecutionBackend",
    "LocalExecutionBackend",
    # Orchestration
    "OrchestrationBackend",
    "LocalSequentialOrchestrator",
]
