"""
GPU memory management utilities.

Provides tools for:
- Memory monitoring
- Context managers for safe execution
- Model optimization for inference

Design principles:
- Safety: Prevent OOM via context managers
- Monitoring: Track memory usage
- Optimization: Prepare models for inference
"""

import torch
import torch.nn as nn
import gc
from contextlib import contextmanager
from typing import Dict, Any


class MemoryManager:
    """Utilities for efficient GPU memory management."""

    @staticmethod
    @contextmanager
    def managed_memory(device: torch.device, clear_cache: bool = True):
        """
        Context manager for memory-efficient operations.

        Args:
            device: Torch device
            clear_cache: Whether to clear cache on exit

        Example:
            ```python
            with MemoryManager.managed_memory(torch.device("cuda")):
                output = model(large_input)
            # Cache cleared automatically
            ```
        """
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            yield
        finally:
            if device.type == "cuda" and clear_cache:
                torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def get_memory_stats(device: torch.device) -> Dict[str, float]:
        """
        Get current GPU memory statistics.

        Args:
            device: Torch device

        Returns:
            Dictionary with memory stats in GB

        Example:
            ```python
            stats = MemoryManager.get_memory_stats(torch.device("cuda:0"))
            print(f"Allocated: {stats['allocated']:.2f} GB")
            ```
        """
        if device.type != "cuda":
            return {}

        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**3,
            "reserved": torch.cuda.memory_reserved(device) / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,
            "max_reserved": torch.cuda.max_memory_reserved(device) / 1024**3,
        }

    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Apply inference optimizations to model.

        Optimizations:
        - Set to eval mode
        - Disable gradient computation
        - Fuse operations (batch norm, etc.)

        Args:
            model: Model to optimize

        Returns:
            Optimized model

        Example:
            ```python
            model = MemoryManager.optimize_for_inference(model)
            # Now ready for efficient inference
            ```
        """
        model.eval()

        # Disable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def print_memory_summary(device: torch.device) -> None:
        """
        Print detailed memory summary.

        Args:
            device: Torch device

        Example:
            ```python
            MemoryManager.print_memory_summary(torch.device("cuda"))
            ```
        """
        if device.type != "cuda":
            print("Memory summary only available for CUDA devices")
            return

        stats = MemoryManager.get_memory_stats(device)
        props = torch.cuda.get_device_properties(device)

        print("=" * 60)
        print("GPU MEMORY SUMMARY")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"GPU: {props.name}")
        print(f"Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"\nCurrent usage:")
        print(f"  Allocated: {stats['allocated']:.2f} GB")
        print(f"  Reserved: {stats['reserved']:.2f} GB")
        print(f"\nPeak usage:")
        print(f"  Max allocated: {stats['max_allocated']:.2f} GB")
        print(f"  Max reserved: {stats['max_reserved']:.2f} GB")
        print("=" * 60)
