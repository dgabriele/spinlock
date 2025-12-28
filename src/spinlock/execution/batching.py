"""
Adaptive batch sizing for memory-efficient GPU execution.

Automatically computes optimal batch sizes based on:
- Available GPU memory
- Model size
- Input dimensions
- Mixed precision usage

Design principles:
- Memory-aware: Never OOM
- Adaptive: Works on 8GB RTX 3000 to 80GB A100
- Performance: Maximize GPU utilization
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    batch_size: int
    grid_size: int
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: Optional[int] = 2


class AdaptiveBatchSizer:
    """
    Compute optimal batch size for available GPU memory.

    Features:
    - Memory estimation based on model and input size
    - Safety factor to prevent OOM
    - Power-of-2 batch sizes for GPU efficiency
    - Fallback for CPU execution

    Example:
        ```python
        sizer = AdaptiveBatchSizer(model, device=torch.device("cuda"))

        config = sizer.compute_batch_size(
            grid_size=256,
            in_channels=3,
            use_mixed_precision=True
        )

        print(f"Optimal batch size: {config.batch_size}")
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        safety_factor: float = 0.85,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
    ):
        """
        Initialize adaptive batch sizer.

        Args:
            model: Neural network model
            device: Torch device
            safety_factor: Fraction of GPU memory to use (0-1)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        """
        self.model = model
        self.device = device
        self.safety_factor = safety_factor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Cache memory info
        self.total_memory = self._get_total_memory()
        self.model_memory = self._estimate_model_memory()

    def _get_total_memory(self) -> int:
        """Get total GPU memory in bytes."""
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            return props.total_memory
        return 8 * 1024**3  # 8GB default for CPU

    def _estimate_model_memory(self) -> int:
        """Estimate model parameter memory in bytes."""
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return param_memory + buffer_memory

    def compute_batch_size(
        self,
        grid_size: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        use_mixed_precision: bool = True,
        num_realizations: int = 1,
        activation_factor: float = 4.0,
    ) -> BatchConfig:
        """
        Compute optimal batch size that fits in memory.

        Args:
            grid_size: Spatial grid dimension (assumes square grids)
            in_channels: Number of input channels
            out_channels: Number of output channels (defaults to in_channels)
            use_mixed_precision: Whether using FP16
            num_realizations: Number of stochastic realizations per sample
            activation_factor: Multiplier for activation memory (typically 3-5x)

        Returns:
            BatchConfig with optimal settings

        Example:
            ```python
            config = sizer.compute_batch_size(
                grid_size=512,
                in_channels=3,
                use_mixed_precision=True
            )
            ```
        """
        out_channels = out_channels or in_channels
        bytes_per_element = 2 if use_mixed_precision else 4

        # Memory per sample (rough estimate)
        # Input: [C_in, H, W]
        input_memory = in_channels * grid_size * grid_size * bytes_per_element

        # Output: [M, C_out, H, W] (M realizations)
        output_memory = num_realizations * out_channels * grid_size * grid_size * bytes_per_element

        # Activations: typically 3-5x input size (intermediate features)
        activation_memory = input_memory * activation_factor

        # Total per sample
        memory_per_sample = input_memory + output_memory + activation_memory

        # Available memory for batching
        available_memory = int(self.total_memory * self.safety_factor) - self.model_memory

        # Compute batch size
        estimated_batch_size = max(1, available_memory // memory_per_sample)
        batch_size = min(estimated_batch_size, self.max_batch_size)
        batch_size = max(batch_size, self.min_batch_size)

        # Round down to nearest power of 2 for optimal GPU utilization
        batch_size_int = int(batch_size)
        batch_size_int = 2 ** (batch_size_int.bit_length() - 1) if batch_size_int > 1 else 1

        # Determine optimal number of data loader workers
        num_workers = (
            min(4, max(2, torch.get_num_threads() // 2)) if self.device.type == "cuda" else 0
        )

        return BatchConfig(
            batch_size=batch_size_int,
            grid_size=grid_size,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def benchmark_batch_size(
        self,
        batch_size: int,
        grid_size: int,
        in_channels: int,
        use_mixed_precision: bool = True,
        num_iterations: int = 10,
    ) -> Tuple[bool, float]:
        """
        Test if a batch size fits in memory and measure throughput.

        Args:
            batch_size: Batch size to test
            grid_size: Spatial grid size
            in_channels: Input channels
            use_mixed_precision: Use FP16
            num_iterations: Number of iterations for benchmarking

        Returns:
            Tuple of (fits_in_memory, avg_time_per_batch_ms)

        Example:
            ```python
            fits, time_ms = sizer.benchmark_batch_size(
                batch_size=32,
                grid_size=256,
                in_channels=3
            )
            print(f"Batch size 32: {time_ms:.2f}ms/batch (fits: {fits})")
            ```
        """
        self.model.eval()

        dtype = torch.float16 if use_mixed_precision else torch.float32
        dummy_input = torch.randn(
            batch_size, in_channels, grid_size, grid_size, device=self.device, dtype=dtype
        )

        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Warmup
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                _ = self.model(dummy_input)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()  # type: ignore
                for _ in range(num_iterations):
                    with torch.cuda.amp.autocast(enabled=use_mixed_precision):  # type: ignore
                        _ = self.model(dummy_input)
                end.record()  # type: ignore

                torch.cuda.synchronize()
                avg_time = start.elapsed_time(end) / num_iterations

                # Check memory
                peak_memory = torch.cuda.max_memory_allocated()
                fits = peak_memory < (self.total_memory * self.safety_factor)
            else:
                # CPU benchmarking
                import time

                start_time = time.time()
                for _ in range(num_iterations):
                    _ = self.model(dummy_input)
                elapsed = time.time() - start_time
                avg_time = (elapsed / num_iterations) * 1000  # Convert to ms
                fits = True

            return fits, avg_time

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False, float("inf")
            raise
        finally:
            del dummy_input
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def find_optimal_batch_size(
        self,
        grid_size: int,
        in_channels: int,
        use_mixed_precision: bool = True,
        target_time_ms: float = 100.0,
    ) -> BatchConfig:
        """
        Binary search for optimal batch size balancing memory and throughput.

        Args:
            grid_size: Spatial grid size
            in_channels: Input channels
            use_mixed_precision: Use FP16
            target_time_ms: Target time per batch (for throughput optimization)

        Returns:
            BatchConfig with optimal batch size

        Example:
            ```python
            config = sizer.find_optimal_batch_size(
                grid_size=512,
                in_channels=3,
                target_time_ms=50.0
            )
            ```
        """
        # Start with estimated batch size
        config = self.compute_batch_size(
            grid_size=grid_size, in_channels=in_channels, use_mixed_precision=use_mixed_precision
        )

        # Try to increase if we're well under target time
        current_bs = config.batch_size
        fits, time_ms = self.benchmark_batch_size(
            current_bs, grid_size, in_channels, use_mixed_precision
        )

        if not fits:
            # Reduce batch size if estimation was too high
            while current_bs > self.min_batch_size:
                current_bs //= 2
                fits, time_ms = self.benchmark_batch_size(
                    current_bs, grid_size, in_channels, use_mixed_precision
                )
                if fits:
                    break

        config.batch_size = current_bs
        return config

    def print_memory_info(self) -> None:
        """Print GPU memory information."""
        print("=" * 60)
        print("MEMORY CONFIGURATION")
        print("=" * 60)
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            print(f"GPU: {props.name}")
            print(f"Total memory: {self.total_memory / 1024**3:.2f} GB")
        print(f"Model memory: {self.model_memory / 1024**3:.2f} GB")
        print(
            f"Available (with safety={self.safety_factor}): "
            f"{(self.total_memory * self.safety_factor - self.model_memory) / 1024**3:.2f} GB"
        )
        print("=" * 60)
