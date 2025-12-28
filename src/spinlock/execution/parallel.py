"""
GPU parallelization strategies for Spinlock.

Modular design allows easy migration from DataParallel → DDP:
- Strategy pattern: Swappable parallelization backends
- Adapter pattern: Unified API regardless of backend
- Factory pattern: Runtime strategy selection

Design principles:
- Single interface for all strategies
- Easy to swap implementations
- Clean separation of concerns
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Union, Optional


class ParallelStrategy(ABC):
    """
    Abstract strategy for model parallelization.

    All strategies must implement:
    - wrap(model) → parallel model
    - unwrap(model) → underlying model
    """

    @abstractmethod
    def wrap(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for parallel execution.

        Args:
            model: Model to parallelize

        Returns:
            Wrapped model
        """
        pass

    @abstractmethod
    def unwrap(self, model: nn.Module) -> nn.Module:
        """
        Extract underlying model from wrapper.

        Args:
            model: Potentially wrapped model

        Returns:
            Underlying model
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources (e.g., process groups for DDP).
        """
        pass


class DataParallelStrategy(ParallelStrategy):
    """
    DataParallel strategy for single-node multi-GPU.

    Simpler than DDP, good for 2-8 GPUs on one machine.
    Lower overhead for small batch sizes.

    Args:
        device_ids: List of GPU device IDs (None = auto-detect)

    Example:
        ```python
        strategy = DataParallelStrategy(device_ids=[0, 1, 2])
        parallel_model = strategy.wrap(model)
        out = parallel_model(x)
        ```
    """

    def __init__(self, device_ids: Optional[List[int]] = None):
        self.device_ids = device_ids or self._get_available_devices()

    def _get_available_devices(self) -> List[int]:
        """Auto-detect available CUDA devices."""
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    def wrap(self, model: nn.Module) -> nn.Module:
        """
        Wrap model with DataParallel.

        Args:
            model: Model to parallelize

        Returns:
            DataParallel model (or original if <2 devices)
        """
        if len(self.device_ids) <= 1:
            # Single device: no parallelization needed
            return model

        # Move to primary device
        if torch.cuda.is_available():
            model = model.cuda(self.device_ids[0])

        return nn.DataParallel(model, device_ids=self.device_ids)

    def unwrap(self, model: nn.Module) -> nn.Module:
        """Extract underlying model from DataParallel wrapper."""
        if isinstance(model, nn.DataParallel):
            return model.module
        return model

    def cleanup(self) -> None:
        """No cleanup needed for DataParallel."""
        pass


class DDPStrategy(ParallelStrategy):
    """
    DistributedDataParallel strategy for multi-node or multi-GPU.

    More scalable than DataParallel, industry standard for production.
    Requires process initialization (torch.distributed).

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ("nccl", "gloo")

    Example:
        ```python
        # Initialize process group first
        torch.distributed.init_process_group(backend="nccl")

        strategy = DDPStrategy(rank=0, world_size=4)
        parallel_model = strategy.wrap(model)
        ```
    """

    def __init__(self, rank: int, world_size: int, backend: str = "nccl"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

        # Verify distributed is initialized
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "DDP requires torch.distributed.init_process_group() to be called first"
            )

    def wrap(self, model: nn.Module) -> nn.Module:
        """
        Wrap model with DDP.

        Args:
            model: Model to parallelize

        Returns:
            DDP-wrapped model
        """
        from torch.nn.parallel import DistributedDataParallel as DDP

        # Move to appropriate device
        device_id = self.rank % torch.cuda.device_count()
        model = model.to(device_id)

        return DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,  # Set True if needed for complex models
        )

    def unwrap(self, model: nn.Module) -> nn.Module:
        """Extract underlying model from DDP wrapper."""
        if hasattr(model, "module"):
            module = getattr(model, "module")
            if isinstance(module, nn.Module):
                return module
        return model

    def cleanup(self) -> None:
        """Destroy process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class NoOpStrategy(ParallelStrategy):
    """
    No-op strategy: does nothing.

    Useful for CPU-only or single-GPU execution.

    Example:
        ```python
        strategy = NoOpStrategy()
        model = strategy.wrap(model)  # Returns model unchanged
        ```
    """

    def wrap(self, model: nn.Module) -> nn.Module:
        return model

    def unwrap(self, model: nn.Module) -> nn.Module:
        return model

    def cleanup(self) -> None:
        pass


def create_parallel_strategy(
    strategy: str,
    device_ids: Union[List[int], None] = None,
    rank: Union[int, None] = None,
    world_size: Union[int, None] = None,
) -> ParallelStrategy:
    """
    Factory function to create parallelization strategy.

    Args:
        strategy: Strategy type ("dp", "ddp", "none")
        device_ids: GPU device IDs (for DataParallel)
        rank: Process rank (for DDP)
        world_size: Total processes (for DDP)

    Returns:
        ParallelStrategy instance

    Example:
        ```python
        # DataParallel
        strategy = create_parallel_strategy("dp", device_ids=[0, 1, 2])

        # DDP
        strategy = create_parallel_strategy("ddp", rank=0, world_size=4)

        # Single device
        strategy = create_parallel_strategy("none")
        ```
    """
    if strategy == "dp" or strategy == "data_parallel":
        return DataParallelStrategy(device_ids=device_ids)

    elif strategy == "ddp" or strategy == "distributed":
        if rank is None or world_size is None:
            raise ValueError("DDP requires rank and world_size")
        return DDPStrategy(rank=rank, world_size=world_size)

    elif strategy == "none" or strategy == "noop":
        return NoOpStrategy()

    else:
        raise ValueError(f"Unknown strategy: {strategy}. " f"Valid options: 'dp', 'ddp', 'none'")


class ParallelExecutor:
    """
    High-level executor for parallel model execution.

    Manages:
    - Strategy creation
    - Model wrapping/unwrapping
    - Resource cleanup

    Example:
        ```python
        executor = ParallelExecutor(strategy="dp", device_ids=[0, 1])

        # Wrap model
        parallel_model = executor.wrap(model)

        # Execute
        output = parallel_model(input)

        # Cleanup
        executor.cleanup()
        ```
    """

    def __init__(self, strategy: str = "dp", device_ids: Union[List[int], None] = None, **kwargs):
        """
        Initialize parallel executor.

        Args:
            strategy: Parallelization strategy
            device_ids: GPU device IDs
            **kwargs: Additional arguments for strategy
        """
        self.strategy = create_parallel_strategy(strategy=strategy, device_ids=device_ids, **kwargs)

    def wrap(self, model: nn.Module) -> nn.Module:
        """Wrap model with parallelization strategy."""
        return self.strategy.wrap(model)

    def unwrap(self, model: nn.Module) -> nn.Module:
        """Unwrap model to get underlying implementation."""
        return self.strategy.unwrap(model)

    def cleanup(self) -> None:
        """Cleanup parallelization resources."""
        self.strategy.cleanup()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.cleanup()
