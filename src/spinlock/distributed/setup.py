"""
Distributed process group setup for PyTorch DDP.
"""

import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Optional

from .config import DistributedConfig


def setup_process_group(
    rank: int,
    world_size: int,
    config: DistributedConfig,
) -> None:
    """
    Initialize distributed process group.

    Args:
        rank: Global rank of this process
        world_size: Total number of processes
        config: Distributed configuration
    """
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize process group
    timeout = timedelta(seconds=config.timeout_seconds)
    dist.init_process_group(
        backend=config.backend,
        init_method=config.init_method,
        world_size=world_size,
        rank=rank,
        timeout=timeout,
    )

    # Set device for this process
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] Process group initialized on {config.backend} backend")
    print(f"[Rank {rank}/{world_size}] Using GPU: cuda:{local_rank}")


def cleanup_process_group() -> None:
    """Cleanup distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(obj, src: int = 0):
    """Broadcast Python object from src rank to all ranks."""
    if not dist.is_available() or not dist.is_initialized():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]
