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
    import sys

    # Set environment variables for distributed training
    master_addr = config.get_master_addr()
    master_port = str(config.master_port)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    print(f"[Rank {rank}] Environment configured: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    sys.stdout.flush()

    # Initialize process group with shorter timeout for faster failure
    timeout = timedelta(seconds=min(config.timeout_seconds, 120))  # Cap at 2 minutes

    print(f"[Rank {rank}] Calling dist.init_process_group (backend={config.backend}, timeout={timeout.seconds}s)...")
    sys.stdout.flush()

    try:
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        print(f"[Rank {rank}] dist.init_process_group completed")
        sys.stdout.flush()
    except Exception as e:
        print(f"[Rank {rank}] ERROR in dist.init_process_group: {e}")
        sys.stdout.flush()
        raise

    # Set device for this process
    if config.backend == "nccl" or torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}/{world_size}] Process group initialized on {config.backend} backend")
        print(f"[Rank {rank}/{world_size}] Using GPU: cuda:{local_rank}")
    else:
        print(f"[Rank {rank}/{world_size}] Process group initialized on {config.backend} backend (CPU)")

    sys.stdout.flush()


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
