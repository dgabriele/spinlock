"""
Distributed training support for Spinlock.

Enables multi-node, multi-GPU training using PyTorch DistributedDataParallel.
"""

from .config import DistributedConfig, NodeConfig
from .setup import (
    setup_process_group,
    cleanup_process_group,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    broadcast_object,
)
from .launcher import launch_distributed_training

__all__ = [
    # Config
    "DistributedConfig",
    "NodeConfig",
    # Setup
    "setup_process_group",
    "cleanup_process_group",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "barrier",
    "broadcast_object",
    # Launcher
    "launch_distributed_training",
]
