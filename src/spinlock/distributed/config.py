"""
Distributed training configuration for Spinlock.

Supports multi-node, multi-GPU training via PyTorch DistributedDataParallel.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class NodeConfig:
    """Configuration for a single compute node."""

    host: str
    """Hostname or IP address"""

    gpus: List[int] = field(default_factory=lambda: [0])
    """List of GPU indices to use on this node"""

    ssh_user: Optional[str] = None
    """SSH username (defaults to current user)"""

    ssh_port: int = 22
    """SSH port"""

    python_path: Optional[str] = None
    """Path to Python executable on remote (defaults to 'poetry run python')"""

    working_dir: Optional[str] = None
    """Working directory on remote (defaults to current directory)"""

    @property
    def is_local(self) -> bool:
        """Check if this is the local node."""
        return self.host in ("localhost", "127.0.0.1")

    @property
    def num_gpus(self) -> int:
        """Number of GPUs on this node."""
        return len(self.gpus)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enabled: bool = False
    """Enable distributed training"""

    backend: str = "nccl"
    """Distributed backend (nccl, gloo, mpi)"""

    nodes: List[NodeConfig] = field(default_factory=list)
    """List of compute nodes"""

    master_port: int = 29500
    """Port for distributed coordination"""

    init_method: str = "env://"
    """Initialization method for distributed training"""

    timeout_seconds: int = 1800
    """Timeout for distributed initialization (seconds)"""

    @property
    def world_size(self) -> int:
        """Total number of processes (GPUs) across all nodes."""
        return sum(node.num_gpus for node in self.nodes)

    @property
    def master_addr(self) -> str:
        """Address of the master node."""
        if not self.nodes:
            return "localhost"
        # First node is master
        return self.nodes[0].host

    def get_rank_mapping(self) -> List[tuple]:
        """
        Get mapping of (node_idx, gpu_idx, global_rank).

        Returns:
            List of (node_idx, local_gpu_idx, global_rank) tuples
        """
        mapping = []
        global_rank = 0

        for node_idx, node in enumerate(self.nodes):
            for local_gpu_idx in node.gpus:
                mapping.append((node_idx, local_gpu_idx, global_rank))
                global_rank += 1

        return mapping

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistributedConfig":
        """Create DistributedConfig from dictionary (YAML config)."""
        if not config_dict.get("enabled", False):
            return cls(enabled=False)

        nodes = []
        for node_dict in config_dict.get("nodes", []):
            nodes.append(NodeConfig(**node_dict))

        return cls(
            enabled=True,
            backend=config_dict.get("backend", "nccl"),
            nodes=nodes,
            master_port=config_dict.get("master_port", 29500),
            init_method=config_dict.get("init_method", "env://"),
            timeout_seconds=config_dict.get("timeout_seconds", 1800),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.enabled:
            return

        if not self.nodes:
            raise ValueError("Distributed training enabled but no nodes specified")

        if self.world_size < 2:
            raise ValueError(f"Distributed training requires at least 2 GPUs, got {self.world_size}")

        if self.backend not in ("nccl", "gloo", "mpi"):
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Ensure first node is local (master node)
        if not self.nodes[0].is_local:
            raise ValueError("First node must be localhost (master node)")
