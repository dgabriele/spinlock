"""
Distributed training configuration for Spinlock.

Supports multi-node, multi-GPU training via PyTorch DistributedDataParallel.
Now includes support for Salad.com cloud GPU platform.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
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
class SaladResourceConfig:
    """Salad.com resource specification."""

    gpu_class: str  # e.g., "rtx_4070"
    num_replicas: int = 1
    cpu: int = 4
    memory: int = 16384  # MB
    storage: int = 10737418240  # bytes


@dataclass
class SaladStorageConfig:
    """Cloud storage configuration for Salad."""

    backend: str  # "s3", "minio", "gcs", "azure"
    bucket: str
    dataset_path: str = ""
    checkpoint_path: str = ""
    endpoint_url: Optional[str] = None  # For MinIO
    region: Optional[str] = None
    use_ssl: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)


@dataclass
class SaladNetworkingConfig:
    """Networking configuration for distributed training."""

    coordination_backend: str = "env"  # "env" or "redis"
    master_port: int = 29500


@dataclass
class SaladContainerConfig:
    """Container image configuration."""

    image: str
    registry_auth: Optional[Dict[str, str]] = None


@dataclass
class SaladMonitoringConfig:
    """Monitoring configuration."""

    log_streaming: bool = True
    poll_interval: int = 30


@dataclass
class SaladConfig:
    """Salad.com distributed training configuration."""

    enabled: bool
    api_key: str
    organization: str
    project: str
    container: SaladContainerConfig
    resources: SaladResourceConfig
    storage: SaladStorageConfig
    networking: SaladNetworkingConfig = field(default_factory=SaladNetworkingConfig)
    monitoring: SaladMonitoringConfig = field(default_factory=SaladMonitoringConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SaladConfig":
        """Create SaladConfig from dictionary."""
        return cls(
            enabled=config_dict.get("enabled", False),
            api_key=config_dict["api_key"],
            organization=config_dict["organization"],
            project=config_dict["project"],
            container=SaladContainerConfig(**config_dict["container"]),
            resources=SaladResourceConfig(**config_dict["resources"]),
            storage=SaladStorageConfig(**config_dict["storage"]),
            networking=SaladNetworkingConfig(**config_dict.get("networking", {})),
            monitoring=SaladMonitoringConfig(**config_dict.get("monitoring", {})),
        )


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enabled: bool = False
    """Enable distributed training"""

    backend: str = "ssh"
    """Distributed backend launcher: 'ssh', 'salad', or 'local'"""

    nodes: List[NodeConfig] = field(default_factory=list)
    """List of compute nodes (for SSH backend)"""

    salad: Optional[SaladConfig] = None
    """Salad.com configuration (for Salad backend)"""

    master_addr: Optional[str] = None
    """Address of the master node (auto-detected from first node if not specified)"""

    master_port: int = 29500
    """Port for distributed coordination"""

    init_method: str = "env://"
    """Initialization method for distributed training"""

    timeout_seconds: int = 1800
    """Timeout for distributed initialization (seconds)"""

    @property
    def world_size(self) -> int:
        """Total number of processes (GPUs) across all nodes."""
        if self.backend == "salad" and self.salad:
            return self.salad.resources.num_replicas
        return sum(node.num_gpus for node in self.nodes)

    def get_master_addr(self) -> str:
        """Get the master address (from config or first node)."""
        if self.master_addr:
            return self.master_addr
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

        backend = config_dict.get("backend", "ssh")

        # Parse nodes (for SSH backend)
        nodes = []
        for node_dict in config_dict.get("nodes", []):
            nodes.append(NodeConfig(**node_dict))

        # Parse Salad config (for Salad backend)
        salad_config = None
        if backend == "salad" and "salad" in config_dict:
            salad_config = SaladConfig.from_dict(config_dict["salad"])

        return cls(
            enabled=True,
            backend=backend,
            nodes=nodes,
            salad=salad_config,
            master_addr=config_dict.get("master_addr"),  # Optional override
            master_port=config_dict.get("master_port", 29500),
            init_method=config_dict.get("init_method", "env://"),
            timeout_seconds=config_dict.get("timeout_seconds", 1800),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if not self.enabled:
            return

        # Validate backend type
        if self.backend not in ("ssh", "salad", "local"):
            raise ValueError(
                f"Unsupported distributed backend: {self.backend}. "
                f"Must be 'ssh', 'salad', or 'local'"
            )

        # Validate SSH backend
        if self.backend == "ssh":
            if not self.nodes:
                raise ValueError("SSH backend enabled but no nodes specified")

            if self.world_size < 2:
                raise ValueError(
                    f"Distributed training requires at least 2 GPUs, got {self.world_size}"
                )

            # Ensure first node is local (master node)
            if not self.nodes[0].is_local:
                raise ValueError("First node must be localhost (master node)")

        # Validate Salad backend
        elif self.backend == "salad":
            if not self.salad or not self.salad.enabled:
                raise ValueError("Salad backend enabled but salad config not provided")

            if self.world_size < 1:
                raise ValueError(
                    f"Salad backend requires at least 1 replica, got {self.world_size}"
                )
