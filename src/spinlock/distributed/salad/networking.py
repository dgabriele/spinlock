"""Networking and coordination for distributed training on Salad."""

from typing import Optional, List
import os
import time
import redis
from dataclasses import dataclass


@dataclass
class NodeInfo:
    """Information about a training node."""

    rank: int
    ip_address: str
    hostname: str
    status: str  # "starting", "ready", "failed"


class CoordinationBackend:
    """Abstract coordination backend."""

    def register_node(self, rank: int, node_info: NodeInfo) -> None:
        """Register node with coordination service."""
        raise NotImplementedError

    def get_master_address(self) -> str:
        """Get master node address."""
        raise NotImplementedError

    def wait_for_all_nodes(self, world_size: int, timeout: int = 600) -> List[NodeInfo]:
        """Wait for all nodes to register."""
        raise NotImplementedError


class RedisCoordination(CoordinationBackend):
    """Redis-based coordination for distributed training."""

    def __init__(self, redis_url: str, job_id: str):
        """
        Initialize Redis coordination.

        Args:
            redis_url: Redis connection URL
            job_id: Unique job identifier
        """
        self.redis_client = redis.from_url(redis_url)
        self.job_id = job_id
        self.prefix = f"salad:training:{job_id}"

    def register_node(self, rank: int, node_info: NodeInfo) -> None:
        """Register node in Redis."""
        key = f"{self.prefix}:nodes:{rank}"
        self.redis_client.hset(
            key,
            mapping={
                "rank": node_info.rank,
                "ip_address": node_info.ip_address,
                "hostname": node_info.hostname,
                "status": node_info.status,
            },
        )
        self.redis_client.expire(key, 3600)  # 1 hour TTL

    def get_master_address(self) -> str:
        """Get rank 0 IP address."""
        key = f"{self.prefix}:nodes:0"
        node_data = self.redis_client.hgetall(key)
        if not node_data:
            raise RuntimeError("Master node (rank 0) not registered")
        return node_data[b"ip_address"].decode("utf-8")

    def wait_for_all_nodes(self, world_size: int, timeout: int = 600) -> List[NodeInfo]:
        """Wait for all nodes to register."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            nodes = []
            all_registered = True

            for rank in range(world_size):
                key = f"{self.prefix}:nodes:{rank}"
                node_data = self.redis_client.hgetall(key)

                if not node_data:
                    all_registered = False
                    break

                nodes.append(
                    NodeInfo(
                        rank=int(node_data[b"rank"]),
                        ip_address=node_data[b"ip_address"].decode("utf-8"),
                        hostname=node_data[b"hostname"].decode("utf-8"),
                        status=node_data[b"status"].decode("utf-8"),
                    )
                )

            if all_registered:
                return nodes

            time.sleep(5)

        raise TimeoutError(f"Not all nodes registered within {timeout}s")


class EnvironmentCoordination(CoordinationBackend):
    """Environment variable-based coordination (simpler, for single-node multi-GPU)."""

    def __init__(self) -> None:
        """Initialize environment-based coordination."""
        self.master_addr = os.environ.get("MASTER_ADDR")

    def register_node(self, rank: int, node_info: NodeInfo) -> None:
        """No-op for environment-based coordination."""
        pass

    def get_master_address(self) -> str:
        """Get master address from environment."""
        if not self.master_addr:
            # For single-node, use localhost
            return "127.0.0.1"
        return self.master_addr

    def wait_for_all_nodes(self, world_size: int, timeout: int = 600) -> List[NodeInfo]:
        """No-op for environment-based coordination."""
        return []


def create_coordination_backend(config: dict, job_id: str) -> CoordinationBackend:
    """
    Factory for coordination backend.

    Args:
        config: Networking configuration dictionary
        job_id: Unique job identifier

    Returns:
        CoordinationBackend instance

    Raises:
        ValueError: If backend type is unknown or configuration is invalid
    """
    backend_type = config.get("coordination_backend", "env")

    if backend_type == "redis":
        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            raise ValueError("REDIS_URL not set in environment")
        return RedisCoordination(redis_url, job_id)
    elif backend_type == "env":
        return EnvironmentCoordination()
    else:
        raise ValueError(f"Unknown coordination backend: {backend_type}")


def setup_distributed_environment(
    rank: int, world_size: int, master_addr: str, master_port: int
) -> None:
    """
    Set up environment variables for PyTorch distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        master_addr: Master node address
        master_port: Master node port
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_DEBUG"] = "INFO"
