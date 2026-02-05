"""Salad.com distributed training integration."""

from .launcher import SaladLauncher, launch_salad_training
from .storage import (
    CloudStorageBackend,
    S3StorageBackend,
    StorageManager,
    create_storage_backend,
)
from .container import ContainerSpecBuilder, build_training_container_spec
from .networking import (
    CoordinationBackend,
    RedisCoordination,
    EnvironmentCoordination,
    create_coordination_backend,
    setup_distributed_environment,
)
from .monitor import SaladJobMonitor

__all__ = [
    # Launcher
    "SaladLauncher",
    "launch_salad_training",
    # Storage
    "CloudStorageBackend",
    "S3StorageBackend",
    "StorageManager",
    "create_storage_backend",
    # Container
    "ContainerSpecBuilder",
    "build_training_container_spec",
    # Networking
    "CoordinationBackend",
    "RedisCoordination",
    "EnvironmentCoordination",
    "create_coordination_backend",
    "setup_distributed_environment",
    # Monitoring
    "SaladJobMonitor",
]
