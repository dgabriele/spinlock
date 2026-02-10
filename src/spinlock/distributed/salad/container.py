"""Container specification builder for Salad.com deployments."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from salad_cloud_sdk.models import (
    ContainerGroupCreationRequest,
    ContainerConfiguration,
    CreateContainerResourceRequirements,
    ContainerRegistryAuthentication,
    ContainerRegistryAuthenticationBasic,
    ContainerRestartPolicy,
)


@dataclass
class ContainerResources:
    """Container resource specification."""

    cpu: int = 4  # vCPUs
    memory: int = 16384  # MB
    gpu_classes: List[str] = field(default_factory=list)  # GPU UUIDs
    storage_amount: int = 10737418240  # 10 GB


@dataclass
class ContainerImage:
    """Container image specification."""

    image: str  # Image name (e.g., "dockerhub/spinlock:latest")
    command: Optional[List[str]] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class RegistryAuth:
    """Container registry authentication."""

    username: str
    password: str
    registry: str = "docker.io"  # Default to Docker Hub


class ContainerSpecBuilder:
    """Builder for Salad.com container specifications."""

    def __init__(self) -> None:
        """Initialize container spec builder."""
        self.image_spec: Optional[ContainerImage] = None
        self.resources = ContainerResources()
        self.registry_auth: Optional[RegistryAuth] = None
        self.restart_policy = "always"
        self.replicas = 1

    def with_image(
        self, image: str, command: Optional[List[str]] = None
    ) -> "ContainerSpecBuilder":
        """
        Set container image and command.

        Args:
            image: Container image name
            command: Optional command to run

        Returns:
            Self for chaining
        """
        self.image_spec = ContainerImage(image=image, command=command)
        return self

    def with_environment(self, env_vars: Dict[str, str]) -> "ContainerSpecBuilder":
        """
        Add environment variables.

        Args:
            env_vars: Environment variables to add

        Returns:
            Self for chaining
        """
        if self.image_spec is None:
            self.image_spec = ContainerImage(image="")
        self.image_spec.environment_variables.update(env_vars)
        return self

    def with_resources(
        self, cpu: int, memory: int, gpu_classes: List[str], storage: int = 10737418240
    ) -> "ContainerSpecBuilder":
        """
        Set resource requirements.

        Args:
            cpu: Number of vCPUs
            memory: Memory in MB
            gpu_classes: List of GPU class IDs
            storage: Storage in bytes

        Returns:
            Self for chaining
        """
        self.resources = ContainerResources(
            cpu=cpu, memory=memory, gpu_classes=gpu_classes, storage_amount=storage
        )
        return self

    def with_registry_auth(
        self, username: str, password: str, registry: str = "docker.io"
    ) -> "ContainerSpecBuilder":
        """
        Set registry authentication.

        Args:
            username: Registry username
            password: Registry password
            registry: Registry URL

        Returns:
            Self for chaining
        """
        self.registry_auth = RegistryAuth(
            username=username, password=password, registry=registry
        )
        return self

    def with_replicas(self, count: int) -> "ContainerSpecBuilder":
        """
        Set number of replicas.

        Args:
            count: Number of replicas

        Returns:
            Self for chaining
        """
        self.replicas = count
        return self

    def build(self, container_group_name: str) -> ContainerGroupCreationRequest:
        """
        Build container specification for Salad API using SDK models.

        Args:
            container_group_name: Name for the container group

        Returns:
            ContainerGroupCreationRequest SDK model

        Raises:
            ValueError: If container image not specified
        """
        if self.image_spec is None:
            raise ValueError("Container image must be specified")

        # Build resource requirements
        resources = CreateContainerResourceRequirements(
            cpu=self.resources.cpu,
            memory=self.resources.memory,
            gpu_classes=self.resources.gpu_classes,
            storage_amount=self.resources.storage_amount,
        )

        # Build container configuration with optional registry auth
        if self.registry_auth:
            basic_auth = ContainerRegistryAuthenticationBasic(
                username=self.registry_auth.username,
                password=self.registry_auth.password,
            )
            registry_auth = ContainerRegistryAuthentication(basic=basic_auth)
        else:
            registry_auth = None

        container_config = ContainerConfiguration(
            image=self.image_spec.image,
            resources=resources,
            command=self.image_spec.command,
            environment_variables=self.image_spec.environment_variables if self.image_spec.environment_variables else None,
            registry_authentication=registry_auth,
        )

        # Map restart policy string to enum
        restart_policy_enum = ContainerRestartPolicy.ALWAYS if self.restart_policy == "always" else ContainerRestartPolicy.NEVER

        # Build container group creation request
        return ContainerGroupCreationRequest(
            name=container_group_name,
            container=container_config,
            replicas=self.replicas,
            autostart_policy=True,
            restart_policy=restart_policy_enum,
        )


def build_training_container_spec(config: dict, rank: int, world_size: int, job_id: str) -> ContainerGroupCreationRequest:
    """
    Build container spec for distributed training.

    Args:
        config: Training configuration dictionary
        rank: Process rank
        world_size: Total number of processes
        job_id: Job identifier for naming

    Returns:
        ContainerGroupCreationRequest SDK model
    """
    salad_config = config["distributed"]["salad"]

    # Build environment variables for distributed training
    env_vars = {
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_PORT": str(salad_config["networking"].get("master_port", 29500)),
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_DEBUG": "INFO",
        # Storage credentials
        "AWS_ACCESS_KEY_ID": salad_config["storage"]["credentials"]["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": salad_config["storage"]["credentials"][
            "aws_secret_access_key"
        ],
        # MinIO endpoint if using MinIO
        "S3_ENDPOINT_URL": salad_config["storage"].get("endpoint_url", ""),
        # Dataset/checkpoint paths
        "DATASET_PATH": salad_config["storage"]["dataset_path"],
        "CHECKPOINT_PATH": salad_config["storage"]["checkpoint_path"],
        "S3_BUCKET": salad_config["storage"]["bucket"],
        # Mark as Salad environment
        "SALAD_MACHINE_ID": "salad-container",
    }

    # Build training command (passed to entrypoint.sh)
    # The entrypoint runs: exec poetry run spinlock "$@"
    # So we just pass the CLI args
    command = [
        "train-meta-operator",
        "--config",
        "/app/config.yaml",
    ]

    # Build container spec
    builder = ContainerSpecBuilder()
    builder.with_image(image=salad_config["container"]["image"], command=command)
    builder.with_environment(env_vars)
    builder.with_resources(
        cpu=salad_config["resources"]["cpu"],
        memory=salad_config["resources"]["memory"],
        gpu_classes=[salad_config["resources"]["gpu_class"]],
        storage=salad_config["resources"].get("storage", 10737418240),
    )

    # Add registry auth if configured
    if "registry_auth" in salad_config["container"]:
        auth = salad_config["container"]["registry_auth"]
        builder.with_registry_auth(username=auth["username"], password=auth["password"])

    builder.with_replicas(1)  # Each rank is a separate container group

    # Build container group with proper name
    container_group_name = f"{job_id}-rank-{rank}"
    return builder.build(container_group_name)
