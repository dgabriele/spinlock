"""Base launcher for Salad.com distributed jobs."""

from abc import ABC, abstractmethod
from typing import List
import uuid
from salad_cloud_sdk import SaladCloudSdk
from salad_cloud_sdk.models import ContainerGroupCreationRequest

from .storage import create_storage_backend, StorageManager
from .networking import create_coordination_backend


class SaladJobLauncher(ABC):
    """Abstract base class for Salad job launchers."""

    def __init__(self, config: dict):
        """
        Initialize Salad job launcher.

        Args:
            config: Full configuration dict with distributed.salad section
        """
        self.config = config
        self.salad_config = config["distributed"]["salad"]

        # Extract Salad API configuration
        self.api_key = self.salad_config["api_key"]
        self.organization = self.salad_config["organization"]
        self.project = self.salad_config["project"]

        # Initialize SDK
        self.sdk = SaladCloudSdk(api_key=self.api_key)

        # Generate unique job ID
        self.job_id = f"spinlock-{uuid.uuid4().hex[:8]}"

        # Initialize storage manager
        storage_backend = create_storage_backend(self.salad_config["storage"])
        self.storage = StorageManager(storage_backend)

        # Initialize coordination backend
        self.coordination = create_coordination_backend(
            self.salad_config["networking"], self.job_id
        )

        print(f"[SaladJobLauncher] Initialized for job {self.job_id}")

    @abstractmethod
    def build_container_specs(self) -> List[ContainerGroupCreationRequest]:
        """
        Build container specifications for this job type.

        Returns:
            List of ContainerGroupCreationRequest objects

        Notes:
            - Training jobs: Create one spec per DDP rank with RANK/WORLD_SIZE env vars
            - Generation jobs: Create independent specs with different Sobol offsets
        """
        pass

    @abstractmethod
    def monitor_job(self) -> None:
        """
        Monitor job execution and handle completion.

        This method is responsible for:
        - Monitoring container status
        - Streaming logs (if enabled)
        - Detecting completion/failure
        - Aggregating results (if needed)
        """
        pass

    def launch(self) -> None:
        """
        Launch job on Salad.com.

        This is the main entry point that orchestrates the workflow:
        1. Build container specifications (job-specific)
        2. Create container groups via Salad API
        3. Monitor job execution (job-specific)
        """
        print(f"[SaladJobLauncher] Launching job {self.job_id}")

        # Step 1: Build container specs (job-specific)
        container_specs = self.build_container_specs()
        print(f"[SaladJobLauncher] Built {len(container_specs)} container specs")

        # Step 2: Create container groups via Salad API
        self._create_container_groups(container_specs)

        # Step 3: Monitor job execution (job-specific)
        print("[SaladJobLauncher] Containers created. Starting monitoring...")
        self.monitor_job()

        print(f"[SaladJobLauncher] ✓ Job {self.job_id} completed")

    def _create_container_groups(
        self, container_specs: List[ContainerGroupCreationRequest]
    ) -> None:
        """
        Create container groups via Salad API.

        Args:
            container_specs: List of container group creation requests
        """
        for idx, spec in enumerate(container_specs):
            group_name = spec.name
            print(f"[SaladJobLauncher] Creating container group {idx+1}/{len(container_specs)}: {group_name}...")

            try:
                self.sdk.container_groups.create_container_group(
                    request_body=spec,
                    organization_name=self.organization,
                    project_name=self.project,
                )
                print(f"  ✓ Container group created: {group_name}")
            except Exception as e:
                print(f"  ✗ Failed to create container group: {e}")
                raise

    def cleanup(self) -> None:
        """Clean up container groups after job completion."""
        print(f"[SaladJobLauncher] Cleaning up job {self.job_id}...")
        # TODO: Implement cleanup based on Salad API
        # For now, containers auto-terminate on completion
        print("  ✓ Cleanup complete")
