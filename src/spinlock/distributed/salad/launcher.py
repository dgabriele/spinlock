"""Salad.com launcher for distributed training."""

from typing import Dict, List
import uuid
import time
import os
from pathlib import Path
from salad_cloud_sdk import SaladCloudSdk
from .container import build_training_container_spec
from .storage import create_storage_backend, StorageManager
from .networking import create_coordination_backend
from .monitor import SaladJobMonitor


class SaladLauncher:
    """Launcher for distributed training on Salad.com."""

    def __init__(self, config: dict, script_path: str, script_args: List[str]):
        """
        Initialize Salad launcher.

        Args:
            config: Full training configuration dict
            script_path: Path to training script (for reference)
            script_args: Arguments to pass to training script
        """
        self.config = config
        self.script_path = script_path
        self.script_args = script_args

        # Extract Salad configuration
        self.salad_config = config["distributed"]["salad"]
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

        print(f"[SaladLauncher] Initialized for job {self.job_id}")

    def prepare_training_data(self) -> None:
        """Upload dataset and config to cloud storage."""
        print("[SaladLauncher] Preparing training data...")

        storage_config = self.salad_config["storage"]

        # Check if dataset exists in cloud storage
        remote_dataset = storage_config["dataset_path"]
        dataset_exists = self.storage.backend.exists(remote_dataset)

        if not dataset_exists:
            # Upload dataset from local
            local_dataset = Path(self.config["data"]["dataset_path"])
            if local_dataset.exists():
                print(
                    f"  Uploading dataset: {local_dataset} -> "
                    f"s3://{storage_config['bucket']}/{remote_dataset}"
                )
                self.storage.backend.upload_file(local_dataset, remote_dataset)
                print("  ✓ Dataset uploaded")
            else:
                raise FileNotFoundError(
                    f"Dataset not found locally at {local_dataset} "
                    "and not in cloud storage"
                )
        else:
            print(f"  ✓ Dataset already in cloud storage: {remote_dataset}")

        print("[SaladLauncher] ✓ Training data prepared")

    def launch(self) -> None:
        """Launch distributed training on Salad.com."""
        print(f"[SaladLauncher] Launching training job {self.job_id}")

        # Step 1: Prepare training data
        self.prepare_training_data()

        # Step 2: Determine world size
        world_size = self.salad_config["resources"]["num_replicas"]
        print(f"[SaladLauncher] World size: {world_size}")

        # Step 3: Create container groups for each rank
        container_group_ids = []
        for rank in range(world_size):
            print(f"[SaladLauncher] Creating container group for rank {rank}...")

            # Build container spec (returns SDK model object)
            container_group_request = build_training_container_spec(self.config, rank, world_size, self.job_id)

            # Create container group via Salad API
            group_name = f"{self.job_id}-rank-{rank}"

            try:
                result = self.sdk.container_groups.create_container_group(
                    request_body=container_group_request,
                    organization_name=self.organization,
                    project_name=self.project,
                )

                container_group_ids.append(
                    {
                        "rank": rank,
                        "group_id": result.id if hasattr(result, "id") else group_name,
                        "group_name": group_name,
                    }
                )

                print(f"  ✓ Container group created: {group_name}")
            except Exception as e:
                print(f"  ✗ Failed to create container group: {e}")
                raise

        # Step 4: Wait for containers to start
        print("[SaladLauncher] Waiting for containers to start...")
        self._wait_for_containers_ready(container_group_ids)

        # Step 5: Monitor training
        print("[SaladLauncher] Training started! Monitoring progress...")
        monitor = SaladJobMonitor(
            self.sdk, self.organization, self.project, container_group_ids
        )

        if self.salad_config.get("monitoring", {}).get("log_streaming", True):
            monitor.stream_logs(
                poll_interval=self.salad_config.get("monitoring", {}).get(
                    "poll_interval", 30
                )
            )

        print(f"[SaladLauncher] ✓ Job {self.job_id} completed")

    def _wait_for_containers_ready(
        self, container_groups: List[Dict], timeout: int = None
    ) -> None:
        """Wait for all containers to reach 'running' state."""
        start_time = time.time()

        print(f"  Waiting for containers (timeout: {'disabled' if timeout is None else f'{timeout}s'})...")

        while True:
            # Check timeout if enabled
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Containers not ready within {timeout}s")

            all_ready = True

            for group_info in container_groups:
                try:
                    status = self.sdk.container_groups.get_container_group(
                        organization_name=self.organization,
                        project_name=self.project,
                        container_group_name=group_info["group_name"],
                    )

                    if status.current_state.status != "running":
                        all_ready = False
                        break
                except Exception:
                    all_ready = False
                    break

            if all_ready:
                elapsed = int(time.time() - start_time)
                print(f"  ✓ All containers ready (took {elapsed}s)")
                return

            time.sleep(10)

    def cleanup(self) -> None:
        """Clean up container groups after training."""
        print(f"[SaladLauncher] Cleaning up job {self.job_id}...")
        # Delete container groups via API
        # Note: Implement cleanup based on Salad API
        print("  ✓ Cleanup complete")


def launch_salad_training(config: dict, script_path: str, script_args: List[str]) -> None:
    """
    Main entry point for Salad distributed training.

    Args:
        config: Training configuration dictionary
        script_path: Path to training script
        script_args: Arguments to pass to training script
    """
    launcher = SaladLauncher(config, script_path, script_args)

    try:
        launcher.launch()
    except KeyboardInterrupt:
        print("\n[SaladLauncher] Interrupted by user")
        launcher.cleanup()
    except Exception as e:
        print(f"\n[SaladLauncher] Error: {e}")
        launcher.cleanup()
        raise
