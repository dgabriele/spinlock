"""Training-specific launcher for Salad.com distributed training."""

from typing import List
import time
from pathlib import Path
from salad_cloud_sdk.models import ContainerGroupCreationRequest

from .base_launcher import SaladJobLauncher
from .container import build_training_container_spec
from .monitor import SaladJobMonitor


class SaladTrainingLauncher(SaladJobLauncher):
    """Launcher for distributed DDP training on Salad.com."""

    def __init__(self, config: dict, script_path: str, script_args: List[str]):
        """
        Initialize Salad training launcher.

        Args:
            config: Full training configuration dict
            script_path: Path to training script (for reference)
            script_args: Arguments to pass to training script
        """
        super().__init__(config)
        self.script_path = script_path
        self.script_args = script_args

    def prepare_training_data(self) -> None:
        """Upload dataset and config to cloud storage."""
        print("[SaladTrainingLauncher] Preparing training data...")

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

        print("[SaladTrainingLauncher] ✓ Training data prepared")

    def build_container_specs(self) -> List[ContainerGroupCreationRequest]:
        """
        Build multi-rank DDP container specs.

        Returns:
            List of container specs (one per DDP rank)
        """
        # Prepare training data before building specs
        self.prepare_training_data()

        # Determine world size from config
        world_size = self.salad_config["resources"]["num_replicas"]
        print(f"[SaladTrainingLauncher] Building specs for world_size={world_size}")

        # Build one container spec per rank
        specs = []
        for rank in range(world_size):
            spec = build_training_container_spec(
                self.config, rank, world_size, self.job_id
            )
            specs.append(spec)

        return specs

    def monitor_job(self) -> None:
        """Monitor DDP training with log streaming."""
        # Wait for containers to start
        print("[SaladTrainingLauncher] Waiting for containers to start...")
        self._wait_for_containers_ready()

        # Start monitoring
        print("[SaladTrainingLauncher] Training started! Monitoring progress...")

        # Build container group info for monitor
        world_size = self.salad_config["resources"]["num_replicas"]
        container_group_ids = [
            {
                "rank": rank,
                "group_id": f"{self.job_id}-rank-{rank}",
                "group_name": f"{self.job_id}-rank-{rank}",
            }
            for rank in range(world_size)
        ]

        monitor = SaladJobMonitor(
            self.sdk, self.organization, self.project, container_group_ids
        )

        if self.salad_config.get("monitoring", {}).get("log_streaming", True):
            monitor.stream_logs(
                poll_interval=self.salad_config.get("monitoring", {}).get(
                    "poll_interval", 30
                )
            )

    def _wait_for_containers_ready(self, timeout: int = None) -> None:
        """
        Wait for all containers to reach 'running' state.

        Args:
            timeout: Optional timeout in seconds
        """
        world_size = self.salad_config["resources"]["num_replicas"]
        container_groups = [
            {"group_name": f"{self.job_id}-rank-{rank}"} for rank in range(world_size)
        ]

        start_time = time.time()
        print(
            f"  Waiting for {world_size} containers "
            f"(timeout: {'disabled' if timeout is None else f'{timeout}s'})..."
        )

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


def launch_salad_training(config: dict, script_path: str, script_args: List[str]) -> None:
    """
    Main entry point for Salad distributed training.

    Args:
        config: Training configuration dictionary
        script_path: Path to training script
        script_args: Arguments to pass to training script
    """
    launcher = SaladTrainingLauncher(config, script_path, script_args)

    try:
        launcher.launch()
    except KeyboardInterrupt:
        print("\n[SaladTrainingLauncher] Interrupted by user")
        launcher.cleanup()
    except Exception as e:
        print(f"\n[SaladTrainingLauncher] Error: {e}")
        launcher.cleanup()
        raise
