"""Job monitoring and logging for Salad container groups."""

from typing import List, Dict
import time
from salad_cloud_sdk import SaladCloudSdk


class SaladJobMonitor:
    """Monitor Salad.com container groups and stream logs."""

    def __init__(
        self,
        sdk: SaladCloudSdk,
        organization: str,
        project: str,
        container_groups: List[Dict],
    ):
        """
        Initialize Salad job monitor.

        Args:
            sdk: Salad Cloud SDK instance
            organization: Organization name
            project: Project name
            container_groups: List of container group info dicts
        """
        self.sdk = sdk
        self.organization = organization
        self.project = project
        self.container_groups = container_groups

    def get_status(self) -> Dict[int, str]:
        """
        Get current status of all containers.

        Returns:
            Dictionary mapping rank to status string
        """
        statuses = {}

        for group_info in self.container_groups:
            try:
                result = self.sdk.container_groups.get_container_group(
                    organization_name=self.organization,
                    project_name=self.project,
                    container_group_name=group_info["group_name"],
                )
                statuses[group_info["rank"]] = result.current_state.status
            except Exception as e:
                statuses[group_info["rank"]] = f"error: {e}"

        return statuses

    def stream_logs(self, poll_interval: int = 30) -> None:
        """
        Stream logs from containers.

        Args:
            poll_interval: Seconds between status polls
        """
        print("[Monitor] Streaming logs (press Ctrl+C to stop)...")

        try:
            while True:
                statuses = self.get_status()

                # Print status summary
                print(f"\n[Monitor] Status at {time.strftime('%H:%M:%S')}:")
                for rank, status in statuses.items():
                    print(f"  Rank {rank}: {status}")

                # Check if all completed
                if all(s in ["stopped", "failed"] for s in statuses.values()):
                    print("[Monitor] All containers finished")
                    break

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n[Monitor] Stopped monitoring")
