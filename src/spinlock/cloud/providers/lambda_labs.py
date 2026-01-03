"""
Lambda Labs API client for instance management.

Provides methods for launching, monitoring, and terminating GPU instances
via the Lambda Labs Cloud API.
"""

import requests
import time
from typing import Dict, List, Optional


class LambdaLabsClient:
    """Lambda Labs API client for instance management."""

    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self, api_key: str):
        """
        Initialize Lambda Labs client.

        Args:
            api_key: Lambda Labs API key
        """
        self._api_key = api_key
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def list_instance_types(self) -> List[Dict]:
        """
        List available instance types.

        Returns:
            List of instance type dictionaries
        """
        response = requests.get(
            f"{self.BASE_URL}/instance-types",
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()["data"]

    def launch_instance(
        self,
        instance_type: str,
        region: str,
        ssh_key_names: List[str],
        name: Optional[str] = None
    ) -> Dict:
        """
        Launch a new GPU instance.

        Args:
            instance_type: Instance type name (e.g., "gpu_1x_a100_sxm4")
            region: Region name (e.g., "us-west-1")
            ssh_key_names: List of SSH key names registered in Lambda Labs
            name: Optional instance name

        Returns:
            Instance details dictionary with id, ip, status
        """
        payload = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": ssh_key_names,
            "quantity": 1
        }

        if name:
            payload["name"] = name

        response = requests.post(
            f"{self.BASE_URL}/instance-operations/launch",
            headers=self._headers,
            json=payload
        )
        response.raise_for_status()

        data = response.json()["data"]
        instance_ids = data["instance_ids"]

        if not instance_ids:
            raise RuntimeError("No instance IDs returned from Lambda Labs API")

        instance_id = instance_ids[0]

        # Wait for instance to be active (Lambda Labs uses "active" not "running")
        return self.wait_for_instance(instance_id, target_status="active", timeout=600)

    def wait_for_instance(
        self,
        instance_id: str,
        target_status: str = "running",
        timeout: int = 300
    ) -> Dict:
        """
        Wait for instance to reach target status.

        Args:
            instance_id: Instance ID
            target_status: Target status (e.g., "running")
            timeout: Maximum wait time in seconds

        Returns:
            Instance details dictionary

        Raises:
            TimeoutError: If instance doesn't reach target status within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            instance = self.get_instance(instance_id)

            if instance["status"] == target_status:
                return instance

            time.sleep(10)

        raise TimeoutError(
            f"Instance {instance_id} did not reach {target_status} within {timeout}s"
        )

    def get_instance(self, instance_id: str) -> Dict:
        """
        Get instance details.

        Args:
            instance_id: Instance ID

        Returns:
            Instance details dictionary with id, ip, status, etc.
        """
        response = requests.get(
            f"{self.BASE_URL}/instances/{instance_id}",
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()["data"]

    def list_instances(self) -> List[Dict]:
        """
        List all instances.

        Returns:
            List of instance dictionaries
        """
        response = requests.get(
            f"{self.BASE_URL}/instances",
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()["data"]

    def terminate_instance(self, instance_id: str) -> bool:
        """
        Terminate an instance.

        Args:
            instance_id: Instance ID

        Returns:
            True if successful
        """
        response = requests.post(
            f"{self.BASE_URL}/instance-operations/terminate",
            headers=self._headers,
            json={"instance_ids": [instance_id]}
        )
        response.raise_for_status()
        return True

    def list_ssh_keys(self) -> List[Dict]:
        """
        List SSH keys.

        Returns:
            List of SSH key dictionaries
        """
        response = requests.get(
            f"{self.BASE_URL}/ssh-keys",
            headers=self._headers
        )
        response.raise_for_status()
        return response.json()["data"]

    def add_ssh_key(self, name: str, public_key: str) -> Dict:
        """
        Add SSH key.

        Args:
            name: SSH key name
            public_key: SSH public key content

        Returns:
            SSH key details dictionary
        """
        response = requests.post(
            f"{self.BASE_URL}/ssh-keys",
            headers=self._headers,
            json={"name": name, "public_key": public_key}
        )
        response.raise_for_status()
        return response.json()["data"]
