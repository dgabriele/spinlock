"""
Execution backend interface for GPU management.

Provides pluggable execution backends for local and cloud GPU execution.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch


class ExecutionBackend(ABC):
    """Abstract interface for GPU execution (local or cloud)."""

    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize execution environment (GPUs, dependencies).

        Args:
            config: Execution configuration (device, precision, etc.)
        """
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """
        Get primary device for execution.

        Returns:
            PyTorch device (cuda:0, cpu, etc.)
        """
        pass

    @abstractmethod
    def get_available_devices(self) -> List[str]:
        """
        List available GPU devices.

        Returns:
            List of device strings (e.g., ["cuda:0", "cuda:1"])
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources (GPUs, memory)."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics (GPU util, memory).

        Returns:
            Dictionary of execution statistics
        """
        pass


class LocalExecutionBackend(ExecutionBackend):
    """Local GPU execution (current implementation)."""

    def __init__(self):
        self._device = None

    def setup(self, config: Dict[str, Any]) -> None:
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device_str)

    def get_device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Execution backend not initialized. Call setup() first.")
        return self._device

    def get_available_devices(self) -> List[str]:
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]

    def cleanup(self) -> None:
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, Any]:
        if self._device and self._device.type == "cuda":
            return {
                "backend": "local_gpu",
                "device": str(self._device),
                "memory_allocated": torch.cuda.memory_allocated(self._device),
                "memory_reserved": torch.cuda.memory_reserved(self._device)
            }
        return {"backend": "local_cpu"}


class LambdaLabsBackend(ExecutionBackend):
    """Lambda Labs GPU Cloud execution."""

    def __init__(self, api_key: str, gpu_type: str = "A100"):
        self._api_key = api_key
        self._gpu_type = gpu_type
        self._instance_id = None
        self._device = None

    def setup(self, config: Dict[str, Any]) -> None:
        # Lambda Labs instances are pre-provisioned, just detect GPU
        # (We'll handle provisioning in CloudJobOrchestrator)
        if torch.cuda.is_available():
            self._device = torch.device("cuda:0")
        else:
            raise RuntimeError("Lambda Labs instance has no GPU!")

    def get_device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("Lambda Labs backend not initialized. Call setup() first.")
        return self._device

    def get_available_devices(self) -> List[str]:
        # Lambda instances typically have 1-8 GPUs
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return []

    def cleanup(self) -> None:
        if self._device and self._device.type == "cuda":
            torch.cuda.empty_cache()
        # Note: Instance shutdown handled by orchestrator

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "lambda_labs",
            "gpu_type": self._gpu_type,
            "device": str(self._device),
            "memory_allocated": torch.cuda.memory_allocated(self._device) if self._device else 0
        }
