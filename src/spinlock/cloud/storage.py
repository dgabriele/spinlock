"""
Storage backend interface for dataset generation.

Provides pluggable storage backends for local and cloud dataset storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class StorageBackend(ABC):
    """Abstract interface for dataset storage (local or cloud)."""

    @abstractmethod
    def initialize(self, output_path: str, config: Dict[str, Any]) -> None:
        """
        Initialize storage (create datasets, set compression).

        Args:
            output_path: Path to output dataset (local path or S3 key)
            config: Storage configuration (compression, chunk_size, etc.)
        """
        pass

    @abstractmethod
    def write_batch(
        self,
        parameters: np.ndarray,
        inputs: np.ndarray,
        outputs: np.ndarray,
        ic_types: Optional[Any] = None,
        evolution_policies: Optional[Any] = None,
        grid_sizes: Optional[Any] = None,
        noise_regimes: Optional[Any] = None,
    ) -> None:
        """
        Write a batch of data.

        Args:
            parameters: [B, P] parameter values
            inputs: [B, C_in, H, W] input fields
            outputs: [B, M, C_out, H, W] or [B, M, T, C_out, H, W] output fields
            ic_types: List of IC type strings (optional)
            evolution_policies: List of evolution policy strings (optional)
            grid_sizes: List of grid sizes (optional)
            noise_regimes: List of noise regime strings (optional)
        """
        pass

    @abstractmethod
    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Write dataset metadata (config, creation_date, etc.).

        Args:
            metadata: Dictionary of metadata key-value pairs
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Finalize and close storage."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics (size, write time, etc.).

        Returns:
            Dictionary of statistics
        """
        pass


class LocalHDF5Backend(StorageBackend):
    """Local HDF5 storage (current implementation)."""

    def __init__(self):
        self._writer = None  # HDF5DatasetWriter instance

    def initialize(self, output_path: str, config: Dict[str, Any]) -> None:
        from spinlock.dataset.storage import HDF5DatasetWriter
        from pathlib import Path

        # Extract all required parameters from config
        # Validate required fields
        required_fields = ["grid_size", "input_channels", "output_channels",
                          "num_realizations", "num_parameter_sets"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required config field missing: {field}")

        self._writer = HDF5DatasetWriter(
            output_path=Path(output_path),
            grid_size=int(config["grid_size"]),
            input_channels=int(config["input_channels"]),
            output_channels=int(config["output_channels"]),
            num_realizations=int(config["num_realizations"]),
            num_parameter_sets=int(config["num_parameter_sets"]),
            compression=config.get("compression", "gzip"),
            compression_opts=config.get("compression_level", 4),
            chunk_size=config.get("chunk_size", 20),
            track_ic_metadata=config.get("track_ic_metadata", True),
            store_trajectories=config.get("store_trajectories", True),
            num_timesteps=config.get("num_timesteps", 1),
        )
        # Enter context manager
        self._writer.__enter__()

    def write_batch(
        self,
        parameters: np.ndarray,
        inputs: np.ndarray,
        outputs: np.ndarray,
        ic_types: Optional[Any] = None,
        evolution_policies: Optional[Any] = None,
        grid_sizes: Optional[Any] = None,
        noise_regimes: Optional[Any] = None,
    ) -> None:
        if self._writer is None:
            raise RuntimeError("Storage backend not initialized. Call initialize() first.")
        self._writer.write_batch(
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            ic_types=ic_types,
            evolution_policies=evolution_policies,
            grid_sizes=grid_sizes,
            noise_regimes=noise_regimes,
        )

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        if self._writer is None:
            raise RuntimeError("Storage backend not initialized. Call initialize() first.")
        self._writer.write_metadata(metadata)

    def close(self) -> None:
        if self._writer:
            # Exit context manager
            self._writer.__exit__(None, None, None)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "local_hdf5",
            "path": str(self._writer.output_path) if self._writer else None
        }


class S3Backend(StorageBackend):
    """S3 storage for cloud-generated datasets."""

    def __init__(self, bucket: str, prefix: str = "datasets/"):
        self._bucket = bucket
        self._prefix = prefix
        self._local_buffer = None  # Buffer locally, upload on close
        self._temp_file = None
        self._local_backend = None
        self._final_s3_key = None

    def initialize(self, output_path: str, config: Dict[str, Any]) -> None:
        import tempfile
        # Buffer to local temp file, then upload to S3 on close
        self._temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        self._local_backend = LocalHDF5Backend()
        self._local_backend.initialize(self._temp_file.name, config)
        self._final_s3_key = f"{self._prefix}{output_path}"

    def write_batch(
        self,
        parameters: np.ndarray,
        inputs: np.ndarray,
        outputs: np.ndarray,
        ic_types: Optional[Any] = None,
        evolution_policies: Optional[Any] = None,
        grid_sizes: Optional[Any] = None,
        noise_regimes: Optional[Any] = None,
    ) -> None:
        # Delegate to local buffer
        if self._local_backend is None:
            raise RuntimeError("S3 backend not initialized. Call initialize() first.")
        self._local_backend.write_batch(
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            ic_types=ic_types,
            evolution_policies=evolution_policies,
            grid_sizes=grid_sizes,
            noise_regimes=noise_regimes,
        )

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        if self._local_backend is None:
            raise RuntimeError("S3 backend not initialized. Call initialize() first.")
        self._local_backend.write_metadata(metadata)

    def close(self) -> None:
        try:
            import boto3  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("boto3 is required for S3 backend. Install with: pip install boto3")

        import os

        # Finalize local file
        if self._local_backend is None or self._temp_file is None:
            raise RuntimeError("S3 backend not initialized properly.")
        self._local_backend.close()

        # Upload to S3
        print(f"Uploading dataset to s3://{self._bucket}/{self._final_s3_key}...")
        s3 = boto3.client("s3")
        s3.upload_file(
            self._temp_file.name,
            self._bucket,
            self._final_s3_key
        )
        print(f"Upload complete!")

        # Cleanup temp file
        os.unlink(self._temp_file.name)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": "s3",
            "bucket": self._bucket,
            "key": self._final_s3_key
        }
