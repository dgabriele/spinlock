"""Cloud storage abstraction for Salad distributed training."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import boto3
from botocore.exceptions import ClientError


class CloudStorageBackend(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file to cloud storage."""
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download file from cloud storage."""
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if remote path exists."""
        pass

    @abstractmethod
    def list_files(self, prefix: str) -> list[str]:
        """List files with given prefix."""
        pass


class S3StorageBackend(CloudStorageBackend):
    """AWS S3 or MinIO storage backend (S3-compatible)."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,  # MinIO endpoint
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        use_ssl: bool = True,
    ):
        """
        Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name
            endpoint_url: Custom endpoint URL (for MinIO, None for AWS S3)
            region: AWS region (default: us-east-1)
            access_key: AWS access key ID
            secret_key: AWS secret access key
            use_ssl: Use SSL/TLS for connections
        """
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,  # None for AWS S3, set for MinIO
            region_name=region or "us-east-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=use_ssl,
            config=boto3.session.Config(signature_version="s3v4"),
        )

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        """Upload file to S3."""
        self.s3.upload_file(str(local_path), self.bucket, remote_path)

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download file from S3."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket, remote_path, str(local_path))

    def exists(self, remote_path: str) -> bool:
        """Check if S3 object exists."""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str) -> list[str]:
        """List S3 objects with prefix."""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]


class StorageManager:
    """High-level storage manager for training data sync."""

    def __init__(self, backend: CloudStorageBackend):
        """
        Initialize storage manager.

        Args:
            backend: Cloud storage backend instance
        """
        self.backend = backend

    def sync_dataset(self, remote_path: str, local_path: Path) -> None:
        """
        Download dataset if not exists locally.

        Args:
            remote_path: Remote storage path
            local_path: Local file path
        """
        if not local_path.exists():
            print(f"Downloading dataset from {remote_path}...")
            self.backend.download_file(remote_path, local_path)
            print(f"✓ Dataset downloaded to {local_path}")
        else:
            print(f"✓ Dataset already exists at {local_path}")

    def sync_checkpoint(self, local_path: Path, remote_path: str) -> None:
        """
        Upload checkpoint to cloud storage.

        Args:
            local_path: Local checkpoint file
            remote_path: Remote storage path
        """
        if local_path.exists():
            print(f"Uploading checkpoint to {remote_path}...")
            self.backend.upload_file(local_path, remote_path)
            print("✓ Checkpoint uploaded")

    def download_checkpoint(self, remote_path: str, local_path: Path) -> bool:
        """
        Download checkpoint if exists.

        Args:
            remote_path: Remote checkpoint path
            local_path: Local file path

        Returns:
            True if checkpoint was downloaded, False if not found
        """
        if self.backend.exists(remote_path):
            print(f"Downloading checkpoint from {remote_path}...")
            self.backend.download_file(remote_path, local_path)
            print("✓ Checkpoint downloaded")
            return True
        return False


def create_storage_backend(config: dict) -> CloudStorageBackend:
    """
    Factory function to create storage backend from config.

    Args:
        config: Storage configuration dictionary

    Returns:
        CloudStorageBackend instance

    Raises:
        ValueError: If backend type is unknown
    """
    backend_type = config.get("backend", "s3").lower()

    if backend_type in ["s3", "minio"]:  # Both use S3-compatible API
        return S3StorageBackend(
            bucket=config["bucket"],
            endpoint_url=config.get("endpoint_url"),  # None for S3, set for MinIO
            region=config.get("region"),
            access_key=config.get("credentials", {}).get("aws_access_key_id"),
            secret_key=config.get("credentials", {}).get("aws_secret_access_key"),
            use_ssl=config.get("use_ssl", True),
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")
