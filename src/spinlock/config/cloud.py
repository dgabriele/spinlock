"""
Cloud provider configuration schemas.

Provides Pydantic models for configuring cloud providers:
- Lambda Labs GPU Cloud
- RunPod
- AWS S3 Storage
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class CloudProviderConfig(BaseModel):
    """Base cloud provider configuration."""
    enabled: bool = Field(default=False, description="Enable cloud provider")
    provider: Literal["lambda_labs", "runpod", "aws", "local"] = Field(
        default="local",
        description="Cloud provider to use"
    )


class LambdaLabsConfig(BaseModel):
    """Lambda Labs GPU Cloud configuration."""
    api_key: str = Field(..., description="Lambda Labs API key (can use ${ENV_VAR} for environment variables)")
    gpu_type: str = Field(default="A100", description="GPU type (A100, A40, RTX4090, etc.)")
    region: str = Field(default="us-west-1", description="Preferred region for instance provisioning")
    instance_type: str = Field(
        default="gpu_1x_a100_sxm4",
        description="Lambda Labs instance type ID"
    )
    ssh_key_name: str = Field(
        default="spinlock-key",
        description="SSH key name (must be registered in Lambda Labs)"
    )

    # Billing and safety limits
    max_cost_per_hour: float = Field(
        default=2.0,
        description="Maximum $/hour safety limit (will refuse to provision if exceeded)"
    )
    auto_shutdown_on_completion: bool = Field(
        default=True,
        description="Automatically terminate instance after job completion"
    )

    # Job execution settings
    remote_working_dir: str = Field(
        default="/home/ubuntu/spinlock",
        description="Working directory on remote instance"
    )
    setup_script: Optional[str] = Field(
        default=None,
        description="Path to custom setup script (optional, runs before job)"
    )


class RunPodConfig(BaseModel):
    """RunPod GPU Cloud configuration."""
    api_key: str = Field(..., description="RunPod API key")
    gpu_type: str = Field(default="RTX4090", description="GPU type (RTX4090, A100, etc.)")
    network_volume_id: Optional[str] = Field(
        default=None,
        description="Network volume ID for persistent storage"
    )
    max_bid_per_hour: float = Field(
        default=1.0,
        description="Maximum bid per hour for spot instances"
    )
    auto_shutdown_on_completion: bool = Field(
        default=True,
        description="Automatically terminate pod after job completion"
    )


class LambdaLabsFileStorageConfig(BaseModel):
    """Lambda Labs built-in file storage configuration."""
    remote_path: str = Field(
        default="/home/ubuntu/datasets",
        description="Remote path on Lambda Labs instance where datasets are stored"
    )
    local_download_path: Optional[str] = Field(
        default=None,
        description="Optional local path to download dataset after generation (None = no download)"
    )


class S3Config(BaseModel):
    """AWS S3 storage configuration."""
    bucket: str = Field(..., description="S3 bucket name")
    prefix: str = Field(
        default="datasets/",
        description="S3 key prefix (subdirectory within bucket)"
    )
    region: str = Field(default="us-west-2", description="AWS region")

    # Optional credentials (defaults to AWS CLI/environment credentials)
    access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID (optional, uses default credentials if not provided)"
    )
    secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key (optional)"
    )

    # Storage class
    storage_class: str = Field(
        default="STANDARD",
        description="S3 storage class (STANDARD, INTELLIGENT_TIERING, GLACIER, etc.)"
    )


class CloudConfig(BaseModel):
    """Cloud provider configuration."""
    provider: CloudProviderConfig = Field(
        default_factory=CloudProviderConfig,
        description="Cloud provider selection and enablement"
    )

    # Provider-specific configs
    lambda_labs: Optional[LambdaLabsConfig] = Field(
        default=None,
        description="Lambda Labs GPU Cloud configuration"
    )
    runpod: Optional[RunPodConfig] = Field(
        default=None,
        description="RunPod GPU Cloud configuration"
    )

    # Storage configs
    lambda_labs_storage: Optional[LambdaLabsFileStorageConfig] = Field(
        default=None,
        description="Lambda Labs built-in file storage (alternative to S3)"
    )
    s3: Optional[S3Config] = Field(
        default=None,
        description="AWS S3 storage configuration"
    )

    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True
