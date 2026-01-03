"""
Orchestration backend interface for job management.

Provides pluggable orchestration backends for local and cloud job execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from pathlib import Path


class OrchestrationBackend(ABC):
    """Abstract interface for job orchestration (local or cloud)."""

    @abstractmethod
    def submit_job(
        self,
        config_path: Path,
        output_path: Path,
        storage_backend: str = "local",
        execution_backend: str = "local"
    ) -> str:
        """
        Submit dataset generation job, return job_id.

        Args:
            config_path: Path to config file
            output_path: Path to output dataset
            storage_backend: Storage backend type ("local", "s3", etc.)
            execution_backend: Execution backend type ("local", "lambda_labs", etc.)

        Returns:
            Job ID for monitoring
        """
        pass

    @abstractmethod
    def monitor_job(self, job_id: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Monitor job progress, optionally call callback with updates.

        Args:
            job_id: Job ID to monitor
            callback: Optional callback function for progress updates

        Returns:
            Job status dictionary
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get current job status.

        Args:
            job_id: Job ID to query

        Returns:
            Job status dictionary
        """
        pass


class LocalSequentialOrchestrator(OrchestrationBackend):
    """Local sequential execution (current implementation)."""

    def submit_job(
        self,
        config_path: Path,
        output_path: Path,
        storage_backend: str = "local",
        execution_backend: str = "local"
    ) -> str:
        # Just run the pipeline directly (blocking)
        from spinlock.config import load_config
        from spinlock.dataset import DatasetGenerationPipeline

        config = load_config(config_path)

        # Inject backends (via factory pattern)
        storage = self._create_storage_backend(storage_backend, config)
        execution = self._create_execution_backend(execution_backend, config)

        pipeline = DatasetGenerationPipeline(
            config=config,
            storage_backend=storage,
            execution_backend=execution
        )

        pipeline.generate()

        return "local-job-completed"

    def monitor_job(self, job_id: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        # Local jobs are synchronous, no monitoring needed
        return {"status": "completed"}

    def cancel_job(self, job_id: str) -> bool:
        return False  # Can't cancel synchronous jobs

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        return {"status": "completed"}

    def _create_storage_backend(self, backend_type: str, config):
        from spinlock.cloud.storage import LocalHDF5Backend, S3Backend

        if backend_type == "local" or backend_type == "lambda_labs":
            # For local execution, both "local" and "lambda_labs" use local HDF5
            # (lambda_labs storage is only for remote execution via SCP)
            return LocalHDF5Backend()
        elif backend_type == "s3":
            s3_config = config.cloud.s3
            return S3Backend(
                bucket=s3_config.bucket,
                prefix=s3_config.prefix
            )
        else:
            raise ValueError(f"Unknown storage backend: {backend_type}")

    def _create_execution_backend(self, backend_type: str, config):
        from spinlock.cloud.execution import LocalExecutionBackend

        if backend_type == "local":
            return LocalExecutionBackend()
        else:
            raise ValueError(f"Unknown execution backend: {backend_type}")


class CloudJobOrchestrator(OrchestrationBackend):
    """Cloud job orchestration (Lambda Labs, RunPod, etc.)."""

    def __init__(self, provider: str = "lambda_labs", config=None):
        self._provider = provider
        self._config = config
        self._jobs = {}  # job_id -> job info
        self._client = None  # Provider API client

        # Initialize provider client
        if provider == "lambda_labs":
            from spinlock.cloud.providers.lambda_labs import LambdaLabsClient
            import os

            api_key = config.lambda_labs.api_key if config and config.lambda_labs else os.getenv("LAMBDA_API_KEY")
            if not api_key:
                raise ValueError("Lambda Labs API key not found in config or environment")

            self._client = LambdaLabsClient(api_key)

    def submit_job(
        self,
        config_path: Path,
        output_path: Path,
        storage_backend: str = "s3",
        execution_backend: str = "lambda_labs"
    ) -> str:
        if self._provider == "lambda_labs":
            return self._submit_lambda_labs_job(config_path, output_path, storage_backend)
        else:
            raise NotImplementedError(f"Provider {self._provider} not implemented")

    def _submit_lambda_labs_job(self, config_path: Path, output_path: Path, storage_backend: str) -> str:
        """Submit job to Lambda Labs via SSH + screen session."""
        import paramiko
        import uuid
        import time
        import socket

        job_id = str(uuid.uuid4())[:8]

        # Step 1: Provision instance (if not already running)
        instance = self._get_or_create_lambda_instance()

        print(f"\nConnecting to instance {instance['ip']}...")

        # Step 2: Wait for SSH to be available (instance may still be booting)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        max_retries = 10
        retry_delay = 15

        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}...")
                ssh.connect(
                    hostname=instance["ip"],
                    username="ubuntu",
                    key_filename=instance["ssh_key_path"],
                    timeout=30,
                    banner_timeout=30,
                    auth_timeout=30
                )
                print("✓ SSH connection established")
                break
            except (paramiko.ssh_exception.NoValidConnectionsError,
                    paramiko.ssh_exception.SSHException,
                    socket.timeout,
                    ConnectionRefusedError) as e:
                if attempt < max_retries - 1:
                    print(f"  Connection failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect to instance after {max_retries} attempts: {e}")

        # Step 3: Setup spinlock on the instance
        print("\nSetting up Spinlock environment on instance...")

        # Upload entire spinlock directory
        import os
        import tarfile
        from pathlib import Path as PathLib

        # Create tarball of local spinlock code
        # Find project root by looking for pyproject.toml
        current_path = PathLib(__file__).resolve()
        local_spinlock_root = None
        for parent in current_path.parents:
            if (parent / "pyproject.toml").exists():
                local_spinlock_root = parent
                break

        if not local_spinlock_root:
            raise RuntimeError("Could not find project root (pyproject.toml not found)")

        tarball_path = f"/tmp/spinlock_{job_id}.tar.gz"

        print(f"  Creating tarball of local code...")
        def exclude_filter(tarinfo):
            # Exclude directories and files we don't want (large data/cache dirs)
            excludes = [
                '.git', '__pycache__', '.pytest_cache', 'datasets', '.venv',
                'node_modules', 'runs', '.mypy_cache', 'checkpoints', 'visualizations'
            ]
            for exclude in excludes:
                if f"/{exclude}/" in tarinfo.name or tarinfo.name.endswith(f"/{exclude}") or tarinfo.name == exclude:
                    return None
            # Also exclude large data files
            if tarinfo.name.endswith(('.h5', '.hdf5', '.pt', '.pth', '.ckpt', '.safetensors')):
                return None
            return tarinfo

        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(local_spinlock_root, arcname="spinlock", filter=exclude_filter)

        # Upload tarball
        print(f"  Uploading code to instance...")
        sftp = ssh.open_sftp()
        remote_tarball = f"/tmp/spinlock_{job_id}.tar.gz"
        sftp.put(tarball_path, remote_tarball)

        # Upload config
        remote_config_path = f"/tmp/spinlock_config_{job_id}.yaml"
        sftp.put(str(config_path), remote_config_path)
        sftp.close()

        # Extract and install
        print(f"  Installing dependencies...")
        setup_cmd = f"""
            cd /tmp &&
            tar -xzf {remote_tarball} &&
            cd spinlock &&
            sudo add-apt-repository -y ppa:deadsnakes/ppa &&
            sudo apt-get update -qq &&
            sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev &&
            curl -sSL https://install.python-poetry.org | python3.11 - &&
            export PATH=$HOME/.local/bin:$PATH &&
            poetry env use python3.11 &&
            poetry install --no-interaction
        """
        stdin, stdout, stderr = ssh.exec_command(setup_cmd, timeout=900)
        exit_code = stdout.channel.recv_exit_status()

        if exit_code != 0:
            error = stderr.read().decode()
            raise RuntimeError(f"Failed to setup environment: {error}")

        print("✓ Environment setup complete")

        # Step 4: Launch generation job in screen session
        print("\nLaunching dataset generation job...")
        screen_name = f"spinlock_{job_id}"
        remote_output = f"/tmp/spinlock_output_{job_id}.h5"

        run_cmd = f"""
            screen -dmS {screen_name} bash -c '
                cd /tmp/spinlock &&
                export PATH=$HOME/.local/bin:$PATH &&
                export LAMBDA_API_KEY={self._config.lambda_labs.api_key} &&
                poetry run spinlock generate \\
                    --config {remote_config_path} \\
                    2>&1 | tee /tmp/spinlock_{job_id}.log
            '
        """

        stdin, stdout, stderr = ssh.exec_command(run_cmd)
        stdout.channel.recv_exit_status()

        print(f"✓ Job launched in screen session: {screen_name}")
        print(f"  Instance: {instance['instance_id']}")
        print(f"  IP: {instance['ip']}")
        print(f"  Log: /tmp/spinlock_{job_id}.log")
        print(f"  Output: {remote_output}")

        # Don't close SSH yet - we'll use it for monitoring
        # ssh.close()

        # Track job
        self._jobs[job_id] = {
            "instance": instance,
            "screen_name": screen_name,
            "log_path": f"/tmp/spinlock_{job_id}.log",
            "output_path": f"/tmp/spinlock_output_{job_id}.h5",
            "status": "running"
        }

        return job_id

    def monitor_job(self, job_id: str, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Monitor job via SSH log tailing."""
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "unknown"}

        # SSH into instance and tail log
        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=job["instance"]["ip"],
            username="ubuntu",
            key_filename=job["instance"]["ssh_key_path"]
        )

        # Check if screen session still running
        stdin, stdout, stderr = ssh.exec_command(f"screen -list | grep {job['screen_name']}")
        screen_exists = bool(stdout.read().decode().strip())

        if not screen_exists:
            # Job finished, check exit code
            stdin, stdout, stderr = ssh.exec_command(f"test -f {job['output_path']} && echo 'success' || echo 'failed'")
            result = stdout.read().decode().strip()

            job["status"] = "completed" if result == "success" else "failed"

            # Download output if successful
            if job["status"] == "completed":
                self._download_output(ssh, job)

        ssh.close()

        return {"status": job["status"], "job_id": job_id}

    def cancel_job(self, job_id: str) -> bool:
        """Cancel job by killing screen session."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        import paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=job["instance"]["ip"],
            username="ubuntu",
            key_filename=job["instance"]["ssh_key_path"]
        )

        ssh.exec_command(f"screen -S {job['screen_name']} -X quit")
        ssh.close()

        job["status"] = "cancelled"
        return True

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        return {"status": job["status"]} if job else {"status": "unknown"}

    def _get_or_create_lambda_instance(self) -> Dict[str, str]:
        """Get existing instance or create new one via Lambda Labs API."""
        import os
        from pathlib import Path

        config = self._config.lambda_labs

        # Check for existing active instances first (Lambda Labs uses "active" status)
        existing_instances = self._client.list_instances()
        running_instances = [
            inst for inst in existing_instances
            if inst["status"] == "active" and inst["instance_type"]["name"] == config.instance_type
        ]

        if running_instances:
            print(f"Found existing running instance: {running_instances[0]['id']}")
            instance = running_instances[0]
        else:
            print(f"Launching new Lambda Labs instance ({config.gpu_type})...")
            instance = self._client.launch_instance(
                instance_type=config.instance_type,
                region=config.region,
                ssh_key_names=[config.ssh_key_name],
                name="spinlock-dataset-generation"
            )
            print(f"✓ Instance launched: {instance['id']}")
            print(f"  IP: {instance['ip']}")
            print(f"  Status: {instance['status']}")

        # Determine SSH key path (use default SSH keys, not the Lambda Labs key name)
        # Try common SSH key locations in order
        possible_keys = [
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_rsa"),
            os.path.expanduser("~/.ssh/id_ecdsa"),
        ]

        ssh_key_path = None
        for key_path in possible_keys:
            if Path(key_path).exists():
                ssh_key_path = key_path
                break

        if not ssh_key_path:
            print(f"Warning: No default SSH key found. Checked: {', '.join(possible_keys)}")
            ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")  # Fallback

        return {
            "ip": instance["ip"],
            "ssh_key_path": ssh_key_path,
            "instance_id": instance["id"]
        }

    def _download_output(self, ssh, job: Dict[str, Any]) -> None:
        """Download generated dataset from Lambda instance to local or S3."""
        import os
        from pathlib import Path

        sftp = ssh.open_sftp()

        # Determine local download path from config
        if self._config and self._config.lambda_labs_storage:
            local_dir = self._config.lambda_labs_storage.local_download_path or "./datasets"
        else:
            local_dir = "./datasets"

        # Create local directory if needed
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Extract filename from remote path
        remote_file = job["output_path"]
        filename = os.path.basename(remote_file)
        local_path = os.path.join(local_dir, filename)

        print(f"Downloading {remote_file} → {local_path}")
        sftp.get(remote_file, local_path)
        sftp.close()

        print(f"✓ Dataset downloaded: {local_path}")

        # Auto-terminate instance if configured
        if self._config and self._config.lambda_labs.auto_shutdown_on_completion:
            instance_id = job['instance']['instance_id']
            print(f"Terminating instance {instance_id}...")
            self._client.terminate_instance(instance_id)
            print(f"✓ Instance terminated")
