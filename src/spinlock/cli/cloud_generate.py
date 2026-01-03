"""
Cloud dataset generation command.

Submits dataset generation jobs to cloud GPU providers (Lambda Labs, RunPod, etc.)
with job monitoring and status tracking.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import time
import sys

from .base import CLICommand
from spinlock.config import load_config
from spinlock.cloud.orchestration import LocalSequentialOrchestrator, CloudJobOrchestrator


class CloudGenerateCommand(CLICommand):
    """Generate datasets using cloud GPU providers."""

    @property
    def name(self) -> str:
        return "cloud-generate"

    @property
    def help(self) -> str:
        return "Generate datasets on cloud GPU providers (Lambda Labs, RunPod, etc.)"

    @property
    def description(self) -> str:
        return """
Generate datasets using cloud GPU providers.

Supports:
- Lambda Labs GPU Cloud (A100, A40, RTX4090)
- RunPod (future)
- AWS SageMaker (future)

Features:
- Automatic instance provisioning
- SSH-based job submission
- Real-time monitoring
- Automatic shutdown on completion
- S3 or Lambda Labs file storage

Examples:
  # Test with 10 operators on Lambda Labs (~2 min, $0.02)
  spinlock cloud-generate \\
      --config configs/cloud/lambda_labs_test_10.yaml \\
      --provider lambda_labs \\
      --monitor

  # Generate 100K dataset with monitoring
  spinlock cloud-generate \\
      --config configs/cloud/lambda_labs_100k.yaml \\
      --provider lambda_labs \\
      --monitor

  # Check existing job status
  spinlock cloud-generate \\
      --provider lambda_labs \\
      --job-id abc12345 \\
      --monitor
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add cloud-generate specific arguments."""
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to cloud generation config YAML file"
        )

        parser.add_argument(
            "--provider",
            type=str,
            default="local",
            choices=["local", "lambda_labs", "runpod"],
            help="Cloud provider to use (default: local for testing)"
        )

        parser.add_argument(
            "--monitor",
            action="store_true",
            help="Monitor job progress in real-time"
        )

        parser.add_argument(
            "--job-id",
            type=str,
            help="Job ID to monitor (for checking existing jobs)"
        )

        parser.add_argument(
            "--storage-backend",
            type=str,
            choices=["local", "s3", "lambda_labs"],
            help="Override storage backend from config"
        )

    def execute(self, args: Namespace) -> int:
        """Execute cloud dataset generation."""
        # Monitor existing job
        if args.job_id:
            return self._monitor_existing_job(args)

        # Validate config is provided for new jobs
        if not args.config:
            return self.error("--config is required for new jobs (or use --job-id to monitor existing job)")

        if not self.validate_file_exists(args.config, "Config file"):
            return 1

        print("=" * 70)
        print("SPINLOCK CLOUD DATASET GENERATION")
        print("=" * 70)

        # Load and validate config
        try:
            config = load_config(args.config)
            print(f"✓ Config loaded: {args.config}")
            print(f"  Provider: {args.provider}")
            print(f"  Samples: {config.sampling.total_samples:,}")
            print(f"  Output: {config.dataset.output_path}")
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Determine storage backend
        storage_backend = self._determine_storage_backend(args, config)
        print(f"  Storage: {storage_backend}")

        # Create orchestrator
        try:
            orchestrator = self._create_orchestrator(args.provider, config)
            print(f"  Orchestrator: {type(orchestrator).__name__}")
        except Exception as e:
            return self.error(f"Failed to create orchestrator: {e}")

        print()

        # Submit job
        try:
            print(f"Submitting job to {args.provider}...")
            job_id = orchestrator.submit_job(
                config_path=args.config,
                output_path=Path(config.dataset.output_path),
                storage_backend=storage_backend,
                execution_backend=args.provider if args.provider != "local" else "local"
            )
            print(f"✓ Job submitted: {job_id}")
        except Exception as e:
            return self.error(f"Failed to submit job: {e}")

        # Monitor if requested
        if args.monitor:
            print()
            return self._monitor_job(orchestrator, job_id)

        print()
        print(f"Job submitted successfully!")
        print(f"To monitor progress, run:")
        print(f"  spinlock cloud-generate --provider {args.provider} --job-id {job_id} --monitor")
        print()

        return 0

    def _determine_storage_backend(self, args: Namespace, config) -> str:
        """Determine which storage backend to use."""
        # CLI override
        if args.storage_backend:
            return args.storage_backend

        # Infer from config
        if config.cloud.s3:
            return "s3"
        elif config.cloud.lambda_labs_storage:
            return "lambda_labs"
        else:
            return "local"

    def _create_orchestrator(self, provider: str, config):
        """Create appropriate orchestrator for provider."""
        if provider == "local":
            return LocalSequentialOrchestrator()
        elif provider in ["lambda_labs", "runpod"]:
            return CloudJobOrchestrator(provider=provider, config=config.cloud)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _monitor_job(self, orchestrator, job_id: str) -> int:
        """Monitor job progress with periodic status checks."""
        print("=" * 70)
        print(f"MONITORING JOB: {job_id}")
        print("=" * 70)
        print("Press Ctrl+C to stop monitoring (job will continue running)\n")

        try:
            last_status = None
            while True:
                # Get current status
                status_info = orchestrator.get_job_status(job_id)
                status = status_info.get("status", "unknown")

                # Print status if changed
                if status != last_status:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] Status: {status}")
                    last_status = status

                # Check if job completed
                if status in ["completed", "failed", "cancelled"]:
                    print()
                    if status == "completed":
                        print("✓ Job completed successfully!")
                        return 0
                    elif status == "failed":
                        print("✗ Job failed")
                        return 1
                    elif status == "cancelled":
                        print("⚠ Job cancelled")
                        return 1

                # Wait before next check
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n")
            print("Monitoring stopped (job is still running)")
            print(f"To resume monitoring, run:")
            print(f"  spinlock cloud-generate --provider <provider> --job-id {job_id} --monitor")
            return 0

    def _monitor_existing_job(self, args: Namespace) -> int:
        """Monitor an existing job by ID."""
        if not args.provider or args.provider == "local":
            return self.error("--provider is required when monitoring jobs")

        print(f"Monitoring job: {args.job_id}")
        print(f"Provider: {args.provider}\n")

        try:
            # Create orchestrator (need config for cloud providers, but we don't have it here)
            # For now, use a minimal config
            from spinlock.config.cloud import CloudConfig
            orchestrator = CloudJobOrchestrator(provider=args.provider, config=CloudConfig())

            return self._monitor_job(orchestrator, args.job_id)
        except Exception as e:
            return self.error(f"Failed to monitor job: {e}")
