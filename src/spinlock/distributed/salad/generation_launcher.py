"""Generation-specific launcher for Salad.com distributed dataset generation."""

from typing import List, Dict, Any
import time
from salad_cloud_sdk.models import ContainerGroupCreationRequest

from .base_launcher import SaladJobLauncher
from .container import ContainerSpecBuilder


class SaladGenerationLauncher(SaladJobLauncher):
    """Launcher for distributed dataset generation on Salad.com."""

    def __init__(self, config: dict, generation_config: Dict[str, Any]):
        """
        Initialize Salad generation launcher.

        Args:
            config: Full configuration dict with distributed.salad section
            generation_config: Generation-specific config with:
                - total_samples: Total number of samples to generate
                - samples_per_job: Number of samples per job
                - dataset_config: Path to dataset config YAML
                - output_prefix: Prefix for output HDF5 files (e.g., "qbm_1m_wide_part")
        """
        super().__init__(config)
        self.generation_config = generation_config

        # Calculate number of jobs based on total_samples and samples_per_job
        self.num_jobs = (
            generation_config["total_samples"] // generation_config["samples_per_job"]
        )

        print(
            f"[SaladGenerationLauncher] Splitting {generation_config['total_samples']} samples "
            f"into {self.num_jobs} jobs of {generation_config['samples_per_job']} samples each"
        )

    def build_container_specs(self) -> List[ContainerGroupCreationRequest]:
        """
        Build independent generation job specs (no DDP coordination).

        Returns:
            List of container specs (one per generation job)

        Notes:
            Each job:
            - Runs independently (no RANK/WORLD_SIZE)
            - Gets unique Sobol offset: job_idx * samples_per_job
            - Uploads to unique S3 path: output_prefix{job_idx:02d}.h5
        """
        specs = []

        for job_idx in range(self.num_jobs):
            # Calculate Sobol offset for this job
            offset = job_idx * self.generation_config["samples_per_job"]

            # Build S3 output path
            s3_output_path = (
                f"{self.salad_config['storage']['dataset_path']}/"
                f"{self.generation_config['output_prefix']}{job_idx:02d}.h5"
            )

            # Build command for dataset generation
            # The entrypoint runs: exec poetry run spinlock "$@"
            command = [
                "generate",
                "--config",
                f"/app/{self.generation_config['dataset_config']}",
                "--output",
                f"/tmp/{self.generation_config['output_prefix']}{job_idx:02d}.h5",
                "--total-samples",
                str(self.generation_config["samples_per_job"]),
                "--sobol-offset",
                str(offset),
            ]

            # Build environment variables (no DDP vars!)
            env_vars = {
                "JOB_INDEX": str(job_idx),
                "AWS_ACCESS_KEY_ID": self.salad_config["storage"]["credentials"][
                    "aws_access_key_id"
                ],
                "AWS_SECRET_ACCESS_KEY": self.salad_config["storage"]["credentials"][
                    "aws_secret_access_key"
                ],
                "S3_ENDPOINT_URL": self.salad_config["storage"].get("endpoint_url", ""),
                "S3_BUCKET": self.salad_config["storage"]["bucket"],
                "S3_OUTPUT_PATH": s3_output_path,
                "SALAD_MACHINE_ID": "salad-container",
            }

            # Use ContainerSpecBuilder (already generic!)
            builder = ContainerSpecBuilder()
            builder.with_image(self.salad_config["container"]["image"], command)
            builder.with_environment(env_vars)

            # Set resources
            gpu_classes = []
            if "gpu_class" in self.salad_config["resources"]:
                gpu_classes = [self.salad_config["resources"]["gpu_class"]]

            builder.with_resources(
                cpu=self.salad_config["resources"]["cpu"],
                memory=self.salad_config["resources"]["memory"],
                gpu_classes=gpu_classes,
                storage=self.salad_config["resources"].get("storage", 21474836480),
            )

            # Add registry auth if configured
            if "registry_auth" in self.salad_config["container"]:
                auth = self.salad_config["container"]["registry_auth"]
                builder.with_registry_auth(
                    username=auth["username"], password=auth["password"]
                )

            # Each job is a single replica
            builder.with_replicas(1)

            # Build container group with unique name
            group_name = f"{self.job_id}-job-{job_idx}"
            spec = builder.build(group_name)
            specs.append(spec)

            print(
                f"  Job {job_idx}: offset={offset}, samples={self.generation_config['samples_per_job']}, "
                f"output={s3_output_path}"
            )

        return specs

    def monitor_job(self) -> None:
        """
        Monitor generation jobs and report progress.

        Unlike training (which streams logs), generation just polls status until complete.
        """
        print("[SaladGenerationLauncher] Monitoring generation jobs...")

        # Build container group list
        container_groups = [
            {"group_name": f"{self.job_id}-job-{job_idx}"}
            for job_idx in range(self.num_jobs)
        ]

        # Wait for all jobs to complete
        self._wait_for_jobs_complete(container_groups)

        print("[SaladGenerationLauncher] ✓ All generation jobs completed")
        print(
            f"[SaladGenerationLauncher] Results uploaded to s3://{self.salad_config['storage']['bucket']}/"
            f"{self.salad_config['storage']['dataset_path']}/"
        )

    def _wait_for_jobs_complete(
        self, container_groups: List[Dict], poll_interval: int = 30
    ) -> None:
        """
        Wait for all generation jobs to complete.

        Args:
            container_groups: List of container group info dicts
            poll_interval: Seconds between status checks
        """
        start_time = time.time()
        completed = set()

        print(
            f"  Waiting for {len(container_groups)} jobs to complete "
            f"(polling every {poll_interval}s)..."
        )

        while len(completed) < len(container_groups):
            for idx, group_info in enumerate(container_groups):
                if idx in completed:
                    continue

                try:
                    status = self.sdk.container_groups.get_container_group(
                        organization_name=self.organization,
                        project_name=self.project,
                        container_group_name=group_info["group_name"],
                    )

                    # Check if job completed successfully
                    if hasattr(status.current_state, "status"):
                        state = status.current_state.status

                        if state == "succeeded":
                            completed.add(idx)
                            elapsed = int(time.time() - start_time)
                            print(
                                f"  ✓ Job {idx} completed ({len(completed)}/{len(container_groups)}) "
                                f"[{elapsed}s elapsed]"
                            )
                        elif state == "failed":
                            print(f"  ✗ Job {idx} failed!")
                            raise RuntimeError(
                                f"Generation job {idx} failed. Check Salad dashboard for logs."
                            )
                except Exception as e:
                    if idx in completed:
                        continue
                    # Ignore transient errors during polling
                    print(f"  Warning: Error checking job {idx}: {e}")

            if len(completed) < len(container_groups):
                time.sleep(poll_interval)

        elapsed = int(time.time() - start_time)
        print(f"  ✓ All jobs completed in {elapsed}s ({elapsed/60:.1f}m)")


def launch_salad_generation(config: dict) -> None:
    """
    Main entry point for Salad distributed generation.

    Args:
        config: Full configuration dictionary with distributed.salad and generation sections
    """
    generation_config = config.get("generation")
    if not generation_config:
        raise ValueError("Config missing 'generation' section")

    launcher = SaladGenerationLauncher(config, generation_config)

    try:
        launcher.launch()
    except KeyboardInterrupt:
        print("\n[SaladGenerationLauncher] Interrupted by user")
        launcher.cleanup()
    except Exception as e:
        print(f"\n[SaladGenerationLauncher] Error: {e}")
        launcher.cleanup()
        raise
