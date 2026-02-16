"""
Launch Salad Cloud dataset generation command.

Handles distributed dataset generation on Salad Cloud with job splitting.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

from .base import CLICommand
import yaml


class LaunchSaladGenerationCommand(CLICommand):
    """Command to launch distributed dataset generation on Salad Cloud."""

    @property
    def name(self) -> str:
        return "launch-salad-generation"

    @property
    def help(self) -> str:
        return "Launch distributed dataset generation on Salad Cloud"

    @property
    def description(self) -> str:
        return """
Launch distributed dataset generation on Salad Cloud.

This command:
1. Splits dataset generation into multiple independent jobs
2. Each job generates a subset with unique Sobol offset
3. Jobs run in parallel on Salad Cloud containers
4. Results are uploaded to S3/MinIO storage

Examples:
  # Launch 1M sample generation (10× 100K jobs)
  spinlock launch-salad-generation --config configs/distributed/salad_qbm_generation.yaml

  # With verbose output
  spinlock launch-salad-generation --config configs/distributed/salad_qbm_generation.yaml --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add launch-salad-generation command arguments."""
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to Salad generation config YAML",
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Print detailed progress information"
        )

    def execute(self, args: Namespace) -> int:
        """Execute Salad generation launch."""
        # Load configuration (YAML dict, not SpinlockConfig)
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            if args.verbose:
                print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return 1

        # Validate required sections
        if "distributed" not in config:
            print("Error: Config missing 'distributed' section", file=sys.stderr)
            return 1

        if "salad" not in config["distributed"]:
            print("Error: Config missing 'distributed.salad' section", file=sys.stderr)
            return 1

        if "generation" not in config:
            print("Error: Config missing 'generation' section", file=sys.stderr)
            return 1

        # Print configuration summary
        if args.verbose:
            self._print_config_summary(config)

        # Launch distributed generation
        try:
            return self._launch_generation(config, args.verbose)
        except KeyboardInterrupt:
            print("\n\nGeneration launch interrupted by user", file=sys.stderr)
            return 130  # Standard SIGINT exit code
        except Exception as e:
            print(f"\nError during launch: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    def _print_config_summary(self, config: dict) -> None:
        """Print configuration summary."""
        gen_config = config["generation"]
        salad_config = config["distributed"]["salad"]

        num_jobs = gen_config["total_samples"] // gen_config["samples_per_job"]

        print("\nConfiguration:")
        print(f"  Total samples: {gen_config['total_samples']:,}")
        print(f"  Samples per job: {gen_config['samples_per_job']:,}")
        print(f"  Number of jobs: {num_jobs}")
        print(f"  Dataset config: {gen_config['dataset_config']}")
        print(f"  Output prefix: {gen_config['output_prefix']}")
        print(
            f"  S3 bucket: s3://{salad_config['storage']['bucket']}/{salad_config['storage']['dataset_path']}/"
        )
        print(f"  GPU class: {salad_config['resources'].get('gpu_class', 'CPU')}")
        print(f"  Memory: {salad_config['resources']['memory']} MB")

    def _launch_generation(self, config: dict, verbose: bool) -> int:
        """
        Launch the Salad generation job.

        Args:
            config: Validated configuration dict
            verbose: Print detailed progress

        Returns:
            Exit code (0 for success)
        """
        from spinlock.distributed.salad.generation_launcher import launch_salad_generation

        print("\n" + "=" * 60)
        print("SALAD CLOUD DATASET GENERATION")
        print("=" * 60)

        # Launch distributed generation
        launch_salad_generation(config)

        # Success summary
        gen_config = config["generation"]
        salad_config = config["distributed"]["salad"]

        print("\n" + "=" * 60)
        print("✓ GENERATION JOBS LAUNCHED")
        print("=" * 60)
        print(f"Total samples: {gen_config['total_samples']:,}")
        print(
            f"Output location: s3://{salad_config['storage']['bucket']}/"
            f"{salad_config['storage']['dataset_path']}/"
        )
        print(
            "\nNext steps:"
            "\n  1. Monitor progress via Salad dashboard"
            "\n  2. Download results: aws s3 sync s3://spinlock-datasets/qbm_1m_wide/ datasets/parts/"
            "\n  3. Merge datasets: spinlock merge-datasets --input 'datasets/parts/*.h5' --output datasets/qbm_1m_wide.h5"
        )
        print("=" * 60)

        return 0
