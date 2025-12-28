"""
Generate command for Spinlock CLI.

Handles dataset generation with configuration loading and pipeline execution.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any
import sys
import time

from .base import ConfigurableCommand


class GenerateCommand(ConfigurableCommand):
    """
    Command to generate Spinlock datasets.

    Loads configuration, applies overrides, and executes the generation pipeline.
    """

    @property
    def name(self) -> str:
        return "generate"

    @property
    def help(self) -> str:
        return "Generate a new dataset"

    @property
    def description(self) -> str:
        return """
Generate a Spinlock dataset from a configuration file.

Examples:
  # Basic generation
  spinlock generate --config configs/experiments/default_10k.yaml

  # With overrides
  spinlock generate --config configs/experiments/default_10k.yaml \\
      --output datasets/custom_10k.h5 \\
      --total-samples 5000 \\
      --device cuda:1

  # Dry run to validate configuration
  spinlock generate --config configs/experiments/default_10k.yaml --dry-run

  # Verbose output
  spinlock generate --config configs/experiments/test_100.yaml --verbose
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add generate command arguments."""
        # Required arguments
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to YAML configuration file",
        )

        # Optional overrides
        override_group = parser.add_argument_group("configuration overrides")

        override_group.add_argument(
            "--output", type=Path, metavar="PATH", help="Override output path from config"
        )

        override_group.add_argument(
            "--device",
            type=str,
            metavar="DEVICE",
            help="Override device from config (e.g., 'cuda', 'cuda:1', 'cpu')",
        )

        override_group.add_argument(
            "--total-samples",
            type=int,
            metavar="N",
            help="Override total number of samples from config",
        )

        override_group.add_argument(
            "--batch-size", type=int, metavar="N", help="Override batch size from config"
        )

        override_group.add_argument(
            "--num-realizations",
            type=int,
            metavar="N",
            help="Override number of stochastic realizations from config",
        )

        override_group.add_argument(
            "--seed", type=int, metavar="SEED", help="Override random seed from config"
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without generating dataset",
        )

        exec_group.add_argument(
            "--verbose", action="store_true", help="Print detailed progress information"
        )

    def execute(self, args: Namespace) -> int:
        """Execute dataset generation."""
        # Load configuration
        config = self.load_config(args.config, verbose=args.verbose)
        if config is None:
            return 1

        # Collect overrides
        overrides: Dict[str, Any] = {
            "dataset.output_path": args.output,
            "simulation.device": args.device,
            "sampling.total_samples": args.total_samples,
            "sampling.batch_size": args.batch_size,
            "simulation.num_realizations": args.num_realizations,
            "sampling.sobol.seed": args.seed,
        }

        # Apply overrides
        if args.verbose and any(v is not None for v in overrides.values()):
            print("\nApplying configuration overrides:")

        self.apply_overrides(config, overrides, verbose=args.verbose)

        # Print configuration summary
        if args.verbose:
            self._print_config_summary(config)

        # Dry run: validate and exit
        if args.dry_run:
            print("\n✓ Configuration valid (dry-run mode, no dataset generated)")
            return 0

        # Execute generation
        try:
            return self._run_generation(config, args.verbose)
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user", file=sys.stderr)
            return 130  # Standard SIGINT exit code
        except Exception as e:
            print(f"\nError during generation: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            return 1

    def _print_config_summary(self, config: Any) -> None:
        """Print configuration summary."""
        print("\nConfiguration:")
        print(f"  Parameter space dimensions: {config.parameter_space.total_dimensions}")
        print(f"  Total samples: {config.sampling.total_samples}")
        print(f"  Batch size: {config.sampling.batch_size}")
        print(f"  Num realizations: {config.simulation.num_realizations}")
        print(f"  Device: {config.simulation.device}")
        print(f"  Output: {config.dataset.output_path}")

    def _run_generation(self, config: Any, verbose: bool) -> int:
        """
        Run the actual dataset generation pipeline.

        Args:
            config: Validated SpinlockConfig
            verbose: Print detailed progress

        Returns:
            Exit code (0 for success)
        """
        from spinlock.dataset import DatasetGenerationPipeline

        print("\n" + "=" * 60)
        print("SPINLOCK DATASET GENERATION")
        print("=" * 60)

        start_time = time.time()

        # Create and execute pipeline
        pipeline = DatasetGenerationPipeline(config)

        if verbose:
            print("\nInitializing pipeline...")

        pipeline.generate()

        elapsed = time.time() - start_time

        # Success summary
        print("\n" + "=" * 60)
        print("✓ GENERATION COMPLETE")
        print("=" * 60)
        print(f"Dataset: {config.dataset.output_path}")
        print(f"Samples: {config.sampling.total_samples}")
        print(f"Realizations per sample: {config.simulation.num_realizations}")
        print(f"Total time: {elapsed:.2f}s ({elapsed / 60:.2f}m)")

        if config.sampling.total_samples > 0:
            rate = config.sampling.total_samples / elapsed
            print(f"Throughput: {rate:.2f} samples/s")

        print("=" * 60)

        return 0
