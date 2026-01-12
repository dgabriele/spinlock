"""
Generate command for NOA feature datasets.

This command generates a feature dataset from a trained NOA model by:
1. Loading a trained NOA checkpoint
2. Sampling the parameter space (same as CNO generation)
3. Generating rollouts using NOA (instead of CNO)
4. Extracting features inline during generation
5. Storing features to HDF5 (feature-only mode, no trajectories)

This is used for the "train tokenizer on simulator's distribution" architecture
where VQ-VAE is trained on NOA's outputs rather than CNO's outputs.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Any
import sys
import time
import torch

from .base import ConfigurableCommand


class GenerateNOAFeaturesCommand(ConfigurableCommand):
    """
    Command to generate feature datasets from trained NOA.

    Loads a trained NOA model and generates a large-scale feature dataset
    by sampling the parameter space and generating rollouts. Features are
    extracted inline (GPU-optimized) and written directly to HDF5.

    This enables training VQ-VAE on NOA's distribution rather than CNO's
    distribution, achieving better alignment by construction.
    """

    @property
    def name(self) -> str:
        return "generate-noa-dataset"

    @property
    def help(self) -> str:
        return "Generate dataset from trained NOA model (same structure as CNO datasets)"

    @property
    def description(self) -> str:
        return """
Generate a feature dataset from a trained NOA model.

This command is used in the "train tokenizer on simulator's distribution"
architecture where VQ-VAE is trained on NOA's outputs for optimal alignment.

Examples:
  # Generate 10K NOA features for VQ-VAE validation
  spinlock generate-noa-features \\
      --noa-checkpoint checkpoints/noa/pure_mse_baseline/best_model.pt \\
      --output datasets/noa_features_10k.h5 \\
      --n-samples 10000 \\
      --config configs/experiments/local_100k_optimized.yaml

  # Generate 100K features for production VQ-VAE training
  spinlock generate-noa-features \\
      --noa-checkpoint checkpoints/noa/pure_mse_baseline/best_model.pt \\
      --output datasets/noa_features_100k.h5 \\
      --n-samples 100000 \\
      --config configs/experiments/local_100k_optimized.yaml \\
      --device cuda \\
      --batch-size 16

  # Dry run to validate configuration
  spinlock generate-noa-features \\
      --noa-checkpoint checkpoints/noa/pure_mse_baseline/best_model.pt \\
      --output datasets/test.h5 \\
      --n-samples 100 \\
      --config configs/experiments/local_100k_optimized.yaml \\
      --dry-run
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add generate-noa-features command arguments."""
        # Required arguments
        parser.add_argument(
            "--noa-checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to trained NOA checkpoint (.pt file)",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Output HDF5 file path for features",
        )

        parser.add_argument(
            "--n-samples",
            type=int,
            required=True,
            metavar="N",
            help="Number of NOA rollouts to generate (e.g., 10000, 100000)",
        )

        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to base config YAML (for parameter space and simulation settings)",
        )

        # Optional overrides
        override_group = parser.add_argument_group("configuration overrides")

        override_group.add_argument(
            "--device",
            type=str,
            metavar="DEVICE",
            default="cuda",
            help="Device for generation (default: cuda)",
        )

        override_group.add_argument(
            "--batch-size",
            type=int,
            metavar="N",
            default=16,
            help="Batch size for NOA rollout generation (default: 16)",
        )

        override_group.add_argument(
            "--num-realizations",
            type=int,
            metavar="N",
            default=1,
            help="Number of stochastic realizations per rollout (default: 1)",
        )

        override_group.add_argument(
            "--seed",
            type=int,
            metavar="SEED",
            default=42,
            help="Random seed for parameter sampling (default: 42)",
        )

        # Execution options
        exec_group = parser.add_argument_group("execution options")

        exec_group.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration and NOA loading without generating dataset",
        )

        exec_group.add_argument(
            "--verbose",
            action="store_true",
            help="Print detailed progress information",
        )

    def execute(self, args: Namespace) -> int:
        """Execute NOA feature generation."""
        # Validate NOA checkpoint exists
        if not args.noa_checkpoint.exists():
            print(f"Error: NOA checkpoint not found: {args.noa_checkpoint}", file=sys.stderr)
            return 1

        # Load base configuration
        config = self.load_config(args.config, verbose=args.verbose)
        if config is None:
            return 1

        # Validate parameter space configuration
        if not hasattr(config, 'parameter_space'):
            print("Error: Config missing 'parameter_space' section", file=sys.stderr)
            print("  NOA feature generation requires operator parameter sampling", file=sys.stderr)
            return 1

        if config.parameter_space.total_dimensions == 0:
            print("Error: Parameter space has 0 dimensions", file=sys.stderr)
            print("  NOA feature generation requires at least 1 parameter dimension", file=sys.stderr)
            return 1

        # Override configuration for NOA generation
        config.sampling.total_samples = args.n_samples
        config.sampling.batch_size = args.batch_size
        config.simulation.device = args.device
        config.simulation.num_realizations = args.num_realizations
        config.sampling.sobol.seed = args.seed
        config.dataset.output_path = str(args.output)

        # Force feature-only mode (don't store trajectories)
        config.dataset.storage.store_trajectories = False

        # Print configuration summary
        if args.verbose or args.dry_run:
            self._print_config_summary(args, config)

        # Dry run: validate and load NOA
        if args.dry_run:
            print("\nValidating NOA checkpoint...")
            try:
                checkpoint = torch.load(args.noa_checkpoint, map_location='cpu', weights_only=False)
                model_config = checkpoint.get('config', {}).get('model', {})
                print(f"  ✓ NOA checkpoint valid")
                print(f"  ✓ Base channels: {model_config.get('base_channels', 'unknown')}")
                print(f"  ✓ Encoder levels: {model_config.get('encoder_levels', 'unknown')}")
                print(f"  ✓ Token conditioning: {model_config.get('token_conditioning', False)}")
                print("\n✓ Configuration valid (dry-run mode, no dataset generated)")
                return 0
            except Exception as e:
                print(f"Error loading NOA checkpoint: {e}", file=sys.stderr)
                return 1

        # Execute generation
        try:
            return self._run_noa_generation(args.noa_checkpoint, config, args.verbose)
        except KeyboardInterrupt:
            print("\n\nGeneration interrupted by user", file=sys.stderr)
            return 130  # Standard SIGINT exit code
        except Exception as e:
            print(f"\nError during generation: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _print_config_summary(self, args: Namespace, config: Any) -> None:
        """Print configuration summary."""
        print("\nNOA Feature Generation Configuration:")
        print(f"  NOA checkpoint: {args.noa_checkpoint}")
        print(f"  Output: {args.output}")
        print(f"  Samples: {args.n_samples}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Realizations: {args.num_realizations}")
        print(f"  Device: {args.device}")
        print(f"  Seed: {args.seed}")
        print(f"  Parameter space dimensions: {config.parameter_space.total_dimensions}")

    def _run_noa_generation(self, noa_checkpoint: Path, config: Any, verbose: bool) -> int:
        """
        Run the actual NOA feature generation pipeline.

        Args:
            noa_checkpoint: Path to trained NOA checkpoint
            config: Validated SpinlockConfig
            verbose: Print detailed progress

        Returns:
            Exit code (0 for success)
        """
        from spinlock.noa.generation_pipeline import NOAFeatureGenerationPipeline

        print("\n" + "=" * 70)
        print("NOA FEATURE GENERATION")
        print("=" * 70)
        print(f"NOA checkpoint: {noa_checkpoint}")
        print(f"Output: {config.dataset.output_path}")
        print(f"Samples: {config.sampling.total_samples}")
        print(f"Device: {config.simulation.device}")
        print("=" * 70 + "\n")

        start_time = time.time()

        # Create and execute pipeline
        pipeline = NOAFeatureGenerationPipeline(
            noa_checkpoint=noa_checkpoint,
            config=config,
            verbose=verbose,
        )

        if verbose:
            print("\nInitializing pipeline...")

        pipeline.generate()

        elapsed = time.time() - start_time

        # Success summary
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        print(f"Output: {config.dataset.output_path}")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Samples: {config.sampling.total_samples}")
        print(f"Throughput: {config.sampling.total_samples / elapsed:.1f} samples/s")
        print("=" * 70 + "\n")

        return 0
