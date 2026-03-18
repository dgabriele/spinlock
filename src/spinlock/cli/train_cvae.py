"""CLI command for training Token-Conditioned CVAE.

This command trains a CVAE that models P(theta, IC | temporal_tokens):
given temporal tokens describing dynamics, generate plausible physical
parameters and initial conditions.

Supports both config file and CLI-based configuration with override capability.
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import h5py

from spinlock.cli.base import CLICommand
from spinlock.tokens.cvae_config import TokenConditionedCVAEConfig, CVAEDataConfig
from spinlock.tokens.cvae_trainer import CVAETrainer


class TrainCVAECommand(CLICommand):
    """Train CVAE: temporal tokens -> (theta, initial_grids)."""

    @property
    def name(self) -> str:
        return "train-cvae"

    @property
    def help(self) -> str:
        return "Train CVAE to generate (theta, IC) from temporal tokens"

    @property
    def description(self) -> str:
        return (
            "Train Token-Conditioned CVAE: P(theta, IC | temporal_tokens). "
            "The CVAE learns to generate plausible physical parameters and "
            "initial conditions given observed temporal dynamics."
        )

    def add_arguments(self, parser: ArgumentParser) -> None:
        # Primary config
        parser.add_argument(
            "--config",
            type=Path,
            required=False,
            help="Path to YAML config file",
        )

        # Data paths (required if no config)
        parser.add_argument(
            "--vq-checkpoint",
            type=Path,
            help="Path to frozen VQTokenizer checkpoint",
        )
        parser.add_argument(
            "--dataset",
            type=Path,
            help="Path to dataset HDF5 (e.g., 50k_baseline.h5)",
        )
        parser.add_argument(
            "--tokenized-dataset",
            type=Path,
            help="Path to pre-tokenized dataset",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            help="Output directory for checkpoints and logs",
        )

        # Data overrides
        parser.add_argument(
            "--truncation-length",
            type=int,
            help="Filter to this truncation length from multi-trunc pretokenized HDF5",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            help="Limit total samples before splitting (for large datasets)",
        )

        # Training overrides
        parser.add_argument(
            "--num-epochs",
            type=int,
            help="Number of training epochs",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Training batch size",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Learning rate",
        )
        parser.add_argument(
            "--latent-dim",
            type=int,
            help="Latent space dimensionality",
        )
        parser.add_argument(
            "--device",
            choices=["cuda", "cpu"],
            help="Device for training",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed",
        )

    def _resolve_dimensions(
        self, data_config: CVAEDataConfig
    ) -> Tuple[int, Tuple[int, int, int]]:
        """Resolve theta_dim and grid_shape from datasets at runtime.

        Args:
            data_config: Data configuration with dataset paths

        Returns:
            Tuple of (theta_dim, grid_shape)
        """
        with h5py.File(data_config.dataset, "r") as f:
            theta_dim = f["/parameters/params"].shape[1]

            fields_shape = f["/inputs/fields"].shape
            grid_shape = (fields_shape[2], fields_shape[3], fields_shape[4])

        return theta_dim, grid_shape

    def execute(self, args: Namespace) -> int:
        """Train TokenConditionedCVAE with config + CLI overrides.

        Args:
            args: Parsed command-line arguments

        Returns:
            Exit code (0 for success)
        """
        print("=" * 80)
        print("Token-Conditioned CVAE Training")
        print("  P(theta, IC | temporal_tokens)")
        print("=" * 80)

        # Load config from YAML if provided
        if args.config:
            print(f"\nLoading config from: {args.config}")
            try:
                config = TokenConditionedCVAEConfig.from_yaml(args.config)
            except Exception as e:
                print(f"Error loading config: {e}", file=sys.stderr)
                return 1
        else:
            if not all([
                args.vq_checkpoint,
                args.dataset,
                args.tokenized_dataset,
                args.output_dir,
            ]):
                print(
                    "Error: Must provide --config or all required CLI args:\n"
                    "  --vq-checkpoint, --dataset, --tokenized-dataset, --output-dir",
                    file=sys.stderr,
                )
                return 1

            config = TokenConditionedCVAEConfig(
                data=CVAEDataConfig(
                    vq_checkpoint=args.vq_checkpoint,
                    dataset=args.dataset,
                    tokenized_dataset=args.tokenized_dataset,
                ),
                output_dir=args.output_dir,
            )

        # Apply CLI overrides — data
        if hasattr(args, 'truncation_length') and args.truncation_length is not None:
            config.data.truncation_length = args.truncation_length
        if hasattr(args, 'max_samples') and args.max_samples is not None:
            config.data.max_samples = args.max_samples

        # Apply CLI overrides — training
        if args.num_epochs is not None:
            config.training.num_epochs = args.num_epochs
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.training.learning_rate = args.learning_rate
        if args.latent_dim is not None:
            config.model.latent_dim = args.latent_dim
        if args.device is not None:
            config.device = args.device
        if args.seed is not None:
            config.seed = args.seed

        # Validate paths exist
        if not config.data.vq_checkpoint.exists():
            print(
                f"Error: VQ checkpoint not found: {config.data.vq_checkpoint}",
                file=sys.stderr,
            )
            return 1
        if not config.data.dataset.exists():
            print(
                f"Error: Dataset not found: {config.data.dataset}",
                file=sys.stderr,
            )
            return 1
        if not config.data.tokenized_dataset.exists():
            print(
                f"Error: Tokenized dataset not found: {config.data.tokenized_dataset}",
                file=sys.stderr,
            )
            return 1

        # Resolve runtime dimensions
        print("\n" + "=" * 80)
        print("Resolving runtime dimensions from datasets...")
        print("=" * 80)
        try:
            theta_dim, grid_shape = self._resolve_dimensions(config.data)
            print(f"  theta_dim: {theta_dim}")
            print(f"  grid_shape: {grid_shape}")
        except Exception as e:
            print(f"Error resolving dimensions: {e}", file=sys.stderr)
            return 1

        # Create trainer
        print("\n" + "=" * 80)
        print("Initializing trainer...")
        print("=" * 80)
        try:
            trainer = CVAETrainer(config, theta_dim, grid_shape)
        except Exception as e:
            print(f"Error initializing trainer: {e}", file=sys.stderr)
            return 1

        # Train model
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print(f"Saving checkpoint to: {config.output_dir / 'interrupted.pt'}")
            trainer.save_checkpoint("interrupted.pt")
            return 1
        except Exception as e:
            print(f"\nError during training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        return 0
