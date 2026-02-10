"""CLI command for training Token-to-Rollout VAE.

This command trains a standalone VAE that learns the inverse mapping:
    tokens → (theta, initial_grids)

Supports both config file and CLI-based configuration with override capability.
"""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import h5py

from spinlock.cli.base import CLICommand
from spinlock.tokens.rollout_vae_config import TokenToRolloutVAEConfig, DataConfig
from spinlock.tokens.rollout_vae_trainer import TokenToRolloutVAETrainer


class TrainTokenToRolloutVAECommand(CLICommand):
    """Train standalone VAE: tokens → (theta, initial_grids)."""

    @property
    def name(self) -> str:
        return "train-token-to-rollout-vae"

    @property
    def help(self) -> str:
        return "Train VAE to decode tokens to generative parameters"

    @property
    def description(self) -> str:
        return "Train VAE to decode tokens to generative parameters (theta, ICs)"

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
            "--cno-dataset",
            type=Path,
            help="Path to CNO dataset (e.g., 50k_baseline.h5)",
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
            "--beta-schedule",
            choices=["linear", "constant"],
            help="KL annealing schedule",
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
        self, data_config: DataConfig
    ) -> Tuple[int, Tuple[int, int, int]]:
        """Resolve theta_dim and grid_shape from datasets at runtime.

        Args:
            data_config: Data configuration with dataset paths

        Returns:
            Tuple of (theta_dim, grid_shape)
        """
        with h5py.File(data_config.cno_dataset, "r") as f:
            # Theta dimensionality from /parameters/params [N, theta_dim]
            theta_dim = f["/parameters/params"].shape[1]

            # Grid shape from /inputs/fields [N, M, C, H, W]
            fields_shape = f["/inputs/fields"].shape
            grid_shape = (fields_shape[2], fields_shape[3], fields_shape[4])

        return theta_dim, grid_shape

    def execute(self, args: Namespace) -> int:
        """Train TokenToRolloutVAE with config + CLI overrides.

        Args:
            args: Parsed command-line arguments

        Returns:
            Exit code (0 for success)
        """
        print("=" * 80)
        print("Token-to-Rollout VAE Training")
        print("=" * 80)

        # Load config from YAML if provided
        if args.config:
            print(f"\nLoading config from: {args.config}")
            try:
                config = TokenToRolloutVAEConfig.from_yaml(args.config)
            except Exception as e:
                print(f"Error loading config: {e}", file=sys.stderr)
                return 1
        else:
            # Must provide required fields via CLI
            if not all(
                [
                    args.vq_checkpoint,
                    args.cno_dataset,
                    args.tokenized_dataset,
                    args.output_dir,
                ]
            ):
                print(
                    "Error: Must provide --config or all required CLI args:\n"
                    "  --vq-checkpoint, --cno-dataset, --tokenized-dataset, --output-dir",
                    file=sys.stderr,
                )
                return 1

            # Create config from CLI args
            config = TokenToRolloutVAEConfig(
                data=DataConfig(
                    vq_checkpoint=args.vq_checkpoint,
                    cno_dataset=args.cno_dataset,
                    tokenized_dataset=args.tokenized_dataset,
                ),
                output_dir=args.output_dir,
            )

        # Apply CLI overrides (take precedence over config file)
        if args.num_epochs is not None:
            config.training.num_epochs = args.num_epochs
        if args.batch_size is not None:
            config.training.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.training.learning_rate = args.learning_rate
        if args.latent_dim is not None:
            config.model.latent_dim = args.latent_dim
        if args.beta_schedule is not None:
            config.training.beta_schedule = args.beta_schedule
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
        if not config.data.cno_dataset.exists():
            print(
                f"Error: CNO dataset not found: {config.data.cno_dataset}",
                file=sys.stderr,
            )
            return 1
        if not config.data.tokenized_dataset.exists():
            print(
                f"Error: Tokenized dataset not found: {config.data.tokenized_dataset}",
                file=sys.stderr,
            )
            return 1

        # Resolve runtime dimensions from datasets
        print("\n" + "=" * 80)
        print("Resolving runtime dimensions from datasets...")
        print("=" * 80)
        try:
            theta_dim, grid_shape = self._resolve_dimensions(config.data)
            print(f"  ✓ theta_dim: {theta_dim}")
            print(f"  ✓ grid_shape: {grid_shape}")
        except Exception as e:
            print(f"Error resolving dimensions: {e}", file=sys.stderr)
            return 1

        # Create trainer
        print("\n" + "=" * 80)
        print("Initializing trainer...")
        print("=" * 80)
        try:
            trainer = TokenToRolloutVAETrainer(config, theta_dim, grid_shape)
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
