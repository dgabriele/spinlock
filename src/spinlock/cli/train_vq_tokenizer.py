"""
Train VQ Tokenizer command for Spinlock CLI (V2 system).

Trains VQ tokenizer using the V2 Pydantic-based configuration system.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
import yaml

from .base import CLICommand


class TrainVQTokenizerCommand(CLICommand):
    """
    Command to train VQ tokenizer using V2 configuration system.

    Uses Pydantic-based TokenizerConfig with multi-family encoder architecture:
    - Initial encoder (CNN with optional pretrained weights)
    - Temporal encoder (PyramidTemporalEncoder for variable-length sequences)
    - Hierarchical vector quantization
    """

    @property
    def name(self) -> str:
        return "train-vq-tokenizer"

    @property
    def help(self) -> str:
        return "Train VQ tokenizer using V2 configuration system"

    @property
    def description(self) -> str:
        return """
Train VQ tokenizer for discrete representation learning.

V2 Architecture:
- Multi-family encoders (initial + temporal)
- Pretrained CNN support for initial encoder
- PyramidTemporalEncoder for variable-length temporal sequences
- Hierarchical vector quantization with EMA updates
- Flexible loss configuration (reconstruction, orthogonality, informativeness)

Configuration:
  Uses Pydantic-based TokenizerConfig (configs/v2/*.yaml)
  - encoder: Initial and temporal encoder settings
  - quantizer: VQ codebook configuration
  - hierarchy: Multi-level quantization
  - training: Optimizer, scheduler, early stopping
  - loss: Loss weights and normalization

Examples:
  # Train with V2 config
  spinlock train-vq-tokenizer --config configs/v2/vqvae_50k.yaml

  # Override dataset path
  spinlock train-vq-tokenizer \\
      --config configs/v2/vqvae_50k.yaml \\
      --dataset datasets/custom.h5

  # Override training parameters
  spinlock train-vq-tokenizer \\
      --config configs/v2/vqvae_50k.yaml \\
      --epochs 500 \\
      --batch-size 128

  # Verbose logging
  spinlock train-vq-tokenizer \\
      --config configs/v2/vqvae_50k.yaml \\
      --verbose

Output:
  Checkpoints saved to directory specified in config or --output:
    - tokenizer_epoch_XXX.pt: Periodic checkpoints
    - tokenizer_best.pt: Best model based on validation loss
    - training_history.json: Training metrics
    - config.yaml: Resolved configuration
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add train-vq-tokenizer command arguments."""
        # Required arguments
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to V2 tokenizer config YAML",
        )

        # Optional overrides
        parser.add_argument(
            "--dataset",
            type=Path,
            metavar="PATH",
            help="Override dataset path from config",
        )

        parser.add_argument(
            "--output",
            type=Path,
            metavar="PATH",
            help="Override output directory from config",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            metavar="N",
            help="Override number of training epochs",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            metavar="N",
            help="Override batch size",
        )

        parser.add_argument(
            "--learning-rate",
            type=float,
            metavar="LR",
            help="Override learning rate",
        )

        parser.add_argument(
            "--device",
            type=str,
            choices=["cuda", "cpu", "auto"],
            help="Override device (cuda/cpu/auto)",
        )

        # Execution options
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

        parser.add_argument(
            "--resume",
            type=Path,
            metavar="PATH",
            help="Resume training from checkpoint",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the train-vq-tokenizer command."""
        import logging
        import torch
        from spinlock.v2.data import SpinlockDataset
        from spinlock.v2.tokens.config import TokenizerConfig
        from spinlock.v2.tokens.tokenizer import VQTokenizer

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Validate config file
        if not self.validate_file_exists(args.config, "Configuration file"):
            return 1

        # Load config
        logger.info(f"Loading config from {args.config}")
        try:
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Extract dataset path
        dataset_path = args.dataset if args.dataset else config_dict.pop("dataset_path", None)
        if not dataset_path:
            return self.error("No dataset path specified (use --dataset or set in config)")

        dataset_path = Path(dataset_path)
        if not self.validate_file_exists(dataset_path, "Dataset"):
            return 1

        # Apply CLI overrides
        if args.epochs:
            config_dict.setdefault("training", {})["num_epochs"] = args.epochs
        if args.batch_size:
            config_dict.setdefault("training", {})["batch_size"] = args.batch_size
        if args.learning_rate:
            config_dict.setdefault("training", {})["learning_rate"] = args.learning_rate
        if args.device:
            config_dict.setdefault("training", {})["device"] = args.device
        if args.verbose:
            config_dict["verbose"] = True
            config_dict["log_level"] = "DEBUG"

        # Create Pydantic config
        logger.info("Parsing configuration")
        try:
            config = TokenizerConfig(**config_dict)
        except Exception as e:
            return self.error(f"Invalid config: {e}")

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            dataset = SpinlockDataset.from_file(dataset_path)
        except Exception as e:
            return self.error(f"Failed to load dataset: {e}")

        # Create tokenizer
        logger.info("Creating VQ tokenizer")
        try:
            tokenizer = VQTokenizer(config)
        except Exception as e:
            return self.error(f"Failed to create tokenizer: {e}")

        # Resume from checkpoint if specified
        if args.resume:
            if not self.validate_file_exists(args.resume, "Checkpoint"):
                return 1
            logger.info(f"Resuming from checkpoint: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location="cpu")
                tokenizer.load_state_dict(checkpoint["model_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                logger.info(f"Resumed from epoch {start_epoch}")
            except Exception as e:
                return self.error(f"Failed to load checkpoint: {e}")
        else:
            start_epoch = 0

        # Determine output directory
        if args.output:
            output_dir = args.output
        else:
            # Use checkpoint_dir from config if available
            output_dir = Path("checkpoints/v2/vqvae")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Train tokenizer
        logger.info("=" * 80)
        logger.info("Starting VQ Tokenizer Training")
        logger.info("=" * 80)
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Samples: {len(dataset)}")
        logger.info(f"Epochs: {config.training.num_epochs}")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.learning_rate}")
        logger.info(f"Device: {config.training.device}")
        logger.info("=" * 80)

        try:
            with dataset.open():
                history = tokenizer.train(
                    dataset=dataset,
                    output_dir=output_dir,
                    start_epoch=start_epoch,
                )
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            return 130
        except Exception as e:
            return self.error(f"Training failed: {e}")

        # Print results
        print()
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best epoch: {history.get('best_epoch', 'N/A')}")
        print(f"Best val loss: {history.get('best_val_loss', 'N/A'):.6f}")
        print(f"Final train loss: {history.get('final_train_loss', 'N/A'):.6f}")
        print(f"Final val loss: {history.get('final_val_loss', 'N/A'):.6f}")
        print(f"Checkpoints saved to: {output_dir}")
        print("=" * 80)

        return 0
