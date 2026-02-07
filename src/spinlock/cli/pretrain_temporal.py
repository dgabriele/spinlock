"""
Pretrain temporal encoder command for Spinlock CLI.

Pretrain PyramidTemporalEncoder via autoencoder reconstruction task.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

from .base import CLICommand


class PretrainTemporalCommand(CLICommand):
    """
    Command to pretrain PyramidTemporalEncoder.

    Trains an autoencoder on temporal features to create a pretrained
    encoder for use in VQ-VAE training.
    """

    @property
    def name(self) -> str:
        return "pretrain-temporal-encoder"

    @property
    def help(self) -> str:
        return "Pretrain PyramidTemporalEncoder via autoencoder"

    @property
    def description(self) -> str:
        return """
Pretrain the PyramidTemporalEncoder via autoencoder reconstruction.

This command trains an autoencoder (encoder + decoder) to reconstruct temporal
sequences, then saves only the encoder weights for use in VQ-VAE training.

The pretrained encoder provides better initialization than random weights,
improving convergence and representation quality.

Examples:
  # Pretrain with default config
  spinlock pretrain-temporal-encoder

  # Pretrain with custom config
  spinlock pretrain-temporal-encoder \\
      --config configs/v2/temporal_pretraining.yaml

  # Custom dataset and output path
  spinlock pretrain-temporal-encoder \\
      --dataset datasets/custom_dataset.h5 \\
      --output checkpoints/v2/custom_temporal.pt
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add pretrain-temporal-encoder command arguments."""
        parser.add_argument(
            "--config",
            type=Path,
            default=Path("configs/v2/temporal_pretraining.yaml"),
            metavar="PATH",
            help="Path to temporal pretraining config YAML (default: configs/v2/temporal_pretraining.yaml)",
        )

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
            help="Override output checkpoint path from config",
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the pretrain-temporal-encoder command."""
        import logging
        import yaml
        import torch
        from spinlock.v2.data import SpinlockDataset
        from spinlock.v2.tokens.config import TemporalPretrainingConfig
        from spinlock.v2.tokens.pretraining import TemporalPretrainer

        # Setup logging
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)

        # Validate config file exists
        if not self.validate_file_exists(args.config, "Configuration file"):
            return 1

        # Load config
        logger.info(f"Loading config from {args.config}")
        try:
            with open(args.config) as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            return self.error(f"Failed to load config: {e}")

        # Extract dataset path and output path
        dataset_path = args.dataset if args.dataset else config_dict.pop("dataset_path", None)
        output_path = args.output if args.output else config_dict.pop("output_path", None)

        if not dataset_path:
            return self.error("No dataset path specified (use --dataset or set in config)")

        if not output_path:
            return self.error("No output path specified (use --output or set in config)")

        dataset_path = Path(dataset_path)
        output_path = Path(output_path)

        # Validate dataset exists
        if not self.validate_file_exists(dataset_path, "Dataset"):
            return 1

        # Create Pydantic config
        try:
            config = TemporalPretrainingConfig(**config_dict)
        except Exception as e:
            return self.error(f"Invalid config: {e}")

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        try:
            dataset = SpinlockDataset.from_file(dataset_path)
        except Exception as e:
            return self.error(f"Failed to load dataset: {e}")

        # Extract temporal features
        logger.info("Extracting temporal features")
        try:
            with dataset.open():
                if hasattr(dataset.features, 'temporal'):
                    temporal = dataset.features.temporal.load_all()
                else:
                    return self.error("Dataset does not contain temporal features")
        except Exception as e:
            return self.error(f"Failed to extract temporal features: {e}")

        logger.info(f"Temporal features shape: {temporal.shape}")

        if temporal.ndim != 3:
            return self.error(f"Expected temporal features with shape [N, T, D], got {temporal.shape}")

        N, T, D = temporal.shape
        logger.info(f"Loaded {N} samples with {T} timesteps and {D} features")

        # Convert to torch tensor
        temporal_tensor = torch.from_numpy(temporal).float()

        # Normalize data
        temporal_flat = temporal_tensor.reshape(-1, D)
        mean = temporal_flat.mean(dim=0)
        std = temporal_flat.std(dim=0)
        std = torch.where(std > 1e-8, std, torch.ones_like(std))
        temporal_tensor = (temporal_tensor - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)

        logger.info(f"Normalized data: mean shape={mean.shape}, std shape={std.shape}")

        # Create validity masks
        masks = torch.ones(N, T, dtype=torch.bool)
        lengths = torch.full((N,), T, dtype=torch.long)

        # Create pretrainer
        logger.info("Creating temporal pretrainer")
        try:
            pretrainer = TemporalPretrainer(config, input_dim=D, max_seq_length=T)
        except Exception as e:
            return self.error(f"Failed to create pretrainer: {e}")

        # Train
        logger.info(f"Starting training for {config.num_epochs} epochs")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            history = pretrainer.train(
                temporal_features=temporal_tensor,
                output_path=output_path,
                masks=masks,
                lengths=lengths,
                normalization_stats={
                    'temporal_mean': mean.cpu().numpy(),
                    'temporal_std': std.cpu().numpy(),
                },
            )
        except Exception as e:
            return self.error(f"Training failed: {e}")

        # Print results
        print("=" * 80)
        print("Training complete!")
        print(f"Best epoch: {history['best_epoch'] + 1}")
        print(f"Best val MSE: {history['best_val_mse']:.6f}")

        best_idx = history['best_epoch']
        print(f"Best val RMSE: {history['val_metrics']['rmse'][best_idx]:.6f}")
        print(f"Best val Rel-L2: {history['val_metrics']['rel_l2'][best_idx]:.6f}")

        print(f"Final train MSE: {history['train_metrics']['mse'][-1]:.6f}")
        print(f"Final train RMSE: {history['train_metrics']['rmse'][-1]:.6f}")
        print(f"Final train Rel-L2: {history['train_metrics']['rel_l2'][-1]:.6f}")

        print(f"Final val MSE: {history['val_metrics']['mse'][-1]:.6f}")
        print(f"Final val RMSE: {history['val_metrics']['rmse'][-1]:.6f}")
        print(f"Final val Rel-L2: {history['val_metrics']['rel_l2'][-1]:.6f}")

        print(f"Checkpoint saved to: {output_path}")
        print("=" * 80)

        return 0
