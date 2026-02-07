#!/usr/bin/env python3
"""V2 CNN pretraining script.

Pretrain the InitialCNNEncoder via autoencoder reconstruction task.
"""

import logging
import yaml
from pathlib import Path

import torch

from spinlock.v2.data import SpinlockDataset
from spinlock.v2.tokens.config import PretrainingConfig
from spinlock.v2.tokens.pretraining import CNNPretrainer


# Custom formatter for concise logging
class ConciseFormatter(logging.Formatter):
    """Concise log formatter with shortened paths and level codes."""

    LEVEL_CODES = {
        'DEBUG': 'DD',
        'INFO': 'II',
        'WARNING': 'WW',
        'ERROR': 'EE',
        'CRITICAL': 'CC',
    }

    def format(self, record):
        # Shorten module name to last 2 components
        name_parts = record.name.split('.')
        if len(name_parts) > 2:
            record.name = f"...{'.'.join(name_parts[-2:])}"

        # Use 2-letter level code
        level_code = self.LEVEL_CODES.get(record.levelname, record.levelname[:2])

        # Format: MM/DD/YY HH:MM:SS - ..module.name - LL - message
        return f"{self.formatTime(record, '%m/%d/%y %H:%M:%S')} - {record.name} - {level_code} - {record.getMessage()}"


# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(ConciseFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def main():
    """Run CNN pretraining."""
    # Load config
    config_path = Path("configs/v2/cnn_pretraining.yaml")
    logger.info(f"Loading config from {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract dataset path and output path
    dataset_path = config_dict.pop("dataset_path")
    output_path = Path(config_dict.pop("output_path"))

    # Create Pydantic config
    config = PretrainingConfig(**config_dict)

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = SpinlockDataset.from_file(dataset_path)

    # Extract initial conditions
    logger.info("Extracting initial conditions")
    with dataset.open():
        if hasattr(dataset, 'initial_conditions'):
            ics = dataset.initial_conditions.load_all()  # [N, H, W]
        elif hasattr(dataset, 'inputs'):
            # Fallback to inputs if initial_conditions not available
            ics = dataset.inputs.load_all()
            logger.info("Using inputs as initial conditions")
        else:
            raise ValueError("Dataset does not contain initial_conditions or inputs")

    logger.info(f"Initial conditions shape: {ics.shape}")

    # Handle different input formats
    if ics.ndim == 5:
        # Shape: [N, C, R, H, W] where R is number of realizations
        # Flatten realizations into batch dimension to train on all realizations
        N, C, R, H, W = ics.shape
        logger.info(f"Detected multi-realization format with {R} realizations per sample")
        ics = ics.transpose(0, 2, 1, 3, 4).reshape(N * R, C, H, W)  # [N*R, C, H, W]
        logger.info(f"Flattened realizations into batch: {ics.shape} ({N * R} total samples)")
    elif ics.ndim == 3:
        # Shape: [N, H, W] - add channel dimension
        ics = ics[:, None, :, :]  # [N, 1, H, W]
        logger.info(f"Added channel dimension: {ics.shape}")
    elif ics.ndim == 4:
        # Shape: [N, C, H, W] - already correct
        logger.info(f"Input already in correct format: {ics.shape}")
    else:
        raise ValueError(f"Unexpected input shape: {ics.shape}")

    # Convert to torch tensor
    ics_tensor = torch.from_numpy(ics).float()
    logger.info(f"Converted to tensor: {ics_tensor.shape}")

    # Auto-detect in_channels and spatial_size from data
    _, in_channels, spatial_h, spatial_w = ics_tensor.shape
    if spatial_h != spatial_w:
        raise ValueError(f"Expected square spatial dimensions, got {spatial_h}x{spatial_w}")

    logger.info(f"Auto-detected: in_channels={in_channels}, spatial_size={spatial_h}")

    # Create pretrainer with runtime-detected parameters
    logger.info("Creating CNN pretrainer")
    pretrainer = CNNPretrainer(config, in_channels=in_channels, spatial_size=spatial_h)

    # Train
    logger.info(f"Starting training for {config.num_epochs} epochs")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history = pretrainer.train(
        initial_conditions=ics_tensor,
        output_path=output_path,
    )

    # Print results
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best epoch: {history['best_epoch'] + 1}")
    logger.info(f"Best val loss: {history['best_val_loss']:.6f}")
    logger.info(f"Final train loss: {history['train_losses'][-1]:.6f}")
    logger.info(f"Final val loss: {history['val_losses'][-1]:.6f}")
    logger.info(f"Checkpoint saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
