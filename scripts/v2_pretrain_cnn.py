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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        else:
            raise ValueError("Dataset does not contain initial_conditions")

    logger.info(f"Initial conditions shape: {ics.shape}")

    # Add channel dimension if needed
    if ics.ndim == 3:
        ics = ics[:, None, :, :]  # [N, 1, H, W]

    # Convert to torch tensor
    ics_tensor = torch.from_numpy(ics).float()
    logger.info(f"Converted to tensor: {ics_tensor.shape}")

    # Create pretrainer
    logger.info("Creating CNN pretrainer")
    pretrainer = CNNPretrainer(config)

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
