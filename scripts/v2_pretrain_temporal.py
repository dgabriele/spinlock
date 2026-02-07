#!/usr/bin/env python3
"""V2 Temporal Encoder pretraining script.

Pretrain the PyramidTemporalEncoder via autoencoder reconstruction task.
"""

import logging
import yaml
from pathlib import Path

import torch
import numpy as np

from spinlock.v2.data import SpinlockDataset
from spinlock.v2.tokens.config import TemporalPretrainingConfig
from spinlock.v2.tokens.pretraining import TemporalPretrainer


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
    """Run temporal encoder pretraining."""
    # Load config
    config_path = Path("configs/v2/temporal_pretraining.yaml")
    logger.info(f"Loading config from {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Extract dataset path and output path
    dataset_path = config_dict.pop("dataset_path")
    output_path = Path(config_dict.pop("output_path"))

    # Create Pydantic config
    config = TemporalPretrainingConfig(**config_dict)

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = SpinlockDataset.from_file(dataset_path)

    # Extract temporal features
    logger.info("Extracting temporal features")
    with dataset.open():
        if hasattr(dataset.features, 'temporal'):
            temporal = dataset.features.temporal.load_all()  # [N, T, D]
        else:
            raise ValueError("Dataset does not contain temporal features")

    logger.info(f"Temporal features shape: {temporal.shape}")

    # Handle different input formats
    if temporal.ndim != 3:
        raise ValueError(f"Expected temporal features with shape [N, T, D], got {temporal.shape}")

    N, T, D = temporal.shape
    logger.info(f"Loaded {N} samples with {T} timesteps and {D} features")

    # Convert to torch tensor
    temporal_tensor = torch.from_numpy(temporal).float()
    logger.info(f"Converted to tensor: {temporal_tensor.shape}")

    # Normalize data (per-feature standardization over all samples and timesteps)
    # Reshape to [N*T, D] for normalization
    temporal_flat = temporal_tensor.reshape(-1, D)
    mean = temporal_flat.mean(dim=0)  # [D]
    std = temporal_flat.std(dim=0)    # [D]

    # Avoid division by zero
    std = torch.where(std > 1e-8, std, torch.ones_like(std))

    # Normalize
    temporal_tensor = (temporal_tensor - mean.unsqueeze(0).unsqueeze(0)) / std.unsqueeze(0).unsqueeze(0)

    logger.info(f"Normalized data: mean shape={mean.shape}, std shape={std.shape}")
    logger.info(f"After normalization: mean={temporal_tensor.mean():.6f}, std={temporal_tensor.std():.6f}")

    # Create validity masks (assume all timesteps are valid by default)
    # In a real scenario, you might load these from the dataset
    masks = torch.ones(N, T, dtype=torch.bool)
    lengths = torch.full((N,), T, dtype=torch.long)

    logger.info("Created validity masks (all timesteps valid)")

    # Auto-detect input_dim from data
    input_dim = D
    logger.info(f"Auto-detected: input_dim={input_dim}, max_seq_length={T}")

    # Create pretrainer with runtime-detected parameters
    logger.info("Creating Temporal pretrainer")
    pretrainer = TemporalPretrainer(config, input_dim=input_dim, max_seq_length=T)

    # Train
    logger.info(f"Starting training for {config.num_epochs} epochs")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Print results
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Best epoch: {history['best_epoch'] + 1}")
    logger.info(f"Best val MSE: {history['best_val_mse']:.6f}")

    best_idx = history['best_epoch']
    logger.info(f"Best val RMSE: {history['val_metrics']['rmse'][best_idx]:.6f}")
    logger.info(f"Best val Rel-L2: {history['val_metrics']['rel_l2'][best_idx]:.6f}")

    logger.info(f"Final train MSE: {history['train_metrics']['mse'][-1]:.6f}")
    logger.info(f"Final train RMSE: {history['train_metrics']['rmse'][-1]:.6f}")
    logger.info(f"Final train Rel-L2: {history['train_metrics']['rel_l2'][-1]:.6f}")

    logger.info(f"Final val MSE: {history['val_metrics']['mse'][-1]:.6f}")
    logger.info(f"Final val RMSE: {history['val_metrics']['rmse'][-1]:.6f}")
    logger.info(f"Final val Rel-L2: {history['val_metrics']['rel_l2'][-1]:.6f}")

    logger.info(f"Checkpoint saved to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
