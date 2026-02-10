"""
Train InitialInverseCNN: encoded initial features → spatial ICs.

This script trains a supervised inverse decoder that maps from the VQTokenizer's
encoded initial space [B, 426] back to spatial initial condition grids [B, 3, 64, 64].

Usage:
    poetry run python scripts/train_initial_inverse.py \
        --tokenizer checkpoints/vq_tokenizer_best.pt \
        --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
        --raw-dataset datasets/50k_baseline.h5 \
        --output checkpoints/initial_inverse.pt \
        --epochs 100 \
        --batch-size 128
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.inverse_models import InitialInverseCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InitialInverseDataset(Dataset):
    """Dataset for training initial inverse decoder."""

    def __init__(
        self,
        initial_encoded: torch.Tensor,
        ics_true: torch.Tensor,
    ):
        """
        Args:
            initial_encoded: [N, encoded_dim] encoded initial features from tokenizer
            ics_true: [N, channels, H, W] ground truth initial conditions
        """
        assert initial_encoded.shape[0] == ics_true.shape[0], \
            f"Size mismatch: {initial_encoded.shape[0]} vs {ics_true.shape[0]}"

        self.initial_encoded = initial_encoded
        self.ics_true = ics_true

    def __len__(self) -> int:
        return self.initial_encoded.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.initial_encoded[idx], self.ics_true[idx]


def load_data(
    tokenizer_path: Path,
    tokenized_dataset_path: Path,
    raw_dataset_path: Path,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load initial_encoded and ics_true from datasets.

    Args:
        tokenizer_path: Path to trained VQTokenizer checkpoint
        tokenized_dataset_path: Path to pretokenized dataset with tokens
        raw_dataset_path: Path to raw dataset with ground truth ICs
        device: Device to load data on
        batch_size: Batch size for processing (to avoid memory issues)

    Returns:
        Tuple of (initial_encoded [N, 426], ics_true [N, 3, 64, 64])
    """
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = VQTokenizer.from_checkpoint(tokenizer_path)
    tokenizer.model.to(device)
    tokenizer.model.eval()

    # Load ALL tokens from pretokenized dataset (decoder needs all families)
    logger.info(f"Loading tokens from {tokenized_dataset_path}")
    tokens = {}
    with h5py.File(tokenized_dataset_path, 'r') as f:
        token_keys = [k for k in f['tokens'].keys()]
        for key in token_keys:
            tokens[key] = torch.from_numpy(f[f'tokens/{key}'][:]).long()

    initial_token_keys = [k for k in tokens.keys() if k.startswith('initial_')]
    logger.info(f"Loaded {len(tokens)} total token categories ({len(initial_token_keys)} initial categories)")

    # Load ground truth ICs
    logger.info(f"Loading ground truth ICs from {raw_dataset_path}")
    with h5py.File(raw_dataset_path, 'r') as f:
        # Load /inputs/fields [N, M, C, H, W] and aggregate over realizations
        ics = f['inputs/fields'][:]
        if ics.ndim == 5:  # [N, M, C, H, W]
            ics = ics.mean(axis=1)  # [N, C, H, W]
        elif ics.ndim == 4:  # [N, C, H, W]
            pass
        elif ics.ndim == 3:  # [N, H, W]
            ics = ics[:, None, :, :]  # [N, 1, H, W]

        ics_true = torch.from_numpy(ics).float()

    logger.info(f"Loaded {ics_true.shape[0]} samples with shape {ics_true.shape}")

    # Extract initial_encoded from tokenizer decoder in batches
    logger.info("Extracting initial_encoded from tokenizer (processing in batches)")
    num_samples = ics_true.shape[0]
    initial_dim = tokenizer.model.initial_dim
    temporal_dim = tokenizer.model.temporal_dim
    initial_encoded_list = []

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            # Get batch of tokens
            tokens_batch = {
                key: val[start_idx:end_idx].to(device)
                for key, val in tokens.items()
            }

            # Extract embeddings from codebooks
            quantized = tokenizer._extract_embeddings(tokens_batch)

            # Pass through decoder
            reconstructed = tokenizer.model.decoder(quantized)

            # Extract initial portion (after temporal, before theta)
            offset = temporal_dim
            initial_encoded_batch = reconstructed[:, offset:offset+initial_dim].cpu()
            initial_encoded_list.append(initial_encoded_batch)

            if (start_idx // batch_size) % 10 == 0:
                logger.info(f"  Processed {end_idx}/{num_samples} samples")

    # Concatenate all batches
    initial_encoded = torch.cat(initial_encoded_list, dim=0)
    logger.info(f"Extracted initial_encoded: {initial_encoded.shape}")

    return initial_encoded, ics_true


def train_epoch(
    model: InitialInverseCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for initial_encoded, ics_true in dataloader:
        initial_encoded = initial_encoded.to(device)
        ics_true = ics_true.to(device)

        # Forward pass
        ics_pred = model(initial_encoded)
        loss = criterion(ics_pred, ics_true)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: InitialInverseCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_true = []

    for initial_encoded, ics_true in dataloader:
        initial_encoded = initial_encoded.to(device)
        ics_true = ics_true.to(device)

        # Forward pass
        ics_pred = model(initial_encoded)
        loss = criterion(ics_pred, ics_true)

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(ics_pred.cpu())
        all_true.append(ics_true.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)

    # Compute metrics
    mse = torch.mean((all_preds - all_true) ** 2).item()
    mae = torch.mean(torch.abs(all_preds - all_true)).item()
    max_error = torch.max(torch.abs(all_preds - all_true)).item()

    # Per-channel metrics
    channel_mse = torch.mean((all_preds - all_true) ** 2, dim=[0, 2, 3]).tolist()

    return {
        'loss': total_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'channel_mse': channel_mse,
    }


def main():
    parser = argparse.ArgumentParser(description='Train initial inverse decoder')
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Path to trained VQTokenizer checkpoint')
    parser.add_argument('--tokenized-dataset', type=Path, required=True,
                        help='Path to pretokenized dataset with tokens')
    parser.add_argument('--raw-dataset', type=Path, required=True,
                        help='Path to raw dataset with ground truth ICs')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save trained inverse model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split fraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    initial_encoded, ics_true = load_data(
        args.tokenizer,
        args.tokenized_dataset,
        args.raw_dataset,
        device,
    )

    # Create dataset
    dataset = InitialInverseDataset(initial_encoded, ics_true)

    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    logger.info(f"Train size: {train_size}, Val size: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    encoded_dim = initial_encoded.shape[1]
    channels = ics_true.shape[1]
    spatial_size = ics_true.shape[2]

    logger.info(f"Creating InitialInverseCNN: {encoded_dim} → [{channels}, {spatial_size}, {spatial_size}]")
    model = InitialInverseCNN(
        encoded_dim=encoded_dim,
        channels=channels,
        spatial_size=spatial_size,
    ).to(device)

    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training loop
    best_val_mse = float('inf')
    patience = 10
    patience_counter = 0

    logger.info("Starting training")
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device
        )

        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.6f} - "
            f"Val MSE: {val_metrics['mse']:.6f} - "
            f"Val MAE: {val_metrics['mae']:.6f} - "
            f"Max Error: {val_metrics['max_error']:.6f}"
        )

        # Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0

            logger.info(f"New best model! MSE: {best_val_mse:.6f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoded_dim': encoded_dim,
                'channels': channels,
                'spatial_size': spatial_size,
                'val_mse': best_val_mse,
                'val_metrics': val_metrics,
                'epoch': epoch,
            }, args.output)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete! Best Val MSE: {best_val_mse:.6f}")
    logger.info(f"Model saved to {args.output}")

    # Final validation
    logger.info("Running final validation on best model")
    checkpoint = torch.load(args.output)
    model.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = validate(model, val_loader, criterion, device)
    logger.info(f"Final metrics: {final_metrics}")
    logger.info(f"Per-channel MSE: {final_metrics['channel_mse']}")

    # Check success criteria
    if final_metrics['mse'] < 0.05:
        logger.info("✅ SUCCESS: MSE < 0.05 - Initial inverse decoder works!")
    else:
        logger.warning(
            f"⚠️  WARNING: MSE = {final_metrics['mse']:.6f} > 0.05 - "
            f"May need end-to-end retraining (Plan B)"
        )


if __name__ == '__main__':
    main()
