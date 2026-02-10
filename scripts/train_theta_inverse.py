"""
Train ThetaInverseMLP: encoded theta features → actual parameters.

This script trains a supervised inverse decoder that maps from the VQTokenizer's
encoded theta space [B, 32] back to actual operator parameters [B, 14] in [0,1].

Usage:
    poetry run python scripts/train_theta_inverse.py \
        --tokenizer checkpoints/vq_tokenizer_best.pt \
        --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
        --raw-dataset datasets/50k_baseline.h5 \
        --output checkpoints/theta_inverse.pt \
        --epochs 100 \
        --batch-size 256
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
from spinlock.tokens.inverse_models import ThetaInverseMLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThetaInverseDataset(Dataset):
    """Dataset for training theta inverse decoder."""

    def __init__(
        self,
        theta_encoded: torch.Tensor,
        theta_true: torch.Tensor,
    ):
        """
        Args:
            theta_encoded: [N, encoded_dim] encoded theta features from tokenizer
            theta_true: [N, param_dim] ground truth parameters in [0,1]
        """
        assert theta_encoded.shape[0] == theta_true.shape[0], \
            f"Size mismatch: {theta_encoded.shape[0]} vs {theta_true.shape[0]}"

        self.theta_encoded = theta_encoded
        self.theta_true = theta_true

    def __len__(self) -> int:
        return self.theta_encoded.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.theta_encoded[idx], self.theta_true[idx]


def load_data(
    tokenizer_path: Path,
    tokenized_dataset_path: Path,
    raw_dataset_path: Path,
    device: torch.device,
    batch_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load theta_encoded and theta_true from datasets.

    Args:
        tokenizer_path: Path to trained VQTokenizer checkpoint
        tokenized_dataset_path: Path to pretokenized dataset with tokens
        raw_dataset_path: Path to raw dataset with ground truth parameters
        device: Device to load data on
        batch_size: Batch size for processing (to avoid memory issues)

    Returns:
        Tuple of (theta_encoded [N, 32], theta_true [N, 14])
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

    theta_token_keys = [k for k in tokens.keys() if k.startswith('theta_')]
    logger.info(f"Loaded {len(tokens)} total token categories ({len(theta_token_keys)} theta categories)")

    # Load ground truth parameters
    logger.info(f"Loading ground truth from {raw_dataset_path}")
    with h5py.File(raw_dataset_path, 'r') as f:
        theta_true = torch.from_numpy(f['parameters/params'][:]).float()

    logger.info(f"Loaded {theta_true.shape[0]} samples with {theta_true.shape[1]} parameters")

    # Extract theta_encoded from tokenizer decoder in batches
    logger.info("Extracting theta_encoded from tokenizer (processing in batches)")
    num_samples = theta_true.shape[0]
    theta_dim = tokenizer.model.theta_dim
    theta_encoded_list = []

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

            # Extract theta portion
            # Family order: temporal, initial, theta
            offset = 0
            if "temporal" in tokenizer.model.families:
                offset += tokenizer.model.temporal_dim
            if "initial" in tokenizer.model.families:
                offset += tokenizer.model.initial_dim

            theta_encoded_batch = reconstructed[:, offset:offset+theta_dim].cpu()
            theta_encoded_list.append(theta_encoded_batch)

            if (start_idx // batch_size) % 10 == 0:
                logger.info(f"  Processed {end_idx}/{num_samples} samples")

    # Concatenate all batches
    theta_encoded = torch.cat(theta_encoded_list, dim=0)
    logger.info(f"Extracted theta_encoded: {theta_encoded.shape}")

    return theta_encoded, theta_true


def train_epoch(
    model: ThetaInverseMLP,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for theta_encoded, theta_true in dataloader:
        theta_encoded = theta_encoded.to(device)
        theta_true = theta_true.to(device)

        # Forward pass
        theta_pred = model(theta_encoded)
        loss = criterion(theta_pred, theta_true)

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
    model: ThetaInverseMLP,
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

    for theta_encoded, theta_true in dataloader:
        theta_encoded = theta_encoded.to(device)
        theta_true = theta_true.to(device)

        # Forward pass
        theta_pred = model(theta_encoded)
        loss = criterion(theta_pred, theta_true)

        total_loss += loss.item()
        num_batches += 1

        all_preds.append(theta_pred.cpu())
        all_true.append(theta_true.cpu())

    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_true = torch.cat(all_true, dim=0)

    # Compute metrics
    mse = torch.mean((all_preds - all_true) ** 2).item()
    mae = torch.mean(torch.abs(all_preds - all_true)).item()
    max_error = torch.max(torch.abs(all_preds - all_true)).item()

    # Check parameter ranges
    in_range = torch.all((all_preds >= 0) & (all_preds <= 1)).item()

    return {
        'loss': total_loss / num_batches,
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'in_range': in_range,
    }


def main():
    parser = argparse.ArgumentParser(description='Train theta inverse decoder')
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Path to trained VQTokenizer checkpoint')
    parser.add_argument('--tokenized-dataset', type=Path, required=True,
                        help='Path to pretokenized dataset with tokens')
    parser.add_argument('--raw-dataset', type=Path, required=True,
                        help='Path to raw dataset with ground truth parameters')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save trained inverse model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
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
    theta_encoded, theta_true = load_data(
        args.tokenizer,
        args.tokenized_dataset,
        args.raw_dataset,
        device,
    )

    # Create dataset
    dataset = ThetaInverseDataset(theta_encoded, theta_true)

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
    encoded_dim = theta_encoded.shape[1]
    param_dim = theta_true.shape[1]

    logger.info(f"Creating ThetaInverseMLP: {encoded_dim} → {param_dim}")
    model = ThetaInverseMLP(
        encoded_dim=encoded_dim,
        param_dim=param_dim,
        hidden_dim=64,
        dropout=0.1,
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
            f"Max Error: {val_metrics['max_error']:.6f} - "
            f"In Range: {val_metrics['in_range']}"
        )

        # Save best model
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            patience_counter = 0

            logger.info(f"New best model! MSE: {best_val_mse:.6f}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoded_dim': encoded_dim,
                'param_dim': param_dim,
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

    # Check success criteria
    if final_metrics['mse'] < 0.01:
        logger.info("✅ SUCCESS: MSE < 0.01 - Theta inverse decoder works!")
    else:
        logger.warning(
            f"⚠️  WARNING: MSE = {final_metrics['mse']:.6f} > 0.01 - "
            f"May need end-to-end retraining (Plan B)"
        )


if __name__ == '__main__':
    main()
