"""
Train ThetaInverseMLP with roundtrip consistency loss.

Instead of training to reconstruct the original theta, this trains the inverse
decoder to produce self-consistent outputs: decode(encode(decode(tokens))) = decode(tokens)

The key insight: VQ tokens represent an equivalence class of (theta, IC) pairs.
The inverse decoder should produce a REPRESENTATIVE that re-encodes to the same tokens.

Loss: Latent space consistency - roundtrip latents should match original token embeddings.

Usage:
    poetry run python scripts/train_theta_inverse_roundtrip.py \
        --tokenizer checkpoints/v2/vqvae/vq_tokenizer_best.pt \
        --tokenized-dataset datasets/50k_tokenized_v2_cleaned.h5 \
        --output checkpoints/theta_inverse_roundtrip.pt \
        --epochs 100 \
        --batch-size 256
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.inverse_models import ThetaInverseMLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThetaRoundtripDataset(Dataset):
    """Dataset for training theta inverse with roundtrip loss."""

    def __init__(self, tokens: Dict[str, torch.Tensor]):
        """
        Args:
            tokens: Dict of all tokens from pretokenized dataset
        """
        # Get batch size from any token tensor
        self.batch_size = next(iter(tokens.values())).shape[0]
        self.tokens = tokens

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Return tokens for single sample
        return {key: val[idx] for key, val in self.tokens.items()}


def collate_tokens(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function to batch tokens."""
    if not batch:
        return {}

    # Stack tokens
    keys = batch[0].keys()
    return {
        key: torch.stack([sample[key] for sample in batch])
        for key in keys
    }


def load_all_tokens(tokenized_dataset_path: Path) -> Dict[str, torch.Tensor]:
    """Load all tokens from pretokenized dataset."""
    logger.info(f"Loading all tokens from {tokenized_dataset_path}")
    tokens = {}
    with h5py.File(tokenized_dataset_path, 'r') as f:
        for key in f['tokens'].keys():
            tokens[key] = torch.from_numpy(f[f'tokens/{key}'][:]).long()

    logger.info(f"Loaded {len(tokens)} token categories with {tokens[next(iter(tokens))].shape[0]} samples")
    return tokens


def compute_roundtrip_loss(
    tokenizer: VQTokenizer,
    tokens_batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute roundtrip consistency loss.

    Flow:
    1. tokens → embeddings → decoder → theta_encoded
    2. theta_encoded → theta_inverse → theta_recon
    3. theta_recon → theta_encoder → theta_encoded_rt
    4. theta_encoded_rt → projectors → latents_rt
    5. Loss: latents_rt should match target embeddings (what tokens represent)

    Args:
        tokenizer: VQTokenizer with inverse models loaded
        tokens_batch: Batch of tokens
        device: Device to compute on

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    # Move tokens to device
    tokens_batch = {k: v.to(device) for k, v in tokens_batch.items()}

    # Step 1: Get theta_encoded from original tokens
    with torch.no_grad():
        quantized = tokenizer._extract_embeddings(tokens_batch)
        reconstructed = tokenizer.model.decoder(quantized)

        # Extract theta portion
        offset = 0
        if "temporal" in tokenizer.model.families:
            offset += tokenizer.model.temporal_dim
        if "initial" in tokenizer.model.families:
            offset += tokenizer.model.initial_dim

        theta_encoded = reconstructed[:, offset:offset+tokenizer.model.theta_dim]

    # Step 2: Decode through inverse (with gradients)
    theta_recon = tokenizer.model.theta_inverse(theta_encoded)

    # Step 3: Re-encode through theta encoder
    theta_encoded_rt = tokenizer.model.theta_encoder(theta_recon)

    # Step 4 & 5: Project to latents and compute loss against target embeddings
    losses = []
    metrics = {}

    for family_cat, indices in tokenizer.group_indices.items():
        if not family_cat.startswith('theta_'):
            continue

        # Extract category features
        cat_features_rt = theta_encoded_rt[:, indices]

        # Project to hierarchical latents
        projector = tokenizer.model.projectors[family_cat]
        latents_rt = projector(cat_features_rt)

        # For each level, compute loss against target embeddings
        for level_idx, latent_rt in enumerate(latents_rt):
            quantizer_key = f"{family_cat}_L{level_idx}"
            quantizer = tokenizer.model.quantizers[quantizer_key]

            # Get target embeddings (what the original tokens represent)
            target_tokens = tokens_batch[quantizer_key]
            target_embeddings = quantizer.embedding(target_tokens)

            # Loss: roundtrip latents should match target embeddings
            loss = nn.functional.mse_loss(latent_rt, target_embeddings)
            losses.append(loss)

            metrics[quantizer_key] = loss.item()

    # Total loss
    total_loss = torch.stack(losses).mean()
    metrics['total'] = total_loss.item()

    return total_loss, metrics


@torch.no_grad()
def evaluate_roundtrip_consistency(
    tokenizer: VQTokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate roundtrip token consistency.

    Returns:
        Dict with token match rates per quantizer
    """
    tokenizer.model.eval()

    total_matches = {}
    total_counts = {}

    for tokens_batch in dataloader:
        # Move to device
        tokens_batch = {k: v.to(device) for k, v in tokens_batch.items()}

        # Get theta_encoded
        quantized = tokenizer._extract_embeddings(tokens_batch)
        reconstructed = tokenizer.model.decoder(quantized)

        offset = 0
        if "temporal" in tokenizer.model.families:
            offset += tokenizer.model.temporal_dim
        if "initial" in tokenizer.model.families:
            offset += tokenizer.model.initial_dim

        theta_encoded = reconstructed[:, offset:offset+tokenizer.model.theta_dim]

        # Decode and re-encode
        theta_recon = tokenizer.model.theta_inverse(theta_encoded)
        theta_encoded_rt = tokenizer.model.theta_encoder(theta_recon)

        # Project and quantize
        for family_cat, indices in tokenizer.group_indices.items():
            if not family_cat.startswith('theta_'):
                continue

            cat_features_rt = theta_encoded_rt[:, indices]
            projector = tokenizer.model.projectors[family_cat]
            latents_rt = projector(cat_features_rt)

            for level_idx, latent_rt in enumerate(latents_rt):
                quantizer_key = f"{family_cat}_L{level_idx}"
                quantizer = tokenizer.model.quantizers[quantizer_key]

                # Quantize roundtrip latents
                distances = torch.cdist(latent_rt, quantizer.embedding.weight, p=2.0)
                tokens_rt = distances.argmin(dim=1)

                # Compare with original
                tokens_orig = tokens_batch[quantizer_key]
                matches = (tokens_orig == tokens_rt).sum().item()
                count = tokens_orig.numel()

                if quantizer_key not in total_matches:
                    total_matches[quantizer_key] = 0
                    total_counts[quantizer_key] = 0

                total_matches[quantizer_key] += matches
                total_counts[quantizer_key] += count

    # Compute match rates
    results = {}
    for key in total_matches.keys():
        results[key] = total_matches[key] / total_counts[key]

    # Overall
    overall_matches = sum(total_matches.values())
    overall_counts = sum(total_counts.values())
    results['overall'] = overall_matches / overall_counts if overall_counts > 0 else 0.0

    return results


def train_epoch(
    tokenizer: VQTokenizer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    tokenizer.model.theta_inverse.train()

    # Freeze everything except theta_inverse
    for param in tokenizer.model.parameters():
        param.requires_grad = False
    for param in tokenizer.model.theta_inverse.parameters():
        param.requires_grad = True

    total_loss = 0.0
    num_batches = 0

    for tokens_batch in dataloader:
        # Compute roundtrip loss
        loss, metrics = compute_roundtrip_loss(tokenizer, tokens_batch, device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return {'loss': total_loss / num_batches}


def main():
    parser = argparse.ArgumentParser(description='Train theta inverse with roundtrip loss')
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Path to trained VQTokenizer checkpoint')
    parser.add_argument('--tokenized-dataset', type=Path, required=True,
                        help='Path to pretokenized dataset with tokens')
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

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = VQTokenizer.from_checkpoint(args.tokenizer)
    tokenizer.model.to(device)

    # Create theta inverse model
    theta_dim = tokenizer.model.theta_dim
    logger.info(f"Creating ThetaInverseMLP: {theta_dim} → 14")
    tokenizer.model.theta_inverse = ThetaInverseMLP(
        encoded_dim=theta_dim,
        param_dim=14,
        hidden_dim=64,
        dropout=0.1,
    ).to(device)

    # Load all tokens
    all_tokens = load_all_tokens(args.tokenized_dataset)

    # Create dataset
    dataset = ThetaRoundtripDataset(all_tokens)

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
        collate_fn=collate_tokens,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_tokens,
        num_workers=4,
        pin_memory=True,
    )

    # Create optimizer (only for theta_inverse)
    optimizer = optim.Adam(tokenizer.model.theta_inverse.parameters(), lr=args.lr)

    # Training loop
    best_consistency = 0.0
    patience = 15
    patience_counter = 0

    logger.info("Starting training with roundtrip consistency loss")
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(tokenizer, train_loader, optimizer, device)

        # Evaluate roundtrip consistency
        consistency_metrics = evaluate_roundtrip_consistency(
            tokenizer, val_loader, device
        )

        # Log
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.6f} - "
            f"Val Consistency: {consistency_metrics['overall']*100:.2f}%"
        )

        # Save best model (based on consistency, not loss)
        if consistency_metrics['overall'] > best_consistency:
            best_consistency = consistency_metrics['overall']
            patience_counter = 0

            logger.info(f"New best consistency! {best_consistency*100:.2f}%")
            torch.save({
                'model_state_dict': tokenizer.model.theta_inverse.state_dict(),
                'encoded_dim': theta_dim,
                'param_dim': 14,
                'val_consistency': best_consistency,
                'consistency_metrics': consistency_metrics,
                'epoch': epoch,
            }, args.output)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete! Best Val Consistency: {best_consistency*100:.2f}%")
    logger.info(f"Model saved to {args.output}")

    # Final evaluation
    logger.info("Running final evaluation")
    checkpoint = torch.load(args.output)
    tokenizer.model.theta_inverse.load_state_dict(checkpoint['model_state_dict'])

    final_metrics = evaluate_roundtrip_consistency(tokenizer, val_loader, device)
    logger.info(f"Final consistency metrics:")
    for key, value in sorted(final_metrics.items()):
        logger.info(f"  {key}: {value*100:.2f}%")

    # Check success criteria
    if final_metrics['overall'] > 0.95:
        logger.info("✅ EXCELLENT: >95% roundtrip consistency achieved!")
    elif final_metrics['overall'] > 0.80:
        logger.info("⚠️  MODERATE: 80-95% roundtrip consistency")
    else:
        logger.info(f"❌ POOR: {final_metrics['overall']*100:.2f}% roundtrip consistency")


if __name__ == '__main__':
    main()
