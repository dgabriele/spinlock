#!/usr/bin/env python3
"""Pre-train the CNN encoder as an autoencoder on initial conditions.

This trains the CNN to learn meaningful representations of ICs before
using it for VQ-VAE training. The goal is to have the CNN produce
high-variance, informative embeddings rather than random low-variance outputs.

Architecture:
    IC (3×64×64) → CNN Encoder (256D) → Decoder → Reconstructed IC (3×64×64)

Training:
    - Loss: MSE reconstruction loss
    - Optimizer: Adam
    - Data: 50K ICs from dataset
    - Output: Trained encoder weights saved to checkpoint

Usage:
    python scripts/pretrain_cnn_encoder.py --dataset datasets/50k_baseline.h5 \\
        --embedding-dim 256 --epochs 50 --batch-size 128 \\
        --output checkpoints/cnn_pretrain/encoder.pt
"""

import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.encoding.encoders.initial_cnn import InitialCNNEncoder


class ICDataset(Dataset):
    """Dataset of initial conditions from HDF5."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        with h5py.File(dataset_path, 'r') as f:
            # ICs are stored as [N, M, C, H, W], we take first realization
            ics = f['inputs/fields'][:, 0]  # [N, C, H, W]
            self.ics = torch.from_numpy(ics).float()

    def __len__(self):
        return len(self.ics)

    def __getitem__(self, idx):
        return self.ics[idx]


class CNNDecoder(nn.Module):
    """Decoder to reconstruct ICs from CNN embeddings.

    Architecture mirrors the encoder in reverse:
        embedding_dim → 256×4×4 → upsample stages → 3×64×64
    """

    def __init__(self, embedding_dim: int = 256, out_channels: int = 3):
        super().__init__()

        # Project embedding to spatial features
        self.fc = nn.Linear(embedding_dim, 256 * 4 * 4)
        self.bn_in = nn.BatchNorm1d(256 * 4 * 4)

        # Upsampling stages (transpose convolutions)
        # 4×4 → 8×8
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 8×8 → 16×16
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 16×16 → 32×32
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 32×32 → 64×64
        self.stage4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final convolution to get correct channels
        self.conv_out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, embedding_dim] embeddings

        Returns:
            [B, out_channels, 64, 64] reconstructed ICs
        """
        # Project and reshape
        x = self.fc(x)              # [B, 256*4*4]
        x = self.bn_in(x)
        x = x.view(-1, 256, 4, 4)   # [B, 256, 4, 4]

        # Upsample stages
        x = self.stage1(x)          # [B, 128, 8, 8]
        x = self.stage2(x)          # [B, 64, 16, 16]
        x = self.stage3(x)          # [B, 32, 32, 32]
        x = self.stage4(x)          # [B, 32, 64, 64]

        # Final convolution (no activation - continuous values)
        x = self.conv_out(x)        # [B, out_channels, 64, 64]

        return x


class CNNAutoencoder(nn.Module):
    """CNN Autoencoder for IC pre-training."""

    def __init__(self, embedding_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.encoder = InitialCNNEncoder(embedding_dim=embedding_dim, in_channels=in_channels, use_final_batchnorm=True)
        self.decoder = CNNDecoder(embedding_dim=embedding_dim, out_channels=in_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] input ICs

        Returns:
            embeddings: [B, embedding_dim]
            reconstructed: [B, C, H, W]
        """
        embeddings = self.encoder(x)
        reconstructed = self.decoder(embeddings)
        return embeddings, reconstructed


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)

        # Forward pass
        embeddings, reconstructed = model(batch)

        # MSE loss
        loss = F.mse_loss(reconstructed, batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate reconstruction quality."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            embeddings, reconstructed = model(batch)
            loss = F.mse_loss(reconstructed, batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Pre-train CNN encoder on ICs")
    parser.add_argument("--dataset", type=str, default="datasets/50k_baseline.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="CNN embedding dimension")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/cnn_pretrain/encoder.pt",
                        help="Output path for trained encoder")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    print("="*70)
    print("CNN ENCODER PRE-TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = ICDataset(args.dataset)

    # Split into train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"IC shape: {dataset.ics.shape[1:]}")
    print()

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CNNAutoencoder(embedding_dim=args.embedding_dim,
                          in_channels=dataset.ics.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training loop
    best_val_loss = float('inf')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Val loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'val_loss': val_loss,
                'embedding_dim': args.embedding_dim,
            }, output_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
        print()

    print("="*70)
    print(f"Training complete! Best val loss: {best_val_loss:.6f}")
    print(f"Encoder weights saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
