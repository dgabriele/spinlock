#!/usr/bin/env python
"""NOA Phase 1 Training on Real Data.

Trains NOA backbone on real dataset features (SUMMARY + TEMPORAL).

Key considerations:
1. 256-step rollouts are memory-intensive - use gradient checkpointing or chunking
2. SUMMARY features are per_trajectory (M=1), not aggregated (M>1)
3. All dimensions resolved dynamically from dataset
4. Uses FeaturePreprocessor to filter NaN features (DRY with VQ-VAE pipeline)

Usage:
    poetry run python scripts/dev/train_noa_real_data.py --n-samples 1000 --epochs 10
    poetry run python scripts/dev/train_noa_real_data.py --n-samples 10000 --epochs 50 --batch-size 8
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import NOABackbone, NOARealDataset, NOAFeatureExtractor
from spinlock.features import FeaturePreprocessor


def create_parser():
    parser = argparse.ArgumentParser(description="Train NOA on real data")

    # Dataset
    parser.add_argument(
        "--dataset", type=str,
        default="datasets/100k_full_features.h5",
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of samples to load"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Validation split fraction"
    )

    # Model
    parser.add_argument(
        "--base-channels", type=int, default=32,
        help="Base channel count for NOA"
    )
    parser.add_argument(
        "--encoder-levels", type=int, default=3,
        help="Number of encoder levels"
    )
    parser.add_argument(
        "--modes", type=int, default=16,
        help="Number of Fourier modes"
    )
    parser.add_argument(
        "--afno-blocks", type=int, default=4,
        help="Number of AFNO blocks"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--summary-weight", type=float, default=1.0,
        help="Weight for SUMMARY loss"
    )
    parser.add_argument(
        "--temporal-weight", type=float, default=1.0,
        help="Weight for TEMPORAL loss"
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=None,
        help="Max timesteps per rollout (None = use all from dataset)"
    )
    parser.add_argument(
        "--num-realizations", type=int, default=1,
        help="Number of independent realizations (M) to generate per IC"
    )

    # System
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to train on"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    return parser


def train_epoch(
    noa: NOABackbone,
    feature_extractor: NOAFeatureExtractor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    summary_weight: float,
    temporal_weight: float,
    max_timesteps: int | None,
    num_realizations: int = 1,
) -> dict:
    """Train for one epoch.

    Uses gradient checkpointing for memory-efficient training.
    """
    noa.train()
    total_loss = 0.0
    total_summary = 0.0
    total_temporal = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        ic = batch['ic'].to(device)
        summary_gt = batch['summary'].to(device)
        temporal_gt = batch['temporal'].to(device)

        B = ic.shape[0]
        T_total = temporal_gt.shape[1]

        # Use all timesteps unless limited
        T_use = T_total if max_timesteps is None else min(max_timesteps, T_total)

        # Generate rollout with configurable num_realizations
        pred_trajectory = noa(ic, steps=T_use, return_all_steps=True, num_realizations=num_realizations)

        # Handle output shape based on num_realizations
        if num_realizations == 1:
            pred_rollout = pred_trajectory[:, 1:, :, :, :]  # Skip IC: [B, T, C, H, W]
        else:
            pred_rollout = pred_trajectory[:, :, 1:, :, :, :]  # Skip IC: [B, M, T, C, H, W]

        # Extract features from rollout
        pred_features = feature_extractor.extract(pred_rollout)

        pred_summary = pred_features['summary']
        pred_temporal = pred_features['temporal']

        # Handle dimension mismatch for summary
        if pred_summary.shape[-1] != summary_gt.shape[-1]:
            gt_dim = pred_summary.shape[-1]
            summary_gt_use = summary_gt[:, :gt_dim]
        else:
            summary_gt_use = summary_gt

        # Handle timestep mismatch for temporal
        if pred_temporal.shape[1] != temporal_gt.shape[1]:
            temporal_gt_use = temporal_gt[:, :pred_temporal.shape[1], :]
        else:
            temporal_gt_use = temporal_gt

        # Compute losses
        summary_loss = F.mse_loss(pred_summary, summary_gt_use)
        temporal_loss = F.mse_loss(pred_temporal, temporal_gt_use)
        loss = summary_weight * summary_loss + temporal_weight * temporal_loss

        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(noa.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_summary += summary_loss.item()
        total_temporal += temporal_loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={avg:.4f}")

    epoch_time = time.time() - start_time
    return {
        "total": total_loss / max(num_batches, 1),
        "summary": total_summary / max(num_batches, 1),
        "temporal": total_temporal / max(num_batches, 1),
        "time": epoch_time,
    }


@torch.no_grad()
def validate(
    noa: NOABackbone,
    feature_extractor: NOAFeatureExtractor,
    dataloader: DataLoader,
    device: str,
    summary_weight: float,
    temporal_weight: float,
    max_timesteps: int | None,
    num_realizations: int = 1,
) -> float:
    """Validate on a dataset."""
    noa.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        ic = batch['ic'].to(device)
        summary_gt = batch['summary'].to(device)
        temporal_gt = batch['temporal'].to(device)

        T_total = temporal_gt.shape[1]
        T_use = T_total if max_timesteps is None else min(max_timesteps, T_total)

        pred_trajectory = noa(ic, steps=T_use, return_all_steps=True, num_realizations=num_realizations)

        # Handle output shape based on num_realizations
        if num_realizations == 1:
            pred_rollout = pred_trajectory[:, 1:, :, :, :]  # Skip IC: [B, T, C, H, W]
        else:
            pred_rollout = pred_trajectory[:, :, 1:, :, :, :]  # Skip IC: [B, M, T, C, H, W]

        pred_features = feature_extractor.extract(pred_rollout)

        pred_summary = pred_features['summary']
        pred_temporal = pred_features['temporal']

        if pred_summary.shape[-1] != summary_gt.shape[-1]:
            summary_gt_use = summary_gt[:, :pred_summary.shape[-1]]
        else:
            summary_gt_use = summary_gt

        if pred_temporal.shape[1] != temporal_gt.shape[1]:
            temporal_gt_use = temporal_gt[:, :pred_temporal.shape[1], :]
        else:
            temporal_gt_use = temporal_gt

        summary_loss = F.mse_loss(pred_summary, summary_gt_use)
        temporal_loss = F.mse_loss(pred_temporal, temporal_gt_use)
        loss = summary_weight * summary_loss + temporal_weight * temporal_loss

        if not torch.isnan(loss):
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("NOA Phase 1 Training on Real Data")
    print("=" * 60)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Device: {device}")

    # Create feature preprocessor (detects and filters NaN features)
    print(f"\nCreating feature preprocessor from: {args.dataset}")
    preprocessor = FeaturePreprocessor.from_dataset(args.dataset)
    preproc_info = preprocessor.get_info()
    print(f"  NaN features detected:")
    for family, info in preproc_info.items():
        if info['nan'] > 0:
            print(f"    {family}: {info['nan']} NaN / {info['total']} total → {info['valid']} valid")
            print(f"      NaN indices: {info['nan_indices']}")
        else:
            print(f"    {family}: no NaN ({info['valid']} valid)")

    # Load dataset with preprocessor
    print(f"\nLoading dataset: {args.dataset}")
    print(f"  n_samples: {args.n_samples}")

    dataset = NOARealDataset(
        args.dataset,
        n_samples=args.n_samples,
        use_per_trajectory=True,  # Match extractor output
        preprocessor=preprocessor,  # Clean NaN features
    )

    dims = dataset.get_dimension_info()
    print(f"  Dataset dimensions:")
    for k, v in dims.items():
        print(f"    {k}: {v}")

    # Split dataset
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create NOA backbone
    print(f"\nCreating NOA backbone:")
    print(f"  base_channels: {args.base_channels}")
    print(f"  encoder_levels: {args.encoder_levels}")
    print(f"  modes: {args.modes}")
    print(f"  afno_blocks: {args.afno_blocks}")

    noa = NOABackbone(
        in_channels=dims['ic_channels'],
        out_channels=dims['ic_channels'],
        base_channels=args.base_channels,
        encoder_levels=args.encoder_levels,
        modes=args.modes,
        afno_blocks=args.afno_blocks,
    ).to(device)
    print(f"  Parameters: {noa.num_parameters:,}")

    # Create feature extractor with preprocessor (same NaN filtering as dataset)
    feature_extractor = NOAFeatureExtractor(device=device, preprocessor=preprocessor)

    # Probe dimensions (use dataset timesteps if max_timesteps not specified)
    probe_timesteps = dims['temporal_steps'] if args.max_timesteps is None else min(args.max_timesteps, dims['temporal_steps'])
    probe_dims = feature_extractor.probe_dimensions(
        timesteps=probe_timesteps,
        channels=dims['ic_channels'],
        height=dims['ic_height'],
        width=dims['ic_width'],
    )
    print(f"\nFeature extractor output dimensions:")
    for k, v in probe_dims.items():
        print(f"  {k}: {v}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        noa.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    timesteps_str = f"{args.max_timesteps}" if args.max_timesteps else f"all ({dims['temporal_steps']})"
    print(f"\nTraining:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  timesteps: {timesteps_str}")
    print(f"  num_realizations: {args.num_realizations}")
    print(f"  summary_weight: {args.summary_weight}")
    print(f"  temporal_weight: {args.temporal_weight}")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_result = train_epoch(
            noa=noa,
            feature_extractor=feature_extractor,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            summary_weight=args.summary_weight,
            temporal_weight=args.temporal_weight,
            max_timesteps=args.max_timesteps,
            num_realizations=args.num_realizations,
        )
        history["train_loss"].append(train_result["total"])

        # Validate
        val_loss = validate(
            noa=noa,
            feature_extractor=feature_extractor,
            dataloader=val_loader,
            device=device,
            summary_weight=args.summary_weight,
            temporal_weight=args.temporal_weight,
            max_timesteps=args.max_timesteps,
            num_realizations=args.num_realizations,
        )
        history["val_loss"].append(val_loss)

        # Log
        print(f"  Train: loss={train_result['total']:.4f} "
              f"(sum={train_result['summary']:.4f}, temp={train_result['temporal']:.4f}) "
              f"[{train_result['time']:.1f}s]")
        print(f"  Val: loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best! (val_loss={best_val_loss:.4f})")

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
