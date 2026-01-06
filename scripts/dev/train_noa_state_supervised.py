#!/usr/bin/env python
"""NOA Training with State-Level Supervision via CNO Replay.

This script trains NOA using direct state-level supervision:
- CNO operators are reconstructed from stored parameter vectors
- CNO trajectories are replayed on-the-fly to produce target states
- NOA learns by minimizing MSE on actual trajectory states (not features)

This is the standard neural operator training approach:
    loss = MSE(NOA_trajectory, CNO_trajectory)

Key insight: Feature-based loss (MSE on extracted features) doesn't provide
sufficient gradient signal for the model to learn. State-level supervision
gives direct pixel-wise feedback.

Usage:
    poetry run python scripts/dev/train_noa_state_supervised.py --n-samples 500 --epochs 10
    poetry run python scripts/dev/train_noa_state_supervised.py --n-samples 5000 --epochs 50 --batch-size 8
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import h5py
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import NOABackbone, CNOReplayer


class NOAStateDataset(Dataset):
    """Dataset that loads ICs and parameter vectors for CNO replay.

    Provides:
    - ic: Initial condition [C, H, W]
    - params: Sobol parameter vector [d,] for CNO reconstruction
    - sample_idx: Original index (for debugging/tracking)
    """

    def __init__(
        self,
        dataset_path: str,
        n_samples: int | None = None,
        realization_idx: int = 0,  # Which realization's IC to use
    ):
        """Initialize dataset.

        Args:
            dataset_path: Path to HDF5 dataset
            n_samples: Number of samples to load (None = all)
            realization_idx: Which realization to use for IC (0 to M-1)
        """
        self.dataset_path = Path(dataset_path)
        self.realization_idx = realization_idx

        with h5py.File(self.dataset_path, "r") as f:
            total = f["inputs/fields"].shape[0]
            n = n_samples if n_samples is not None else total
            n = min(n, total)

            # Load ICs [N, M, H, W] -> take one realization -> [N, H, W]
            inputs = f["inputs/fields"][:n, realization_idx, :, :]

            # Add channel dimension [N, H, W] -> [N, 1, H, W]
            self.ics = torch.from_numpy(inputs).float().unsqueeze(1)

            # Load parameter vectors [N, d]
            self.params = torch.from_numpy(f["parameters/params"][:n]).float()

        self.n_samples = n

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "ic": self.ics[idx],
            "params": self.params[idx],
            "sample_idx": idx,
        }


def train_epoch(
    noa: NOABackbone,
    replayer: CNOReplayer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    timesteps: int,
    state_weight: float = 1.0,
    clip_grad: float = 1.0,
) -> dict:
    """Train for one epoch with state-level supervision.

    For each batch:
    1. Replay CNO from IC -> target trajectory
    2. Run NOA from same IC -> predicted trajectory
    3. Compute MSE loss on trajectory states
    """
    noa.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        ic = batch["ic"].to(device)  # [B, C, H, W]
        params = batch["params"]  # [B, d] - keep on CPU for replayer

        B = ic.shape[0]

        # Generate NOA trajectory: [B, T+1, C, H, W]
        pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

        # Replay CNO trajectories for each sample in batch
        target_trajectories = []
        for b in range(B):
            target_traj = replayer.rollout(
                params_vector=params[b].numpy(),
                ic=ic[b:b+1],  # [1, C, H, W]
                timesteps=timesteps,
                num_realizations=1,
                return_all_steps=True,
            )  # [1, T+1, C, H, W]
            target_trajectories.append(target_traj)

        # Stack: [B, T+1, C, H, W]
        target_trajectory = torch.cat(target_trajectories, dim=0)

        # Compute state-level MSE loss
        # Skip IC (index 0) since it's given, only supervise predicted states
        pred_states = pred_trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]
        target_states = target_trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]

        state_loss = F.mse_loss(pred_states, target_states)
        loss = state_weight * state_loss

        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg = total_loss / num_batches
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: state_loss={avg:.6f}")

    epoch_time = time.time() - start_time
    return {
        "total": total_loss / max(num_batches, 1),
        "time": epoch_time,
    }


@torch.no_grad()
def validate(
    noa: NOABackbone,
    replayer: CNOReplayer,
    dataloader: DataLoader,
    device: str,
    timesteps: int,
) -> float:
    """Validate with state-level loss."""
    noa.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        ic = batch["ic"].to(device)
        params = batch["params"]

        B = ic.shape[0]

        pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

        target_trajectories = []
        for b in range(B):
            target_traj = replayer.rollout(
                params_vector=params[b].numpy(),
                ic=ic[b:b+1],
                timesteps=timesteps,
                num_realizations=1,
                return_all_steps=True,
            )
            target_trajectories.append(target_traj)

        target_trajectory = torch.cat(target_trajectories, dim=0)

        pred_states = pred_trajectory[:, 1:, :, :, :]
        target_states = target_trajectory[:, 1:, :, :, :]

        loss = F.mse_loss(pred_states, target_states)

        if not torch.isnan(loss):
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train NOA with state-level supervision")

    # Dataset
    parser.add_argument(
        "--dataset", type=str,
        default="datasets/100k_full_features.h5",
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/experiments/local_100k_optimized.yaml",
        help="Config file used for dataset generation (for CNO reconstruction)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
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
        "--timesteps", type=int, default=32,
        help="Number of timesteps to supervise (shorter = faster training)"
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

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("NOA Training with State-Level Supervision")
    print("=" * 60)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Device: {device}")

    # Create CNO replayer from config
    print(f"\nLoading CNO replayer from: {args.config}")
    replayer = CNOReplayer.from_config(args.config, device=device, cache_size=8)
    print(f"  Parameter space loaded")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    print(f"  n_samples: {args.n_samples}")

    dataset = NOAStateDataset(
        args.dataset,
        n_samples=args.n_samples,
    )
    print(f"  IC shape: {dataset.ics[0].shape}")
    print(f"  Params shape: {dataset.params[0].shape}")

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
        in_channels=1,
        out_channels=1,
        base_channels=args.base_channels,
        encoder_levels=args.encoder_levels,
        modes=args.modes,
        afno_blocks=args.afno_blocks,
    ).to(device)
    print(f"  Parameters: {noa.num_parameters:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        noa.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    print(f"\nTraining:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  timesteps: {args.timesteps}")
    print(f"  lr: {args.lr}")

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_result = train_epoch(
            noa=noa,
            replayer=replayer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            timesteps=args.timesteps,
        )
        history["train_loss"].append(train_result["total"])

        # Validate
        val_loss = validate(
            noa=noa,
            replayer=replayer,
            dataloader=val_loader,
            device=device,
            timesteps=args.timesteps,
        )
        history["val_loss"].append(val_loss)

        # Log
        print(f"  Train: loss={train_result['total']:.6f} [{train_result['time']:.1f}s]")
        print(f"  Val: loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best! (val_loss={best_val_loss:.6f})")

        # Clear replayer cache periodically to manage memory
        replayer.clear_cache()

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
