#!/usr/bin/env python
"""NOA Training with State-Level Supervision and Optional VQ-VAE Alignment.

This script trains NOA using a two-loss structure:
1. L_traj: MSE on trajectories (primary, non-negotiable)
2. L_commit: VQ commitment loss (optional, for manifold adherence)

Loss = L_traj + λ * L_commit

The VQ-VAE alignment is optional and enables NOA to "think" in terms of
the VQ token vocabulary learned from CNO rollouts.

**Truncated BPTT**: For long rollouts (T > 32), gradients can explode through
the autoregressive chain. We use truncated backpropagation through time (TBPTT):
- Warmup phase: Roll out T - bptt_window steps WITHOUT gradient tracking
- Supervised phase: Roll out bptt_window steps WITH gradient tracking
This limits gradient flow to the last bptt_window steps while still supervising
the full trajectory via the target from CNO replay.

Usage:
    # State-only training (short sequences)
    poetry run python scripts/dev/train_noa_state_supervised.py --n-samples 500 --epochs 10

    # Long sequences with truncated BPTT
    poetry run python scripts/dev/train_noa_state_supervised.py \
        --n-samples 500 --epochs 10 --timesteps 256 --bptt-window 32

    # With VQ-VAE alignment
    poetry run python scripts/dev/train_noa_state_supervised.py \
        --n-samples 500 --epochs 10 \
        --vqvae-path checkpoints/production/100k_3family_v1 \
        --lambda-commit 0.5
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

from spinlock.noa import NOABackbone, CNOReplayer, VQVAEAlignmentLoss


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
    n_realizations: int = 1,
    state_weight: float = 1.0,
    clip_grad: float = 1.0,
    alignment: VQVAEAlignmentLoss | None = None,
    lambda_commit: float = 0.5,
    lambda_latent: float = 0.1,
    bptt_window: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    save_fn = None,
    save_every: int = 0,
    global_step: int = 0,
    log_every: int = 1,
    checkpoint_state: dict = None,
) -> dict:
    """Train for one epoch with state-level supervision and optional VQ-VAE alignment.

    For each batch:
    1. Replay CNO from IC -> target trajectory
    2. Run NOA from same IC -> predicted trajectory (with truncated BPTT if needed)
    3. Compute MSE loss on trajectory states (L_traj)
    4. Optionally compute VQ-VAE commitment loss (L_commit)

    Args:
        bptt_window: If set, use truncated BPTT - only backprop through the last
                    bptt_window steps. This prevents gradient explosion for long
                    sequences (T > 32). The full trajectory is still generated
                    and compared against the target, but gradients only flow
                    through the last bptt_window steps.
    """
    noa.train()
    total_loss = 0.0
    total_traj = 0.0  # Trajectory MSE loss (physics fidelity)
    total_commit = 0.0
    total_latent = 0.0
    num_batches = 0
    start_time = time.time()

    # Determine if we need truncated BPTT
    use_tbptt = bptt_window is not None and bptt_window < timesteps

    for batch_idx, batch in enumerate(dataloader):
        ic = batch["ic"].to(device)  # [B, C, H, W]
        params = batch["params"]  # [B, d] - keep on CPU for replayer

        B = ic.shape[0]

        # Generate NOA trajectory with truncated BPTT if needed
        if use_tbptt:
            # Truncated BPTT: warmup without grad, then supervised with grad
            warmup_steps = timesteps - bptt_window

            # Phase 1: Warmup (no gradients)
            x = ic.clone()
            with torch.no_grad():
                for _ in range(warmup_steps):
                    x = noa.single_step(x)
            warmup_state = x.clone()  # Detached from graph

            # Phase 2: Supervised (with gradients)
            # Use rollout() to leverage gradient checkpointing
            supervised_traj = noa.rollout(warmup_state, steps=bptt_window, return_all_steps=True)

            # pred_trajectory only contains supervised portion [B, bptt_window+1, C, H, W]
            pred_trajectory = supervised_traj
        else:
            # Standard rollout (full gradient flow)
            pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

        # Replay CNO trajectories for each sample in batch
        target_trajectories = []
        for b in range(B):
            target_traj = replayer.rollout(
                params_vector=params[b].numpy(),
                ic=ic[b:b+1],  # [1, C, H, W]
                timesteps=timesteps,
                num_realizations=n_realizations,
                return_all_steps=True,
            )  # [1, M, T+1, C, H, W] if M>1 else [1, T+1, C, H, W]
            target_trajectories.append(target_traj)

        # Stack: [B, M, T+1, C, H, W] if M>1 else [B, T+1, C, H, W]
        target_trajectory = torch.cat(target_trajectories, dim=0)

        # For M > 1, take mean across realizations for supervision
        if n_realizations > 1 and target_trajectory.dim() == 6:
            target_trajectory = target_trajectory.mean(dim=1)  # [B, T+1, C, H, W]

        # Compute state-level MSE loss
        if use_tbptt:
            # Match predicted states to corresponding target window
            # pred_trajectory: [B, bptt_window+1, C, H, W]
            # target_trajectory: [B, T+1, C, H, W] -> take last bptt_window+1 states
            pred_states = pred_trajectory[:, 1:, :, :, :]  # Skip first (warmup final state)
            target_states = target_trajectory[:, -(bptt_window):, :, :, :]
        else:
            # Skip IC (index 0) since it's given, only supervise predicted states
            pred_states = pred_trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]
            target_states = target_trajectory[:, 1:, :, :, :]  # [B, T, C, H, W]

        state_loss = F.mse_loss(pred_states, target_states)
        loss = state_weight * state_loss

        # Add VQ-VAE commitment loss if enabled
        commit_loss = torch.tensor(0.0, device=device)
        latent_loss = torch.tensor(0.0, device=device)

        if alignment is not None:
            try:
                align_losses = alignment.compute_losses(
                    pred_trajectory=pred_states,
                    target_trajectory=target_states,
                    ic=ic,
                )
                commit_loss = align_losses['commit']
                loss = loss + lambda_commit * commit_loss

                # Add L_latent if enabled
                if 'latent' in align_losses:
                    latent_loss = align_losses['latent']
                    loss = loss + lambda_latent * latent_loss
            except Exception as e:
                if batch_idx == 0:
                    print(f"  Warning: VQ-VAE alignment failed: {e}")

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at batch {batch_idx}")
            continue

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check for NaN in gradients (prevents weight corruption)
        has_nan_grad = False
        for name, param in noa.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break

        if has_nan_grad:
            print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping update")
            optimizer.zero_grad()  # Clear corrupted gradients
            continue

        torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        total_loss += loss.item()
        total_traj += state_loss.item()
        total_commit += commit_loss.item()
        total_latent += latent_loss.item()
        num_batches += 1

        # Periodic checkpoint
        if save_every > 0 and save_fn is not None and global_step % save_every == 0:
            if checkpoint_state is not None:
                checkpoint_state['global_step'] = global_step
            save_fn(f"step_{global_step}")

        # Logging
        if (batch_idx + 1) % log_every == 0:
            batch_time = (time.time() - start_time) / num_batches
            avg_traj = total_traj / num_batches
            avg_total = total_loss / num_batches
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

            if alignment is not None:
                avg_commit = total_commit / num_batches
                avg_latent = total_latent / num_batches
                if alignment.enable_latent_loss:
                    print(f"  [{batch_idx + 1}/{len(dataloader)}] "
                          f"loss={avg_total:.4f} traj={avg_traj:.4f} commit={avg_commit:.6f} "
                          f"latent={avg_latent:.6f} "
                          f"lr={lr:.2e} {batch_time:.1f}s/b")
                else:
                    print(f"  [{batch_idx + 1}/{len(dataloader)}] "
                          f"loss={avg_total:.4f} traj={avg_traj:.4f} commit={avg_commit:.6f} "
                          f"lr={lr:.2e} {batch_time:.1f}s/b")
            else:
                print(f"  [{batch_idx + 1}/{len(dataloader)}] "
                      f"loss={avg_total:.4f} traj={avg_traj:.4f} "
                      f"lr={lr:.2e} {batch_time:.1f}s/b")

    epoch_time = time.time() - start_time
    return {
        "total": total_loss / max(num_batches, 1),
        "traj": total_traj / max(num_batches, 1),
        "commit": total_commit / max(num_batches, 1),
        "latent": total_latent / max(num_batches, 1),
        "time": epoch_time,
        "global_step": global_step,
    }


@torch.no_grad()
def validate(
    noa: NOABackbone,
    replayer: CNOReplayer,
    dataloader: DataLoader,
    device: str,
    timesteps: int,
    n_realizations: int = 1,
    alignment: VQVAEAlignmentLoss | None = None,
    lambda_commit: float = 0.5,
    lambda_latent: float = 0.1,
    bptt_window: int | None = None,
) -> dict:
    """Validate with state-level loss and optional VQ-VAE alignment."""
    noa.eval()
    total_loss = 0.0
    total_traj = 0.0
    total_commit = 0.0
    total_latent = 0.0
    num_batches = 0

    # For validation, we use same window as training for fair comparison
    use_tbptt = bptt_window is not None and bptt_window < timesteps

    for batch in dataloader:
        ic = batch["ic"].to(device)
        params = batch["params"]

        B = ic.shape[0]

        # Generate full trajectory (no grad, so no memory issue)
        pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

        target_trajectories = []
        for b in range(B):
            target_traj = replayer.rollout(
                params_vector=params[b].numpy(),
                ic=ic[b:b+1],
                timesteps=timesteps,
                num_realizations=n_realizations,
                return_all_steps=True,
            )
            target_trajectories.append(target_traj)

        target_trajectory = torch.cat(target_trajectories, dim=0)

        # For M > 1, take mean across realizations for supervision
        if n_realizations > 1 and target_trajectory.dim() == 6:
            target_trajectory = target_trajectory.mean(dim=1)

        # Use same window as training for consistent evaluation
        if use_tbptt:
            pred_states = pred_trajectory[:, -(bptt_window):, :, :, :]
            target_states = target_trajectory[:, -(bptt_window):, :, :, :]
        else:
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = target_trajectory[:, 1:, :, :, :]

        state_loss = F.mse_loss(pred_states, target_states)
        loss = state_loss

        commit_loss = torch.tensor(0.0, device=device)
        latent_loss = torch.tensor(0.0, device=device)

        if alignment is not None:
            try:
                align_losses = alignment.compute_losses(
                    pred_trajectory=pred_states,
                    target_trajectory=target_states,
                    ic=ic,
                )
                commit_loss = align_losses['commit']
                loss = loss + lambda_commit * commit_loss

                if 'latent' in align_losses:
                    latent_loss = align_losses['latent']
                    loss = loss + lambda_latent * latent_loss
            except Exception:
                pass

        if not torch.isnan(loss):
            total_loss += loss.item()
            total_traj += state_loss.item()
            total_commit += commit_loss.item()
            total_latent += latent_loss.item()
            num_batches += 1

    return {
        "total": total_loss / max(num_batches, 1),
        "traj": total_traj / max(num_batches, 1),
        "commit": total_commit / max(num_batches, 1),
        "latent": total_latent / max(num_batches, 1),
    }


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
    parser.add_argument(
        "--n-realizations", type=int, default=1,
        help="Number of stochastic realizations for CNO rollout (M > 1 enables realization aggregation)"
    )
    parser.add_argument(
        "--bptt-window", type=int, default=None,
        help="Truncated BPTT window size (only backprop through last N steps). "
             "Required for long sequences (T > 32) to prevent gradient explosion. "
             "Set to 32 for T=256, or leave None for full backprop on short sequences."
    )

    # LR scheduling
    parser.add_argument(
        "--warmup-steps", type=int, default=0,
        help="Number of warmup steps for LR scheduler"
    )
    parser.add_argument(
        "--lr-schedule", type=str, default="cosine",
        choices=["cosine", "constant"],
        help="Learning rate schedule"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/noa",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (e.g., checkpoints/noa/step_200.pt)"
    )
    parser.add_argument(
        "--save-every", type=int, default=1000,
        help="Save checkpoint every N batches (0 = only save at epoch end)"
    )
    parser.add_argument(
        "--log-every", type=int, default=1,
        help="Log progress every N batches"
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=2,
        help="Stop if no improvement for N epochs (0 = disabled)"
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

    # VQ-VAE alignment (optional)
    parser.add_argument(
        "--vqvae-path", type=str, default=None,
        help="Path to VQ-VAE checkpoint for alignment loss (optional)"
    )
    parser.add_argument(
        "--lambda-commit", type=float, default=0.5,
        help="Weight for VQ commitment loss"
    )
    parser.add_argument(
        "--enable-latent-loss", action="store_true",
        help="Enable L_latent (NOA-VQ latent alignment)"
    )
    parser.add_argument(
        "--lambda-latent", type=float, default=0.1,
        help="Weight for latent alignment loss (default: 0.1)"
    )
    parser.add_argument(
        "--latent-sample-steps", type=int, default=3,
        help="Number of timesteps to sample for latent loss (default: 3, use -1 for all)"
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
        use_checkpointing=True,  # Keep enabled to reduce activation memory
    ).to(device)
    print(f"  Parameters: {noa.num_parameters:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        noa.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create LR scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = None
    if args.lr_schedule == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        if args.warmup_steps > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Setup checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Will be populated by save_checkpoint closure
    _checkpoint_state = {
        'epoch': 0,
        'global_step': 0,
        'history': None,
        'best_val_loss': float('inf'),
        'alignment': None,
    }

    def save_checkpoint(name: str):
        path = checkpoint_dir / f"{name}.pt"
        checkpoint = {
            "model_state_dict": noa.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": _checkpoint_state['epoch'],
            "global_step": _checkpoint_state['global_step'],
            "history": _checkpoint_state['history'],
            "best_val_loss": _checkpoint_state['best_val_loss'],
            "config": {
                'base_channels': args.base_channels,
                'encoder_levels': args.encoder_levels,
                'modes': args.modes,
                'afno_blocks': args.afno_blocks,
            },
            "args": vars(args),
        }

        # Save alignment/projector state if enabled
        if _checkpoint_state['alignment'] is not None and _checkpoint_state['alignment'].latent_projector is not None:
            checkpoint["alignment_state"] = _checkpoint_state['alignment'].latent_projector.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    # Initialize VQ-VAE alignment (optional)
    alignment = None
    if args.vqvae_path is not None:
        print(f"\nLoading VQ-VAE alignment from: {args.vqvae_path}")
        try:
            alignment = VQVAEAlignmentLoss.from_checkpoint(
                vqvae_path=args.vqvae_path,
                device=device,
                noa=noa,
                enable_latent_loss=args.enable_latent_loss,
                latent_sample_steps=args.latent_sample_steps,
            )
            print(f"  VQ-VAE alignment enabled")
            print(f"  λ_commit: {args.lambda_commit}")
            if args.enable_latent_loss:
                print(f"  λ_latent: {args.lambda_latent}")
                sample_info = f"all timesteps" if args.latent_sample_steps <= 0 else f"{args.latent_sample_steps} sampled"
                print(f"  Latent sampling: {sample_info}")
                print(f"  VQ latent dim: {alignment.latent_projector.vq_latent_dim}")
                print(f"  NOA latent dim: {alignment.latent_projector.noa_latent_dim}")
                print(f"  Projector parameters: {alignment.latent_projector.num_parameters:,}")
        except Exception as e:
            print(f"  Warning: Failed to load VQ-VAE: {e}")
            print(f"  Continuing without VQ-VAE alignment")

    # Resume from checkpoint (optional)
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history = {"train_loss": [], "val_loss": []}

    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        # Load model state
        noa.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Loaded model weights")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  ✓ Loaded optimizer state")

        # Load scheduler state
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  ✓ Loaded scheduler state")

        # Load training state
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        history = checkpoint.get('history', {"train_loss": [], "val_loss": []})
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  ✓ Resuming from epoch {start_epoch + 1}, step {global_step}")
        print(f"  ✓ Best val loss so far: {best_val_loss:.6f}")

        # Load alignment/projector state if available
        if alignment is not None and alignment.latent_projector is not None:
            if 'alignment_state' in checkpoint:
                alignment.latent_projector.load_state_dict(checkpoint['alignment_state'])
                print(f"  ✓ Loaded latent projector weights")
            else:
                print(f"  ⚠ No projector weights in checkpoint (will start from scratch)")

    # Update checkpoint state for save_checkpoint closure
    _checkpoint_state['alignment'] = alignment
    _checkpoint_state['history'] = history
    _checkpoint_state['best_val_loss'] = best_val_loss
    _checkpoint_state['global_step'] = global_step

    # Training loop
    print(f"\nTraining:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  timesteps: {args.timesteps}")
    print(f"  n_realizations: {args.n_realizations}")
    print(f"  lr: {args.lr}")
    if args.bptt_window is not None:
        print(f"  bptt_window: {args.bptt_window} (truncated BPTT enabled)")
    else:
        print(f"  bptt_window: None (full backprop)")
    print(f"  lr_schedule: {args.lr_schedule}")
    if args.warmup_steps > 0:
        print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")
    print(f"  save_every: {args.save_every} batches")
    print(f"  log_every: {args.log_every} batches")
    print(f"  early_stop_patience: {args.early_stop_patience} epochs")
    if args.resume is not None:
        print(f"  resuming: from epoch {start_epoch + 1}, step {global_step}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_result = train_epoch(
            noa=noa,
            replayer=replayer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            timesteps=args.timesteps,
            n_realizations=args.n_realizations,
            alignment=alignment,
            lambda_commit=args.lambda_commit,
            lambda_latent=args.lambda_latent,
            bptt_window=args.bptt_window,
            scheduler=scheduler,
            save_fn=save_checkpoint,
            save_every=args.save_every,
            global_step=global_step,
            log_every=args.log_every,
            checkpoint_state=_checkpoint_state,
        )
        global_step = train_result["global_step"]
        history["train_loss"].append(train_result["total"])

        # Validate
        val_result = validate(
            noa=noa,
            replayer=replayer,
            dataloader=val_loader,
            device=device,
            timesteps=args.timesteps,
            n_realizations=args.n_realizations,
            alignment=alignment,
            lambda_commit=args.lambda_commit,
            lambda_latent=args.lambda_latent,
            bptt_window=args.bptt_window,
        )
        history["val_loss"].append(val_result["total"])

        # Log
        if alignment is not None:
            if alignment.enable_latent_loss:
                print(f"  Train: total={train_result['total']:.6f} "
                      f"traj={train_result['traj']:.6f} "
                      f"commit={train_result['commit']:.6f} "
                      f"latent={train_result['latent']:.6f} [{train_result['time']:.1f}s]")
                print(f"  Val:   total={val_result['total']:.6f} "
                      f"traj={val_result['traj']:.6f} "
                      f"commit={val_result['commit']:.6f} "
                      f"latent={val_result['latent']:.6f}")
            else:
                print(f"  Train: total={train_result['total']:.6f} "
                      f"traj={train_result['traj']:.6f} "
                      f"commit={train_result['commit']:.6f} [{train_result['time']:.1f}s]")
                print(f"  Val:   total={val_result['total']:.6f} "
                      f"traj={val_result['traj']:.6f} "
                      f"commit={val_result['commit']:.6f}")
        else:
            print(f"  Train: loss={train_result['total']:.6f} [{train_result['time']:.1f}s]")
            print(f"  Val:   loss={val_result['total']:.6f}")

        # Update checkpoint state
        _checkpoint_state['epoch'] = epoch
        _checkpoint_state['global_step'] = global_step
        _checkpoint_state['history'] = history
        _checkpoint_state['best_val_loss'] = best_val_loss

        if val_result["total"] < best_val_loss:
            best_val_loss = val_result["total"]
            epochs_without_improvement = 0
            _checkpoint_state['best_val_loss'] = best_val_loss
            print(f"  New best! (val_loss={best_val_loss:.6f})")
            save_checkpoint("best_model")
        else:
            epochs_without_improvement += 1

        # Save epoch checkpoint
        save_checkpoint(f"epoch_{epoch + 1}")

        # Early stopping
        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"\n  Early stopping: no improvement for {args.early_stop_patience} epochs")
            break

        # Clear replayer cache periodically to manage memory
        replayer.clear_cache()

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
