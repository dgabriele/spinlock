#!/usr/bin/env python
"""Unified NOA Training with Configurable Loss Modes.

This script trains NOA using one of two training paradigms:

MSE-led (Physics First):
    Loss = λ_traj * L_traj + λ_commit * L_commit + λ_latent * L_latent
           ═══════════════
              PRIMARY
    Use when exact trajectory matching is critical.

VQ-led (Creative Observer):
    Loss = λ_recon * L_recon + λ_commit * L_commit + λ_traj * L_traj
           ════════════════
              PRIMARY
    Use when symbolic coherence matters more than exact matching.
    Enables NOA to generate "creative" interpretations of dynamics.

**Truncated BPTT**: For long rollouts (T > 32), gradients can explode through
the autoregressive chain. We use truncated backpropagation through time (TBPTT):
- Warmup phase: Roll out T - bptt_window steps WITHOUT gradient tracking
- Supervised phase: Roll out bptt_window steps WITH gradient tracking

Usage:
    # MSE-led (physics first)
    poetry run python scripts/dev/train_noa_unified.py \
        --loss-mode mse_led \
        --vqvae-path checkpoints/production/100k_3family_v1 \
        --n-samples 500 --epochs 10

    # VQ-led (creative observer)
    poetry run python scripts/dev/train_noa_unified.py \
        --loss-mode vq_led \
        --vqvae-path checkpoints/production/100k_3family_v1 \
        --n-samples 500 --epochs 10 \
        --lambda-traj 0.3
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import h5py
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import (
    NOABackbone,
    CNOReplayer,
    VQVAEAlignmentLoss,
    BaseNOALoss,
    LossOutput,
)
from spinlock.noa.losses import MSELedLoss, VQLedLoss


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
        realization_idx: int = 0,
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

            inputs = f["inputs/fields"][:n, realization_idx, :, :]
            self.ics = torch.from_numpy(inputs).float().unsqueeze(1)
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
    loss_fn: BaseNOALoss,
    replayer: CNOReplayer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    timesteps: int,
    n_realizations: int = 1,
    clip_grad: float = 1.0,
    bptt_window: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    save_fn=None,
    save_every: int = 0,
    global_step: int = 0,
    log_every: int = 1,
    checkpoint_state: dict = None,
) -> dict:
    """Train for one epoch with abstracted loss computation.

    The key difference from train_noa_state_supervised.py is that loss
    computation is delegated to the loss_fn (BaseNOALoss), enabling
    togglable training paradigms without code duplication.

    Args:
        noa: NOA backbone to train
        loss_fn: Loss function (MSELedLoss or VQLedLoss)
        replayer: CNO replayer for target generation
        dataloader: Training data
        optimizer: Optimizer
        device: Computation device
        timesteps: Number of timesteps to rollout
        n_realizations: Number of realizations for CNO rollout
        clip_grad: Gradient clipping value
        bptt_window: Truncated BPTT window (None = full backprop)
        scheduler: LR scheduler (optional)
        save_fn: Checkpoint save function
        save_every: Checkpoint interval (batches)
        global_step: Current global step
        log_every: Logging interval (batches)
        checkpoint_state: Mutable state for checkpointing

    Returns:
        Dictionary with epoch statistics
    """
    noa.train()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0
    start_time = time.time()

    use_tbptt = bptt_window is not None and bptt_window < timesteps
    batches_to_skip = global_step % len(dataloader) if global_step > 0 else 0

    if batches_to_skip > 0:
        print(f"  Skipping first {batches_to_skip} batches (already processed)...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < batches_to_skip:
            if (batch_idx + 1) % 50 == 0:
                print(f"    Skipped {batch_idx + 1}/{batches_to_skip} batches...")
            continue

        # TEMPORARY: Skip problematic batches that cause CUDA hangs
        # TODO: Identify root cause of these specific parameter vectors
        if batch_idx in [83, 84, 85]:
            print(f"  [SKIP] Batch {batch_idx} (known CUDA hang issue)")
            continue

        ic = batch["ic"].to(device)
        params = batch["params"]
        B = ic.shape[0]

        # Generate NOA trajectory with truncated BPTT if needed
        if use_tbptt:
            warmup_steps = timesteps - bptt_window
            x = ic.clone()
            with torch.no_grad():
                for _ in range(warmup_steps):
                    x = noa.single_step(x)
            warmup_state = x.clone()
            pred_trajectory = noa.rollout(warmup_state, steps=bptt_window, return_all_steps=True)
        else:
            pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

        # Replay CNO trajectories with error handling and detailed logging
        target_trajectories = []
        skip_batch = False
        for b in range(B):
            try:
                # Log which sample we're about to process (for debugging hangs)
                if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                    print(f"  [DEBUG] Batch {batch_idx}, sample {b}/{B}, params hash: {hash(tuple(params[b].numpy().tolist()))}")

                target_traj = replayer.rollout(
                    params_vector=params[b].numpy(),
                    ic=ic[b:b+1],
                    timesteps=timesteps,
                    num_realizations=n_realizations,
                    return_all_steps=True,
                )
                target_trajectories.append(target_traj)
            except (ValueError, RuntimeError) as e:
                # CNO rollout failed (NaN/Inf, CUDA error, or abnormal values)
                print(f"  Warning: CNO rollout failed for sample {b} in batch {batch_idx}: {e}")
                skip_batch = True
                break

        if skip_batch:
            continue

        target_trajectory = torch.cat(target_trajectories, dim=0)

        if n_realizations > 1 and target_trajectory.dim() == 6:
            target_trajectory = target_trajectory.mean(dim=1)

        # Extract states for loss computation
        if use_tbptt:
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = target_trajectory[:, -(bptt_window):, :, :, :]
        else:
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = target_trajectory[:, 1:, :, :, :]

        # Compute loss using abstracted interface
        try:
            loss_output: LossOutput = loss_fn.compute(
                pred_trajectory=pred_states,
                target_trajectory=target_states,
                ic=ic,
                noa=noa,
            )
        except Exception as e:
            if batch_idx == 0:
                print(f"  Warning: Loss computation failed: {e}")
            continue

        if torch.isnan(loss_output.total) or torch.isinf(loss_output.total):
            print(f"Warning: NaN/Inf loss at batch {batch_idx}")
            continue

        # Backward
        optimizer.zero_grad()
        loss_output.total.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for name, param in noa.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break

        if has_nan_grad:
            print(f"Warning: NaN/Inf gradients at batch {batch_idx}, skipping update")
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(noa.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        total_loss += loss_output.total.item()
        num_batches += 1

        # Accumulate metrics
        for key, value in loss_output.metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value

        # Periodic checkpoint
        if save_every > 0 and save_fn is not None and global_step % save_every == 0:
            if checkpoint_state is not None:
                checkpoint_state['global_step'] = global_step
            save_fn(f"step_{global_step}")

        # Logging using loss_fn's format helper
        if (batch_idx + 1) % log_every == 0:
            batch_time = (time.time() - start_time) / num_batches
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

            log_str = loss_fn.format_log_string(avg_metrics)
            print(f"  [{batch_idx + 1}/{len(dataloader)}] {log_str} lr={lr:.2e} {batch_time:.1f}s/b")

    epoch_time = time.time() - start_time
    result = {
        "time": epoch_time,
        "global_step": global_step,
    }

    # Add averaged metrics
    for key, value in total_metrics.items():
        result[key] = value / max(num_batches, 1)

    return result


@torch.no_grad()
def validate(
    noa: NOABackbone,
    loss_fn: BaseNOALoss,
    replayer: CNOReplayer,
    dataloader: DataLoader,
    device: str,
    timesteps: int,
    n_realizations: int = 1,
    bptt_window: int | None = None,
) -> dict:
    """Validate with abstracted loss computation."""
    noa.eval()
    total_metrics = {}
    num_batches = 0

    use_tbptt = bptt_window is not None and bptt_window < timesteps

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
                num_realizations=n_realizations,
                return_all_steps=True,
            )
            target_trajectories.append(target_traj)
        target_trajectory = torch.cat(target_trajectories, dim=0)

        if n_realizations > 1 and target_trajectory.dim() == 6:
            target_trajectory = target_trajectory.mean(dim=1)

        if use_tbptt:
            pred_states = pred_trajectory[:, -(bptt_window):, :, :, :]
            target_states = target_trajectory[:, -(bptt_window):, :, :, :]
        else:
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = target_trajectory[:, 1:, :, :, :]

        try:
            loss_output: LossOutput = loss_fn.compute(
                pred_trajectory=pred_states,
                target_trajectory=target_states,
                ic=ic,
                noa=noa,
            )
        except Exception:
            continue

        if not torch.isnan(loss_output.total):
            for key, value in loss_output.metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in total_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Train NOA with configurable loss modes")

    # Loss mode selection
    parser.add_argument(
        "--loss-mode", type=str, default="mse_led",
        choices=["mse_led", "vq_led"],
        help="Training paradigm: mse_led (physics first) or vq_led (creative observer)"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str,
        default="datasets/100k_full_features.h5",
        help="Path to HDF5 dataset"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/experiments/local_100k_optimized.yaml",
        help="Config file for CNO reconstruction"
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    # Model
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels")
    parser.add_argument("--encoder-levels", type=int, default=3, help="Encoder levels")
    parser.add_argument("--modes", type=int, default=16, help="Fourier modes")
    parser.add_argument("--afno-blocks", type=int, default=4, help="AFNO blocks")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--timesteps", type=int, default=32, help="Timesteps to supervise")
    parser.add_argument("--n-realizations", type=int, default=1, help="CNO realizations")
    parser.add_argument("--bptt-window", type=int, default=None, help="Truncated BPTT window")

    # LR scheduling
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "constant"])

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/noa", help="Checkpoint dir")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--save-every", type=int, default=1000, help="Save interval (batches)")
    parser.add_argument("--log-every", type=int, default=1, help="Log interval (batches)")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="Early stop patience")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # VQ-VAE alignment (required for vq_led)
    parser.add_argument("--vqvae-path", type=str, default=None, help="VQ-VAE checkpoint")

    # Loss weights
    parser.add_argument("--lambda-traj", type=float, default=1.0, help="Trajectory loss weight")
    parser.add_argument("--lambda-recon", type=float, default=1.0, help="VQ recon loss weight (vq_led)")
    parser.add_argument("--lambda-commit", type=float, default=0.5, help="Commitment loss weight")
    parser.add_argument("--lambda-latent", type=float, default=0.1, help="Latent loss weight (mse_led)")
    parser.add_argument("--enable-latent-loss", action="store_true", help="Enable L_latent (mse_led only)")
    parser.add_argument("--latent-sample-steps", type=int, default=3, help="Latent sampling steps")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print(f"NOA Unified Training - {args.loss_mode.upper()} Mode")
    print("=" * 60)

    if args.loss_mode == "mse_led":
        print("Paradigm: Physics First (MSE-led)")
        print("  Primary: L_traj (trajectory matching)")
        print("  Auxiliary: L_commit, L_latent")
    else:
        print("Paradigm: Creative Observer (VQ-led)")
        print("  Primary: L_recon (VQ reconstruction)")
        print("  Auxiliary: L_commit, L_traj (regularizer)")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Device: {device}")

    # CNO replayer
    print(f"\nLoading CNO replayer from: {args.config}")
    replayer = CNOReplayer.from_config(args.config, device=device, cache_size=8)

    # Dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = NOAStateDataset(args.dataset, n_samples=args.n_samples)
    print(f"  IC shape: {dataset.ics[0].shape}")
    print(f"  Params shape: {dataset.params[0].shape}")

    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # NOA backbone
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
        use_checkpointing=True,
    ).to(device)
    print(f"  Parameters: {noa.num_parameters:,}")

    # VQ-VAE alignment
    alignment = None
    if args.vqvae_path is not None:
        print(f"\nLoading VQ-VAE alignment from: {args.vqvae_path}")
        enable_latent = args.enable_latent_loss and args.loss_mode == "mse_led"
        alignment = VQVAEAlignmentLoss.from_checkpoint(
            vqvae_path=args.vqvae_path,
            device=device,
            noa=noa,
            enable_latent_loss=enable_latent,
            latent_sample_steps=args.latent_sample_steps,
        )
        print(f"  VQ-VAE alignment enabled")
    elif args.loss_mode == "vq_led":
        raise ValueError("vq_led mode requires --vqvae-path")

    # Create loss function based on mode
    print(f"\nLoss configuration:")
    if args.loss_mode == "mse_led":
        loss_fn = MSELedLoss(
            lambda_traj=args.lambda_traj,
            lambda_commit=args.lambda_commit,
            lambda_latent=args.lambda_latent,
            vqvae_alignment=alignment,
        )
        print(f"  MSELedLoss:")
        print(f"    λ_traj: {args.lambda_traj} (primary)")
        print(f"    λ_commit: {args.lambda_commit}")
        print(f"    λ_latent: {args.lambda_latent}")
    else:
        loss_fn = VQLedLoss(
            lambda_recon=args.lambda_recon,
            lambda_commit=args.lambda_commit,
            lambda_traj=args.lambda_traj,
            vqvae_alignment=alignment,
        )
        print(f"  VQLedLoss:")
        print(f"    λ_recon: {args.lambda_recon} (primary)")
        print(f"    λ_commit: {args.lambda_commit}")
        print(f"    λ_traj: {args.lambda_traj} (regularizer)")

    # Optimizer
    optimizer = torch.optim.AdamW(noa.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
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

    # Checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _checkpoint_state = {
        'epoch': 0,
        'global_step': 0,
        'history': None,
        'best_val_loss': float('inf'),
        'alignment': alignment,
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
            "loss_mode": args.loss_mode,
            "config": {
                'base_channels': args.base_channels,
                'encoder_levels': args.encoder_levels,
                'modes': args.modes,
                'afno_blocks': args.afno_blocks,
            },
            "args": vars(args),
        }

        if _checkpoint_state['alignment'] is not None and _checkpoint_state['alignment'].latent_projector is not None:
            checkpoint["alignment_state"] = _checkpoint_state['alignment'].latent_projector.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    history = {"train_loss": [], "val_loss": []}

    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        noa.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        history = checkpoint.get('history', {"train_loss": [], "val_loss": []})
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"  Resumed from epoch {start_epoch + 1}, step {global_step}")

        if alignment is not None and alignment.latent_projector is not None:
            if 'alignment_state' in checkpoint:
                alignment.latent_projector.load_state_dict(checkpoint['alignment_state'])

    _checkpoint_state['alignment'] = alignment
    _checkpoint_state['history'] = history
    _checkpoint_state['best_val_loss'] = best_val_loss
    _checkpoint_state['global_step'] = global_step

    # Training loop
    print(f"\nTraining:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  timesteps: {args.timesteps}")
    print(f"  bptt_window: {args.bptt_window or 'None (full backprop)'}")
    print(f"  lr_schedule: {args.lr_schedule}")
    print(f"  checkpoint_dir: {args.checkpoint_dir}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_result = train_epoch(
            noa=noa,
            loss_fn=loss_fn,
            replayer=replayer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            timesteps=args.timesteps,
            n_realizations=args.n_realizations,
            bptt_window=args.bptt_window,
            scheduler=scheduler,
            save_fn=save_checkpoint,
            save_every=args.save_every,
            global_step=global_step,
            log_every=args.log_every,
            checkpoint_state=_checkpoint_state,
        )
        global_step = train_result["global_step"]
        history["train_loss"].append(train_result.get("total", 0.0))

        val_result = validate(
            noa=noa,
            loss_fn=loss_fn,
            replayer=replayer,
            dataloader=val_loader,
            device=device,
            timesteps=args.timesteps,
            n_realizations=args.n_realizations,
            bptt_window=args.bptt_window,
        )
        history["val_loss"].append(val_result.get("total", 0.0))

        # Log using loss function's format
        train_log = loss_fn.format_log_string(train_result)
        val_log = loss_fn.format_log_string(val_result)
        print(f"  Train: {train_log} [{train_result['time']:.1f}s]")
        print(f"  Val:   {val_log}")

        _checkpoint_state['epoch'] = epoch
        _checkpoint_state['global_step'] = global_step
        _checkpoint_state['history'] = history

        val_loss = val_result.get("total", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _checkpoint_state['best_val_loss'] = best_val_loss
            print(f"  New best! (val_loss={best_val_loss:.6f})")
            save_checkpoint("best_model")
        else:
            epochs_without_improvement += 1

        save_checkpoint(f"epoch_{epoch + 1}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"\n  Early stopping: no improvement for {args.early_stop_patience} epochs")
            break

        replayer.clear_cache()

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Loss mode: {args.loss_mode}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
