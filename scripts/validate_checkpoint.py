#!/usr/bin/env python3
"""Run validation on a saved checkpoint to diagnose issues."""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlock.data import SpinlockDataset
from spinlock.mno import MNOBackbone, TruncatedBPTT, CNOReplayer
from spinlock.mno.losses import MSELedLoss


def load_checkpoint(checkpoint_path):
    """Load checkpoint and return state dict."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def create_dataloader(dataset_path, batch_size=2, val_split=0.1, num_workers=4, max_samples=10000):
    """Create validation dataloader."""
    dataset = SpinlockDataset(dataset_path, max_samples=max_samples)
    total_samples = len(dataset)
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size

    _, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return val_loader


def validate_one_batch(mno, noa_rollout, loss_fn, replayer, batch, device, timesteps):
    """Run validation on a single batch with detailed error reporting."""
    mno.eval()

    with torch.no_grad():
        ic = batch["ic"].to(device)
        params = batch["params"].to(device)
        B = ic.shape[0]

        print(f"\n  Batch shapes: ic={ic.shape}, params={params.shape}")

        # Generate MNO rollout
        print("  Generating MNO rollout...")
        try:
            pred_trajectory = noa_rollout.rollout(ic, params=params, tokens=None)
            print(f"  ✓ MNO rollout shape: {pred_trajectory.shape}")
        except Exception as e:
            print(f"  ✗ MNO rollout failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Check if loss needs target trajectories
        needs_target = not hasattr(loss_fn, 'needs_target_trajectory') or loss_fn.needs_target_trajectory
        print(f"  Loss needs target: {needs_target}")

        if needs_target:
            # Generate ground truth trajectories
            print("  Generating ground truth rollout...")
            try:
                if hasattr(replayer, 'rollout_batch'):
                    target_trajectory = replayer.rollout_batch(
                        params_batch=params.cpu().numpy(),
                        num_realizations=1,
                        num_timesteps=timesteps,
                        timesteps=timesteps,
                        return_all_steps=True,
                    )
                    target_trajectory = target_trajectory.squeeze(1)
                else:
                    # Fallback: sequential rollout
                    target_trajectories = []
                    for b in range(B):
                        target_traj = replayer.rollout(
                            params_vector=params[b].cpu().numpy(),
                            ic=ic[b:b+1],
                            timesteps=timesteps,
                            num_realizations=1,
                            return_all_steps=True,
                        )
                        target_trajectories.append(target_traj)
                    target_trajectory = torch.cat(target_trajectories, dim=0)

                print(f"  ✓ Ground truth rollout shape: {target_trajectory.shape}")
            except Exception as e:
                print(f"  ✗ Ground truth rollout failed: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Align states
            print("  Aligning predicted and target states...")
            try:
                pred_states, target_states = noa_rollout.align_for_loss(
                    pred_trajectory,
                    target_trajectory,
                    skip_ic=True,
                )
                print(f"  ✓ Aligned: pred={pred_states.shape}, target={target_states.shape}")
            except Exception as e:
                print(f"  ✗ Alignment failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            pred_states = pred_trajectory[:, 1:, :, :, :]
            target_states = None
            print(f"  ✓ Using pred states only: {pred_states.shape}")

        # Compute loss
        print("  Computing loss...")
        try:
            loss_output = loss_fn.compute(
                pred_trajectory=pred_states.float() if pred_states is not None else None,
                target_trajectory=target_states,
                ic=ic,
                noa=mno,
                params=params,
            )
            print(f"  ✓ Loss computed: {loss_output.total.item():.6f}")
            print(f"    Components: {loss_output.components}")
            print(f"    Metrics: {loss_output.metrics}")
            return loss_output
        except Exception as e:
            print(f"  ✗ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(description="Validate a saved checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to training config")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to validate")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Checkpoint Validation")
    print(f"{'='*70}\n")

    # Load config
    print(f"Loading config: {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Keys: {list(checkpoint.keys())}")

    # Create dataset and dataloader
    dataset_path = config["data"]["dataset_path"]
    print(f"\nCreating validation dataloader from: {dataset_path}")
    val_loader = create_dataloader(
        dataset_path,
        batch_size=config["training"]["batch_size"],
        val_split=config["data"]["val_split"],
        num_workers=2,  # Reduce workers for debugging
        max_samples=config["training"].get("n_samples", 10000),
    )
    print(f"  Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating MNO model...")
    device = torch.device(args.device)

    # Infer dimensions from dataset
    spinlock_dataset, config = SpinlockDataset.infer_and_update_config(
        config_dict=config,
        dataset_path=Path(dataset_path),
        verbose=True
    )
    dims = spinlock_dataset.infer_mno_dimensions()
    print(f"  Auto-detected dimensions: {dims}")

    # Manually apply dimensions to config (infer_and_update_config bug)
    config["model"].update(dims["model"])

    # Create MNO model
    model_config = dict(config["model"])

    # Handle FiLM config transformation
    if "film" in model_config:
        model_config["film_config"] = model_config.pop("film")

    mno = MNOBackbone(**model_config).to(device)
    mno.load_state_dict(checkpoint["model_state_dict"])
    print(f"  ✓ Model loaded ({sum(p.numel() for p in mno.parameters()):,} parameters)")

    # Create rollout wrapper
    noa_rollout = TruncatedBPTT(
        model=mno,
        timesteps=config["training"]["timesteps"],
        bptt_window=config["training"]["bptt_window"],
    )

    # Create loss function
    print("\nCreating loss function...")
    loss_fn = MSELedLoss(
        lambda_traj=config["loss"]["lambda_traj"],
        lambda_ic=config["loss"]["lambda_ic"],
    )
    print(f"  ✓ Loss function: {type(loss_fn).__name__}")

    # Create replayer
    print("\nCreating ground truth replayer...")
    replayer_config_path = config["data"]["config"]
    replayer = CNOReplayer.from_config(
        config_path=replayer_config_path,
        device=device,
        cache_size=config["training"].get("replayer_cache_size", 4),
    )
    print(f"  ✓ Replayer: {type(replayer).__name__}")

    # Run validation on first few batches
    print(f"\n{'='*70}")
    print(f"Running validation on {args.num_batches} batch(es)")
    print(f"{'='*70}")

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n[Batch {batch_idx + 1}/{args.num_batches}]")
        result = validate_one_batch(
            mno=mno,
            noa_rollout=noa_rollout,
            loss_fn=loss_fn,
            replayer=replayer,
            batch=batch,
            device=device,
            timesteps=config["training"]["timesteps"],
        )

        if result is None:
            print(f"\n✗ Validation failed for batch {batch_idx}")
            return 1

    print(f"\n{'='*70}")
    print("✓ Validation successful")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
