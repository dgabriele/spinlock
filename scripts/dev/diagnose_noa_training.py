#!/usr/bin/env python
"""Diagnostic script to analyze NaN gradients in NOA training.

This script loads a checkpoint and provides detailed transparency into:
1. Which specific layers/parameters have NaN gradients
2. Gradient magnitudes and statistics across the model
3. Trajectory value ranges and potential numerical issues
4. Feature extraction outputs for debugging

Usage:
    poetry run python scripts/dev/diagnose_noa_training.py \
        --checkpoint checkpoints/noa/step_500.pt \
        --dataset datasets/100k_full_features.h5 \
        --config configs/experiments/local_100k_optimized.yaml \
        --n-samples 5
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.noa import NOABackbone, CNOReplayer, VQVAEAlignmentLoss
import h5py
from torch.utils.data import Dataset


class NOAStateDataset(Dataset):
    """Dataset that loads ICs and parameter vectors for CNO replay."""

    def __init__(
        self,
        dataset_path: str,
        n_samples: int | None = None,
        realization_idx: int = 0,
    ):
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


def analyze_tensor(name: str, tensor: torch.Tensor) -> dict:
    """Analyze a tensor for NaN/Inf and compute statistics."""
    stats = {
        "name": name,
        "shape": tuple(tensor.shape),
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "min": tensor.min().item() if not torch.isnan(tensor).any() else float('nan'),
        "max": tensor.max().item() if not torch.isnan(tensor).any() else float('nan'),
        "mean": tensor.mean().item() if not torch.isnan(tensor).any() else float('nan'),
        "std": tensor.std().item() if not torch.isnan(tensor).any() else float('nan'),
    }

    if stats["has_nan"]:
        stats["nan_count"] = torch.isnan(tensor).sum().item()
        stats["nan_percentage"] = 100 * stats["nan_count"] / tensor.numel()

    if stats["has_inf"]:
        stats["inf_count"] = torch.isinf(tensor).sum().item()
        stats["inf_percentage"] = 100 * stats["inf_count"] / tensor.numel()

    return stats


def diagnose_forward_pass(
    noa: NOABackbone,
    replayer: CNOReplayer,
    ic: torch.Tensor,
    params: torch.Tensor,
    timesteps: int,
    device: str,
    alignment: VQVAEAlignmentLoss = None,
):
    """Run a forward pass and analyze all intermediate tensors."""
    print("\n" + "=" * 80)
    print("FORWARD PASS DIAGNOSTICS")
    print("=" * 80)

    # Generate NOA trajectory
    print("\n1. NOA TRAJECTORY GENERATION")
    pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)

    stats = analyze_tensor("pred_trajectory", pred_trajectory)
    print(f"  Shape: {stats['shape']}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

    if stats['has_nan'] or stats['has_inf']:
        print(f"  ⚠️  WARNING: Trajectory contains NaN/Inf!")
        if stats['has_nan']:
            print(f"     NaN: {stats['nan_count']} values ({stats['nan_percentage']:.2f}%)")
        if stats['has_inf']:
            print(f"     Inf: {stats['inf_count']} values ({stats['inf_percentage']:.2f}%)")

    # Generate CNO target trajectory
    print("\n2. CNO TARGET TRAJECTORY")
    target_trajectory = replayer.rollout(
        params_vector=params[0].numpy(),
        ic=ic[0:1],
        timesteps=timesteps,
        num_realizations=1,
        return_all_steps=True,
    )

    stats = analyze_tensor("target_trajectory", target_trajectory)
    print(f"  Shape: {stats['shape']}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

    # Compute state loss
    print("\n3. STATE-LEVEL MSE LOSS")
    pred_states = pred_trajectory[:, 1:, :, :, :]
    target_states = target_trajectory[:, 1:, :, :, :]
    state_loss = F.mse_loss(pred_states, target_states)
    print(f"  State loss: {state_loss.item():.6f}")
    print(f"  Has NaN: {torch.isnan(state_loss).item()}")
    print(f"  Has Inf: {torch.isinf(state_loss).item()}")

    # Analyze VQ-VAE alignment if enabled
    if alignment is not None:
        print("\n4. VQ-VAE ALIGNMENT FEATURES")
        try:
            # Extract features
            pred_result = alignment.feature_extractor(pred_states, ic=ic)

            if isinstance(pred_result, tuple):
                pred_features, pred_raw_ics = pred_result
            else:
                pred_features = pred_result

            stats = analyze_tensor("pred_features", pred_features)
            print(f"  Shape: {stats['shape']}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

            if stats['has_nan'] or stats['has_inf']:
                print(f"  ⚠️  WARNING: Features contain NaN/Inf!")
                if stats['has_nan']:
                    print(f"     NaN: {stats['nan_count']} values ({stats['nan_percentage']:.2f}%)")
                if stats['has_inf']:
                    print(f"     Inf: {stats['inf_count']} values ({stats['inf_percentage']:.2f}%)")

            # Normalize features
            print("\n5. FEATURE NORMALIZATION")
            pred_norm = alignment._normalize_features(pred_features)

            stats = analyze_tensor("pred_norm", pred_norm)
            print(f"  Shape: {stats['shape']}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

            if stats['has_nan'] or stats['has_inf']:
                print(f"  ⚠️  WARNING: Normalized features contain NaN/Inf!")

            # VQ encoding
            print("\n6. VQ-VAE ENCODING")
            if alignment._is_hybrid_model and pred_raw_ics is not None:
                z_pred_list = alignment.vqvae.encode(pred_norm, raw_ics=pred_raw_ics)
            else:
                z_pred_list = alignment.vqvae.encode(pred_norm)

            z_pred = torch.cat(z_pred_list, dim=1)

            stats = analyze_tensor("z_pred (pre-quant)", z_pred)
            print(f"  Shape: {stats['shape']}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

            # VQ quantization
            print("\n7. VQ-VAE QUANTIZATION")
            z_q_pred_list, _, _ = alignment.vqvae.quantize(z_pred_list)
            z_q_pred = torch.cat(z_q_pred_list, dim=1)

            stats = analyze_tensor("z_q_pred (quantized)", z_q_pred)
            print(f"  Shape: {stats['shape']}")
            print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"  Mean±Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"  NaN: {stats['has_nan']}, Inf: {stats['has_inf']}")

            # Commitment loss
            print("\n8. COMMITMENT LOSS")
            commit_loss = F.mse_loss(z_pred, z_q_pred.detach())
            print(f"  Commit loss: {commit_loss.item():.6f}")
            print(f"  Has NaN: {torch.isnan(commit_loss).item()}")
            print(f"  Has Inf: {torch.isinf(commit_loss).item()}")

        except Exception as e:
            print(f"  ⚠️  ERROR during VQ-VAE alignment: {e}")
            import traceback
            traceback.print_exc()


def diagnose_backward_pass(
    noa: NOABackbone,
    replayer: CNOReplayer,
    ic: torch.Tensor,
    params: torch.Tensor,
    timesteps: int,
    device: str,
    alignment: VQVAEAlignmentLoss = None,
    lambda_commit: float = 0.5,
):
    """Run a backward pass and analyze gradients."""
    print("\n" + "=" * 80)
    print("BACKWARD PASS DIAGNOSTICS")
    print("=" * 80)

    # Forward pass
    pred_trajectory = noa(ic, steps=timesteps, return_all_steps=True)
    target_trajectory = replayer.rollout(
        params_vector=params[0].numpy(),
        ic=ic[0:1],
        timesteps=timesteps,
        num_realizations=1,
        return_all_steps=True,
    )

    # Compute loss
    pred_states = pred_trajectory[:, 1:, :, :, :]
    target_states = target_trajectory[:, 1:, :, :, :]
    state_loss = F.mse_loss(pred_states, target_states)
    loss = state_loss

    # Add VQ alignment if enabled
    commit_loss = torch.tensor(0.0, device=device)
    if alignment is not None:
        try:
            align_losses = alignment.compute_losses(
                pred_trajectory=pred_states,
                target_trajectory=target_states,
                ic=ic,
            )
            commit_loss = align_losses['commit']
            loss = loss + lambda_commit * commit_loss
        except Exception as e:
            print(f"  ⚠️  ERROR: VQ-VAE alignment failed: {e}")

    print(f"\nTotal loss: {loss.item():.6f}")
    print(f"  State loss: {state_loss.item():.6f}")
    print(f"  Commit loss: {commit_loss.item():.6f}")

    # Backward pass
    print("\nComputing gradients...")
    noa.zero_grad()
    loss.backward()

    # Analyze gradients
    print("\n" + "=" * 80)
    print("GRADIENT ANALYSIS")
    print("=" * 80)

    nan_grads = []
    inf_grads = []
    large_grads = []

    for name, param in noa.named_parameters():
        if param.grad is not None:
            grad = param.grad

            has_nan = torch.isnan(grad).any().item()
            has_inf = torch.isinf(grad).any().item()

            if has_nan:
                nan_count = torch.isnan(grad).sum().item()
                nan_pct = 100 * nan_count / grad.numel()
                nan_grads.append((name, nan_count, nan_pct, grad.shape))

            if has_inf:
                inf_count = torch.isinf(grad).sum().item()
                inf_pct = 100 * inf_count / grad.numel()
                inf_grads.append((name, inf_count, inf_pct, grad.shape))

            if not (has_nan or has_inf):
                grad_norm = grad.norm().item()
                if grad_norm > 1000:
                    large_grads.append((name, grad_norm, grad.shape))

    if nan_grads:
        print(f"\n🔴 FOUND {len(nan_grads)} PARAMETERS WITH NaN GRADIENTS:")
        for name, count, pct, shape in nan_grads[:20]:
            print(f"  {name}")
            print(f"    Shape: {shape}")
            print(f"    NaN: {count} values ({pct:.2f}%)")
    else:
        print("\n✅ No NaN gradients found")

    if inf_grads:
        print(f"\n🔴 FOUND {len(inf_grads)} PARAMETERS WITH Inf GRADIENTS:")
        for name, count, pct, shape in inf_grads[:20]:
            print(f"  {name}")
            print(f"    Shape: {shape}")
            print(f"    Inf: {count} values ({pct:.2f}%)")
    else:
        print("\n✅ No Inf gradients found")

    if large_grads:
        print(f"\n⚠️  FOUND {len(large_grads)} PARAMETERS WITH LARGE GRADIENTS (>1000):")
        for name, norm, shape in sorted(large_grads, key=lambda x: -x[1])[:10]:
            print(f"  {name}: norm={norm:.2e}, shape={shape}")

    # Gradient norm statistics
    print("\n" + "=" * 80)
    print("GRADIENT NORM STATISTICS")
    print("=" * 80)

    grad_norms = []
    for name, param in noa.named_parameters():
        if param.grad is not None and not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
            grad_norms.append(param.grad.norm().item())

    if grad_norms:
        grad_norms = np.array(grad_norms)
        print(f"\nGradient norms across {len(grad_norms)} parameters:")
        print(f"  Min: {grad_norms.min():.6e}")
        print(f"  Max: {grad_norms.max():.6e}")
        print(f"  Mean: {grad_norms.mean():.6e}")
        print(f"  Median: {np.median(grad_norms):.6e}")
        print(f"  Std: {grad_norms.std():.6e}")
        print(f"  95th percentile: {np.percentile(grad_norms, 95):.6e}")
        print(f"  99th percentile: {np.percentile(grad_norms, 99):.6e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose NOA training NaN issues")

    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint to diagnose"
    )
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
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override timesteps (default: use checkpoint value)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("=" * 80)
    print("NOA TRAINING DIAGNOSTIC TOOL")
    print("=" * 80)

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = checkpoint['args']

    print(f"\nCheckpoint training configuration:")
    print(f"  n_samples: {ckpt_args['n_samples']}")
    print(f"  batch_size: {ckpt_args['batch_size']}")
    print(f"  timesteps: {ckpt_args['timesteps']}")
    print(f"  bptt_window: {ckpt_args.get('bptt_window', 'None')}")
    print(f"  vqvae_path: {ckpt_args.get('vqvae_path', 'None')}")
    print(f"  lambda_commit: {ckpt_args.get('lambda_commit', 0.0)}")

    # Check for TBPTT issue
    timesteps = args.timesteps if args.timesteps is not None else ckpt_args['timesteps']
    bptt_window = ckpt_args.get('bptt_window')

    print("\n" + "=" * 80)
    print("CONFIGURATION CHECK")
    print("=" * 80)

    if timesteps > 32 and bptt_window is None:
        print("\n🔴 CRITICAL ISSUE DETECTED:")
        print(f"  Training with timesteps={timesteps} WITHOUT truncated BPTT!")
        print(f"  This WILL cause gradient explosion and NaN gradients.")
        print(f"\n  SOLUTION: Add --bptt-window 32 to training command")
        print(f"\n  Recommended command:")
        print(f"  poetry run python scripts/dev/train_noa_state_supervised.py \\")
        print(f"      --dataset {ckpt_args['dataset']} \\")
        print(f"      --vqvae-path {ckpt_args.get('vqvae_path', 'checkpoints/production/100k_3family_v1')} \\")
        print(f"      --n-samples {ckpt_args['n_samples']} \\")
        print(f"      --epochs {ckpt_args['epochs']} \\")
        print(f"      --batch-size {ckpt_args['batch_size']} \\")
        print(f"      --lr {ckpt_args['lr']} \\")
        print(f"      --warmup-steps {ckpt_args.get('warmup_steps', 500)} \\")
        print(f"      --timesteps {timesteps} \\")
        print(f"      --bptt-window 32 \\  # <-- ADD THIS!")
        print(f"      --save-every {ckpt_args.get('save_every', 100)} \\")
        print(f"      --early-stop-patience {ckpt_args.get('early_stop_patience', 1)}")
    elif timesteps > 32 and bptt_window is not None:
        print(f"\n✅ Configuration OK: Using TBPTT with window={bptt_window}")
    else:
        print(f"\n✅ Configuration OK: Short sequences (T={timesteps}) don't require TBPTT")

    # Create CNO replayer
    print(f"\nLoading CNO replayer from: {args.config}")
    replayer = CNOReplayer.from_config(args.config, device=device, cache_size=8)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = NOAStateDataset(args.dataset, n_samples=args.n_samples)

    # Create NOA backbone
    print(f"\nCreating NOA backbone")
    noa = NOABackbone(
        in_channels=1,
        out_channels=1,
        base_channels=ckpt_args['base_channels'],
        encoder_levels=ckpt_args['encoder_levels'],
        modes=ckpt_args['modes'],
        afno_blocks=ckpt_args['afno_blocks'],
    ).to(device)

    # Load weights
    noa.load_state_dict(checkpoint['model_state_dict'])
    noa.train()  # Set to training mode

    # Load VQ-VAE alignment if enabled
    alignment = None
    if ckpt_args.get('vqvae_path') is not None:
        print(f"\nLoading VQ-VAE alignment from: {ckpt_args['vqvae_path']}")
        try:
            alignment = VQVAEAlignmentLoss.from_checkpoint(
                vqvae_path=ckpt_args['vqvae_path'],
                device=device,
            )
            print("  VQ-VAE alignment enabled")
        except Exception as e:
            print(f"  ⚠️  Failed to load VQ-VAE: {e}")

    # Get test sample
    sample = dataset[0]
    ic = sample['ic'].unsqueeze(0).to(device)
    params = sample['params'].unsqueeze(0)

    # Run diagnostics
    with torch.enable_grad():
        diagnose_forward_pass(noa, replayer, ic, params, timesteps, device, alignment)
        diagnose_backward_pass(
            noa, replayer, ic, params, timesteps, device, alignment,
            lambda_commit=ckpt_args.get('lambda_commit', 0.5)
        )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
