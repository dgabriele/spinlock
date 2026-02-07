#!/usr/bin/env python3
"""
Generate CNO reference features from MNO-stored ICs and parameters.

For VQ-VAE training with reference feature regularization, this script:
1. Loads ICs and parameters from MNO feature dataset
2. Generates CNO rollouts using those exact (params, IC) pairs
3. Extracts per_trajectory features from CNO trajectories
4. Reshapes from [N, M, D] to [N, M*D] (matching MNO format)
5. Stores as reference features in MNO dataset

This ensures reference features correspond to the EXACT same problem instances
as MNO features (same parameters + same initial conditions).

Note: Uses per_trajectory features instead of aggregated to bypass corrupted
aggregated features (std/cv blocks have near-zero variance across realizations).

Usage:
    python scripts/generate_cno_reference_features.py \
        --mno-dataset datasets/noa_features/mno_v3_10k.h5 \
        --config configs/experiments/local_100k_optimized.yaml \
        --device cuda \
        --batch-size 16
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

from spinlock.mno.cno_replay import CNOReplayer
from spinlock.features.summary.extractors import SummaryExtractor
from spinlock.features.summary.config import SummaryConfig


def generate_cno_reference_features(
    mno_dataset_path: Path,
    config_path: Path,
    device: str = "cuda",
    batch_size: int = 16,
    timesteps: int = 256,
) -> int:
    """Generate CNO reference features for MNO dataset.

    Args:
        mno_dataset_path: Path to MNO feature dataset
        config_path: Path to experiment config (for parameter space)
        device: Computation device
        batch_size: Batch size for CNO rollout generation
        timesteps: Number of timesteps for CNO rollout

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Validate inputs
    if not mno_dataset_path.exists():
        print(f"Error: MNO dataset not found: {mno_dataset_path}", file=sys.stderr)
        return 1

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        return 1

    print(f"{'='*70}")
    print(f"CNO REFERENCE FEATURE GENERATION")
    print(f"{'='*70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Timesteps: {timesteps}\n")

    # Load MNO dataset
    print("\nLoading MNO dataset...")
    with h5py.File(mno_dataset_path, 'r') as f:
        # Check required fields
        required_fields = [
            'inputs/fields',
            'features/summary/per_trajectory/features',
        ]

        for field in required_fields:
            if field not in f:
                print(f"Error: MNO dataset missing '{field}'", file=sys.stderr)
                return 1

        # Check parameters (can be at two locations)
        has_params = ('parameters/params' in f) or ('features/parameters' in f)
        if not has_params:
            print("Error: MNO dataset missing parameters", file=sys.stderr)
            return 1

        # Load MNO data
        mno_ics = f['inputs/fields'][:]  # [N, M, H, W] - all realizations

        # Parameters can be at /parameters/params or /features/parameters
        if 'parameters/params' in f:
            mno_params = f['parameters/params'][:]  # [N, D]
        else:
            mno_params = f['features/parameters'][:]  # [N, D]

        # Get MNO per_trajectory shape and compute expected reshaped dimension
        mno_per_traj_shape = f['features/summary/per_trajectory/features'].shape
        n_samples = mno_ics.shape[0]
        num_realizations = mno_ics.shape[1]

        # Expected shape after reshape: [N, M*D]
        N, M, D = mno_per_traj_shape
        expected_dim = M * D
        mno_feature_shape = (N, expected_dim)

        print(f"  MNO samples: {n_samples}")
        print(f"  MNO num_realizations: {num_realizations}")
        print(f"  MNO per_trajectory shape: {mno_per_traj_shape} -> reshaped to {mno_feature_shape}")
        print(f"  Target feature dimension: {expected_dim}")
        print(f"  IC shape: {mno_ics.shape}")
        print(f"  Parameter shape: {mno_params.shape}")

    # Initialize CNO replayer
    print(f"\nInitializing CNO replayer...")
    try:
        replayer = CNOReplayer.from_config(
            config_path=str(config_path),
            device=device,
            cache_size=8,
        )
        print(f"  ✓ CNO replayer initialized")
    except Exception as e:
        print(f"Error initializing CNO replayer: {e}", file=sys.stderr)
        return 1

    # Initialize feature extractor
    print(f"\nInitializing feature extractor...")
    try:
        # Use default SummaryConfig (matches MNO generation)
        feature_config = SummaryConfig()
        feature_extractor = SummaryExtractor(
            device=torch.device(device),
            config=feature_config,
        )
        print(f"  ✓ Feature extractor initialized")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}", file=sys.stderr)
        return 1

    # Allocate output array
    reference_features = np.zeros((n_samples, mno_feature_shape[1]), dtype=np.float32)

    # Generate CNO rollouts and extract features
    print(f"\nGenerating {n_samples} CNO rollouts and extracting features...")
    print(f"  This will take approximately {n_samples * 0.05 / 60:.1f} minutes (~50ms/rollout)\n")

    start_time = time.time()
    n_batches = (n_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="CNO generation", unit="batch"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_samples)
            current_batch_size = batch_end - batch_start

            try:
                # Get batch data
                batch_params = mno_params[batch_start:batch_end]  # [B, D]
                batch_ics = mno_ics[batch_start:batch_end]  # [B, M, H, W]

                # Convert to torch: [B, M, H, W]
                batch_ics_torch = torch.from_numpy(batch_ics).float().to(device)

                # Generate CNO rollouts for each sample and each realization
                batch_trajectories = []
                for i in range(current_batch_size):
                    sample_realizations = []
                    for realization_idx in range(num_realizations):
                        # Get IC for this realization: [H, W]
                        ic = batch_ics_torch[i, realization_idx]  # [H, W]
                        # Add channel dimension: [1, H, W]
                        ic = ic.unsqueeze(0)

                        # Generate trajectory: [1, T+1, 1, H, W] (includes IC at t=0)
                        trajectory = replayer.rollout(
                            params_vector=batch_params[i],
                            ic=ic,
                            timesteps=timesteps,
                            num_realizations=1,
                            return_all_steps=True,
                        )
                        sample_realizations.append(trajectory)

                    # Stack realizations: [M, 1, T+1, 1, H, W]
                    sample_trajectories = torch.stack(sample_realizations, dim=0)  # [M, 1, T+1, 1, H, W]
                    # Squeeze the num_realizations=1 dimension: [M, T+1, 1, H, W]
                    sample_trajectories = sample_trajectories.squeeze(1)  # [M, T+1, 1, H, W]
                    batch_trajectories.append(sample_trajectories)

                # Stack batch: [B, M, T+1, 1, H, W]
                batch_trajectories = torch.stack(batch_trajectories, dim=0)

                # Extract features
                # SummaryExtractor expects: [N, M, T, C, H, W]
                # We have: [B, M, T+1, C, H, W] - includes IC at t=0
                # Remove IC to get: [B, M, T, C, H, W]
                batch_trajectories = batch_trajectories[:, :, 1:, :, :]  # Skip t=0 (IC)

                # Extract per_trajectory features (bypasses corrupted aggregated features)
                # per_trajectory: [B, M, D] where M=num_realizations (matching MNO)
                per_trajectory = feature_extractor.extract_per_trajectory(batch_trajectories)

                # Reshape from [B, M, D] to [B, M*D]
                B, M, D = per_trajectory.shape
                batch_features = per_trajectory.reshape(B, M * D).cpu().numpy()  # [B, M*D]

                # Store
                reference_features[batch_start:batch_end] = batch_features

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}", file=sys.stderr)
                print(f"  Samples {batch_start}-{batch_end}", file=sys.stderr)
                return 1

    elapsed = time.time() - start_time
    print(f"\n✓ CNO generation complete: {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/n_samples*1000:.1f}ms per rollout")

    # Validate feature dimensions
    print(f"\nValidating feature dimensions...")
    print(f"  Expected (MNO per_trajectory reshaped): {mno_feature_shape[1]}")
    print(f"  Generated (CNO per_trajectory reshaped): {reference_features.shape[1]}")

    if reference_features.shape[1] != mno_feature_shape[1]:
        print(f"\nERROR: Feature dimension mismatch!", file=sys.stderr)
        print(f"  MNO and CNO per_trajectory features must have same dimensions.", file=sys.stderr)
        print(f"  This indicates different feature extraction configs were used.", file=sys.stderr)
        return 1

    # Store in MNO dataset
    print(f"\nStoring reference features in MNO dataset...")
    try:
        with h5py.File(mno_dataset_path, 'a') as f:
            # Check if already exists
            if 'features/reference_features' in f:
                print("  Warning: 'features/reference_features' already exists, overwriting...")
                del f['features/reference_features']

            # Create dataset
            f.create_dataset(
                'features/reference_features',
                data=reference_features,
                compression='gzip',
                compression_opts=4,
            )

            # Add metadata
            f['features/reference_features'].attrs['description'] = (
                f'CNO per_trajectory reference features (reshaped to [N, M*D]) generated from MNO ICs and parameters. '
                f'{n_samples} samples generated with exact IC/param matching. '
                f'Uses per_trajectory features to bypass corrupted aggregated features.'
            )
            f['features/reference_features'].attrs['config_path'] = str(config_path)
            f['features/reference_features'].attrs['timesteps'] = timesteps
            f['features/reference_features'].attrs['generation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            f['features/reference_features'].attrs['elapsed_seconds'] = elapsed

        print(f"  ✓ Reference features stored")
    except Exception as e:
        print(f"Error storing features: {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 70}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference features stored at: 'features/reference_features'")
    print(f"Shape: {reference_features.shape}")
    print(f"Samples: {n_samples}")
    print(f"Generation time: {elapsed/60:.1f} minutes")
    print(f"Average per sample: {elapsed/n_samples*1000:.1f}ms")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate CNO reference features from MNO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mno-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to MNO feature dataset (.h5 file)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to experiment config (for parameter space)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        metavar="DEVICE",
        help="Computation device (default: cuda)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Batch size for rollout generation (default: 16)",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=256,
        metavar="T",
        help="Number of timesteps for CNO rollout (default: 256)",
    )

    args = parser.parse_args()

    return generate_cno_reference_features(
        args.mno_dataset,
        args.config,
        args.device,
        args.batch_size,
        args.timesteps,
    )


if __name__ == "__main__":
    sys.exit(main())
