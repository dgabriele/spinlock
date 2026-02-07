#!/usr/bin/env python3
"""
Generate CNO reference features using DRY infrastructure.

This script composes existing components to generate CNO reference features:
- CNOReplayer: Generate CNO rollouts
- SummaryExtractor: Extract features (SAME config as MNO)
- InitialManualExtractor: Extract IC features

NO custom feature extraction logic - everything uses existing components.

Usage:
    python scripts/generate_cno_reference_features_dry.py \
        --mno-dataset datasets/noa_features/mno_v3_10k.h5 \
        --cno-dataset datasets/local_100k_optimized.h5 \
        --config configs/experiments/local_100k_optimized.yaml \
        --device cuda \
        --batch-size 4
"""

import argparse
import sys
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Import existing components (DRY principle)
from spinlock.mno.cno_replay import CNOReplayer
from spinlock.features.summary.extractors import SummaryExtractor
from spinlock.features.summary.config import SummaryConfig
from spinlock.features.initial.manual_extractors import InitialManualExtractor
from spinlock.config import load_config


def generate_cno_reference_features(
    mno_dataset_path: Path,
    cno_dataset_path: Path,
    config_path: Path,
    device: str = "cuda",
    batch_size: int = 4,
    timesteps: int = 256,
) -> int:
    """Generate CNO reference features using existing infrastructure.

    Args:
        mno_dataset_path: Path to MNO feature dataset
        cno_dataset_path: Path to 100K CNO dataset
        config_path: Experiment config (for parameter space and feature config)
        device: cuda or cpu
        batch_size: Batch size for generation
        timesteps: Number of timesteps

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Validate inputs
    if not mno_dataset_path.exists():
        print(f"Error: MNO dataset not found: {mno_dataset_path}", file=sys.stderr)
        return 1
    if not cno_dataset_path.exists():
        print(f"Error: CNO dataset not found: {cno_dataset_path}", file=sys.stderr)
        return 1
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        return 1

    print(f"{'='*70}")
    print(f"CNO REFERENCE FEATURES (DRY INFRASTRUCTURE)")
    print(f"{'='*70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"CNO dataset: {cno_dataset_path}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}\n")

    # Load MNO metadata
    print("Loading MNO metadata...")
    with h5py.File(mno_dataset_path, 'r') as f:
        generation_indices = f['metadata/generation_indices'][:]
        num_realizations = f['inputs/fields'].shape[1]
        n_samples = len(generation_indices)
        print(f"  Generation indices: [0, 10, 20, ..., {generation_indices[-1]}]")
        print(f"  Num samples: {n_samples}")
        print(f"  Num realizations: {num_realizations}")

    # Filter to ONLY interpolated samples (not training samples)
    # MNO was trained on indices with stride=50: [0, 50, 100, ..., 99950]
    # We only want CNO features for interpolated samples (indices NOT divisible by 50)
    print("\nFiltering to interpolated samples only...")
    training_indices_set = set(range(0, 100000, 50))
    is_interpolated_mask = np.array([idx not in training_indices_set for idx in generation_indices])
    interpolated_indices = generation_indices[is_interpolated_mask]
    n_interpolated = len(interpolated_indices)
    n_training = n_samples - n_interpolated

    print(f"  Total samples: {n_samples}")
    print(f"  Training samples (stride=50): {n_training}")
    print(f"  Interpolated samples: {n_interpolated}")
    print(f"  Will generate CNO features for {n_interpolated} interpolated samples only")

    # Load CNO ICs and params ONLY for interpolated indices
    print("\nLoading CNO data for interpolated indices...")
    with h5py.File(cno_dataset_path, 'r') as f:
        cno_params = f['parameters/params'][interpolated_indices]
        cno_ics = f['inputs/fields'][interpolated_indices]
        print(f"  CNO params: {cno_params.shape}")
        print(f"  CNO ICs: {cno_ics.shape}")

    # Load experiment config
    print("\nLoading experiment config...")
    config = load_config(config_path)

    # Initialize CNO replayer (existing component)
    print("\nInitializing CNO replayer...")
    try:
        replayer = CNOReplayer.from_config(
            config_path=str(config_path),
            device=device,
            cache_size=8,
        )
        print(f"  ✓ CNOReplayer initialized")
    except Exception as e:
        print(f"Error initializing CNOReplayer: {e}", file=sys.stderr)
        return 1

    # Initialize feature extractors (existing components, SAME config as MNO)
    print("\nInitializing feature extractors (SAME config as MNO)...")
    try:
        summary_extractor = SummaryExtractor(
            device=torch.device(device),
            config=SummaryConfig.from_schema_config(config.features.summary),
        )
        initial_extractor = InitialManualExtractor(device=torch.device(device))
        print(f"  ✓ SummaryExtractor initialized")
        print(f"  ✓ InitialManualExtractor initialized")
    except Exception as e:
        print(f"Error initializing extractors: {e}", file=sys.stderr)
        return 1

    # Calculate expected dimensions
    initial_dim = 14
    summary_dim = num_realizations * 110  # 3 * 110 = 330
    temporal_dim = timesteps * 63  # 256 * 63 = 16128
    total_dim = initial_dim + summary_dim + temporal_dim  # 16472

    print(f"\nExpected feature dimensions:")
    print(f"  INITIAL: {initial_dim}D")
    print(f"  SUMMARY: {summary_dim}D ({num_realizations} realizations × 110)")
    print(f"  TEMPORAL: {temporal_dim}D ({timesteps} timesteps × 63)")
    print(f"  TOTAL: {total_dim}D")

    # Allocate output - FULL array for all samples (with zeros for training samples)
    reference_features_full = np.zeros((n_samples, total_dim), dtype=np.float32)
    # Temporary array for interpolated features only
    reference_features_interpolated = np.zeros((n_interpolated, total_dim), dtype=np.float32)

    # Generate CNO rollouts and extract features ONLY for interpolated samples
    print(f"\nGenerating {n_interpolated} CNO rollouts (interpolated samples only)...")
    print(f"  Estimated time: ~{n_interpolated * num_realizations * 0.05 / 60:.1f} minutes\n")

    start_time = time.time()
    n_batches = (n_interpolated + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="CNO generation", unit="batch"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_interpolated)
            current_batch_size = batch_end - batch_start

            try:
                # Get batch data
                batch_params = cno_params[batch_start:batch_end]
                batch_ics = cno_ics[batch_start:batch_end]

                # Convert to torch
                batch_ics_torch = torch.from_numpy(batch_ics).float().to(device)

                # Generate CNO rollouts for each sample and realization
                batch_trajectories = []
                for i in range(current_batch_size):
                    sample_realizations = []
                    for m in range(num_realizations):
                        ic = batch_ics_torch[i, m]  # [H, W]
                        ic = ic.unsqueeze(0)  # Add channel: [1, H, W]

                        # Use CNOReplayer (existing component)
                        trajectory = replayer.rollout(
                            params_vector=batch_params[i],
                            ic=ic,
                            timesteps=timesteps,
                            num_realizations=1,
                            return_all_steps=True,
                        )  # [1, T+1, 1, H, W]

                        sample_realizations.append(trajectory.squeeze(0))  # [T+1, 1, H, W]

                    batch_trajectories.append(torch.stack(sample_realizations))  # [M, T+1, 1, H, W]

                trajectories = torch.stack(batch_trajectories)  # [B, M, T+1, 1, H, W]
                trajectories = trajectories[:, :, 1:, :, :]  # Remove IC (t=0)

                # Extract INITIAL features (from ICs)
                batch_ics_for_initial = batch_ics_torch.unsqueeze(2)  # [B, M, 1, H, W]
                initial_per_realization = initial_extractor.extract_all(batch_ics_for_initial)  # [B, M, 14]
                initial_features = initial_per_realization.mean(dim=1)  # [B, 14]

                # Extract SUMMARY features (from trajectories)
                per_traj = summary_extractor.extract_per_trajectory(trajectories)  # [B, M, D]
                summary_features = per_traj.reshape(current_batch_size, -1)  # [B, M*D]

                # Extract TEMPORAL features (from mean trajectory)
                trajectories_mean = trajectories.mean(dim=1, keepdim=True)  # [B, 1, T, 1, H, W]
                per_timestep = summary_extractor.extract_per_timestep(trajectories_mean)  # [B, 1, T, D]
                temporal_features = per_timestep.squeeze(1).reshape(current_batch_size, -1)  # [B, T*D]

                # Concatenate all families
                batch_features = torch.cat([
                    initial_features,
                    summary_features,
                    temporal_features,
                ], dim=1)  # [B, 16472]

                reference_features_interpolated[batch_start:batch_end] = batch_features.cpu().numpy()

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}", file=sys.stderr)
                print(f"  Samples {batch_start}-{batch_end}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1

    elapsed = time.time() - start_time
    print(f"\n✓ CNO generation complete: {elapsed/60:.1f} minutes")
    print(f"  Average: {elapsed/n_interpolated*1000:.1f}ms per sample")

    # Fill interpolated positions in full array
    print(f"\nFilling full reference array...")
    interpolated_positions = np.where(is_interpolated_mask)[0]
    reference_features_full[interpolated_positions] = reference_features_interpolated
    print(f"  Filled {n_interpolated} interpolated positions (training positions remain zero)")

    # Validate dimensions
    print(f"\nValidating feature dimensions...")
    print(f"  Expected: {total_dim}D")
    print(f"  Generated: {reference_features_full.shape[1]}D")

    if reference_features_full.shape[1] != total_dim:
        print(f"\nERROR: Feature dimension mismatch!", file=sys.stderr)
        return 1

    # Store in MNO dataset
    print(f"\nStoring reference features and interpolation mask in MNO dataset...")
    try:
        with h5py.File(mno_dataset_path, 'a') as f:
            # Store reference features
            if 'features/reference_features' in f:
                print("  Warning: 'features/reference_features' exists, overwriting...")
                del f['features/reference_features']

            f.create_dataset(
                'features/reference_features',
                data=reference_features_full,
                compression='gzip',
                compression_opts=4,
            )

            # Add metadata
            f['features/reference_features'].attrs['description'] = (
                f'CNO reference features generated using DRY infrastructure. '
                f'Composes CNOReplayer + SummaryExtractor + InitialManualExtractor. '
                f'{n_samples} total samples ({n_interpolated} interpolated, {n_training} training). '
                f'Training samples have zero features (only interpolated samples generated).'
            )
            f['features/reference_features'].attrs['config_path'] = str(config_path)
            f['features/reference_features'].attrs['timesteps'] = timesteps
            f['features/reference_features'].attrs['generation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            f['features/reference_features'].attrs['elapsed_seconds'] = elapsed
            f['features/reference_features'].attrs['n_interpolated'] = n_interpolated
            f['features/reference_features'].attrs['n_training'] = n_training

            # Store is_interpolated mask
            if 'metadata/is_interpolated' in f:
                print("  Warning: 'metadata/is_interpolated' exists, overwriting...")
                del f['metadata/is_interpolated']

            f.create_dataset(
                'metadata/is_interpolated',
                data=is_interpolated_mask,
                compression='gzip',
                compression_opts=4,
            )

            f['metadata/is_interpolated'].attrs['description'] = (
                f'Boolean mask indicating which samples are interpolated (True) vs training (False). '
                f'Training samples: indices divisible by 50. Interpolated: all others.'
            )

        print(f"  ✓ Reference features stored ({n_interpolated} interpolated, {n_training} training zeros)")
        print(f"  ✓ Interpolation mask stored")
    except Exception as e:
        print(f"Error storing features: {e}", file=sys.stderr)
        return 1

    print(f"\n{'=' * 70}")
    print(f"GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference features: 'features/reference_features'")
    print(f"Interpolation mask: 'metadata/is_interpolated'")
    print(f"Shape: {reference_features_full.shape}")
    print(f"  Interpolated samples: {n_interpolated} (CNO features)")
    print(f"  Training samples: {n_training} (zeros)")
    print(f"Dimensions: INITIAL ({initial_dim}D) + SUMMARY ({summary_dim}D) + TEMPORAL ({temporal_dim}D) = {total_dim}D")
    print(f"Generation time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate CNO reference features using DRY infrastructure",
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
        "--cno-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to 100K CNO dataset (.h5 file)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to experiment config (for parameter space and feature config)",
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
        default=4,
        metavar="N",
        help="Batch size for rollout generation (default: 4)",
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
        args.cno_dataset,
        args.config,
        args.device,
        args.batch_size,
        args.timesteps,
    )


if __name__ == "__main__":
    sys.exit(main())
