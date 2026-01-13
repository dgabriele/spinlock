#!/usr/bin/env python3
"""
Extract reference features from full dataset or generate via CNO for MNO dataset alignment.

For VQ-VAE training with reference feature regularization, this script:
1. Checks if reference dataset has matching (params, IC) pairs
2. If match found: extracts features directly (fast)
3. If no match: generates CNO rollout and extracts features (slower)

This ensures reference features correspond to the EXACT same problem instances
(same parameters + same initial conditions) as MNO features.

Usage:
    python scripts/extract_reference_features.py \\
        --mno-dataset datasets/noa_features/mno_v3_10k.h5 \\
        --reference-dataset datasets/local_100k_optimized.h5 \\
        --config configs/experiments/local_100k_optimized.yaml \\
        --device cuda
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add spinlock to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlock.noa.cno_replay import CNOReplayer
from spinlock.features.extraction import SummaryExtractor
from spinlock.features.config import SummaryConfig
from spinlock.config import ExperimentConfig


def find_matching_sample(
    target_params: np.ndarray,
    target_ic: np.ndarray,
    reference_params: np.ndarray,
    reference_ics: np.ndarray,
    param_tol: float = 1e-6,
    ic_tol: float = 1e-6,
) -> Optional[int]:
    """Find reference sample matching target params and IC.

    Args:
        target_params: Target parameter vector [D]
        target_ic: Target initial condition [H, W]
        reference_params: Reference parameter array [N, D]
        reference_ics: Reference IC array [N, M, H, W]
        param_tol: Tolerance for parameter matching
        ic_tol: Tolerance for IC matching

    Returns:
        Index of matching sample, or None if no match
    """
    # Find samples with matching parameters
    param_diffs = np.abs(reference_params - target_params[None, :])
    param_matches = np.all(param_diffs < param_tol, axis=1)

    if not param_matches.any():
        return None

    # Among param matches, find IC match
    matching_indices = np.where(param_matches)[0]

    for idx in matching_indices:
        # Check all realizations for this sample
        for r in range(reference_ics.shape[1]):
            ref_ic = reference_ics[idx, r]
            ic_diff = np.abs(ref_ic - target_ic)
            if np.all(ic_diff < ic_tol):
                return int(idx)

    return None


def extract_reference_features(
    mno_dataset_path: Path,
    reference_dataset_path: Path,
    config_path: Path,
    device: str = "cuda",
    batch_size: int = 8,
) -> int:
    """Extract reference features with IC/param matching and CNO generation fallback.

    Args:
        mno_dataset_path: Path to MNO feature dataset
        reference_dataset_path: Path to reference dataset (100K CNO/UAFNO/etc.)
        config_path: Path to experiment config (for CNO replay)
        device: Device for CNO generation ("cuda" or "cpu")
        batch_size: Batch size for CNO generation

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Validate inputs
    if not mno_dataset_path.exists():
        print(f"Error: MNO dataset not found: {mno_dataset_path}", file=sys.stderr)
        return 1

    if not reference_dataset_path.exists():
        print(f"Error: Reference dataset not found: {reference_dataset_path}", file=sys.stderr)
        return 1

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        return 1

    print(f"{'='*70}")
    print(f"REFERENCE FEATURE EXTRACTION WITH IC/PARAM MATCHING")
    print(f"{'='*70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference dataset: {reference_dataset_path}")
    print(f"Config: {config_path}")
    print(f"Device: {device}\n")

    # Load MNO dataset
    print("Loading MNO dataset...")
    with h5py.File(mno_dataset_path, 'r') as f:
        # Check required fields
        required_fields = [
            'metadata/generation_indices',
            'metadata/is_interpolated',
            'inputs/fields',
            'parameters/params',
            'features/summary/aggregated/features',
        ]

        for field in required_fields:
            if field not in f:
                print(f"Error: MNO dataset missing '{field}'", file=sys.stderr)
                return 1

        # Load MNO data
        mno_ics = f['inputs/fields'][:, 0]  # [N, H, W] - first realization
        mno_params = f['parameters/params'][:]  # [N, D]
        generation_indices = f['metadata/generation_indices'][:]
        is_interpolated = f['metadata/is_interpolated'][:]
        mno_feature_shape = f['features/summary/aggregated/features'].shape

        n_samples = len(mno_ics)

        print(f"  MNO samples: {n_samples}")
        print(f"  Exact training points: {(~is_interpolated).sum()}")
        print(f"  Interpolated points: {is_interpolated.sum()}")
        print(f"  Feature dimension: {mno_feature_shape[1]}")
        print(f"  IC shape: {mno_ics.shape}")

    # Load reference dataset
    print(f"\nLoading reference dataset...")
    with h5py.File(reference_dataset_path, 'r') as f:
        if 'inputs/fields' not in f or 'parameters/params' not in f:
            print(f"Error: Reference dataset missing inputs/fields or parameters/params", file=sys.stderr)
            return 1

        if 'features/summary/aggregated/features' not in f:
            print(f"Error: Reference dataset missing aggregated features", file=sys.stderr)
            return 1

        reference_ics = f['inputs/fields'][:]  # [N, M, H, W]
        reference_params = f['parameters/params'][:]  # [N, D]
        reference_features = f['features/summary/aggregated/features']  # [N, D_feat]

        ref_n_samples = len(reference_ics)
        ref_feature_dim = reference_features.shape[1]

        print(f"  Reference samples: {ref_n_samples}")
        print(f"  Feature dimension: {ref_feature_dim}")
        print(f"  IC shape: {reference_ics.shape}")

        # Validate feature dimensions
        if ref_feature_dim != mno_feature_shape[1]:
            print(f"Error: Feature dimension mismatch", file=sys.stderr)
            print(f"  MNO: {mno_feature_shape[1]}, Reference: {ref_feature_dim}", file=sys.stderr)
            return 1

    # Match MNO samples to reference dataset
    print(f"\nMatching MNO samples to reference dataset...")
    matched_indices = []  # Reference indices for matched samples
    unmatched_indices = []  # MNO indices for unmatched samples

    for i in tqdm(range(n_samples), desc="Matching samples"):
        match_idx = find_matching_sample(
            mno_params[i],
            mno_ics[i],
            reference_params,
            reference_ics,
        )

        if match_idx is not None:
            matched_indices.append((i, match_idx))
        else:
            unmatched_indices.append(i)

    print(f"\n  Matched samples: {len(matched_indices)}")
    print(f"  Unmatched samples: {len(unmatched_indices)}")

    # Extract features for matched samples
    output_features = np.zeros((n_samples, mno_feature_shape[1]), dtype=np.float32)

    if matched_indices:
        print(f"\nExtracting features for {len(matched_indices)} matched samples...")
        with h5py.File(reference_dataset_path, 'r') as f:
            ref_features = f['features/summary/aggregated/features']
            for mno_idx, ref_idx in tqdm(matched_indices, desc="Extracting features"):
                output_features[mno_idx] = ref_features[ref_idx]

    # Generate CNO rollouts for unmatched samples
    if unmatched_indices:
        print(f"\nGenerating CNO rollouts for {len(unmatched_indices)} unmatched samples...")
        print("  This may take several minutes...")

        # Load config and initialize CNO replayer
        config = ExperimentConfig.from_yaml(config_path)
        replayer = CNOReplayer(reference_dataset_path, config, device=device)

        # Initialize feature extractor
        summary_config = SummaryConfig.from_schema_config(config.features.summary)
        feature_extractor = SummaryExtractor(device=device, config=summary_config)

        # Process in batches
        num_batches = (len(unmatched_indices) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating rollouts"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(unmatched_indices))
            batch_mno_indices = unmatched_indices[batch_start:batch_end]

            # Prepare batch data
            batch_params = torch.from_numpy(mno_params[batch_mno_indices]).to(device)
            batch_ics = torch.from_numpy(mno_ics[batch_mno_indices]).unsqueeze(1).to(device)  # [B, 1, H, W]

            # Generate CNO rollouts
            with torch.no_grad():
                trajectories_list = []
                for i in range(len(batch_mno_indices)):
                    traj = replayer.rollout(
                        batch_params[i],
                        batch_ics[i, 0],
                        timesteps=config.simulation.num_timesteps,
                    )  # [T, 1, H, W]
                    trajectories_list.append(traj)

                # Stack trajectories: [B, T, 1, H, W]
                trajectories = torch.stack(trajectories_list, dim=0)

                # Add realization dimension for feature extractor: [B, 1, T, 1, H, W]
                trajectories_with_realizations = trajectories.unsqueeze(1)

                # Extract features
                per_trajectory = feature_extractor.extract_per_trajectory(
                    trajectories_with_realizations
                )  # [B, 1, D]

                # Aggregate (mean over single realization)
                aggregated = feature_extractor.aggregate_realizations(
                    per_trajectory, method='mean'
                ).cpu().numpy()  # [B, D]

            # Store features
            for i, mno_idx in enumerate(batch_mno_indices):
                output_features[mno_idx] = aggregated[i]

    # Store in MNO dataset
    print(f"\nStoring reference features in MNO dataset...")
    with h5py.File(mno_dataset_path, 'a') as f:
        # Check if already exists
        if 'features/reference_features' in f:
            print("  Warning: 'features/reference_features' already exists, overwriting...")
            del f['features/reference_features']

        # Create dataset
        f.create_dataset(
            'features/reference_features',
            data=output_features,
            compression='gzip',
            compression_opts=4,
        )

        # Add metadata
        f['features/reference_features'].attrs['description'] = (
            f'Reference solver features with IC/param matching. '
            f'{len(matched_indices)} extracted from reference dataset, '
            f'{len(unmatched_indices)} generated via CNO replay.'
        )
        f['features/reference_features'].attrs['source_dataset'] = str(reference_dataset_path)
        f['features/reference_features'].attrs['matched_samples'] = len(matched_indices)
        f['features/reference_features'].attrs['generated_samples'] = len(unmatched_indices)

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"MNO dataset: {mno_dataset_path}")
    print(f"Reference features stored at: 'features/reference_features'")
    print(f"Shape: {output_features.shape}")
    print(f"Matched (extracted): {len(matched_indices)}")
    print(f"Unmatched (generated): {len(unmatched_indices)}")
    print(f"Interpolated samples: {is_interpolated.sum()} / {len(is_interpolated)}")
    print(f"{'=' * 70}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference features with IC/param matching and CNO generation",
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
        "--reference-dataset",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to reference dataset (100K CNO/UAFNO/etc., .h5 file)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to experiment config (.yaml file) for CNO replay",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for CNO generation (default: cuda)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for CNO generation (default: 8)",
    )

    args = parser.parse_args()

    return extract_reference_features(
        args.mno_dataset,
        args.reference_dataset,
        args.config,
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    sys.exit(main())
