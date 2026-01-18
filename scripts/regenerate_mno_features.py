#!/usr/bin/env python3
"""
Regenerate MNO features with enhanced temporal architecture.

Extracts 193D per-timestep features from MNO rollout dataset:
- Spatial (24D)
- Spectral (27D)
- Cross-channel (12D)
- Enhanced temporal (130D)

This replaces the old architecture which used trajectory-level SUMMARY features.
"""

import sys
from pathlib import Path
import torch
import h5py
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spinlock.features.temporal import (
    TemporalFeatureOrchestrator,
    TemporalFeatureConfig,
    SpatialConfig,
    SpectralConfig,
    CrossChannelConfig,
    TemporalConfig,
)
from spinlock.features.storage import HDF5FeatureWriter


def regenerate_mno_features(
    input_dataset: Path,
    output_dataset: Path,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Regenerate MNO features with enhanced temporal architecture.

    Args:
        input_dataset: Path to MNO dataset (e.g., mno_rollouts_100k.h5)
        output_dataset: Path to output dataset with features
        batch_size: Batch size for feature extraction
        device: Device for computation
    """
    print(f"Regenerating MNO features: {input_dataset} -> {output_dataset}")
    print(f"Device: {device}, Batch size: {batch_size}\n")

    # Initialize feature extractor with new architecture
    config = TemporalFeatureConfig(
        spatial=SpatialConfig(enabled=True, include_percentiles=True),
        spectral=SpectralConfig(enabled=True, num_fft_scales=5),
        cross_channel=CrossChannelConfig(enabled=True),
        temporal=TemporalConfig(
            enabled=True,
            window_size=5,
            short_window=5,
            medium_window=20,
            long_window=50,
        ),
    )

    extractor = TemporalFeatureOrchestrator(device=torch.device(device), config=config)
    registry = extractor.get_feature_registry()

    print("Feature Configuration:")
    print(f"  Spatial: {len(registry.get_features_by_category('spatial'))}D")
    print(f"  Spectral: {len(registry.get_features_by_category('spectral'))}D")
    print(f"  Cross-channel: {len(registry.get_features_by_category('cross_channel'))}D")
    print(f"  Temporal: {len(registry.get_features_by_category('temporal'))}D")
    print(f"  Total: {registry.num_features}D\n")

    # Load input dataset
    print(f"Loading dataset: {input_dataset}")
    with h5py.File(input_dataset, 'r') as f_in:
        # MNO rollouts may have different structure - check what's available
        if 'trajectories' in f_in:
            trajectories = f_in['trajectories'][:]  # [N, M, T, C, H, W]
        elif 'rollouts' in f_in:
            trajectories = f_in['rollouts'][:]
        else:
            raise ValueError(f"Could not find trajectories or rollouts in {input_dataset}")

        # Handle different trajectory formats
        if trajectories.ndim == 5:  # [N, T, C, H, W]
            # Add realization dimension
            trajectories = trajectories[:, np.newaxis, :, :, :, :]  # [N, 1, T, C, H, W]
        elif trajectories.ndim != 6:
            raise ValueError(f"Unexpected trajectory shape: {trajectories.shape}")

        N, M, T, C, H, W = trajectories.shape
        print(f"  Shape: {trajectories.shape}")
        print(f"  N={N} samples, M={M} realizations, T={T} timesteps, C={C} channels, H={H}, W={W}\n")

    # Create output dataset
    print(f"Creating output dataset: {output_dataset}")
    writer = HDF5FeatureWriter(output_dataset, overwrite=True)

    with writer.open_for_writing():
        # Create temporal feature group
        writer.create_temporal_group(
            num_samples=N,
            num_timesteps=T,
            num_realizations=M,
            registry=registry,
            config=config,
            compression="gzip",
            compression_opts=4,
            chunk_size=batch_size,
            param_dim=0,
        )

        # Extract features in batches
        print(f"Extracting features (batch_size={batch_size})...")
        for batch_start in tqdm(range(0, N, batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, N)
            batch_trajectories = trajectories[batch_start:batch_end]  # [B, M, T, C, H, W]

            # Convert to tensor
            batch_tensor = torch.from_numpy(batch_trajectories).float().to(device)

            # Extract per-timestep features [B, T, 193]
            with torch.no_grad():
                # Reset temporal extractor buffers for new batch
                extractor.temporal_extractor.reset()
                features = extractor.extract_per_timestep(batch_tensor)

            # Write to HDF5
            writer.write_temporal_batch(
                batch_idx=batch_start,
                per_timestep=features.cpu().numpy(),
            )

    print(f"\n✓ Feature extraction complete!")
    print(f"  Output: {output_dataset}")
    print(f"  Features shape: [{N}, {T}, {registry.num_features}]")


def main():
    parser = argparse.ArgumentParser(description="Regenerate MNO features with enhanced temporal architecture")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/mno_rollouts_100k.h5"),
        help="Input MNO dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/mno_features_100k_enhanced.h5"),
        help="Output dataset path with features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
    )

    args = parser.parse_args()

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input dataset not found: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    regenerate_mno_features(
        input_dataset=args.input,
        output_dataset=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
