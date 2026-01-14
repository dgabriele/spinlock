#!/usr/bin/env python3
"""Extract manual INITIAL features from raw ICs for VQ-VAE training."""

import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.features.initial import InitialManualExtractor

def extract_initial_features(dataset_path: str):
    """Extract manual initial features from raw ICs."""
    dataset_path = Path(dataset_path)

    print(f"Extracting INITIAL features from: {dataset_path}")

    with h5py.File(dataset_path, 'a') as f:
        # Load raw ICs
        if 'inputs/fields' not in f:
            print("Error: No raw ICs found at /inputs/fields")
            return 1

        ics = f['inputs/fields'][:]  # [N, M, H, W]
        print(f"  Raw ICs shape: {ics.shape}")

        # Take first realization [N, 1, H, W]
        ics_first = torch.from_numpy(ics[:, 0:1]).float()  # [N, 1, H, W]
        # Add channel dim → [N, 1, 1, H, W]
        ics_first = ics_first.unsqueeze(2)  # [N, M=1, C=1, H, W]

        # Extract features in batches
        print("  Extracting manual features...")
        extractor = InitialManualExtractor(device='cpu')

        batch_size = 100
        features_list = []
        for i in range(0, len(ics_first), batch_size):
            batch = ics_first[i:i+batch_size]  # [B, 1, 1, H, W]
            batch_features = extractor.extract_all(batch)  # [B, 14]
            features_list.append(batch_features)

        features = torch.cat(features_list, dim=0)  # [N, M, 14]
        # Remove realization dimension (M=1)
        features = features.squeeze(1)  # [N, 14]

        print(f"  Extracted features shape: {features.shape}")

        # Store in dataset
        if 'features/initial' in f:
            del f['features/initial']

        init_group = f.create_group('features/initial')
        agg_group = init_group.create_group('aggregated')
        agg_group.create_dataset(
            'features',
            data=features.cpu().numpy(),
            compression='gzip',
            compression_opts=4
        )

        print(f"  ✓ Stored at /features/initial/aggregated/features")

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset.h5>")
        sys.exit(1)

    sys.exit(extract_initial_features(sys.argv[1]))
