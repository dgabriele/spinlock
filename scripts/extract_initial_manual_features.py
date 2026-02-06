#!/usr/bin/env python3
"""Extract manual INITIAL features from raw ICs for VQ-VAE training."""

import sys
import h5py
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spinlock.features.initial.ic_feature_extractors import InitialConditionsFeatureExtractor

def extract_initial_features(dataset_path: str, use_statistical: bool = True):
    """Extract manual initial features from raw ICs.

    Args:
        dataset_path: Path to HDF5 dataset
        use_statistical: Use new statistical features (66D) vs old pattern features (42D)
    """
    dataset_path = Path(dataset_path)

    print(f"Extracting INITIAL features from: {dataset_path}")
    print(f"  Feature type: {'Statistical (42D: distributional + energy, no spatial)' if use_statistical else 'Pattern (42D)'}")

    with h5py.File(dataset_path, 'a') as f:
        # Load raw ICs
        if 'inputs/fields' not in f:
            print("Error: No raw ICs found at /inputs/fields")
            return 1

        ics = f['inputs/fields'][:]  # [N, M, C, H, W] or [N, M, H, W]
        print(f"  Raw ICs shape: {ics.shape}")

        # Handle shape: need [N, C, H, W] for statistical extractor
        if len(ics.shape) == 4:
            # [N, M, H, W] - add channel dim
            ics_torch = torch.from_numpy(ics[:, 0]).float()  # [N, H, W]
            ics_torch = ics_torch.unsqueeze(1)  # [N, 1, H, W]
        elif len(ics.shape) == 5:
            # [N, M, C, H, W] - take first realization
            ics_torch = torch.from_numpy(ics[:, 0]).float()  # [N, C, H, W]
        else:
            raise ValueError(f"Unexpected IC shape: {ics.shape}")

        print(f"  Prepared ICs shape: {ics_torch.shape}")

        # Extract features in batches
        print("  Extracting features...")
        if use_statistical:
            # Disable spatial features (gradients, Laplacian) - they have collapsed variance
            extractor = InitialConditionsFeatureExtractor(device='cpu', include_spatial=False)
        else:
            from spinlock.features.initial import InitialManualExtractor
            extractor = InitialManualExtractor(device='cpu')
            # Need to reshape for old extractor: [N, M=1, C, H, W]
            ics_torch = ics_torch.unsqueeze(1)

        batch_size = 100
        features_list = []
        for i in range(0, len(ics_torch), batch_size):
            batch = ics_torch[i:i+batch_size]
            if use_statistical:
                # Statistical extractor: [B, C, H, W] -> [B, D]
                with torch.no_grad():
                    batch_features = extractor(batch)  # [B, 66] for C=3
            else:
                # Old extractor: [B, M, C, H, W] -> [B, M, 14*C]
                batch_features = extractor.extract_all(batch)  # [B, M, D]
                batch_features = batch_features.squeeze(1)  # [B, D]
            features_list.append(batch_features)

        features = torch.cat(features_list, dim=0)  # [N, D]

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
    import argparse
    parser = argparse.ArgumentParser(description="Extract initial features from dataset")
    parser.add_argument("dataset", help="Path to HDF5 dataset")
    parser.add_argument("--statistical", action="store_true", default=True,
                        help="Use statistical features (66D) instead of pattern features (default: True)")
    parser.add_argument("--pattern", dest="statistical", action="store_false",
                        help="Use old pattern features (42D)")
    args = parser.parse_args()

    sys.exit(extract_initial_features(args.dataset, use_statistical=args.statistical))
