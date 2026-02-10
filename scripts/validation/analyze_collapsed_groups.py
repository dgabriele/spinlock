#!/usr/bin/env python3
"""Analyze which feature groups collapsed and why."""

import h5py
import numpy as np
from pathlib import Path

# Load original dataset to check feature diversity
dataset_path = Path("datasets/50k_baseline.h5")
tokenized_path = Path("datasets/50k_baseline_tokenized_theta.h5")

print("="*80)
print("ANALYZING COLLAPSED GROUPS")
print("="*80)

# Identify fully collapsed groups (all 3 levels have 1 token)
fully_collapsed = [
    'temporal_group_5',
    'temporal_group_8',
    'temporal_group_9',
    'temporal_group_10',
    'temporal_group_12',
    'temporal_group_13',
    'temporal_group_14',
    'temporal_group_19',
]

print(f"\nFully collapsed groups (1 token across L0/L1/L2): {len(fully_collapsed)}")
for group in fully_collapsed:
    print(f"  {group}")

# Load tokenized dataset to verify
print("\n" + "="*80)
print("VERIFYING TOKEN DIVERSITY IN PRETOKENIZED DATASET:")
print("="*80)

with h5py.File(tokenized_path, 'r') as f:
    tokens_group = f['tokens']

    for group_name in fully_collapsed:
        print(f"\n{group_name}:")
        for level in ['L0', 'L1', 'L2']:
            key = f"{group_name}_{level}"
            if key in tokens_group:
                tokens = tokens_group[key][:]
                unique_tokens = np.unique(tokens)
                print(f"  {level}: {len(unique_tokens)} unique tokens → {unique_tokens}")

# Now check the ORIGINAL features for these groups
print("\n" + "="*80)
print("CHECKING ORIGINAL FEATURE DIVERSITY:")
print("="*80)

with h5py.File(dataset_path, 'r') as f:
    # The grouping info is stored in the VQTokenizer checkpoint
    # For temporal features, check if variance is zero or very low

    if 'features/temporal/features' in f:
        temporal_features = f['features/temporal/features'][:]  # [N, T, D]
        print(f"\nTemporal features shape: {temporal_features.shape}")

        # Aggregate to [N, D] by taking mean over time
        temporal_agg = temporal_features.mean(axis=1)  # [N, D]

        print("\nChecking per-feature variance:")
        print("(Low variance means all samples have similar values → explains collapse)")

        # The groups are learned via hierarchical clustering
        # But we can check which individual features have low variance
        feature_variance = temporal_agg.var(axis=0)  # [D]

        # Find low-variance features
        zero_var_indices = np.where(feature_variance < 1e-6)[0]
        low_var_indices = np.where((feature_variance >= 1e-6) & (feature_variance < 0.01))[0]

        print(f"\nZero variance features (<1e-6): {len(zero_var_indices)}/{len(feature_variance)}")
        if len(zero_var_indices) > 0:
            print(f"  Indices: {zero_var_indices[:20].tolist()}")

        print(f"Low variance features (1e-6 to 0.01): {len(low_var_indices)}/{len(feature_variance)}")
        if len(low_var_indices) > 0:
            print(f"  Indices: {low_var_indices[:20].tolist()}")

        # Show distribution of variances
        print(f"\nVariance distribution:")
        print(f"  Min:    {feature_variance.min():.6e}")
        print(f"  10%ile: {np.percentile(feature_variance, 10):.6e}")
        print(f"  25%ile: {np.percentile(feature_variance, 25):.6e}")
        print(f"  Median: {np.median(feature_variance):.6e}")
        print(f"  75%ile: {np.percentile(feature_variance, 75):.6e}")
        print(f"  90%ile: {np.percentile(feature_variance, 90):.6e}")
        print(f"  Max:    {feature_variance.max():.6e}")

print("\n" + "="*80)
print("HYPOTHESIS:")
print("="*80)
print("""
The 50K training dataset has low diversity in certain temporal feature dimensions.
When features have near-zero variance (all samples look the same), the VQTokenizer
learns to map them all to the same code.

This is NOT mode collapse - it's CORRECT behavior for low-diversity data.

The codebooks have capacity (100% utilization during training), but the dataset
doesn't need it because many features are constant or nearly constant across all
50K samples.

SOLUTION:
1. Accept this as correct behavior (tokenizer is working as designed)
2. OR generate more diverse training data with wider parameter/IC variation
3. OR check if feature extraction is broken (producing constant features)
""")
