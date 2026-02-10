#!/usr/bin/env python3
"""
Investigate the contradiction:
- Features have zero variance across 50K samples
- But ema_cluster_size shows all codes are used

Possible explanations:
1. Training uses data augmentation (adds noise)
2. VQ layer initialization artifacts
3. Checkpoint trained on different dataset
4. Bug in analysis
"""

import torch
import h5py
import numpy as np
from pathlib import Path

checkpoint_path = Path("checkpoints/vqvae/theta_baseline_50k/vq_tokenizer_final.pt")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*80)
print("INVESTIGATING EMA_CLUSTER_SIZE CONTRADICTION")
print("="*80)

# Check checkpoint metadata
print("\nCheckpoint Metadata:")
print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")
if 'metadata' in checkpoint:
    print(f"  Metadata: {checkpoint['metadata']}")
if 'initial_input_dim' in checkpoint and 'temporal_input_dim' in checkpoint:
    print(f"  Input dims: initial={checkpoint['initial_input_dim']}, temporal={checkpoint['temporal_input_dim']}")

# Look at a collapsed group's EMA values in detail
state_dict = checkpoint['model_state_dict']

collapsed_groups = [
    'temporal_group_5',
    'temporal_group_10',
    'temporal_group_19',
]

print("\n" + "="*80)
print("DETAILED EMA ANALYSIS FOR COLLAPSED GROUPS")
print("="*80)

for group_name in collapsed_groups:
    print(f"\n{group_name}:")

    for level in ['L0', 'L1', 'L2']:
        ema_key = f"quantizers.{group_name}_{level}.ema_cluster_size"

        if ema_key in state_dict:
            ema_cluster = state_dict[ema_key]
            values = ema_cluster.numpy()

            print(f"\n  {level}:")
            print(f"    Shape: {ema_cluster.shape}")
            print(f"    Active codes: {(values > 0).sum()}/{len(values)}")
            print(f"    Sum: {values.sum():.2f}")
            print(f"    Mean: {values.mean():.2f}")
            print(f"    Std: {values.std():.2f}")
            print(f"    Min: {values.min():.2f}")
            print(f"    Max: {values.max():.2f}")

            # Check for duplicate values (sign of initialization artifacts)
            unique_values = np.unique(values)
            print(f"    Unique values: {len(unique_values)}/{len(values)}")

            # Show distribution of values
            hist, bin_edges = np.histogram(values, bins=5)
            print(f"    Distribution: {hist.tolist()}")

            # Show actual values
            print(f"    Values: {values.tolist()}")

# Check if checkpoint has training config that mentions data augmentation
print("\n" + "="*80)
print("CHECKING FOR DATA AUGMENTATION IN CONFIG")
print("="*80)

if 'config' in checkpoint:
    config = checkpoint['config']
    print(f"Config keys: {list(config.keys())}")

    # Look for augmentation settings
    for key in config.keys():
        if 'aug' in key.lower() or 'noise' in key.lower():
            print(f"  {key}: {config[key]}")

# Check optimizer state for clues
print("\n" + "="*80)
print("CHECKING VQ-VAE TRAINING DYNAMICS")
print("="*80)

# The EMA decay parameter affects how cluster sizes accumulate
print("\nLooking for EMA decay parameter...")
if 'config' in checkpoint:
    config = checkpoint['config']
    if 'levels' in config:
        first_cat = list(config['levels'].keys())[0] if config['levels'] else None
        if first_cat:
            levels = config['levels'][first_cat]
            print(f"Level configs for {first_cat}:")
            for i, level_config in enumerate(levels):
                print(f"  L{i}: {level_config}")

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

print("""
Let's test each hypothesis:

Hypothesis 1: Training adds noise/augmentation
  → Check: config for augmentation settings
  → Status: See above

Hypothesis 2: EMA initialization artifacts
  → Check: Do ema_cluster_size values look like uniform initialization?
  → Status: See distributions above - if all ~equal, likely initialization

Hypothesis 3: Checkpoint trained on different dataset
  → Check: metadata, training history
  → Status: See metadata above

Hypothesis 4: Features aren't actually zero-variance during training
  → Check: Re-compute variance on features vs. variance on embeddings
  → Need: Run features through encoder to see embedding diversity

Hypothesis 5: Commitment loss creates artificial diversity
  → VQ-VAE has commitment loss: ||z - sg(e)||^2
  → This might create small variations in embeddings even for identical inputs
  → During training, gradient noise + commitment loss might activate multiple codes
""")

# Most likely scenario
print("\n" + "="*80)
print("MOST LIKELY EXPLANATION")
print("="*80)
print("""
Looking at the EMA values, many codes have VERY SIMILAR values (duplicates).
This suggests:

1. The EMA is tracking cumulative assignments over training
2. Gradient noise + initialization might cause temporary code assignments
3. Even if features converge to near-identical embeddings, early training
   (before convergence) might activate multiple codes
4. EMA decay parameter (typically 0.99) means old assignments persist

VERIFICATION NEEDED:
- Check if EMA values are artifacts from initialization
- Run forward pass through trained encoder to see actual code assignments
- Compare to pretokenized dataset (which uses trained, frozen encoder)
""")

print("\n" + "="*80)
print("NEXT STEP: FORWARD PASS VALIDATION")
print("="*80)
print("""
We need to:
1. Load the trained VQTokenizer model
2. Run the 50K dataset through it
3. Record which codes are ACTUALLY selected (not just EMA)
4. Compare to ema_cluster_size to see if they match

If pretokenized dataset shows 1 token but EMA shows all codes used,
then EMA is misleading (includes training transients, not final behavior).
""")
