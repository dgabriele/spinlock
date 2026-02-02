#!/usr/bin/env python
"""Debug category correlation NaN issue."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
import yaml
import h5py
from torch.utils.data import TensorDataset, DataLoader

# Load the trained model
checkpoint_dir = Path('checkpoints/vqvae_1k_full_integrated_metrics')
model_path = checkpoint_dir / 'best_model.pt'

# Load dataset
dataset_path = Path('datasets/test_1k_inline_features.h5')
with h5py.File(dataset_path, 'r') as f:
    features = f['features'][:]

# Create dataloader
dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

# Load model checkpoint
checkpoint = torch.load(model_path, map_location='cpu')
model = checkpoint['model']
model.eval()
model = model.to('cuda')

# Test correlation computation with debugging
category_names = sorted(model.config.group_indices.keys())

# Collect latent activations per category
latent_activations = {cat: [] for cat in category_names}

max_batches = 10
with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        # Extract features from batch
        features_batch = batch[0].to('cuda')

        # Forward pass
        output = model(features_batch)
        latents = output["latents"]

        # Collect latent vectors per category
        idx = 0
        for cat_name in category_names:
            cat_levels = model.config.category_levels[cat_name]
            num_levels = len(cat_levels)

            cat_latents = []
            for level_idx in range(num_levels):
                cat_latents.append(latents[idx])
                idx += 1

            # Concatenate along feature dim
            cat_concat = torch.cat(cat_latents, dim=1)

            # Mean pool across batch
            cat_avg_batch = cat_concat.mean(dim=0)
            latent_activations[cat_name].append(cat_avg_batch.cpu().numpy())

# Check variance
print("=" * 70)
print("CATEGORY CORRELATION DEBUG")
print("=" * 70)
print(f"\nProcessed {max_batches} batches")
print(f"Categories: {len(category_names)}")

print("\n" + "=" * 70)
print("Analyzing latent activation variance per category:")
print("=" * 70)

for cat_name in category_names[:5]:  # Just first 5
    acts = np.stack(latent_activations[cat_name])  # [batches, latent_dim]
    batch_means = acts.mean(axis=1)  # Mean over latent dim -> [batches]

    print(f"\n{cat_name}:")
    print(f"  Latent activations shape: {acts.shape}")
    print(f"  Batch means: {batch_means}")
    print(f"  Batch means variance: {batch_means.var():.10f}")
    print(f"  Batch means std: {batch_means.std():.10f}")

    if batch_means.std() < 1e-10:
        print(f"  ⚠️  NEAR-ZERO VARIANCE - will cause NaN in corrcoef!")

    # Also check the raw activation variance
    print(f"  Raw activation variance (all dims): {acts.var():.10f}")
    print(f"  Raw activation std (all dims): {acts.std():.10f}")

print("\n" + "=" * 70)
print("Testing corrcoef on first two categories:")
print("=" * 70)

cat1, cat2 = category_names[0], category_names[1]
acts_1 = np.stack(latent_activations[cat1])
acts_2 = np.stack(latent_activations[cat2])

batch_means_1 = acts_1.mean(axis=1)
batch_means_2 = acts_2.mean(axis=1)

print(f"\n{cat1} batch means: {batch_means_1}")
print(f"{cat2} batch means: {batch_means_2}")

try:
    corr = np.corrcoef(batch_means_1, batch_means_2)[0, 1]
    print(f"\nCorrelation: {corr}")
except Exception as e:
    print(f"\n❌ Error computing correlation: {e}")

print("\n" + "=" * 70)
