#!/usr/bin/env python
"""Test category correlation fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Load model and dataset
checkpoint_dir = Path('checkpoints/vqvae_1k_full_integrated_metrics')
model_path = checkpoint_dir / 'final_model.pt'

# Load dataset features
import h5py
dataset_path = Path('datasets/test_1k_inline_features.h5')
with h5py.File(dataset_path, 'r') as f:
    features = np.array(f['features/sdf/aggregated/features'])

print(f"Loaded features: shape={features.shape}")

# Create dataloader
dataset = TensorDataset(torch.tensor(features, dtype=torch.float32))
val_loader = DataLoader(dataset, batch_size=256, shuffle=False)

# Load model
from spinlock.encoding import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
model_config_dict = checkpoint['model_config']

# Convert dict to config object
model_config = CategoricalVQVAEConfig(**model_config_dict)

# Reconstruct model
model = CategoricalHierarchicalVQVAE(model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to('cuda')

print(f"Model loaded: {type(model).__name__}")
print(f"Categories: {len(model.config.group_indices)}")

# Test category correlation
from spinlock.encoding.training.metrics import compute_category_correlation

print("\nComputing category correlation...")
result = compute_category_correlation(model, val_loader, device='cuda', max_batches=10)

print("\n" + "=" * 70)
print("CATEGORY CORRELATION RESULTS")
print("=" * 70)
print(f"Categories: {len(result['category_names'])}")
print(f"Max off-diagonal: {result['max_off_diagonal']:.4f}")
print(f"Mean off-diagonal: {result['mean_off_diagonal']:.4f}")

print("\nCorrelation matrix (first 5×5):")
corr_matrix = result['correlation_matrix']
for i in range(min(5, len(result['category_names']))):
    row = [f"{corr_matrix[i,j]:.3f}" for j in range(min(5, len(result['category_names'])))]
    print(f"  {result['category_names'][i]:12s}: {' '.join(row)}")

# Check for NaN values
has_nan = np.isnan(corr_matrix).any()
print(f"\nContains NaN: {'YES ❌' if has_nan else 'NO ✅'}")

if not has_nan:
    print("\n✅ Category correlation fix SUCCESSFUL!")
else:
    print("\n❌ Category correlation still has NaN values")

print("=" * 70)
