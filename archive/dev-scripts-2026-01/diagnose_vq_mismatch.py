#!/usr/bin/env python3
"""Diagnose why VQ alignment fails during NOA training.

Investigates:
1. Feature distribution mismatch (VQ-VAE training data vs NOA outputs)
2. Loss magnitude imbalance (latent vs traj)
3. Encoder semantic drift when unfrozen
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '/home/daniel/projects/spinlock/src')

from spinlock.mno.vqvae_alignment import VQVAEAlignmentLoss
from spinlock.mno import NOABackbone, CNOReplayer
import h5py

print("=" * 70)
print("VQ-VAE Alignment Diagnostic")
print("=" * 70)

device = 'cuda'

# Load VQ-VAE checkpoint to understand training data
print("\n1. VQ-VAE Checkpoint Analysis")
print("-" * 70)

checkpoint_path = Path('checkpoints/production/100k_3family_v1/best_model.pt')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model_config = checkpoint.get('model_config', {})
training_config = checkpoint.get('config', {})

print(f"Input dimension: {model_config.get('input_dim', 'NOT FOUND')}")
print(f"Number of features after cleaning: {model_config.get('input_dim', 'NOT FOUND')}")
print(f"Group indices: {len(model_config.get('group_indices', {}))} categories")
print(f"Families trained on: {training_config.get('families', {}).keys()}")

# Check what features were used
group_indices = model_config.get('group_indices', {})
total_features = sum(len(indices) for indices in group_indices.values())
print(f"Total features selected: {total_features}")

print("\n2. Feature Extractor Output Analysis")
print("-" * 70)

# Load VQ-VAE alignment
alignment = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path='checkpoints/production/100k_3family_v1',
    device=device,
    freeze_vqvae=True,
)

# Load dataset to get sample trajectories
print("Loading sample from dataset...")
with h5py.File('datasets/100k_full_features.h5', 'r') as f:
    # Load IC and params
    ic = torch.from_numpy(f['inputs/fields'][0, 0, :, :]).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    params = torch.from_numpy(f['parameters/params'][0]).float().unsqueeze(0).to(device)  # [1, d]

print(f"Loaded IC shape: {ic.shape}")
print(f"Loaded params shape: {params.shape}")

# Load CNO replayer to get ground truth trajectory
print("Loading CNO replayer...")
replayer = CNOReplayer('configs/experiments/local_100k_optimized.yaml', device=device)

# Generate CNO trajectory (ground truth)
print("Generating CNO trajectory...")
with torch.no_grad():
    cno_trajectory = replayer.rollout(
        params_vector=params[0].cpu().numpy(),
        ic=ic,
        timesteps=32,
        num_realizations=1,
        return_all_steps=True,
    )  # [T, 1, H, W]
    cno_trajectory = cno_trajectory.unsqueeze(0)  # [1, T, 1, H, W]

# Create NOA
noa = NOABackbone(
    spatial_dim=64,
    in_channels=1,
    out_channels=1,
    base_channels=32,
    encoder_levels=3,
    modes=16,
    afno_blocks=4,
).to(device)

print("Generating sample trajectories...")

# Generate trajectories
with torch.no_grad():
    # NOA trajectory (random initialization)
    noa_trajectory = noa(ic, steps=32, return_all_steps=True)

print(f"CNO trajectory shape: {cno_trajectory.shape}")
print(f"NOA trajectory shape: {noa_trajectory.shape}")

# Extract features from both
print("\nExtracting features from CNO trajectory...")
cno_result = alignment.feature_extractor(cno_trajectory, ic=ic)
if isinstance(cno_result, tuple):
    cno_features, _ = cno_result
else:
    cno_features = cno_result

print("\nExtracting features from NOA trajectory...")
noa_result = alignment.feature_extractor(noa_trajectory, ic=ic)
if isinstance(noa_result, tuple):
    noa_features, _ = noa_result
else:
    noa_features = noa_result

print(f"CNO features shape: {cno_features.shape}")
print(f"NOA features shape: {noa_features.shape}")

# Normalize features
cno_features_norm = alignment._normalize_features(cno_features)
noa_features_norm = alignment._normalize_features(noa_features)

# Apply feature cleaning
cno_features_clean = alignment._apply_feature_cleaning(cno_features_norm)
noa_features_clean = alignment._apply_feature_cleaning(noa_features_norm)

print(f"CNO features after cleaning: {cno_features_clean.shape}")
print(f"NOA features after cleaning: {noa_features_clean.shape}")

print("\n3. Feature Distribution Comparison")
print("-" * 70)

# Compare feature statistics
cno_mean = cno_features_clean.mean().item()
cno_std = cno_features_clean.std().item()
cno_min = cno_features_clean.min().item()
cno_max = cno_features_clean.max().item()

noa_mean = noa_features_clean.mean().item()
noa_std = noa_features_clean.std().item()
noa_min = noa_features_clean.min().item()
noa_max = noa_features_clean.max().item()

print(f"CNO features - mean: {cno_mean:.6f}, std: {cno_std:.6f}, range: [{cno_min:.6f}, {cno_max:.6f}]")
print(f"NOA features - mean: {noa_mean:.6f}, std: {noa_std:.6f}, range: [{noa_min:.6f}, {noa_max:.6f}]")

# Encode to latent space
print("\n4. Latent Space Analysis")
print("-" * 70)

with torch.no_grad():
    # CNO latents
    cno_z_list = alignment.vqvae.encode(cno_features_clean)
    cno_z = torch.cat(cno_z_list, dim=1)

    # NOA latents (random init)
    noa_z_list = alignment.vqvae.encode(noa_features_clean)
    noa_z = torch.cat(noa_z_list, dim=1)

print(f"CNO latent shape: {cno_z.shape}")
print(f"NOA latent shape: {noa_z.shape}")

cno_z_mean = cno_z.mean().item()
cno_z_std = cno_z.std().item()

noa_z_mean = noa_z.mean().item()
noa_z_std = noa_z.std().item()

print(f"CNO latents - mean: {cno_z_mean:.6f}, std: {cno_z_std:.6f}")
print(f"NOA latents - mean: {noa_z_mean:.6f}, std: {noa_z_std:.6f}")

# Compute losses
latent_loss = torch.nn.functional.mse_loss(noa_z, cno_z)
feature_loss = torch.nn.functional.mse_loss(noa_features_clean, cno_features_clean)
traj_loss = torch.nn.functional.mse_loss(noa_trajectory, cno_trajectory)

print("\n5. Loss Magnitude Analysis")
print("-" * 70)

print(f"Latent loss (raw):   {latent_loss.item():.6f}")
print(f"Feature loss (raw):  {feature_loss.item():.6f}")
print(f"Trajectory loss (raw): {traj_loss.item():.6f}")

print(f"\nWith current weights (λ_recon=1.0, λ_traj=0.3):")
print(f"  Weighted latent: {1.0 * latent_loss.item():.6f}")
print(f"  Weighted traj:   {0.3 * traj_loss.item():.6f}")
print(f"  Ratio (latent/traj): {(1.0 * latent_loss.item()) / (0.3 * traj_loss.item() + 1e-10):.4f}")

print(f"\nActual magnitude difference:")
print(f"  Traj loss is {traj_loss.item() / latent_loss.item():.1f}x larger than latent loss")

print("\n6. Diagnosis")
print("=" * 70)

if feature_loss.item() > 1.0:
    print("❌ PROBLEM: Feature-space distance is large (>1.0)")
    print("   → NOA outputs are far from CNO in feature space")
    print("   → This is expected for random init, but should decrease with training")

if traj_loss.item() / latent_loss.item() > 100:
    print("❌ PROBLEM: Trajectory loss dominates latent loss by >100x")
    print("   → Even with λ_traj=0.3, trajectory gradient will dominate")
    print("   → Recommend: Increase λ_recon or decrease λ_traj")

if abs(noa_z_mean - cno_z_mean) > 0.5:
    print("❌ PROBLEM: Latent space means differ significantly")
    print("   → NOA and CNO occupy different regions of latent space")
    print("   → This suggests fundamental distribution mismatch")

print("\n7. Recommendations")
print("=" * 70)

# Calculate better weight ratio
ideal_ratio = traj_loss.item() / latent_loss.item()
print(f"Current raw loss ratio (traj/latent): {ideal_ratio:.1f}")
print(f"To balance gradients, try:")
print(f"  Option 1: λ_recon={ideal_ratio:.1f}, λ_traj=1.0")
print(f"  Option 2: λ_recon=10.0, λ_traj={10.0/ideal_ratio:.3f}")
print(f"  Option 3: λ_recon=100.0, λ_traj={100.0/ideal_ratio:.3f}")

print("\n" + "=" * 70)
print("Diagnostic complete")
print("=" * 70)
