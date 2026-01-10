#!/usr/bin/env python3
"""Test MSE-led latent loss to verify it works when enabled.

The investigation revealed that MSE-led latent loss shows 0.000 because:
1. It requires --enable-latent-loss flag (defaults to False)
2. When disabled, VQVAEAlignmentLoss.compute_losses() doesn't return 'latent'
3. MSELedLoss gets 0.0 from align_out.get('latent', 0.0)

This test verifies that when enabled, the latent loss is actually computed.
"""

import torch
import sys

sys.path.insert(0, '/home/daniel/projects/spinlock/src')

from spinlock.noa.losses import MSELedLoss
from spinlock.noa.vqvae_alignment import VQVAEAlignmentLoss
from spinlock.noa import NOABackbone

print("=" * 70)
print("MSE-Led Latent Loss Test")
print("=" * 70)

device = 'cuda'

# Create NOA
print("\n1. Creating NOA backbone...")
noa = NOABackbone(
    spatial_dim=64,
    in_channels=1,
    out_channels=1,
    base_channels=32,
    encoder_levels=3,
    modes=16,
    afno_blocks=4,
).to(device)
print(f"   ✓ NOA created ({sum(p.numel() for p in noa.parameters()):,} parameters)")

# Load VQ-VAE alignment with latent loss DISABLED (default)
print("\n2. Loading VQ-VAE alignment (latent loss DISABLED)...")
alignment_disabled = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path='checkpoints/production/100k_3family_v1',
    device=device,
    noa=noa,
    enable_latent_loss=False,  # Default behavior
)
print(f"   ✓ Alignment loaded (enable_latent_loss=False)")

# Load VQ-VAE alignment with latent loss ENABLED
print("\n3. Loading VQ-VAE alignment (latent loss ENABLED)...")
alignment_enabled = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path='checkpoints/production/100k_3family_v1',
    device=device,
    noa=noa,
    enable_latent_loss=True,  # Enable it!
    latent_sample_steps=3,
)
print(f"   ✓ Alignment loaded (enable_latent_loss=True)")
print(f"   ✓ LatentProjector initialized: {alignment_enabled.latent_projector is not None}")

# Create MSE-led losses
print("\n4. Creating MSE-led loss functions...")
loss_disabled = MSELedLoss(
    lambda_traj=1.0,
    lambda_commit=0.5,
    lambda_latent=0.1,
    vqvae_alignment=alignment_disabled,
)
loss_enabled = MSELedLoss(
    lambda_traj=1.0,
    lambda_commit=0.5,
    lambda_latent=0.1,
    vqvae_alignment=alignment_enabled,
)
print("   ✓ Loss functions created")

# Generate random trajectories
print("\n5. Generating sample trajectories...")
B, T, C, H, W = 2, 32, 1, 64, 64
ic = torch.randn(B, C, H, W, device=device)
pred_traj = torch.randn(B, T, C, H, W, device=device)
target_traj = torch.randn(B, T, C, H, W, device=device)
print(f"   ✓ Trajectories generated: {pred_traj.shape}")

# Compute losses with latent DISABLED
print("\n6. Computing MSE-led loss (latent loss DISABLED)...")
output_disabled = loss_disabled.compute(pred_traj, target_traj, ic, noa)
print(f"   traj:   {output_disabled.metrics['traj']:.6f}")
print(f"   commit: {output_disabled.metrics['commit']:.6f}")
print(f"   latent: {output_disabled.metrics['latent']:.6f} ← Should be 0.000")
print(f"   total:  {output_disabled.metrics['total']:.6f}")

# Compute losses with latent ENABLED
print("\n7. Computing MSE-led loss (latent loss ENABLED)...")
output_enabled = loss_enabled.compute(pred_traj, target_traj, ic, noa)
print(f"   traj:   {output_enabled.metrics['traj']:.6f}")
print(f"   commit: {output_enabled.metrics['commit']:.6f}")
print(f"   latent: {output_enabled.metrics['latent']:.6f} ← Should be non-zero!")
print(f"   total:  {output_enabled.metrics['total']:.6f}")

# Verify
print("\n8. Verification:")
if output_disabled.metrics['latent'] == 0.0:
    print("   ✓ DISABLED: Latent loss is 0.0 (expected)")
else:
    print(f"   ✗ DISABLED: Latent loss is {output_disabled.metrics['latent']} (unexpected!)")

if output_enabled.metrics['latent'] > 0.0:
    print(f"   ✓ ENABLED: Latent loss is {output_enabled.metrics['latent']:.6f} (expected)")
else:
    print("   ✗ ENABLED: Latent loss is 0.0 (BUG!)")

print("\n" + "=" * 70)
print("Test complete")
print("=" * 70)
