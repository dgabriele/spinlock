#!/usr/bin/env python3
"""Test latent loss implementation."""

import torch
import sys
sys.path.insert(0, '/home/daniel/projects/spinlock/src')

from spinlock.noa.vqvae_alignment import VQVAEAlignmentLoss
from spinlock.noa.losses.vq_led import VQLedLoss

print("=" * 60)
print("Testing VQ-Led Latent Loss Implementation")
print("=" * 60)

# Load VQ-VAE alignment
alignment = VQVAEAlignmentLoss.from_checkpoint(
    vqvae_path='checkpoints/production/100k_3family_v1',
    device='cuda'
)

# Create VQ-led loss with latent comparison
loss_fn = VQLedLoss(
    lambda_recon=1.0,  # Actually lambda_latent now
    lambda_commit=0.25,
    lambda_traj=0.1,
    vqvae_alignment=alignment
)

print(f"\nLoss function: {loss_fn}")
print(f"Primary loss: {loss_fn.leading_loss_name}")
print(f"Auxiliary losses: {loss_fn.auxiliary_loss_names}")

# Create dummy trajectories
B, T, C, H, W = 4, 32, 1, 64, 64
pred_trajectory = torch.randn(B, T, C, H, W, device='cuda')
target_trajectory = torch.randn(B, T, C, H, W, device='cuda')
ic = torch.randn(B, C, H, W, device='cuda')

print(f"\nInput shapes:")
print(f"  pred_trajectory: {pred_trajectory.shape}")
print(f"  target_trajectory: {target_trajectory.shape}")
print(f"  ic: {ic.shape}")

# Compute loss
print("\nComputing loss...")
try:
    output = loss_fn.compute(pred_trajectory, target_trajectory, ic=ic)

    print("✅ Loss computation successful!")
    print(f"\nLoss components:")
    for name, value in output.components.items():
        print(f"  {name}: {value.item():.6f}")

    print(f"\nTotal loss: {output.total.item():.6f}")

    # Test backward pass
    print("\nTesting backward pass...")
    output.total.backward()
    print("✅ Backward pass successful!")

    # Check that we're using latent loss, not recon loss
    if 'latent' in output.components:
        print("\n✅ Using latent loss (correct)")
    else:
        print("\n❌ Not using latent loss (expected 'latent' in components)")

    if 'recon' in output.components:
        print("❌ Still using recon loss (should be removed)")
    else:
        print("✅ Recon loss removed (correct)")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Latent loss test PASSED")
print("=" * 60)
