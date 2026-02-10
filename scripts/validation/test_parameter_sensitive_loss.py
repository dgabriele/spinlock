#!/usr/bin/env python3
"""Test script for parameter-sensitive loss components.

Validates that all modular components can be instantiated and run
without errors. This is a smoke test to catch import/API issues
before full training.
"""

import torch
import torch.nn as nn

def test_rollout_feature_extractor():
    """Test RolloutFeatureExtractor."""
    from spinlock.mno.features import RolloutFeatureExtractor

    print("Testing RolloutFeatureExtractor...")

    # Create dummy rollout [B, T, C, H, W]
    rollout = torch.randn(2, 32, 3, 64, 64)

    extractor = RolloutFeatureExtractor()
    features = extractor(rollout)

    assert features.shape == (2, 32), f"Expected shape (2, 32), got {features.shape}"
    print(f"  ✓ Feature extraction: {rollout.shape} → {features.shape}")

    # Test with realizations [B, M, T, C, H, W]
    rollout_with_realizations = torch.randn(2, 3, 32, 3, 64, 64)
    features = extractor(rollout_with_realizations)
    assert features.shape == (2, 32), f"Expected shape (2, 32), got {features.shape}"
    print(f"  ✓ Feature extraction with realizations: {rollout_with_realizations.shape} → {features.shape}")


def test_parameter_reconstructor():
    """Test ParameterReconstructor."""
    from spinlock.mno.modules import ParameterReconstructor

    print("\nTesting ParameterReconstructor...")

    rollout = torch.randn(2, 32, 3, 64, 64)
    params_true = torch.randn(2, 14)

    reconstructor = ParameterReconstructor(param_dim=14)
    params_pred = reconstructor(rollout)

    assert params_pred.shape == (2, 14), f"Expected shape (2, 14), got {params_pred.shape}"
    print(f"  ✓ Parameter prediction: {rollout.shape} → {params_pred.shape}")

    loss = reconstructor.reconstruction_loss(rollout, params_true)
    assert loss.ndim == 0, "Loss should be scalar"
    print(f"  ✓ Reconstruction loss: {loss.item():.6f}")


def test_contrastive_similarity():
    """Test ContrastiveSimilarity."""
    from spinlock.mno.modules import ContrastiveSimilarity

    print("\nTesting ContrastiveSimilarity...")

    rollout = torch.randn(4, 32, 3, 64, 64)
    params = torch.randn(4, 14)

    similarity_module = ContrastiveSimilarity(param_dim=14, embed_dim=128)

    rollout_embed, param_embed = similarity_module(rollout, params)
    assert rollout_embed.shape == (4, 128), f"Expected shape (4, 128), got {rollout_embed.shape}"
    assert param_embed.shape == (4, 128), f"Expected shape (4, 128), got {param_embed.shape}"
    print(f"  ✓ Embedding: rollout {rollout_embed.shape}, params {param_embed.shape}")

    # Test embeddings are normalized
    norms = torch.norm(rollout_embed, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), "Embeddings should be L2-normalized"
    print(f"  ✓ Embeddings are L2-normalized")

    # Test similarity matrix
    sim_matrix = similarity_module.compute_similarity_matrix(rollout_embed, param_embed)
    assert sim_matrix.shape == (4, 4), f"Expected shape (4, 4), got {sim_matrix.shape}"
    print(f"  ✓ Similarity matrix: {sim_matrix.shape}")


def test_loss_components():
    """Test individual loss components."""
    from spinlock.mno.losses.components import (
        ParameterReconstructionLoss,
        ContrastiveLoss,
    )

    print("\nTesting Loss Components...")

    rollout = torch.randn(4, 32, 3, 64, 64)
    params = torch.randn(4, 14)

    # Parameter reconstruction loss
    print("  Testing ParameterReconstructionLoss...")
    param_recon_loss = ParameterReconstructionLoss(param_dim=14)
    param_recon_out = param_recon_loss(rollout, params)
    assert 'loss' in param_recon_out, "Should return loss"
    assert 'accuracy' in param_recon_out, "Should return accuracy"
    print(f"    ✓ Loss: {param_recon_out['loss'].item():.6f}, Accuracy: {param_recon_out['accuracy'].item():.2%}")

    # Contrastive loss
    print("  Testing ContrastiveLoss...")
    contrastive_loss = ContrastiveLoss(param_dim=14, temperature=0.1)
    contrastive_out = contrastive_loss(rollout, params)
    assert 'loss' in contrastive_out, "Should return loss"
    assert 'accuracy' in contrastive_out, "Should return accuracy"
    print(f"    ✓ Loss: {contrastive_out['loss'].item():.6f}, Accuracy: {contrastive_out['accuracy'].item():.2%}")


def test_parameter_sensitive_loss():
    """Test composed ParameterSensitiveLoss."""
    from spinlock.mno.losses import ParameterSensitiveLoss

    print("\nTesting ParameterSensitiveLoss...")

    pred_trajectory = torch.randn(2, 32, 3, 64, 64)
    target_trajectory = torch.randn(2, 32, 3, 64, 64)
    ic = torch.randn(2, 3, 64, 64)
    params = torch.randn(2, 14)

    loss_fn = ParameterSensitiveLoss(
        lambda_traj=1.0,
        lambda_ic=0.3,
        lambda_param_recon=0.5,
        lambda_contrastive=0.3,
        lambda_sensitivity=0.0,  # Disable sensitivity for simple test (needs MNO)
    )

    # Test without MNO (sensitivity disabled)
    loss_output = loss_fn.compute(
        pred_trajectory=pred_trajectory,
        target_trajectory=target_trajectory,
        ic=ic,
        noa=None,
        params=params,
    )

    assert loss_output.total is not None, "Should return total loss"
    assert 'traj' in loss_output.components, "Should have trajectory loss"
    assert 'param_recon' in loss_output.components, "Should have param reconstruction loss"
    assert 'contrastive' in loss_output.components, "Should have contrastive loss"

    print(f"  ✓ Total loss: {loss_output.total.item():.6f}")
    print(f"  ✓ Components:")
    for name, value in loss_output.metrics.items():
        if name in ['traj', 'ic', 'param_recon', 'contrastive', 'sensitivity']:
            print(f"    {name}: {value:.6f}")

    # Test backward pass
    loss_output.total.backward()
    print(f"  ✓ Backward pass successful")


def main():
    """Run all tests."""
    print("="*60)
    print("Parameter-Sensitive Loss Components Test")
    print("="*60)

    try:
        test_rollout_feature_extractor()
        test_parameter_reconstructor()
        test_contrastive_similarity()
        test_loss_components()
        test_parameter_sensitive_loss()

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
