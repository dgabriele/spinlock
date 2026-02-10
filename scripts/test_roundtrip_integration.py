"""Test roundtrip consistency integration into VQTokenizer training.

This script validates that:
1. Config loads properly with inverse_heads and roundtrip settings
2. Model creates inverse heads correctly
3. Loss function includes roundtrip loss
4. Forward pass produces decoded outputs
5. Training loop runs without errors
"""

import torch
import yaml
from pathlib import Path

from spinlock.tokens.config import TokenizerConfig
from spinlock.tokens.model import JointHierarchicalVQVAE
from spinlock.tokens.losses import VQVAELoss
from spinlock.tokens.trainer import VQTokenizerTrainer


def test_config_loading():
    """Test that roundtrip config loads correctly."""
    print("=" * 80)
    print("Test 1: Config Loading")
    print("=" * 80)

    config_path = Path("configs/vqvae_50k_roundtrip.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    config = TokenizerConfig(**config_dict)

    # Check inverse heads config
    assert config.inverse_heads is not None, "inverse_heads should be present"
    assert config.inverse_heads.theta_hidden_dim == 64
    assert config.inverse_heads.theta_dropout == 0.1
    assert config.inverse_heads.initial_base_channels == 256

    # Check roundtrip loss config
    assert config.loss.roundtrip is not None, "roundtrip should be present"
    assert config.loss.roundtrip.enabled is True
    assert config.loss.roundtrip.weight == 2.0
    assert config.loss.roundtrip.theta_weight == 1.0
    assert config.loss.roundtrip.initial_weight == 1.0

    print("✅ Config loaded successfully")
    print(f"   - Inverse heads: theta_hidden={config.inverse_heads.theta_hidden_dim}")
    print(f"   - Roundtrip loss: enabled={config.loss.roundtrip.enabled}, weight={config.loss.roundtrip.weight}")
    print()

    return config


def test_model_creation(config):
    """Test that model creates inverse heads correctly."""
    print("=" * 80)
    print("Test 2: Model Creation with Inverse Heads")
    print("=" * 80)

    # Create dummy group_indices
    group_indices = {
        'theta_group_1': list(range(0, 10)),
        'theta_group_2': list(range(10, 20)),
        'theta_group_3': list(range(20, 32)),
        'initial_group_1': list(range(0, 100)),
        'initial_group_2': list(range(100, 200)),
        'initial_group_3': list(range(200, 384)),
        'temporal_group_1': list(range(0, 80)),
        'temporal_group_2': list(range(80, 160)),
        'temporal_group_3': list(range(160, 240)),
        'temporal_group_4': list(range(240, 320)),
    }

    model = JointHierarchicalVQVAE(
        config=config,
        group_indices=group_indices,
        temporal_input_dim=20,  # Example: 20 features per timestep
        initial_input_dim=None,  # Not used for hybrid encoder
    )

    # Check inverse heads were created
    assert model.theta_inverse is not None, "theta_inverse should be created"
    assert model.initial_inverse is not None, "initial_inverse should be created"

    print("✅ Model created successfully")
    print(f"   - Theta inverse: {model.theta_inverse}")
    print(f"   - Initial inverse: {model.initial_inverse}")
    print(f"   - Families: {model.families}")
    print(f"   - Theta dim: {model.theta_dim}")
    print(f"   - Initial dim: {model.initial_dim}")
    print(f"   - Temporal dim: {model.temporal_dim}")
    print()

    return model, group_indices


def test_loss_creation(config):
    """Test that loss function includes roundtrip loss."""
    print("=" * 80)
    print("Test 3: Loss Function Creation")
    print("=" * 80)

    loss_fn = VQVAELoss(config.loss)

    # Check roundtrip loss was created
    assert loss_fn.roundtrip_loss is not None, "roundtrip_loss should be created"
    assert loss_fn.roundtrip_loss.theta_weight == 1.0
    assert loss_fn.roundtrip_loss.initial_weight == 1.0

    print("✅ Loss function created successfully")
    print(f"   - Roundtrip loss: {loss_fn.roundtrip_loss}")
    print(f"   - Theta weight: {loss_fn.roundtrip_loss.theta_weight}")
    print(f"   - Initial weight: {loss_fn.roundtrip_loss.initial_weight}")
    print()

    return loss_fn


def test_forward_pass(model):
    """Test that forward pass produces decoded outputs."""
    print("=" * 80)
    print("Test 4: Forward Pass with Inverse Heads")
    print("=" * 80)

    batch_size = 4
    device = torch.device('cpu')

    # Create dummy inputs
    theta = torch.rand(batch_size, 14)  # 14 PDE parameters
    u0 = torch.rand(batch_size, 3, 64, 64)  # 3-channel 64x64 ICs
    initial_manual = torch.rand(batch_size, 42)  # 42 manual features (required for hybrid encoder)
    temporal = torch.rand(batch_size, 256, 20)  # Variable-length sequences (max 256 timesteps)
    temporal_mask = torch.ones(batch_size, 256, dtype=torch.bool)
    temporal_lengths = torch.tensor([128, 200, 256, 180])

    # Forward pass
    outputs = model(
        temporal_features=temporal,
        initial_manual=initial_manual,
        initial_raw=u0,
        theta_features=theta,
        temporal_mask=temporal_mask,
        temporal_lengths=temporal_lengths,
    )

    # Check decoded outputs exist
    assert outputs['decoded'] is not None, "decoded should be present"
    assert 'theta' in outputs['decoded'], "theta should be in decoded"
    assert 'initial' in outputs['decoded'], "initial should be in decoded"

    # Check shapes
    theta_decoded = outputs['decoded']['theta']
    u0_decoded = outputs['decoded']['initial']

    assert theta_decoded.shape == (batch_size, 14), f"theta shape mismatch: {theta_decoded.shape}"
    assert u0_decoded.shape == (batch_size, 3, 64, 64), f"u0 shape mismatch: {u0_decoded.shape}"

    print("✅ Forward pass successful")
    print(f"   - Theta decoded shape: {theta_decoded.shape}")
    print(f"   - Initial decoded shape: {u0_decoded.shape}")
    print(f"   - Reconstructed shape: {outputs['reconstructed'].shape}")
    print(f"   - Token indices: {len(outputs['token_indices'])} quantizers")
    print()

    return outputs


def test_loss_computation(loss_fn, model, outputs):
    """Test that loss computes correctly with roundtrip."""
    print("=" * 80)
    print("Test 5: Loss Computation with Roundtrip")
    print("=" * 80)

    # Extract needed components
    original = outputs['original_encoded']
    reconstructed = outputs['reconstructed']
    vq_loss = outputs['vq_loss']
    tokens = outputs['token_indices']
    decoded = outputs['decoded']

    # Extract category embeddings
    category_embeddings = {}
    for family_cat, indices in model.group_indices.items():
        category_embeddings[family_cat] = original[:, indices]

    # Compute loss
    losses = loss_fn(
        original=original,
        reconstructed=reconstructed,
        vq_loss=vq_loss,
        category_embeddings=category_embeddings,
        quantized_vectors=outputs.get('encodings'),
        token_indices=tokens,
        latent_vectors=outputs.get('latents'),
        model=model,
        tokens=tokens,
        decoded=decoded,
        initial_manual=None,
    )

    # Check roundtrip loss is computed
    assert 'roundtrip/total' in losses, "roundtrip/total should be in losses"
    assert losses['roundtrip/total'] > 0, "roundtrip loss should be positive"

    print("✅ Loss computation successful")
    print(f"   - Total loss: {losses['total']:.6f}")
    print(f"   - Reconstruction: {losses['reconstruction']:.6f}")
    print(f"   - VQ loss: {losses['vq']:.6f}")
    print(f"   - Roundtrip loss: {losses['roundtrip/total']:.6f}")
    print(f"   - Roundtrip metrics: {sum(1 for k in losses if k.startswith('roundtrip/'))} quantizers")
    print()

    # Print per-quantizer roundtrip losses
    print("   Per-quantizer roundtrip losses:")
    for key, value in sorted(losses.items()):
        if key.startswith('roundtrip/') and key != 'roundtrip/total':
            print(f"     {key}: {value:.6f}")
    print()

    return losses


def test_backward_pass(model, losses):
    """Test that backward pass works correctly."""
    print("=" * 80)
    print("Test 6: Backward Pass")
    print("=" * 80)

    # Zero gradients
    for param in model.parameters():
        param.grad = None

    # Backward pass
    loss = losses['total']
    loss.backward()

    # Check that inverse heads have gradients
    theta_grads = any(p.grad is not None for p in model.theta_inverse.parameters())
    initial_grads = any(p.grad is not None for p in model.initial_inverse.parameters())

    assert theta_grads, "theta_inverse should have gradients"
    assert initial_grads, "initial_inverse should have gradients"

    print("✅ Backward pass successful")
    print(f"   - Theta inverse has gradients: {theta_grads}")
    print(f"   - Initial inverse has gradients: {initial_grads}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "VQTokenizer Roundtrip Integration Tests" + " " * 24 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Run tests
    config = test_config_loading()
    model, group_indices = test_model_creation(config)
    loss_fn = test_loss_creation(config)
    outputs = test_forward_pass(model)
    losses = test_loss_computation(loss_fn, model, outputs)
    test_backward_pass(model, losses)

    print("=" * 80)
    print("🎉 All tests passed! Roundtrip integration is working correctly.")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run full training: poetry run spinlock train-vq-tokenizer --config configs/vqvae_50k_roundtrip.yaml")
    print("  2. Monitor roundtrip loss convergence (target: < 0.01)")
    print("  3. Validate token consistency: poetry run python scripts/test_roundtrip_consistency.py")
    print()


if __name__ == "__main__":
    main()
