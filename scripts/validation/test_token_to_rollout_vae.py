"""Quick test script for Token-to-Rollout VAE implementation.

Tests model initialization and forward pass without requiring training.
"""

from pathlib import Path

import torch

from spinlock.tokens.rollout_dataset import TokenToRolloutDataset
from spinlock.tokens.rollout_vae import TokenToRolloutVAE
from spinlock.tokens.rollout_vae_config import TokenToRolloutVAEConfig


def test_model_initialization():
    """Test that model initializes correctly with runtime dimensions."""
    print("=" * 80)
    print("Test 1: Model Initialization")
    print("=" * 80)

    # TODO: Load config
    config = TokenToRolloutVAEConfig.from_yaml(
        Path("configs/token_to_rollout_vae.yaml")
    )

    # TODO: Resolve dimensions from datasets
    import h5py
    with h5py.File(config.data.cno_dataset, "r") as f:
        theta_dim = f["/parameters/params"].shape[1]
        fields_shape = f["/inputs/fields"].shape
        grid_shape = (fields_shape[2], fields_shape[3], fields_shape[4])

    print(f"  Runtime dimensions:")
    print(f"    theta_dim: {theta_dim}")
    print(f"    grid_shape: {grid_shape}")

    # TODO: Create model
    model = TokenToRolloutVAE(
        vq_checkpoint=config.data.vq_checkpoint,
        theta_dim=theta_dim,
        grid_shape=grid_shape,
        latent_dim=config.model.latent_dim,
        encoder_hidden_dims=config.model.encoder.hidden_dims,
        param_decoder_hidden_dims=config.model.param_decoder.hidden_dims,
        grid_decoder_hidden_channels=config.model.grid_decoder.hidden_channels,
        dropout=config.model.encoder.dropout,
    )

    print(f"  ✓ Model initialized successfully")
    print(f"    Embedding dim: {model.embedding_extractor.total_embedding_dim}")
    print(f"    Latent dim: {model.latent_dim}")
    print(f"    Num token groups: {model.embedding_extractor.num_token_groups}")

    return model, theta_dim, grid_shape, config


def test_forward_pass(model, theta_dim, grid_shape, config):
    """Test forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)

    # TODO: Load a real sample from dataset
    train_dataset, _ = TokenToRolloutDataset.create_splits(
        config.data.cno_dataset,
        config.data.tokenized_dataset,
        train_split=0.9,
        seed=42,
    )

    sample = train_dataset[0]
    print(f"  Sample loaded:")
    print(f"    Tokens: {len(sample['tokens'])} keys")
    print(f"    Theta: {sample['theta'].shape}")
    print(f"    Grids: {sample['grids'].shape}")

    # TODO: Create batch (size 4)
    batch_tokens = {k: torch.stack([sample["tokens"][k]] * 4) for k in sample["tokens"].keys()}

    # TODO: Forward pass
    with torch.no_grad():
        outputs = model(batch_tokens)

    print(f"  ✓ Forward pass successful")
    print(f"    Theta output: {outputs['theta'].shape}")
    print(f"    Grids output: {outputs['grids'].shape}")
    print(f"    Mu: {outputs['mu'].shape}")
    print(f"    Logvar: {outputs['logvar'].shape}")
    print(f"    Z: {outputs['z'].shape}")

    # TODO: Verify shapes
    assert outputs["theta"].shape == (4, theta_dim), f"Expected (4, {theta_dim}), got {outputs['theta'].shape}"
    assert outputs["grids"].shape == (4, *grid_shape), f"Expected (4, {grid_shape}), got {outputs['grids'].shape}"
    assert outputs["mu"].shape == (4, model.latent_dim), f"Expected (4, {model.latent_dim}), got {outputs['mu'].shape}"

    print(f"  ✓ All shapes correct")


def test_sobol_sampling(model, config):
    """Test Sobol sampling."""
    print("\n" + "=" * 80)
    print("Test 3: Sobol Sampling")
    print("=" * 80)

    from spinlock.tokens.sobol_sampler import generate_sobol_variations

    # TODO: Load a sample
    train_dataset, _ = TokenToRolloutDataset.create_splits(
        config.data.cno_dataset,
        config.data.tokenized_dataset,
        train_split=0.9,
        seed=42,
    )

    sample = train_dataset[0]
    tokens = {k: v.unsqueeze(0) for k, v in sample["tokens"].items()}

    # TODO: Generate variations
    variations = generate_sobol_variations(
        model,
        tokens,
        n_variations=50,
        device="cpu",
    )

    print(f"  ✓ Sobol sampling successful")
    print(f"    Theta variations: {variations['theta'].shape}")
    print(f"    Grids variations: {variations['grids'].shape}")
    print(f"    Z variations: {variations['z'].shape}")

    # TODO: Check diversity
    theta_std = variations["theta"].std(axis=0).mean()
    print(f"    Theta diversity (avg std): {theta_std:.6f}")

    assert theta_std > 0.01, "Theta variations should be diverse"
    print(f"  ✓ Variations are diverse")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Token-to-Rollout VAE Implementation Tests")
    print("=" * 80)

    try:
        # TODO: Test 1: Initialization
        model, theta_dim, grid_shape, config = test_model_initialization()

        # TODO: Test 2: Forward pass
        test_forward_pass(model, theta_dim, grid_shape, config)

        # TODO: Test 3: Sobol sampling
        test_sobol_sampling(model, config)

        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
