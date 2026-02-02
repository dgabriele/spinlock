#!/usr/bin/env python3
"""Quick integration test for compilation wrapper with training.

This script verifies that:
1. Fixed-length models train correctly with full compilation
2. Variable-length models train correctly with selective compilation
3. Learnable models train correctly with selective compilation
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spinlock.encoding.categorical_vqvae import CategoricalHierarchicalVQVAE, CategoricalVQVAEConfig
from spinlock.encoding.training.trainer import VQVAETrainer
from spinlock.encoding.training.learnable_trainer import LearnableVQVAETrainer
from spinlock.encoding.training.annealing import TemperatureScheduler
from spinlock.encoding.learnable_assignment import SoftAssignmentMatrix


def create_simple_model(input_dim=100, use_learnable=False):
    """Create a simple VQ-VAE model for testing."""
    config = CategoricalVQVAEConfig(
        input_dim=input_dim,
        group_indices={
            "category_1": list(range(input_dim // 2)),
            "category_2": list(range(input_dim // 2, input_dim)),
        },
        group_embedding_dim=32,
        group_hidden_dim=64,
        decoder_hidden_dim=128,
        levels={
            "category_1": [{"num_tokens": 16, "latent_dim": 16}],
            "category_2": [{"num_tokens": 16, "latent_dim": 16}],
        },
        dropout=0.0,
    )
    model = CategoricalHierarchicalVQVAE(config)

    if use_learnable:
        # Add assignment matrix for learnable mode
        model.assignment_matrix = SoftAssignmentMatrix(
            input_dim=input_dim,
            num_categories=2,
        )

    return model


def create_simple_temporal_encoder():
    """Create a simple temporal encoder."""
    class SimpleTemporalEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(10, 50)

        def forward(self, x, mask=None, lengths=None):
            if mask is not None:
                x_masked = x * mask.unsqueeze(-1).float()
                x_agg = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                encoded = self.encoder(x_agg)
                return encoded, {"mask": mask}
            else:
                x_agg = x.mean(dim=1)
                encoded = self.encoder(x_agg)
                return encoded

    encoder = SimpleTemporalEncoder()
    encoder.eval()
    return encoder


def test_fixed_length_training():
    """Test fixed-length model training with full compilation."""
    print("\n" + "=" * 70)
    print("TEST 1: Fixed-Length Model Training (Full Compilation)")
    print("=" * 70)

    model = create_simple_model(input_dim=100)

    # Create dummy data
    batch_size = 32
    num_samples = 128
    features = torch.randn(num_samples, 100)

    dataset = TensorDataset(features)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create trainer with compilation
    print("\n1. Creating trainer with torch.compile enabled...")
    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        device="cpu",  # Use CPU for testing
        use_torch_compile=True,
        val_every_n_epochs=1,
        verbose=False,
    )

    # Train for 2 epochs
    print("2. Training for 2 epochs...")
    try:
        history = trainer.train(epochs=2)
        print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        print("\n✓ PASSED: Fixed-length training completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: Training error: {e}")
        return False


def test_variable_length_training():
    """Test variable-length model training with selective compilation."""
    print("\n" + "=" * 70)
    print("TEST 2: Variable-Length Model Training (Selective Compilation)")
    print("=" * 70)

    model = create_simple_model(input_dim=100)
    temporal_encoder = create_simple_temporal_encoder()

    # Create dummy variable-length data
    batch_size = 32
    num_samples = 128
    max_time = 16
    temporal_dim = 10

    features = torch.randn(num_samples, max_time, temporal_dim)
    lengths = torch.randint(4, max_time + 1, (num_samples,))
    masks = torch.zeros(num_samples, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        masks[i, :length] = True
    encoded_initial = torch.randn(num_samples, 50)

    # Create dataset
    dataset = TensorDataset(features, masks, lengths, encoded_initial)

    def collate_fn(batch):
        features, masks, lengths, encoded_initial = zip(*batch)
        return {
            "features": torch.stack(features),
            "mask": torch.stack(masks),
            "length": torch.stack(lengths),
            "encoded_initial_features": torch.stack(encoded_initial),
        }

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Create trainer with compilation and temporal encoder
    print("\n1. Creating trainer with torch.compile and temporal encoder...")
    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        device="cpu",
        use_torch_compile=True,
        temporal_encoder=temporal_encoder,
        temporal_encoder_output_dim=50,
        encoded_initial_features=encoded_initial.numpy(),
        val_every_n_epochs=1,
        verbose=False,
    )

    # Train for 2 epochs
    print("2. Training for 2 epochs...")
    try:
        history = trainer.train(epochs=2)
        print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        print("\n✓ PASSED: Variable-length training completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learnable_variable_length_training():
    """Test learnable model training with variable-length and selective compilation."""
    print("\n" + "=" * 70)
    print("TEST 3: Learnable Variable-Length Training (Selective Compilation)")
    print("=" * 70)

    model = create_simple_model(input_dim=100, use_learnable=True)
    temporal_encoder = create_simple_temporal_encoder()

    # Create dummy variable-length data
    batch_size = 32
    num_samples = 128
    max_time = 16
    temporal_dim = 10

    features = torch.randn(num_samples, max_time, temporal_dim)
    lengths = torch.randint(4, max_time + 1, (num_samples,))
    masks = torch.zeros(num_samples, max_time, dtype=torch.bool)
    for i, length in enumerate(lengths):
        masks[i, :length] = True
    encoded_initial = torch.randn(num_samples, 50)

    # Create dataset
    dataset = TensorDataset(features, masks, lengths, encoded_initial)

    def collate_fn(batch):
        features, masks, lengths, encoded_initial = zip(*batch)
        return {
            "features": torch.stack(features),
            "mask": torch.stack(masks),
            "length": torch.stack(lengths),
            "encoded_initial_features": torch.stack(encoded_initial),
        }

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Create temperature scheduler
    temp_scheduler = TemperatureScheduler(start=1.0, end=0.1, total_epochs=10)

    # Create learnable trainer
    print("\n1. Creating learnable trainer with torch.compile...")
    trainer = LearnableVQVAETrainer(
        model=model,
        temp_scheduler=temp_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        assignment_lr=1e-3,
        device="cpu",
        use_torch_compile=True,
        temporal_encoder=temporal_encoder,
        temporal_encoder_output_dim=50,
        encoded_initial_features=encoded_initial.numpy(),
        val_every_n_epochs=1,
        verbose=False,
    )

    # Train for 2 epochs
    print("2. Training for 2 epochs...")
    try:
        history = trainer.train(epochs=2)
        print(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"   Final val loss: {history['val_loss'][-1]:.6f}")
        print("\n✓ PASSED: Learnable variable-length training completed successfully")
        return True
    except Exception as e:
        print(f"\n✗ FAILED: Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: torch.compile with Training")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Fixed-length training", test_fixed_length_training()))
    results.append(("Variable-length training", test_variable_length_training()))
    results.append(("Learnable variable-length training", test_learnable_variable_length_training()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("torch.compile integration works correctly with training")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review failures above")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
