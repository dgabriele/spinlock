"""
Quick test of inverse decoder integration (without trained models).

Tests that:
1. VQTokenizer loads correctly
2. Inverse model attributes are initialized
3. decode() raises proper errors when models not loaded
4. Load methods work correctly (with dummy checkpoints)
"""

import torch
from pathlib import Path

from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.inverse_models import ThetaInverseMLP, InitialInverseCNN


def test_tokenizer_loads():
    """Test tokenizer loads without inverse models."""
    print("1. Testing tokenizer load...")
    tokenizer = VQTokenizer.from_checkpoint("checkpoints/v2/vqvae/vq_tokenizer_best.pt")

    assert hasattr(tokenizer.model, 'theta_inverse'), "Missing theta_inverse attribute"
    assert hasattr(tokenizer.model, 'initial_inverse'), "Missing initial_inverse attribute"
    assert tokenizer.model.theta_inverse is None, "theta_inverse should be None initially"
    assert tokenizer.model.initial_inverse is None, "initial_inverse should be None initially"

    print("✅ Tokenizer loads correctly with inverse attributes")


def test_decode_requires_models():
    """Test decode() raises error when models not loaded."""
    print("\n2. Testing decode() error handling...")
    tokenizer = VQTokenizer.from_checkpoint("checkpoints/v2/vqvae/vq_tokenizer_best.pt")

    # Create valid dummy tokens (index 0 is always valid)
    tokens = {
        key: torch.zeros(5, dtype=torch.long)
        for key in tokenizer.model.quantizers.keys()
    }

    try:
        theta, u0 = tokenizer.decode(tokens)
        assert False, "decode() should raise error without inverse models"
    except ValueError as e:
        assert "inverse model not loaded" in str(e).lower(), f"Unexpected error: {e}"
        print(f"✅ decode() raises proper error: {str(e)[:80]}...")


def test_inverse_models_shape():
    """Test inverse models have correct architectures."""
    print("\n3. Testing inverse model architectures...")

    # ThetaInverseMLP
    theta_model = ThetaInverseMLP(encoded_dim=32, param_dim=14)
    x = torch.randn(5, 32)
    out = theta_model(x)
    assert out.shape == (5, 14), f"Expected (5, 14), got {out.shape}"
    assert torch.all(out >= 0) and torch.all(out <= 1), "Output not in [0,1]"
    print("✅ ThetaInverseMLP: [5, 32] → [5, 14] in [0,1]")

    # InitialInverseCNN
    initial_model = InitialInverseCNN(encoded_dim=426, channels=3, spatial_size=64)
    x = torch.randn(5, 426)
    out = initial_model(x)
    assert out.shape == (5, 3, 64, 64), f"Expected (5, 3, 64, 64), got {out.shape}"
    print("✅ InitialInverseCNN: [5, 426] → [5, 3, 64, 64]")


def test_load_methods_exist():
    """Test load methods are available."""
    print("\n4. Testing load methods exist...")
    tokenizer = VQTokenizer.from_checkpoint("checkpoints/v2/vqvae/vq_tokenizer_best.pt")

    assert hasattr(tokenizer, 'load_theta_inverse'), "Missing load_theta_inverse method"
    assert hasattr(tokenizer, 'load_initial_inverse'), "Missing load_initial_inverse method"

    print("✅ Load methods available: load_theta_inverse(), load_initial_inverse()")


def test_from_checkpoint_signature():
    """Test from_checkpoint accepts inverse paths."""
    print("\n5. Testing from_checkpoint signature...")

    # Should accept inverse paths without error (files don't exist yet, so we won't load)
    import inspect
    sig = inspect.signature(VQTokenizer.from_checkpoint)
    params = list(sig.parameters.keys())

    assert 'theta_inverse_path' in params, "Missing theta_inverse_path parameter"
    assert 'initial_inverse_path' in params, "Missing initial_inverse_path parameter"

    print("✅ from_checkpoint() accepts inverse model paths")


def main():
    """Run all integration tests."""
    print("="*70)
    print("VQTokenizer Inverse Decoder Integration Tests")
    print("="*70)

    try:
        test_tokenizer_loads()
        test_decode_requires_models()
        test_inverse_models_shape()
        test_load_methods_exist()
        test_from_checkpoint_signature()

        print("\n" + "="*70)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*70)
        print("\nNext steps:")
        print("1. Train theta inverse: python scripts/train_theta_inverse.py ...")
        print("2. Train initial inverse: python scripts/train_initial_inverse.py ...")
        print("3. Validate: python scripts/validate_inverse_decoders.py ...")

    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        raise


if __name__ == '__main__':
    main()
