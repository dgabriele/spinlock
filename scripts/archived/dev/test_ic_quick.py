#!/usr/bin/env python3
"""Quick validation test for IC feature extraction."""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spinlock.features.initial.config import InitialConfig
from spinlock.features.initial.extractors import InitialExtractor

def test_ic_extraction():
    """Test IC extraction pipeline."""
    print("=" * 60)
    print("IC FEATURE EXTRACTION QUICK TEST")
    print("=" * 60)
    print()

    # Create synthetic ICs [B=2, M=3, C=1, H=128, W=128]
    print("Creating synthetic ICs [2, 3, 1, 128, 128]...")
    ics = torch.randn(2, 3, 1, 128, 128)

    # Create config
    print("Initializing IC extractor...")
    config = InitialConfig()
    extractor = InitialExtractor(config=config, device=torch.device('cpu'))

    print(f"  Manual extractor: {'enabled' if extractor.manual_extractor else 'disabled'}")
    print(f"  CNN extractor: {'enabled' if extractor.cnn_encoder else 'disabled'}")
    print(f"  Generative mode: {extractor.is_generative}")
    print()

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        features = extractor.extract_all(ics)

    # Check outputs
    print("Results:")
    if features['manual'] is not None:
        print(f"  Manual features: {features['manual'].shape}")
        print(f"    Mean: {features['manual'].mean():.3f}, Std: {features['manual'].std():.3f}")

    if features['cnn'] is not None:
        print(f"  CNN features: {features['cnn'].shape}")
        print(f"    Mean: {features['cnn'].mean():.3f}, Std: {features['cnn'].std():.3f}")

    if features['combined'] is not None:
        print(f"  Combined features: {features['combined'].shape}")
        print(f"    Expected: [2, 3, 42]")

    # Check registry
    print()
    print("Feature registry:")
    registry = extractor.get_feature_registry()
    print(f"  Total features: {registry.num_features}")
    print(f"  Categories: {', '.join(registry.categories)}")

    for cat in registry.categories:
        cat_features = registry.get_features_by_category(cat)
        print(f"    {cat}: {len(cat_features)} features")

    # Validate
    print()
    assert features['manual'].shape == (2, 3, 14), f"Expected [2, 3, 14], got {features['manual'].shape}"
    assert features['cnn'].shape == (2, 3, 28), f"Expected [2, 3, 28], got {features['cnn'].shape}"
    assert features['combined'].shape == (2, 3, 42), f"Expected [2, 3, 42], got {features['combined'].shape}"
    assert registry.num_features == 42, f"Expected 42 features, got {registry.num_features}"

    print("✓ All assertions passed!")
    print()

    # Test VAE mode
    print("Testing VAE mode (generative)...")
    config_vae = InitialConfig()
    config_vae.cnn.use_vae = True
    extractor_vae = InitialExtractor(config=config_vae, device=torch.device('cpu'))
    print(f"  Generative mode: {extractor_vae.is_generative}")

    # Sample ICs
    print("  Sampling ICs from prior...")
    sampled_ics = extractor_vae.sample_ics(num_operators=2, num_realizations=3)
    print(f"  Sampled ICs shape: {sampled_ics.shape}")
    assert sampled_ics.shape == (2, 3, 1, 128, 128), f"Expected [2, 3, 1, 128, 128], got {sampled_ics.shape}"
    print("  ✓ IC sampling works!")

    print()
    print("=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    test_ic_extraction()
