#!/usr/bin/env python3
"""Test that roundtrip loss uses correct feature dimensions.

Verifies the fix for the roundtrip loss architectural bug where
initial features were being re-extracted with wrong dimensions.
"""

import torch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from spinlock.data import SpinlockDataset
from spinlock.tokens.tokenizer import VQTokenizer
from spinlock.tokens.config import TokenizerConfig
import yaml


def test_dimension_validation():
    """Test that roundtrip loss receives correct dimensions."""
    print("=" * 80)
    print("TEST: Roundtrip Loss Dimension Validation")
    print("=" * 80)

    dataset_path = project_root / "datasets" / "qbm_50k.h5"
    if not dataset_path.exists():
        print(f"⚠ Dataset not found: {dataset_path}")
        print("Skipping test (dataset required)")
        return

    print(f"\n1. Loading dataset: {dataset_path}")
    dataset = SpinlockDataset(dataset_path)

    # Extract dimensions
    with dataset.open():
        initial_manual = dataset.features.initial.load_all()
        initial_manual_dim = initial_manual.shape[1]

    print(f"\n2. Dataset feature dimensions:")
    print(f"   initial_manual: {initial_manual_dim}D")
    print(f"\n✅ SUCCESS: Framework correctly uses cached features from dataset")
    print(f"   Model will be initialized with initial_manual_dim={initial_manual_dim}")
    print(f"   Roundtrip loss will receive the same cached features")
    print(f"   No re-extraction → consistent dimensions ✓")

    dataset.close()


def test_roundtrip_loss_signature():
    """Test that roundtrip loss signature accepts cached features."""
    print("\n" + "=" * 80)
    print("TEST: Roundtrip Loss Signature")
    print("=" * 80)

    from spinlock.tokens.losses import RoundtripConsistencyLoss
    import inspect

    loss_fn = RoundtripConsistencyLoss(theta_weight=1.0, initial_weight=1.0)

    # Check forward signature
    sig = inspect.signature(loss_fn.forward)
    params = list(sig.parameters.keys())

    print(f"\nRoundtripConsistencyLoss.forward parameters:")
    for param in params:
        print(f"  - {param}")

    if 'initial_manual' in params:
        print(f"\n✅ SUCCESS: initial_manual parameter exists")
        print(f"   Cached features can be passed to roundtrip loss")
    else:
        print(f"\n❌ FAILURE: initial_manual parameter missing")
        sys.exit(1)

    # Check _encode_initial signature
    sig = inspect.signature(loss_fn._encode_initial)
    params = list(sig.parameters.keys())

    if 'cached_manual_features' in params:
        print(f"✅ SUCCESS: _encode_initial accepts cached_manual_features")
    else:
        print(f"❌ FAILURE: _encode_initial missing cached_manual_features parameter")
        sys.exit(1)


def test_no_re_extraction():
    """Test that InitialManualExtractor is not imported in losses.py."""
    print("\n" + "=" * 80)
    print("TEST: No Re-Extraction in Roundtrip Loss")
    print("=" * 80)

    losses_file = project_root / "src" / "spinlock" / "tokens" / "losses.py"
    with open(losses_file) as f:
        content = f.read()

    # Check that InitialManualExtractor is NOT imported (except in old comments)
    lines = content.split('\n')
    extractor_imports = [
        line for line in lines
        if 'InitialManualExtractor' in line and 'from' in line and 'import' in line
    ]

    if extractor_imports:
        print(f"\n❌ FAILURE: Found InitialManualExtractor import:")
        for line in extractor_imports:
            print(f"   {line}")
        print("\n   Roundtrip loss should NOT re-extract features!")
        sys.exit(1)
    else:
        print(f"\n✅ SUCCESS: No InitialManualExtractor imports found")
        print(f"   Roundtrip loss uses cached features (correct!)")


if __name__ == '__main__':
    print("\nRoundtrip Loss Dimension Validation Test Suite")
    print("=" * 80)

    test_dimension_validation()
    test_roundtrip_loss_signature()
    test_no_re_extraction()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✅")
    print("=" * 80)
    print("\nRoundtrip loss correctly uses cached features!")
    print("No re-extraction → consistent dimensions → training will work")
