#!/usr/bin/env python3
"""Quick test of NOP feature extraction."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spinlock.features.config import FeatureExtractionConfig
from spinlock.features.nop.config import NOPConfig
from spinlock.features.extractor import FeatureExtractor
from spinlock.features.storage import HDF5FeatureReader

def test_nop_extraction():
    """Test NOP extraction on small dataset."""

    # Create configuration
    config = FeatureExtractionConfig(
        input_dataset=Path("datasets/test_5_samples.h5"),
        batch_size=5,
        device="cpu",
        overwrite=True,
        max_samples=5,  # Just test on 5 samples
        nop=NOPConfig(
            version="1.0.0",
            architecture=NOPConfig.model_fields['architecture'].default_factory(),
            stochastic=NOPConfig.model_fields['stochastic'].default_factory(),
            operator=NOPConfig.model_fields['operator'].default_factory(),
            evolution=NOPConfig.model_fields['evolution'].default_factory(),
            stratification=NOPConfig.model_fields['stratification'].default_factory(),
        )
    )

    print("=" * 60)
    print("NOP FEATURE EXTRACTION TEST")
    print("=" * 60)
    print()

    # Create extractor
    try:
        extractor = FeatureExtractor(config)
        print("✓ Extractor initialized successfully")
        print(f"  NOP extractor: {'enabled' if extractor.nop_extractor else 'disabled'}")

        if extractor.nop_extractor:
            registry = extractor.nop_extractor.get_feature_registry()
            print(f"  Feature count: {registry.num_features}")
            print(f"  Categories: {', '.join(registry.categories)}")
        print()

    except Exception as e:
        print(f"✗ Failed to initialize extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run extraction
    try:
        print("Running extraction...")
        extractor.extract(verbose=True)
        print()
        print("✓ Extraction completed successfully")
        print()

    except Exception as e:
        print(f"✗ Failed during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify features were written
    try:
        with HDF5FeatureReader(config.input_dataset) as reader:
            if reader.has_nop():
                print("✓ NOP features found in dataset")

                nop_features = reader.get_nop_features()
                print(f"  Shape: {nop_features.shape}")

                registry = reader.get_nop_registry()
                if registry:
                    print(f"  Features: {registry.num_features}")
                    print()
                    print("Feature registry:")
                    for cat in registry.categories:
                        features = registry.get_features_by_category(cat)
                        print(f"  {cat}: {len(features)} features")
                        for feat in features[:3]:  # Show first 3
                            print(f"    - {feat.name}")
                        if len(features) > 3:
                            print(f"    ... and {len(features) - 3} more")
                    print()
                return True
            else:
                print("✗ NOP features not found in dataset")
                return False

    except Exception as e:
        print(f"✗ Failed to read features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_nop_extraction()
    print()
    if success:
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        sys.exit(1)
