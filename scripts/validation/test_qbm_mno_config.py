#!/usr/bin/env python3
"""Verification script for framework-compliant QBM MNO configuration.

Tests:
1. Config loads with None dimensions (framework-compliant)
2. QBMReplayer initializes from substrate config
3. Dimension inference detects all dimensions from dataset
4. MNO config validates successfully after inference
5. No hardcoded dimensions in YAML

Usage:
    poetry run python scripts/validation/test_qbm_mno_config.py
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from spinlock.data import SpinlockDataset
from spinlock.mno.config import MNOConfig
from spinlock.qbm import QBMReplayer


def test_config_loads_without_dimensions():
    """Test 1: Config loads with None dimensions."""
    print("Test 1: Config loads without hardcoded dimensions")

    with open('configs/qbm/mno/baseline.yaml') as f:
        config_dict = yaml.safe_load(f)

    # Verify data-dependent dimensions are NOT hardcoded
    assert config_dict['model'].get('in_channels') is None, "in_channels should be None (auto-detect)"
    assert config_dict['model'].get('out_channels') is None, "out_channels should be None (auto-detect)"
    assert config_dict['model'].get('param_dim') is None, "param_dim should be None (auto-detect)"
    assert config_dict['model'].get('spatial_dim') is None, "spatial_dim should be None (auto-detect)"

    print("  ✓ All data-dependent dimensions are None (framework-compliant)")
    print()
    return config_dict


def test_qbm_replayer_initialization():
    """Test 2: QBMReplayer initializes from substrate config."""
    print("Test 2: QBMReplayer initialization")

    replayer = QBMReplayer.from_config('configs/qbm/substrate.yaml')

    # Verify replayer components
    assert replayer.grid_size == 64
    assert replayer.domain_size == 10.0
    assert len(replayer.param_names) >= 6  # At least 6 core parameters
    assert 'gamma' in replayer.param_names
    assert 'kT' in replayer.param_names
    assert 'potential_type' in replayer.param_names

    print(f"  ✓ QBMReplayer initialized: {replayer}")
    print(f"    Parameters: {replayer.param_names[:6]}")
    print()


def test_dimension_inference(config_dict):
    """Test 3: Dimension inference detects all dimensions."""
    print("Test 3: Dimension inference on QBM dataset")

    # Check if dataset exists
    dataset_path = Path('datasets/qbm_50k.h5')
    if not dataset_path.exists():
        print(f"  ⚠ Dataset not found: {dataset_path}")
        print("  Skipping dimension inference test (requires dataset)")
        return config_dict

    # Infer dimensions
    dataset, config_dict = SpinlockDataset.infer_and_update_config(
        config_dict,
        dataset_path,
        verbose=False
    )

    # Apply MNO-specific dimension inference
    mno_dimensions = dataset.infer_mno_dimensions()
    for key, value in mno_dimensions.items():
        if isinstance(value, dict):
            config_dict.setdefault(key, {}).update(value)
        else:
            config_dict[key] = value

    # Verify dimensions were detected
    assert config_dict['model']['in_channels'] == 2, "QBM should have 2 channels (Re/Im)"
    assert config_dict['model']['out_channels'] == 2
    assert config_dict['model']['param_dim'] == 9, "QBM should have 9 parameters"
    assert config_dict['model']['spatial_dim'] == 64, "QBM should have 64x64 grid"
    assert config_dict['training']['max_timesteps'] == 256, "QBM should have 256 timesteps"

    print("  ✓ All dimensions detected:")
    print(f"    in_channels: {config_dict['model']['in_channels']}")
    print(f"    out_channels: {config_dict['model']['out_channels']}")
    print(f"    param_dim: {config_dict['model']['param_dim']}")
    print(f"    spatial_dim: {config_dict['model']['spatial_dim']}")
    print(f"    max_timesteps: {config_dict['training']['max_timesteps']}")
    print()

    dataset.close()
    return config_dict


def test_mno_config_validation(config_dict):
    """Test 4: MNO config validates successfully."""
    print("Test 4: Pydantic config validation")

    # Create and validate config
    config = MNOConfig(**config_dict)
    max_timesteps = config_dict.get('training', {}).get('max_timesteps')
    config.validate(max_timesteps=max_timesteps)

    # Verify config structure
    assert config.model.in_channels is not None, "in_channels should be resolved"
    assert config.model.param_dim is not None, "param_dim should be resolved"
    assert config.model.base_channels == 40, "Algorithmic choice preserved"
    assert config.model.conditioning_mode == "film", "Algorithmic choice preserved"

    print("  ✓ Config validated successfully")
    print(f"    Model: {config.model.in_channels} channels, {config.model.param_dim} params")
    print(f"    Training: {config.training.timesteps} timesteps, batch_size={config.training.batch_size}")
    print()


def main():
    """Run all verification tests."""
    print("=" * 80)
    print("QBM MNO Configuration Verification")
    print("=" * 80)
    print()

    try:
        # Run tests
        config_dict = test_config_loads_without_dimensions()
        test_qbm_replayer_initialization()
        config_dict = test_dimension_inference(config_dict)
        test_mno_config_validation(config_dict)

        # Success
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print()
        print("Framework compliance verified:")
        print("  - No hardcoded dimensions in config")
        print("  - All dimensions auto-detected from dataset")
        print("  - QBMReplayer works with substrate config")
        print("  - MNO config validates successfully")
        print()
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
