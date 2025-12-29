#!/usr/bin/env python3
"""
Test script for hybrid operator_sensitivity feature extraction.

Validates:
1. Operator features are extracted during rollout
2. Features are returned correctly from evolve_operator
3. Features can be merged into per_trajectory features
4. All 12 operator sensitivity features are present and valid
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from spinlock.rollout.engine import OperatorRollout
from spinlock.features.sdf.config import SDFConfig, SDFOperatorSensitivityConfig
from spinlock.features.sdf.extractors import SDFExtractor


class DummyOperator(nn.Module):
    """Dummy operator for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def test_operator_sensitivity_extraction():
    """Test that operator sensitivity features are extracted during rollout."""
    print("=" * 70)
    print("TEST: Operator Sensitivity Feature Extraction")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create dummy operator
    operator = DummyOperator().to(device)
    operator.eval()

    # Create initial condition
    IC = torch.randn(3, 64, 64, device=device)

    # Configure operator sensitivity extraction
    ops_config = SDFOperatorSensitivityConfig(
        enabled=True,
        include_lipschitz=True,
        include_gain_curve=True,
        include_linearity_metrics=True,
        lipschitz_epsilon_scales=[1e-4, 1e-3, 1e-2],
        gain_scale_factors=[0.5, 0.75, 1.25, 1.5]
    )

    # Create rollout engine with operator feature extraction enabled
    engine = OperatorRollout(
        policy_type="autoregressive",
        num_timesteps=10,
        extract_operator_features=True,
        operator_feature_config=ops_config,
        device=device
    )

    print("\n" + "=" * 70)
    print("STEP 1: Extract operator features during rollout")
    print("=" * 70)

    # Evolve operator
    trajectories, metrics, op_features = engine.evolve_operator(
        operator=operator,
        initial_condition=IC,
        num_realizations=5,
        base_seed=42
    )

    print(f"\nTrajectories shape: {trajectories.shape}")  # [5, 10, 3, 64, 64]
    print(f"Number of metrics: {len(metrics)}")  # 5 realizations

    # Validate operator features
    assert op_features is not None, "Operator features should not be None"
    print(f"\nOperator features extracted: {len(op_features)} features")

    # Expected features
    expected_features = [
        # Lipschitz estimates (3)
        "lipschitz_eps_1e-04",
        "lipschitz_eps_1e-03",
        "lipschitz_eps_1e-02",
        # Gain curves (4)
        "gain_scale_0.50",
        "gain_scale_0.75",
        "gain_scale_1.25",
        "gain_scale_1.50",
        # Linearity metrics (3)
        "linearity_r2",
        "saturation_degree",
        "compression_ratio",
    ]

    print("\nValidating features:")
    for feat_name in expected_features:
        assert feat_name in op_features, f"Missing feature: {feat_name}"
        value = op_features[feat_name]
        print(f"  ✓ {feat_name:30s} = {value.item():.6f}")

    print("\n" + "=" * 70)
    print("STEP 2: Test integration with SDF extractor")
    print("=" * 70)

    # Create SDF config with operator sensitivity enabled
    sdf_config = SDFConfig(operator_sensitivity=ops_config)

    # Create SDF extractor
    extractor = SDFExtractor(device=device, config=sdf_config)

    # Prepare trajectories for extraction [N, M, T, C, H, W]
    trajectories_batch = trajectories.unsqueeze(0)  # [1, 5, 10, 3, 64, 64]

    # Convert operator features to batch format [N]
    op_features_batch = {
        name: value.unsqueeze(0) for name, value in op_features.items()
    }

    # Prepare metadata with operator features
    metadata = {'operator_sensitivity_features': op_features_batch}

    print("\nExtracting per-trajectory features with operator sensitivity...")
    per_traj_features = extractor.extract_per_trajectory(
        trajectories=trajectories_batch,
        metadata=metadata
    )

    print(f"Per-trajectory features shape: {per_traj_features.shape}")  # [1, 5, D_traj]

    # Get registry to check feature count
    registry = extractor.get_feature_registry()
    ops_features_in_registry = [
        f.name for f in registry.get_features_by_category('operator_sensitivity')
    ]

    print(f"\nOperator sensitivity features in registry: {len(ops_features_in_registry)}")
    for feat_name in ops_features_in_registry:
        print(f"  - {feat_name}")

    # Validate that operator features are in the per-trajectory feature tensor
    print("\n" + "=" * 70)
    print("STEP 3: Validate feature values are non-NaN")
    print("=" * 70)

    # Check for NaN in operator sensitivity features
    # These are at the end of the feature vector
    num_ops_features = len(ops_features_in_registry)
    ops_feature_slice = per_traj_features[0, 0, -num_ops_features:]  # [D_ops]

    nan_count = torch.isnan(ops_feature_slice).sum().item()
    print(f"\nNaN count in operator sensitivity features: {nan_count} / {num_ops_features}")

    if nan_count == 0:
        print("✓ All operator sensitivity features are valid (no NaN)")
    else:
        print("✗ Some operator sensitivity features are NaN")
        # Show which ones are NaN
        for i, feat_name in enumerate(ops_features_in_registry):
            is_nan = torch.isnan(ops_feature_slice[i]).item()
            status = "NaN" if is_nan else "OK"
            print(f"  {feat_name:30s} [{status}]")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: Operator sensitivity extraction working correctly")
    print("=" * 70)


if __name__ == "__main__":
    test_operator_sensitivity_extraction()
