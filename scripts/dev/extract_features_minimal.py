#!/usr/bin/env python3
"""
Minimal feature extraction script for SDF v2.0 features only.

Extracts only the 155 core SDF v2.0 features, disabling expensive
categories (distributional, structural, physics, morphological, multiscale).
"""

from pathlib import Path
from spinlock.features.config import FeatureExtractionConfig
from spinlock.features.sdf.config import SDFConfig
from spinlock.features.extractor import FeatureExtractor

def main():
    # Configure SDF with only v2.0 categories
    sdf_config = SDFConfig(
        # v2.0 core categories (enabled)
        spatial=SDFConfig.model_fields['spatial'].default_factory(),
        spectral=SDFConfig.model_fields['spectral'].default_factory(),
        temporal=SDFConfig.model_fields['temporal'].default_factory(),
        cross_channel=SDFConfig.model_fields['cross_channel'].default_factory(),
        causality=SDFConfig.model_fields['causality'].default_factory(),
        invariant_drift=SDFConfig.model_fields['invariant_drift'].default_factory(),
        operator_sensitivity=SDFConfig.model_fields['operator_sensitivity'].default_factory(),
    )

    # Disable expensive v1.0 categories
    sdf_config.distributional.enabled = False
    sdf_config.structural.enabled = False
    sdf_config.physics.enabled = False
    sdf_config.morphological.enabled = False
    sdf_config.multiscale.enabled = False

    # Create extraction config
    config = FeatureExtractionConfig(
        input_dataset=Path("datasets/vqvae_baseline_10k/vqvae_baseline_10k.h5"),
        output_dataset=None,  # Write in-place
        sdf=sdf_config,
        batch_size=32,
        device="cuda",
        overwrite=True,
        max_samples=None,
        num_workers=4,
        cache_trajectories=True,
    )

    print("="*60)
    print("MINIMAL FEATURE EXTRACTION (SDF v2.0 only)")
    print("="*60)
    print(f"\nDataset: {config.input_dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Device: {config.device}")
    print(f"Overwrite: {config.overwrite}")
    print("\nEnabled categories:")
    print(f"  - spatial: {sdf_config.spatial.enabled}")
    print(f"  - spectral: {sdf_config.spectral.enabled}")
    print(f"  - temporal: {sdf_config.temporal.enabled}")
    print(f"  - cross_channel: {sdf_config.cross_channel.enabled}")
    print(f"  - causality: {sdf_config.causality.enabled}")
    print(f"  - invariant_drift: {sdf_config.invariant_drift.enabled}")
    print(f"  - operator_sensitivity: {sdf_config.operator_sensitivity.enabled}")
    print("\nDisabled categories (expensive):")
    print(f"  - distributional: {sdf_config.distributional.enabled}")
    print(f"  - structural: {sdf_config.structural.enabled}")
    print(f"  - physics: {sdf_config.physics.enabled}")
    print(f"  - morphological: {sdf_config.morphological.enabled}")
    print(f"  - multiscale: {sdf_config.multiscale.enabled}")
    print("="*60 + "\n")

    # Create extractor and run
    extractor = FeatureExtractor(config)
    extractor.extract(verbose=True)

    print("\n✓ Feature extraction complete")

if __name__ == "__main__":
    main()
