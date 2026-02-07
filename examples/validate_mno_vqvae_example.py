#!/usr/bin/env python
"""
Example script demonstrating MNO-VQ-VAE validation.

This script shows how to programmatically use the MNO-VQ-VAE validator
to verify distribution alignment between MNO outputs and VQ-VAE tokenization.
"""

from pathlib import Path
from spinlock.mno.validation import MNOVQVAEValidator, ValidationConfig, ValidationReport


def main():
    """Run MNO-VQ-VAE validation example."""
    # Configuration
    config = ValidationConfig(
        num_samples=100,
        batch_size=8,
        max_reconstruction_ratio=2.0,
        device="cuda",
        rollout_steps=256
    )

    # Checkpoint paths (update these to your actual paths)
    mno_checkpoint = Path("checkpoints/mno/50k_baseline/meta_operator_best.pt")
    vqvae_checkpoint = Path("checkpoints/vqvae/50k_baseline/best_model.pt")
    dataset_path = Path("datasets/cno_50k_v3_1.h5")
    output_dir = Path("validation_results/mno_vqvae")

    # Create validator
    print("Initializing MNO-VQ-VAE validator...")
    validator = MNOVQVAEValidator(
        mno_checkpoint=mno_checkpoint,
        vqvae_checkpoint=vqvae_checkpoint,
        config=config,
        device="cuda"
    )

    # Run validation
    print("\nRunning validation...")
    result = validator.validate(
        dataset_path=dataset_path,
        num_samples=config.num_samples,
        batch_size=config.batch_size
    )

    # Generate report
    print("\nGenerating report...")
    ValidationReport.generate_report(
        result=result,
        output_dir=output_dir,
        save_plots=True
    )

    # Print results
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"\nReconstruction Ratio: {result.reconstruction_ratio:.3f}x")
    print(f"Pass Threshold: {result.pass_threshold}")

    if result.pass_threshold:
        print("\n✓ SUCCESS: VQ-VAE can reliably tokenize MNO outputs")
        print("  → System is ready for downstream NOA experimentation")
    else:
        print("\n✗ FAILURE: Distribution mismatch detected")
        print("  → MNO training needs improvement")
        print(f"  → Current ratio: {result.reconstruction_ratio:.3f}x")
        print(f"  → Target ratio: <{config.max_reconstruction_ratio}x")

    print(f"\nDetailed report saved to: {output_dir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
