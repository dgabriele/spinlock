"""
MNO-VQ-VAE Distribution Alignment Validation

Validates that VQ-VAE (trained on CNO ground truth) can reliably tokenize
MNO-generated rollouts with acceptable reconstruction quality.

Core Components:
- MNOVQVAEValidator: Main validation orchestrator
- ValidationMetrics: Metrics computation
- ValidationReport: Results reporting and visualization
- ValidationConfig: Configuration schemas

Usage:
    from spinlock.mno.validation import MNOVQVAEValidator, ValidationConfig

    validator = MNOVQVAEValidator(
        mno_checkpoint="checkpoints/mno/50k_baseline/meta_operator_best.pt",
        vqvae_checkpoint="checkpoints/vqvae/50k_baseline/best_model.pt"
    )

    result = validator.validate(
        dataset_path="datasets/cno_50k_v3_1.h5",
        num_samples=100
    )

    if result.pass_threshold:
        print("✓ VQ-VAE can tokenize MNO outputs")
"""

from .config import ValidationConfig
from .metrics import ValidationMetrics
from .mno_vqvae_validator import MNOVQVAEValidator, ValidationResult
from .report import ValidationReport

__all__ = [
    "ValidationConfig",
    "ValidationMetrics",
    "ValidationReport",
    "MNOVQVAEValidator",
    "ValidationResult",
]
