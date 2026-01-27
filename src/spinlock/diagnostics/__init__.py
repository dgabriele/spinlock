"""
Diagnostics package for evaluating trained models.

This package provides tools for comprehensive model evaluation including:
- Physics fidelity evaluation (MNO vs CNO comparison)
- Tokenization generalization assessment
- Distribution shift detection
- Performance reporting and visualization
"""

from .mno_evaluator import MNODiagnosticEvaluator
from .metrics import (
    compute_physics_metrics,
    compute_tokenization_metrics,
    compute_distribution_metrics,
)

__all__ = [
    "MNODiagnosticEvaluator",
    "compute_physics_metrics",
    "compute_tokenization_metrics",
    "compute_distribution_metrics",
]
