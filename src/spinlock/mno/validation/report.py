"""Results reporting and visualization for MNO-VQ-VAE validation."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mno_vqvae_validator import ValidationResult


class ValidationReport:
    """Generate validation reports and visualizations."""

    @staticmethod
    def generate_report(
        result: "ValidationResult",
        output_dir: Path,
        save_plots: bool = True
    ) -> str:
        """
        Generate markdown report with validation results.

        Args:
            result: ValidationResult from validator
            output_dir: Directory to save report and plots
            save_plots: Whether to save visualization plots

        Returns:
            Markdown-formatted report string
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate markdown report
        report = ValidationReport._generate_markdown(result)

        # Save report
        report_path = output_dir / "validation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Report saved to: {report_path}")

        return report

    @staticmethod
    def _generate_markdown(result: "ValidationResult") -> str:
        """Generate markdown report."""
        status_emoji = "✓" if result.pass_threshold else "✗"

        report = f"""# MNO-VQ-VAE Distribution Alignment Validation Report

## Summary

**Status**: {status_emoji} {"PASS" if result.pass_threshold else "FAIL"}

**Samples Tested**: {result.num_samples}

## Reconstruction Quality

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **MNO Reconstruction Error** | {result.mno_reconstruction_error:.6f} | <{result.cno_reconstruction_error * 2:.6f} | {status_emoji} |
| **CNO Baseline Error** | {result.cno_reconstruction_error:.6f} | - | Reference |
| **Ratio (MNO/CNO)** | {result.reconstruction_ratio:.3f}x | <2.0x | {"✓" if result.reconstruction_ratio < 2.0 else "✗"} |

## Interpretation

### Reconstruction Ratio: {result.reconstruction_ratio:.3f}x

"""

        if result.reconstruction_ratio <= 1.5:
            report += """
**✓ EXCELLENT** - VQ-VAE tokenizes MNO outputs with quality matching CNO training data.
Distribution alignment is excellent. No action needed.
"""
        elif result.reconstruction_ratio <= 2.0:
            report += """
**✓ GOOD** - VQ-VAE tokenizes MNO outputs with acceptable quality.
Minor distribution drift but within acceptable bounds. System is usable.
"""
        elif result.reconstruction_ratio <= 3.0:
            report += """
**⚠ ACCEPTABLE** - VQ-VAE tokenizes MNO outputs but with degraded quality.
Consider improving MNO training to reduce distribution drift.
"""
        else:
            report += """
**✗ POOR** - Significant distribution mismatch detected.
VQ-VAE cannot reliably tokenize MNO outputs. MNO training needs improvement.

**Action Required**: Improve MNO physics fidelity (reduce L_traj and relative_l2).
"""

        report += f"""

## Recommendation

"""
        if result.pass_threshold:
            report += "System is production-ready for downstream NOA experimentation."
        else:
            report += "Improve MNO training before proceeding with NOA integration."

        return report
