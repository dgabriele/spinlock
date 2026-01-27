"""
Output generation for MNO diagnostics.

Handles generation of JSON summaries, CSV files, and HDF5 outputs.
"""

import json
import csv
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class DiagnosticReporter:
    """Generates diagnostic reports in multiple formats."""

    def __init__(self, output_dir: Path):
        """Initialize reporter.

        Args:
            output_dir: Directory to write outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_summary_json(
        self,
        metadata: Dict[str, Any],
        physics_results: Optional[Dict[str, Any]] = None,
        tokenization_results: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Generate JSON summary file.

        Args:
            metadata: Metadata about the evaluation
            physics_results: Physics fidelity results (optional)
            tokenization_results: Tokenization results (optional)

        Returns:
            Path to generated JSON file
        """
        summary = {
            "metadata": {
                **metadata,
                "timestamp": datetime.now().isoformat(),
            },
        }

        if physics_results is not None:
            # Add physics summary
            summary["physics_fidelity"] = self._format_physics_summary(physics_results)

        if tokenization_results is not None:
            # Add tokenization summary
            summary["tokenization"] = self._format_tokenization_summary(tokenization_results)

        # Add overall assessment
        if physics_results is not None and tokenization_results is not None:
            summary["summary"] = self._generate_assessment(physics_results, tokenization_results)

        # Write to file
        output_path = self.output_dir / "summary.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  ✓ Summary JSON written to {output_path}")
        return output_path

    def generate_physics_csv(
        self,
        results: Dict[str, Any],
        split_name: str = "default",
    ) -> Path:
        """Generate CSV file with per-sample physics metrics.

        Args:
            results: Physics evaluation results
            split_name: Name of the split (for filename)

        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / f"physics_errors_{split_name}.csv"

        per_sample = results.get('per_sample_results', [])
        if not per_sample:
            print(f"  ⚠ No per-sample results to write")
            return output_path

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['split', 'sample_idx', 'final_mse', 'mean_mse', 'final_rel_l2']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for sample in per_sample:
                writer.writerow({
                    'split': split_name,
                    **sample,
                })

        print(f"  ✓ Physics CSV written to {output_path} ({len(per_sample)} samples)")
        return output_path

    def generate_tokenization_csv(
        self,
        mno_metrics: Dict[str, Any],
        cno_metrics: Dict[str, Any],
        split_name: str = "default",
    ) -> Path:
        """Generate CSV file with tokenization metrics.

        Args:
            mno_metrics: MNO tokenization metrics
            cno_metrics: CNO tokenization metrics
            split_name: Name of the split

        Returns:
            Path to generated CSV file
        """
        output_path = self.output_dir / f"tokenization_metrics_{split_name}.csv"

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['split', 'source', 'reconstruction_mse', 'perplexity',
                         'token_entropy', 'unique_tokens']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # MNO row
            writer.writerow({
                'split': split_name,
                'source': 'mno',
                'reconstruction_mse': mno_metrics.get('reconstruction_mse', 0.0),
                'perplexity': mno_metrics.get('perplexity', 0.0),
                'token_entropy': mno_metrics.get('token_entropy', 0.0),
                'unique_tokens': mno_metrics.get('unique_tokens', 0),
            })

            # CNO row
            writer.writerow({
                'split': split_name,
                'source': 'cno',
                'reconstruction_mse': cno_metrics.get('reconstruction_mse', 0.0),
                'perplexity': cno_metrics.get('perplexity', 0.0),
                'token_entropy': cno_metrics.get('token_entropy', 0.0),
                'unique_tokens': cno_metrics.get('unique_tokens', 0),
            })

        print(f"  ✓ Tokenization CSV written to {output_path}")
        return output_path

    def _format_physics_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format physics results for JSON summary."""
        summary = {}

        # Check if we have split-level results
        for split in ['train', 'holdout']:
            if split in results:
                split_results = results[split]
                summary[split] = {
                    'mean_mse': split_results.get('mean_mse', 0.0),
                    'std_mse': split_results.get('std_mse', 0.0),
                    'final_mse': split_results.get('final_mse', 0.0),
                    'relative_l2': split_results.get('relative_l2', 0.0),
                    'nrmse': split_results.get('nrmse', 0.0),
                    'error_growth': split_results.get('error_growth', {}),
                    'per_timestep_mse': split_results.get('per_timestep_mse', []),
                }

        # If no splits, treat as single result
        if not summary:
            summary['overall'] = {
                'mean_mse': results.get('mean_mse', 0.0),
                'std_mse': results.get('std_mse', 0.0),
                'final_mse': results.get('final_mse', 0.0),
                'relative_l2': results.get('relative_l2', 0.0),
                'nrmse': results.get('nrmse', 0.0),
                'error_growth': results.get('error_growth', {}),
            }

        # Add generalization metrics if we have both splits
        if 'train' in summary and 'holdout' in summary:
            train_mse = summary['train']['mean_mse']
            holdout_mse = summary['holdout']['mean_mse']
            summary['generalization'] = {
                'overfitting_ratio': holdout_mse / (train_mse + 1e-10),
                'is_overfitting': (holdout_mse / (train_mse + 1e-10)) > 1.5,
            }

        return summary

    def _format_tokenization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format tokenization results for JSON summary."""
        summary = {}

        # Check if we have split-level results
        for split in ['train', 'holdout']:
            if split in results:
                split_results = results[split]
                summary[f'mno_{split}'] = split_results.get('mno', {})
                summary[f'cno_{split}'] = split_results.get('cno', {})
                summary[f'distribution_comparison_{split}'] = split_results.get('distribution_comparison', {})

        # If no splits, treat as single result
        if not summary:
            summary['mno'] = results.get('mno', {})
            summary['cno'] = results.get('cno', {})
            summary['distribution_comparison'] = results.get('distribution_comparison', {})

        return summary

    def _generate_assessment(
        self,
        physics_results: Dict[str, Any],
        tokenization_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate overall quality assessment and recommendations.

        Args:
            physics_results: Physics evaluation results
            tokenization_results: Tokenization evaluation results

        Returns:
            Dictionary with assessment and recommendations
        """
        # Extract key metrics
        physics_mse = physics_results.get('mean_mse', 0.0)
        physics_rel_l2 = physics_results.get('relative_l2', 0.0)

        # Get tokenization metrics (handle different result structures)
        mno_metrics = tokenization_results.get('mno', tokenization_results.get('mno_train', {}))
        cno_metrics = tokenization_results.get('cno', tokenization_results.get('cno_train', {}))

        mno_recon_mse = mno_metrics.get('reconstruction_mse', 0.0)
        cno_recon_mse = cno_metrics.get('reconstruction_mse', 0.0)

        dist_comparison = tokenization_results.get('distribution_comparison',
                                                   tokenization_results.get('distribution_comparison_train', {}))
        kl_div = dist_comparison.get('kl_mno_to_cno', 0.0)

        # Assess physics quality
        if physics_mse < 0.01:
            physics_quality = "excellent"
        elif physics_mse < 0.1:
            physics_quality = "good"
        elif physics_mse < 0.5:
            physics_quality = "fair"
        else:
            physics_quality = "poor"

        # Assess tokenization quality
        recon_degradation = mno_recon_mse / (cno_recon_mse + 1e-10)
        if recon_degradation < 1.2 and kl_div < 0.1:
            tokenization_quality = "excellent"
        elif recon_degradation < 2.0 and kl_div < 0.3:
            tokenization_quality = "good"
        elif recon_degradation < 3.0 and kl_div < 0.5:
            tokenization_quality = "fair"
        else:
            tokenization_quality = "poor"

        # Generate recommendations
        recommendations = []

        if physics_quality in ["poor", "fair"]:
            recommendations.append(
                f"Physics fidelity is {physics_quality} (MSE={physics_mse:.4f}). "
                "Consider increasing training data or model capacity."
            )

        if tokenization_quality in ["poor", "fair"]:
            recommendations.append(
                f"Tokenization quality is {tokenization_quality} "
                f"(MNO recon MSE={mno_recon_mse:.4f} vs CNO={cno_recon_mse:.4f}). "
                "MNO outputs may not align well with VQ-VAE vocabulary."
            )

        if kl_div > 0.3:
            recommendations.append(
                f"Large distribution shift detected (KL divergence={kl_div:.4f}). "
                "MNO is generating different dynamics than CNO training distribution."
            )

        # Check for overfitting
        gen_metrics = physics_results.get('generalization', {})
        if gen_metrics.get('is_overfitting', False):
            overfitting_ratio = gen_metrics.get('overfitting_ratio', 1.0)
            recommendations.append(
                f"Overfitting detected (holdout/train ratio={overfitting_ratio:.2f}). "
                "Consider regularization or more training data."
            )

        if not recommendations:
            recommendations.append("Model performance looks good! No major issues detected.")

        return {
            'physics_quality': physics_quality,
            'tokenization_quality': tokenization_quality,
            'recommendations': recommendations,
            'key_metrics': {
                'physics_mse': physics_mse,
                'physics_rel_l2': physics_rel_l2,
                'mno_reconstruction_mse': mno_recon_mse,
                'cno_reconstruction_mse': cno_recon_mse,
                'kl_divergence': kl_div,
            }
        }
