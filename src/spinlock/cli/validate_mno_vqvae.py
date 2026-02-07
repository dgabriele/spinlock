"""CLI command for MNO-VQ-VAE distribution alignment validation."""

from pathlib import Path
import argparse

from spinlock.cli.base import CLICommand
from spinlock.mno.validation.mno_vqvae_validator import MNOVQVAEValidator
from spinlock.mno.validation.config import ValidationConfig
from spinlock.mno.validation.report import ValidationReport


class ValidateMNOVQVAECommand(CLICommand):
    """Validate VQ-VAE tokenization quality on MNO rollouts."""

    @property
    def name(self) -> str:
        return "validate-mno-vqvae"

    @property
    def help(self) -> str:
        return "Validate VQ-VAE can tokenize MNO outputs with acceptable quality"

    @property
    def description(self) -> str:
        return """
Validate MNO-VQ-VAE Distribution Alignment

Tests whether the VQ-VAE (trained on CNO ground truth) can reliably tokenize
MNO-generated rollouts with acceptable reconstruction quality.

This validates the core assumption that the MNO's physics fidelity is sufficient
for downstream tokenization and symbolic reasoning.

Pass Criteria:
  - Reconstruction ratio (MNO/CNO) < 2.0x
  - This ensures VQ-VAE token semantics are preserved on MNO outputs

Example:
  spinlock validate-mno-vqvae \\
      --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
      --vqvae-checkpoint checkpoints/vqvae/50k_baseline/best_model.pt \\
      --dataset datasets/cno_50k_v3_1.h5 \\
      --num-samples 100 \\
      --output-dir validation_results/
        """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mno-checkpoint",
            type=Path,
            required=True,
            help="Path to trained MNO checkpoint"
        )

        parser.add_argument(
            "--vqvae-checkpoint",
            type=Path,
            required=True,
            help="Path to trained VQ-VAE checkpoint"
        )

        parser.add_argument(
            "--dataset",
            type=Path,
            required=True,
            help="Path to validation dataset (HDF5)"
        )

        parser.add_argument(
            "--num-samples",
            type=int,
            default=100,
            help="Number of samples to validate (default: 100)"
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=8,
            help="Batch size for inference (default: 8)"
        )

        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("validation_results/mno_vqvae"),
            help="Output directory for reports and plots"
        )

        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="Torch device (default: cuda)"
        )

    def execute(self, args: argparse.Namespace) -> int:
        """Execute validation."""
        # Create config
        config = ValidationConfig(
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device
        )

        # Create validator
        validator = MNOVQVAEValidator(
            mno_checkpoint=args.mno_checkpoint,
            vqvae_checkpoint=args.vqvae_checkpoint,
            config=config,
            device=args.device
        )

        # Run validation
        result = validator.validate(
            dataset_path=args.dataset,
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )

        # Generate report
        ValidationReport.generate_report(
            result=result,
            output_dir=args.output_dir,
            save_plots=True
        )

        # Exit code: 0 if pass, 1 if fail
        return 0 if result.pass_threshold else 1
