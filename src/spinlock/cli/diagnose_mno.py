"""
MNO diagnostics CLI command.

Comprehensive evaluation of trained MNO models including physics fidelity
and VQ-VAE tokenization generalization.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

from spinlock.cli.base import CLICommand
from spinlock.diagnostics.mno_evaluator import MNODiagnosticEvaluator
from spinlock.diagnostics.reporter import DiagnosticReporter
from spinlock.diagnostics.visualizations import (
    plot_physics_fidelity,
    plot_tokenization_quality,
    plot_split_comparison,
)


class DiagnoseMNOCommand(CLICommand):
    """Comprehensive MNO diagnostics with physics and tokenization evaluation."""

    @property
    def name(self) -> str:
        return "diagnose-mno"

    @property
    def help(self) -> str:
        return "Evaluate trained MNO on physics fidelity and VQ tokenization"

    @property
    def description(self) -> str:
        return """
Comprehensive MNO diagnostics that evaluates:
1. Physics fidelity: MNO vs CNO rollout comparison
2. Tokenization generalization: VQ-VAE reconstruction quality on MNO outputs
3. Distribution shift: MNO token distributions vs CNO distributions

Compares performance on training samples (0-10K) vs held-out samples (10K-50K)
to detect overfitting and assess generalization.
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command arguments."""
        # Required arguments
        required = parser.add_argument_group('required arguments')
        required.add_argument(
            '--mno-checkpoint',
            type=Path,
            required=True,
            help='Path to MNO checkpoint (.pt file)',
        )
        required.add_argument(
            '--vqvae-checkpoint',
            type=Path,
            required=True,
            help='Path to VQ-VAE checkpoint directory',
        )
        required.add_argument(
            '--dataset',
            type=Path,
            required=True,
            help='Path to CNO dataset HDF5 file',
        )
        required.add_argument(
            '--substrate-config',
            type=Path,
            required=True,
            help='Path to CNO generation config YAML',
        )

        # Evaluation control
        eval_group = parser.add_argument_group('evaluation control')
        eval_group.add_argument(
            '--mode',
            type=str,
            choices=['physics', 'tokenization', 'both'],
            default='both',
            help='Evaluation mode (default: both)',
        )
        eval_group.add_argument(
            '--split',
            type=str,
            choices=['train', 'holdout', 'both'],
            default='both',
            help='Dataset split to evaluate (default: both)',
        )
        eval_group.add_argument(
            '--n-samples',
            type=int,
            default=1000,
            help='Maximum samples per split (default: 1000)',
        )
        eval_group.add_argument(
            '--train-split-end',
            type=int,
            default=10240,
            help='Training split boundary index (default: 10240)',
        )
        eval_group.add_argument(
            '--num-timesteps',
            type=int,
            default=256,
            help='Number of timesteps to roll out (default: 256)',
        )

        # Output control
        output_group = parser.add_argument_group('output control')
        output_group.add_argument(
            '--output-dir',
            type=Path,
            default=None,
            help='Output directory (default: diagnostics/mno_<timestamp>)',
        )
        output_group.add_argument(
            '--save-rollouts',
            action='store_true',
            help='Save predicted/target rollouts to HDF5',
        )
        output_group.add_argument(
            '--save-tokens',
            action='store_true',
            help='Save VQ tokens for analysis',
        )

        # Performance
        perf_group = parser.add_argument_group('performance')
        perf_group.add_argument(
            '--batch-size',
            type=int,
            default=8,
            help='Batch size for evaluation (default: 8)',
        )
        perf_group.add_argument(
            '--device',
            type=str,
            choices=['cuda', 'cpu'],
            default='cuda',
            help='Device to use (default: cuda)',
        )

        # Visualization
        viz_group = parser.add_argument_group('visualization')
        viz_group.add_argument(
            '--visualize',
            action='store_true',
            help='Generate diagnostic plots',
        )
        viz_group.add_argument(
            '--no-progress',
            action='store_true',
            help='Disable progress bars',
        )

    def execute(self, args: Namespace) -> int:
        """Execute the diagnostic evaluation."""
        # Validate inputs
        if not self._validate_inputs(args):
            return 1

        # Setup output directory
        output_dir = self._setup_output_dir(args)
        print(f"\nOutput directory: {output_dir}")

        # Create evaluator
        print("\nInitializing evaluator...")
        evaluator = MNODiagnosticEvaluator(
            mno_checkpoint=args.mno_checkpoint,
            vqvae_checkpoint=args.vqvae_checkpoint,
            dataset_path=args.dataset,
            substrate_config_path=args.substrate_config,
            train_split_end=args.train_split_end,
            device=args.device,
        )

        # Create reporter
        reporter = DiagnosticReporter(output_dir)

        # Prepare metadata
        metadata = {
            'mno_checkpoint': str(args.mno_checkpoint),
            'vqvae_checkpoint': str(args.vqvae_checkpoint),
            'dataset': str(args.dataset),
            'substrate_config': str(args.substrate_config),
            'mode': args.mode,
            'split': args.split,
            'n_samples': args.n_samples,
            'train_split_end': args.train_split_end,
            'num_timesteps': args.num_timesteps,
            'batch_size': args.batch_size,
            'device': args.device,
        }

        # Run evaluation
        physics_results = None
        tokenization_results = None

        try:
            # Physics evaluation
            if args.mode in ['physics', 'both']:
                physics_results = self._evaluate_physics(
                    evaluator, args, output_dir, reporter
                )

            # Tokenization evaluation
            if args.mode in ['tokenization', 'both']:
                tokenization_results = self._evaluate_tokenization(
                    evaluator, args, output_dir, reporter
                )

            # Generate summary
            print("\nGenerating summary...")
            reporter.generate_summary_json(
                metadata=metadata,
                physics_results=physics_results,
                tokenization_results=tokenization_results,
            )

            print("\n✓ Diagnostics complete!")
            print(f"  Results saved to: {output_dir}")

            return 0

        except Exception as e:
            print(f"\nError during evaluation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    def _validate_inputs(self, args: Namespace) -> bool:
        """Validate input files exist."""
        if not self.validate_file_exists(args.mno_checkpoint, "MNO checkpoint"):
            return False
        if not self.validate_file_exists(args.vqvae_checkpoint, "VQ-VAE checkpoint"):
            return False
        if not self.validate_file_exists(args.dataset, "Dataset"):
            return False
        if not self.validate_file_exists(args.substrate_config, "Ground truth config"):
            return False
        return True

    def _setup_output_dir(self, args: Namespace) -> Path:
        """Setup output directory."""
        if args.output_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"diagnostics/mno_{timestamp}")
        else:
            output_dir = args.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _evaluate_physics(
        self,
        evaluator: MNODiagnosticEvaluator,
        args: Namespace,
        output_dir: Path,
        reporter: DiagnosticReporter,
    ) -> dict:
        """Run physics fidelity evaluation."""
        print("\n" + "="*70)
        print("PHYSICS FIDELITY EVALUATION")
        print("="*70)

        results = {}

        # Determine which splits to evaluate
        splits = self._get_splits(args.split)

        for split_name in splits:
            print(f"\n--- Evaluating {split_name} split ---")

            # Setup rollout save path if requested
            rollouts_path = None
            if args.save_rollouts:
                rollouts_path = output_dir / f"rollouts_{split_name}.h5"

            # Run evaluation
            split_results = evaluator.evaluate_physics_fidelity(
                split=split_name,
                max_samples=args.n_samples,
                batch_size=args.batch_size,
                num_timesteps=args.num_timesteps,
                save_rollouts_path=rollouts_path,
            )

            results[split_name] = split_results

            # Generate CSV
            reporter.generate_physics_csv(split_results, split_name)

            # Print summary
            print(f"\nSummary for {split_name}:")
            print(f"  Mean MSE: {split_results['mean_mse']:.6f}")
            print(f"  Final MSE: {split_results['final_mse']:.6f}")
            print(f"  Relative L2: {split_results['relative_l2']:.6f}")
            print(f"  Error growth: {split_results['error_growth']['growth_type']} "
                  f"(rate={split_results['error_growth']['growth_rate']:.6f})")

            # Generate visualization if requested
            if args.visualize:
                viz_path = output_dir / f"physics_fidelity_{split_name}.png"
                plot_physics_fidelity(split_results, viz_path)

        # Generate comparison plot if both splits evaluated
        if args.visualize and 'train' in results and 'holdout' in results:
            comp_path = output_dir / "physics_comparison.png"
            plot_split_comparison(results['train'], results['holdout'], comp_path)

        return results

    def _evaluate_tokenization(
        self,
        evaluator: MNODiagnosticEvaluator,
        args: Namespace,
        output_dir: Path,
        reporter: DiagnosticReporter,
    ) -> dict:
        """Run tokenization evaluation."""
        print("\n" + "="*70)
        print("TOKENIZATION EVALUATION")
        print("="*70)

        results = {}

        # Determine which splits to evaluate
        splits = self._get_splits(args.split)

        for split_name in splits:
            print(f"\n--- Evaluating {split_name} split ---")

            # Setup tokens save path if requested
            tokens_path = None
            if args.save_tokens:
                tokens_path = output_dir / f"tokens_{split_name}.h5"

            # Run evaluation
            split_results = evaluator.evaluate_tokenization(
                split=split_name,
                max_samples=args.n_samples,
                batch_size=args.batch_size,
                num_timesteps=args.num_timesteps,
                save_tokens_path=tokens_path,
            )

            results[split_name] = split_results

            # Generate CSV
            reporter.generate_tokenization_csv(
                split_results.get('mno'),
                split_results['cno'],
                split_name,
            )

            # Print summary
            print(f"\nSummary for {split_name}:")
            if 'mno' in split_results:
                print(f"  MNO reconstruction MSE: {split_results['mno']['reconstruction_mse']:.6f}")
                print(f"  MNO perplexity: {split_results['mno']['perplexity']:.2f}")
            print(f"  CNO reconstruction MSE: {split_results['cno']['reconstruction_mse']:.6f}")
            print(f"  CNO perplexity: {split_results['cno']['perplexity']:.2f}")

            if 'distribution_comparison' in split_results:
                print(f"  KL divergence (MNO→CNO): "
                      f"{split_results['distribution_comparison'].get('kl_mno_to_cno', 0.0):.6f}")

            if 'note' in split_results:
                print(f"  Note: {split_results['note']}")

            # Generate visualization if requested (only if we have MNO results)
            if args.visualize and 'mno' in split_results and 'distribution_comparison' in split_results:
                viz_path = output_dir / f"tokenization_quality_{split_name}.png"
                plot_tokenization_quality(
                    split_results['mno'],
                    split_results['cno'],
                    split_results['distribution_comparison'],
                    viz_path,
                )

        return results

    def _get_splits(self, split_arg: str) -> list:
        """Get list of splits to evaluate."""
        if split_arg == 'both':
            return ['train', 'holdout']
        else:
            return [split_arg]
