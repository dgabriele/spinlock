"""Generate MNO Dataset command for Spinlock CLI.

Generates HDF5 datasets of MNO rollouts for VQTokenizer training.

This command:
1. Loads a trained MNO checkpoint
2. Samples parameters via Sobol sequences (uniform parameter space coverage)
3. Generates MNO rollouts with multiple realizations
4. Extracts temporal and initial features
5. Saves to HDF5 format compatible with VQTokenizer training

Output HDF5 Structure:
    datasets/mno_rollouts_N.h5:
        inputs/
            fields: [N, M, C, H, W]  # Generated ICs
        parameters/
            params: [N, 12]  # Operator parameters
        features/
            temporal: [N, T, D_t]  # Temporal features
            initial/aggregated: [N, D_i]  # Initial features
        rollouts/
            mno: [N, M, T, C, H, W]  # Full rollouts

Usage Examples:
    # Generate 100k rollouts for tokenizer training
    spinlock generate-mno-dataset \\
        --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
        --num-rollouts 100000 \\
        --batch-size 128 \\
        --output datasets/mno_rollouts_100k.h5

    # Generate smaller dataset for testing
    spinlock generate-mno-dataset \\
        --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
        --num-rollouts 1000 \\
        --batch-size 64 \\
        --output datasets/mno_rollouts_1k.h5

    # Use different number of realizations
    spinlock generate-mno-dataset \\
        --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
        --num-rollouts 50000 \\
        --num-realizations 4 \\
        --output datasets/mno_rollouts_50k_m4.h5
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class GenerateMNODatasetCommand(CLICommand):
    """CLI command for generating MNO rollout datasets."""

    @property
    def name(self) -> str:
        return "generate-mno-dataset"

    @property
    def help(self) -> str:
        return "Generate HDF5 dataset of MNO rollouts for tokenizer training"

    @property
    def description(self) -> str:
        return """
Generate HDF5 dataset of MNO rollouts for VQTokenizer training.

This command generates a comprehensive dataset by:
1. Loading a trained MNO checkpoint
2. Sampling operator parameters via Sobol sequences
3. Generating MNO rollouts with multiple realizations
4. Extracting temporal and initial features
5. Saving to HDF5 format compatible with VQTokenizer

The generated dataset can be used to train a VQTokenizer that compresses
MNO rollouts efficiently, enabling semantic grounding via alignment with
CNO-trained tokenizers.

Output Structure:
    datasets/mno_rollouts_N.h5:
        inputs/fields: [N, M, C, H, W]  # Initial conditions
        parameters/params: [N, 12]       # Operator parameters
        features/temporal: [N, T, D_t]   # Temporal features
        features/initial/aggregated: [N, D_i]  # Initial features
        rollouts/mno: [N, M, T, C, H, W]  # Full trajectories

Performance:
    With batch_size=128 on a modern GPU:
    - ~36 rollouts/sec (including feature extraction)
    - 100k rollouts: ~45 minutes
    - Dataset size: ~40GB (with compression)

Examples:
    # Standard 100k dataset for production tokenizer
    spinlock generate-mno-dataset \\
        --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
        --num-rollouts 100000 \\
        --batch-size 128 \\
        --output datasets/mno_rollouts_100k.h5

    # Quick 10k dataset for prototyping
    spinlock generate-mno-dataset \\
        --mno-checkpoint checkpoints/mno/50k_baseline/meta_operator_best.pt \\
        --num-rollouts 10000 \\
        --batch-size 128 \\
        --output datasets/mno_rollouts_10k.h5
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add generate-mno-dataset command arguments."""
        # Required arguments
        parser.add_argument(
            "--mno-checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to trained MNO checkpoint (.pt file)",
        )

        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to output HDF5 file",
        )

        # Dataset size
        parser.add_argument(
            "--num-rollouts",
            type=int,
            default=100000,
            metavar="N",
            help="Number of rollouts to generate (default: 100000)",
        )

        parser.add_argument(
            "--batch-size",
            type=int,
            default=128,
            metavar="N",
            help="Batch size for generation (default: 128)",
        )

        # Generation parameters
        parser.add_argument(
            "--num-realizations",
            type=int,
            default=3,
            metavar="M",
            help="Number of realizations per operator (default: 3, matches CNO datasets)",
        )

        parser.add_argument(
            "--num-channels",
            type=int,
            default=3,
            metavar="C",
            help="Number of channels in rollouts (default: 3)",
        )

        parser.add_argument(
            "--num-params",
            type=int,
            default=14,
            metavar="P",
            help="Number of operator parameters (default: 14)",
        )

        parser.add_argument(
            "--rollout-steps",
            type=int,
            default=256,
            metavar="T",
            help="Number of timesteps in rollouts (default: 256)",
        )

        parser.add_argument(
            "--checkpoint-interval",
            type=int,
            default=10000,
            metavar="N",
            help="Save checkpoint every N rollouts (default: 10000)",
        )

        # Compute
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda", "cpu"],
            help="Device for computation (default: cuda)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the generate-mno-dataset command."""
        from spinlock.mno.dataset_generation import MNORolloutDatasetGenerator

        # Validate MNO checkpoint exists
        if not self.validate_file_exists(args.mno_checkpoint, "MNO checkpoint"):
            return 1

        # Validate output directory exists or can be created
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Warn if output file already exists
        if output_path.exists():
            print(f"Warning: Output file already exists: {output_path}")
            response = input("Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 0

        try:
            # Create generator
            generator = MNORolloutDatasetGenerator(
                mno_checkpoint=args.mno_checkpoint,
                num_realizations=args.num_realizations,
                num_channels=args.num_channels,
                num_params=args.num_params,
                rollout_steps=args.rollout_steps,
                device=args.device
            )

            # Generate dataset
            generator.generate_dataset(
                num_rollouts=args.num_rollouts,
                batch_size=args.batch_size,
                output_path=output_path,
                checkpoint_interval=args.checkpoint_interval
            )

            print(f"\nSuccess! Dataset saved to: {output_path}")
            return 0

        except Exception as e:
            return self.error(f"Dataset generation failed: {e}")
