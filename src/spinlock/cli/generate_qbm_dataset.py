"""Generate QBM Dataset command for Spinlock CLI.

Generates HDF5 datasets of Quantum Brownian Motion rollouts for U-AFNO training.

This command:
1. Samples QBM parameters via Sobol sequences
2. Generates quantum ICs (wavepackets, coherent states, superpositions)
3. Generates potential energy landscapes from parameters
4. Runs QBM simulations using split-operator method
5. Extracts temporal and initial features
6. Saves to HDF5 format compatible with U-AFNO training

Output HDF5 Structure:
    datasets/qbm_N.h5:
        inputs/
            fields: [N, M, 2, H, W]  # Quantum ICs (Re/Im wavefunctions)
        parameters/
            params: [N, 12]  # QBM parameters
        features/
            temporal/features: [N, T, D_t]  # Temporal features
            initial/aggregated/features: [N, D_i]  # Initial features
        rollouts/ # Optional (only if --store-rollouts)
            qbm: [N, M, T, 2, H, W]  # Full trajectories

Usage Examples:
    # Generate 100k rollouts for U-AFNO training
    spinlock generate-qbm-dataset \\
        --num-rollouts 100000 \\
        --batch-size 2 \\
        --output datasets/qbm_100k.h5

    # Quick test with 100 samples
    spinlock generate-qbm-dataset \\
        --num-rollouts 100 \\
        --batch-size 2 \\
        --output datasets/qbm_test.h5

    # Custom grid size and domain
    spinlock generate-qbm-dataset \\
        --num-rollouts 10000 \\
        --grid-size 128 \\
        --domain-size 15.0 \\
        --output datasets/qbm_10k_highres.h5
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from .base import CLICommand


class GenerateQBMDatasetCommand(CLICommand):
    """CLI command for generating QBM dataset."""

    @property
    def name(self) -> str:
        return "generate-qbm-dataset"

    @property
    def help(self) -> str:
        return "Generate HDF5 dataset of Quantum Brownian Motion rollouts"

    @property
    def description(self) -> str:
        return """
Generate HDF5 dataset of Quantum Brownian Motion rollouts for U-AFNO training.

This command generates a comprehensive dataset by:
1. Sampling QBM parameters (bath coupling, temperature, mass, potential type)
2. Generating quantum initial conditions (wavepackets, coherent states, superpositions)
3. Creating potential energy landscapes (harmonic, double-well, quartic, random)
4. Running QBM simulations using split-operator method with Lindblad dissipation
5. Extracting temporal and initial features
6. Saving to HDF5 format compatible with SpinlockDataset

The generated dataset can be used to train U-AFNO models on open quantum systems
that exhibit decoherence and the quantum-to-classical transition.

Physics:
    The Caldeira-Leggett model describes a quantum particle coupled to a thermal bath:
    H = p²/(2m) + V(x,y) with Lindblad dissipation modeling environmental coupling.

Output Structure:
    datasets/qbm_N.h5:
        inputs/fields: [N, M, 2, H, W]  # Quantum ICs (Re/Im)
        parameters/params: [N, 12]       # QBM parameters
        features/temporal/features: [N, T, D_t]  # Temporal features
        features/initial/aggregated/features: [N, D_i]  # Initial features
        rollouts/qbm: [N, M, T, 2, H, W]  # Full trajectories (optional)

Performance:
    With batch_size=2 on modern GPU (conservative for memory):
    - 100k rollouts: ~2-4 hours (depending on GPU)
    - Dataset size: ~10-15GB (features only, with compression)

Examples:
    # Standard 100k dataset for U-AFNO training
    spinlock generate-qbm-dataset \\
        --num-rollouts 100000 \\
        --batch-size 2 \\
        --output datasets/qbm_100k.h5

    # Quick 1k dataset for prototyping
    spinlock generate-qbm-dataset \\
        --num-rollouts 1000 \\
        --batch-size 2 \\
        --output datasets/qbm_1k.h5
        """

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add generate-qbm-dataset command arguments."""
        # Required arguments
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
            default=2,
            metavar="N",
            help="Batch size for generation (default: 2, conservative for GPU memory)",
        )

        # Generation parameters
        parser.add_argument(
            "--num-realizations",
            type=int,
            default=3,
            metavar="M",
            help="Number of realizations per parameter set (default: 3)",
        )

        parser.add_argument(
            "--num-params",
            type=int,
            default=12,
            metavar="P",
            help="Number of QBM parameters (default: 12)",
        )

        parser.add_argument(
            "--rollout-steps",
            type=int,
            default=256,
            metavar="T",
            help="Number of timesteps in rollouts (default: 256)",
        )

        # Spatial grid parameters
        parser.add_argument(
            "--grid-size",
            type=int,
            default=64,
            metavar="N",
            help="Spatial grid size (NxN) (default: 64)",
        )

        parser.add_argument(
            "--domain-size",
            type=float,
            default=10.0,
            metavar="L",
            help="Physical domain size (default: 10.0)",
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

        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable performance profiling (prints breakdown after first batch)",
        )

        parser.add_argument(
            "--store-rollouts",
            action="store_true",
            default=False,
            help="Store full rollouts in HDF5 (WARNING: uses significantly more disk space). Default: False (features only)",
        )

    def execute(self, args: Namespace) -> int:
        """Execute the generate-qbm-dataset command."""
        from spinlock.qbm.dataset_generation import QBMDatasetGenerator

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
            generator = QBMDatasetGenerator(
                num_realizations=args.num_realizations,
                num_channels=2,  # Fixed: Re/Im for quantum wavefunction
                num_params=args.num_params,
                rollout_steps=args.rollout_steps,
                grid_size=args.grid_size,
                domain_size=args.domain_size,
                device=args.device,
                enable_profiling=args.profile,
                store_rollouts=args.store_rollouts
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
