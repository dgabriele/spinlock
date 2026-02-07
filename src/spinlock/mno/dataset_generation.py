"""MNO Rollout Dataset Generation.

Generates HDF5 datasets of MNO rollouts for tokenizer training.

This module handles:
- MNO checkpoint loading
- Sobol parameter sampling across full parameter space
- Batched rollout generation (GPU-accelerated)
- Feature extraction (temporal + initial)
- HDF5 saving with checkpointing for long runs

Architecture:
    MNORolloutDatasetGenerator:
        - Loads trained MNO checkpoint
        - Samples parameters via Sobol sequences (uniform parameter space coverage)
        - Generates rollouts in batches (GPU efficient)
        - Extracts features via MNOFeatureExtractor + InitialManualExtractor
        - Saves to HDF5 with periodic checkpointing

Output HDF5 Structure:
    datasets/mno_rollouts_N.h5:
        inputs/
            fields: [N, M, C, H, W]  # Generated ICs (M realizations)
            parameters/params: [N, 12]  # Sobol parameter vectors
        features/
            temporal: [N, T, D_t]  # MNO temporal features (aggregated across M)
            initial/aggregated: [N, D_i]  # Initial manual features (aggregated)
        rollouts/
            mno: [N, M, T, C, H, W]  # Full MNO trajectories (for alignment analysis)

Usage:
    generator = MNORolloutDatasetGenerator(
        mno_checkpoint="checkpoints/mno/50k_baseline/meta_operator_best.pt",
        device="cuda"
    )

    generator.generate_dataset(
        num_rollouts=100000,
        batch_size=128,
        output_path="datasets/mno_rollouts_100k.h5"
    )
"""

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm

from spinlock.mno.validation_utils import load_mno_checkpoint
from spinlock.mno.feature_extraction import MNOFeatureExtractor
from spinlock.features.initial.manual_extractors import InitialManualExtractor


class MNORolloutDatasetGenerator:
    """Generate HDF5 datasets of MNO rollouts for tokenizer training.

    Handles end-to-end pipeline:
    1. Load trained MNO checkpoint
    2. Sample parameters via Sobol sequences (full parameter space)
    3. Generate initial conditions from parameters
    4. Generate MNO rollouts (batched for GPU efficiency)
    5. Extract temporal features via MNOFeatureExtractor
    6. Extract initial features via InitialManualExtractor
    7. Save to HDF5 with checkpointing

    Example:
        generator = MNORolloutDatasetGenerator(
            mno_checkpoint="checkpoints/mno/50k_baseline/meta_operator_best.pt",
            device="cuda"
        )

        generator.generate_dataset(
            num_rollouts=100000,
            batch_size=128,
            output_path="datasets/mno_rollouts_100k.h5"
        )
    """

    def __init__(
        self,
        mno_checkpoint: Path,
        num_realizations: int = 8,
        rollout_steps: int = 256,
        device: str = "cuda"
    ):
        """Initialize MNO rollout dataset generator.

        Args:
            mno_checkpoint: Path to trained MNO checkpoint
            num_realizations: Number of realizations per operator (default: 8)
            rollout_steps: Number of timesteps in rollouts (default: 256)
            device: Torch device for inference
        """
        self.device = torch.device(device)
        self.num_realizations = num_realizations
        self.rollout_steps = rollout_steps

        # Load MNO checkpoint
        print(f"Loading MNO checkpoint: {mno_checkpoint}")
        self.mno = load_mno_checkpoint(str(mno_checkpoint), device=str(device))
        self.mno.eval()

        # Initialize feature extractors
        print("Initializing feature extractors...")
        self.temporal_extractor = MNOFeatureExtractor(device=str(device))
        self.initial_extractor = InitialManualExtractor(device=self.device)

        # Probe feature dimensions
        self._probe_feature_dimensions()

    def _probe_feature_dimensions(self):
        """Probe extractors to determine feature dimensions."""
        print("Probing feature dimensions...")

        # Probe temporal extractor
        dims = self.temporal_extractor.probe_dimensions(
            timesteps=self.rollout_steps,
            channels=1,
            height=64,
            width=64,
            batch_size=1
        )
        self.temporal_dim = dims['temporal_dim']
        print(f"  Temporal feature dimension: {self.temporal_dim}")

        # Probe initial extractor
        dummy_ic = torch.randn(1, self.num_realizations, 1, 64, 64, device=self.device)
        initial_features = self.initial_extractor.extract_all(dummy_ic)
        self.initial_dim = initial_features.shape[-1]
        print(f"  Initial feature dimension: {self.initial_dim}")

    def generate_dataset(
        self,
        num_rollouts: int,
        batch_size: int,
        output_path: Path,
        checkpoint_interval: int = 10000
    ):
        """Generate complete MNO rollout dataset with checkpointing.

        Args:
            num_rollouts: Total number of rollouts to generate
            batch_size: Batch size for generation (GPU memory dependent)
            output_path: Path to output HDF5 file
            checkpoint_interval: Save checkpoint every N rollouts
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"MNO Rollout Dataset Generation")
        print(f"{'='*60}")
        print(f"Rollouts: {num_rollouts}")
        print(f"Batch size: {batch_size}")
        print(f"Realizations per rollout: {self.num_realizations}")
        print(f"Timesteps: {self.rollout_steps}")
        print(f"Output: {output_path}")
        print(f"Checkpoint interval: {checkpoint_interval}")
        print(f"{'='*60}\n")

        # Create HDF5 file with dataset structure
        with h5py.File(output_path, 'w') as f:
            # Create dataset groups
            inputs_grp = f.create_group('inputs')
            params_grp = f.create_group('parameters')
            features_grp = f.create_group('features')
            initial_grp = features_grp.create_group('initial')
            rollouts_grp = f.create_group('rollouts')

            # Create datasets (fixed size)
            fields_ds = inputs_grp.create_dataset(
                'fields',
                shape=(num_rollouts, self.num_realizations, 1, 64, 64),
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            params_ds = params_grp.create_dataset(
                'params',
                shape=(num_rollouts, 12),  # 12 operator parameters
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            temporal_ds = features_grp.create_dataset(
                'temporal',
                shape=(num_rollouts, self.rollout_steps, self.temporal_dim),
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            initial_ds = initial_grp.create_dataset(
                'aggregated',
                shape=(num_rollouts, self.initial_dim),
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            rollouts_ds = rollouts_grp.create_dataset(
                'mno',
                shape=(num_rollouts, self.num_realizations, self.rollout_steps, 1, 64, 64),
                dtype='float32',
                compression='gzip',
                compression_opts=4
            )

            # Generate in batches
            num_batches = (num_rollouts + batch_size - 1) // batch_size

            with tqdm(total=num_rollouts, desc="Generating rollouts") as pbar:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_rollouts)
                    current_batch_size = end_idx - start_idx

                    # Generate batch
                    batch_data = self._generate_batch(current_batch_size)

                    # Save to HDF5
                    fields_ds[start_idx:end_idx] = batch_data['fields']
                    params_ds[start_idx:end_idx] = batch_data['params']
                    temporal_ds[start_idx:end_idx] = batch_data['temporal_features']
                    initial_ds[start_idx:end_idx] = batch_data['initial_features']
                    rollouts_ds[start_idx:end_idx] = batch_data['rollouts']

                    pbar.update(current_batch_size)

                    # Periodic checkpoint (flush to disk)
                    if end_idx % checkpoint_interval == 0:
                        f.flush()
                        print(f"\n  Checkpoint: {end_idx}/{num_rollouts} rollouts saved")

        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"Saved to: {output_path}")
        print(f"Total rollouts: {num_rollouts}")
        print(f"{'='*60}\n")

    def _generate_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Generate a batch of MNO rollouts with features.

        Args:
            batch_size: Number of rollouts to generate

        Returns:
            Dictionary containing:
                - fields: [B, M, C, H, W] initial conditions
                - params: [B, 12] operator parameters
                - temporal_features: [B, T, D_t] temporal features
                - initial_features: [B, D_i] initial features
                - rollouts: [B, M, T, C, H, W] full MNO rollouts
        """
        with torch.no_grad():
            # Sample parameters via Sobol sequence
            params = self._sample_parameters(batch_size)  # [B, 12]

            # Generate initial conditions from parameters
            # For stochastic operators, we need M realizations
            ics = self._generate_initial_conditions(params)  # [B, M, C, H, W]

            # Generate MNO rollouts for all realizations
            rollouts_list = []
            for m in range(self.num_realizations):
                # Get ICs for this realization
                ic_m = ics[:, m, :, :, :]  # [B, C, H, W]

                # Generate rollout
                rollout_m = self.mno.rollout(
                    ic_m,
                    steps=self.rollout_steps,
                    return_all_steps=True,
                    params=params
                )[:, 1:, ...]  # Remove IC from output: [B, T, C, H, W]

                rollouts_list.append(rollout_m.unsqueeze(1))  # [B, 1, T, C, H, W]

            # Concatenate all realizations: [B, M, T, C, H, W]
            rollouts = torch.cat(rollouts_list, dim=1)

            # Extract temporal features (aggregate across realizations)
            # MNOFeatureExtractor expects [B, M, T, C, H, W]
            temporal_features = self._extract_temporal_features(rollouts)  # [B, T, D_t]

            # Extract initial features (aggregate across realizations)
            initial_features = self._extract_initial_features(ics)  # [B, D_i]

            return {
                'fields': ics.cpu().numpy(),
                'params': params.cpu().numpy(),
                'temporal_features': temporal_features.cpu().numpy(),
                'initial_features': initial_features.cpu().numpy(),
                'rollouts': rollouts.cpu().numpy()
            }

    def _sample_parameters(self, batch_size: int) -> torch.Tensor:
        """Sample operator parameters via Sobol sequence.

        Args:
            batch_size: Number of parameter vectors to sample

        Returns:
            Parameter tensor [B, 12]
        """
        # Use Sobol sequence for better parameter space coverage
        sobol_engine = torch.quasirandom.SobolEngine(dimension=12, scramble=True)
        params_unit = sobol_engine.draw(batch_size)  # [B, 12] in [0, 1]

        # Map to parameter ranges (example ranges - adjust based on your operators)
        # These are typical ranges for diffusion-reaction systems
        param_ranges = torch.tensor([
            [0.001, 0.1],   # diffusion coefficient 1
            [0.001, 0.1],   # diffusion coefficient 2
            [0.01, 1.0],    # reaction rate 1
            [0.01, 1.0],    # reaction rate 2
            [-1.0, 1.0],    # parameter 5
            [-1.0, 1.0],    # parameter 6
            [0.1, 2.0],     # parameter 7
            [0.1, 2.0],     # parameter 8
            [-0.5, 0.5],    # parameter 9
            [-0.5, 0.5],    # parameter 10
            [0.5, 2.0],     # parameter 11
            [0.5, 2.0],     # parameter 12
        ], device=self.device)

        # Linearly map [0, 1] to parameter ranges
        params = params_unit.to(self.device) * (param_ranges[:, 1] - param_ranges[:, 0]) + param_ranges[:, 0]

        return params

    def _generate_initial_conditions(self, params: torch.Tensor) -> torch.Tensor:
        """Generate stochastic initial conditions from parameters.

        Args:
            params: Parameter vectors [B, 12]

        Returns:
            Initial conditions [B, M, C, H, W]
        """
        batch_size = params.shape[0]

        # Generate M realizations per parameter set
        # Use Gaussian random fields with parameter-dependent statistics
        ics_list = []
        for _ in range(self.num_realizations):
            # Base random field
            ic = torch.randn(batch_size, 1, 64, 64, device=self.device)

            # Apply parameter-dependent scaling (example)
            # In practice, you might use more sophisticated IC generation
            scale = params[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            ic = ic * scale

            ics_list.append(ic.unsqueeze(1))  # [B, 1, C, H, W]

        # Concatenate realizations: [B, M, C, H, W]
        ics = torch.cat(ics_list, dim=1)

        return ics

    def _extract_temporal_features(self, rollouts: torch.Tensor) -> torch.Tensor:
        """Extract temporal features from MNO rollouts.

        Args:
            rollouts: MNO trajectories [B, M, T, C, H, W]

        Returns:
            Temporal features [B, T, D_t]
        """
        # MNOFeatureExtractor expects [B, M, T, C, H, W] or [B, T, C, H, W]
        # It will aggregate across M realizations internally
        extracted = self.temporal_extractor.extract(rollouts)
        return extracted['temporal']  # [B, T, D_t]

    def _extract_initial_features(self, ics: torch.Tensor) -> torch.Tensor:
        """Extract initial features from initial conditions.

        Args:
            ics: Initial conditions [B, M, C, H, W]

        Returns:
            Initial features [B, D_i] (aggregated across M)
        """
        # InitialManualExtractor returns [B, M, D_i]
        features = self.initial_extractor.extract_all(ics)  # [B, M, D_i]

        # Aggregate across realizations (mean)
        features_aggregated = features.mean(dim=1)  # [B, D_i]

        return features_aggregated
