"""QBM Rollout Dataset Generation.

Generates HDF5 datasets of Quantum Brownian Motion rollouts for U-AFNO training.

This module handles:
- Sobol parameter sampling across QBM parameter space
- Quantum IC generation (wavepackets, coherent states, superpositions)
- Potential energy landscape generation
- QBM simulation (GPU-accelerated split-operator method)
- Feature extraction (temporal + initial)
- HDF5 saving with checkpointing for long runs

Architecture:
    QBMDatasetGenerator:
        - Initializes QBM simulator, IC generator, potential generator
        - Samples parameters via Sobol sequences (uniform parameter space coverage)
        - Generates quantum ICs and potentials from parameters
        - Runs QBM simulations in batches (GPU efficient)
        - Extracts features via MNOFeatureExtractor + InitialManualExtractor
        - Saves to HDF5 with periodic checkpointing

Output HDF5 Structure (matches SpinlockDataset schema):
    datasets/qbm_N.h5:
        inputs/
            fields: [N, M, 2, H, W]  # Quantum ICs (Re, Im wavefunctions)
        parameters/
            params: [N, P]  # QBM parameter vectors
        features/
            temporal/
                features: [N, T, D_t]  # Temporal features from rollouts
            initial/
                aggregated/
                    features: [N, D_i]  # Initial features from ICs
        rollouts/ # Optional (only if --store-rollouts flag used)
            qbm: [N, M, T, 2, H, W]  # Full QBM trajectories

Usage:
    generator = QBMDatasetGenerator(device="cuda")

    generator.generate_dataset(
        num_rollouts=100000,
        batch_size=2,
        output_path="datasets/qbm_100k.h5"
    )
"""

import gc
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import time

from spinlock.qbm.simulator import QuantumBrownianSimulator
from spinlock.qbm.initial_conditions import QuantumICGenerator
from spinlock.qbm.potentials import PotentialGenerator
from spinlock.mno.feature_extraction import MNOFeatureExtractor
from spinlock.features.initial.manual_extractors import InitialManualExtractor
from spinlock.utils.profiler import PerformanceProfiler
from spinlock.utils.async_hdf5_writer import AsyncHDF5Writer


class QBMDatasetGenerator:
    """Generate HDF5 datasets of QBM rollouts for U-AFNO training.

    Handles end-to-end pipeline:
    1. Sample parameters via Sobol sequences (full parameter space)
    2. Generate quantum ICs from sampled IC distribution
    3. Generate potentials from parameters
    4. Run QBM simulations (batched for GPU efficiency)
    5. Extract temporal features via MNOFeatureExtractor
    6. Extract initial features via InitialManualExtractor
    7. Save to HDF5 with checkpointing

    IC Type Distribution (quantum-specific):
        - 40% Gaussian wavepackets (fundamental quantum states)
        - 30% Coherent states (minimum uncertainty states)
        - 15% 2-component superpositions (quantum interference)
        - 15% 3-component superpositions (complex interference)

    Example:
        generator = QBMDatasetGenerator(device="cuda")

        generator.generate_dataset(
            num_rollouts=100000,
            batch_size=2,
            output_path="datasets/qbm_100k.h5"
        )
    """

    # IC Type Configuration (Quantum Distribution)
    IC_TYPE_CONFIG = [
        # Gaussian wavepackets: 40%
        (0.40, 'quantum_gaussian_wavepacket', {'sigma': 1.0}),

        # Coherent states: 30%
        (0.70, 'quantum_coherent_state', {}),

        # 2-component superpositions: 15%
        (0.85, 'quantum_superposition_2', {'sigma': 0.8, 'separation': 2.0}),

        # 3-component superpositions: 15%
        (1.00, 'quantum_superposition_3', {'sigma': 0.8, 'separation': 2.0}),
    ]

    def __init__(
        self,
        num_realizations: int = 3,
        num_channels: int = 2,  # Re, Im
        num_params: int = 12,    # QBM parameters
        rollout_steps: int = 256,
        grid_size: int = 64,
        domain_size: float = 10.0,
        device: str = "cuda",
        enable_profiling: bool = False,
        store_rollouts: bool = False
    ):
        """Initialize QBM dataset generator.

        Args:
            num_realizations: Number of realizations per operator (default: 3)
            num_channels: Number of channels (2 for Re/Im wavefunction)
            num_params: Number of QBM parameters (default: 12)
            rollout_steps: Number of timesteps in rollouts (default: 256)
            grid_size: Spatial grid size (default: 64)
            domain_size: Physical domain size (default: 10.0)
            device: Torch device for computation
            enable_profiling: Whether to enable performance profiling
            store_rollouts: Whether to store full rollouts (default: False)
        """
        self.device = torch.device(device)
        self.num_realizations = num_realizations
        self.num_channels = num_channels
        self.num_params = num_params
        self.rollout_steps = rollout_steps
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.store_rollouts = store_rollouts
        self.profiler = PerformanceProfiler() if enable_profiling else None

        # Initialize QBM simulator
        print(f"Initializing QBM simulator (grid: {grid_size}, domain: {domain_size})...")
        self.simulator = QuantumBrownianSimulator(
            grid_size=grid_size,
            domain_size=domain_size,
            device=str(device)
        )

        # Initialize quantum IC generator
        print("Initializing quantum IC generator...")
        self.ic_generator = QuantumICGenerator(
            grid_size=grid_size,
            domain_size=domain_size,
            device=self.device
        )

        # Initialize potential generator
        print("Initializing potential generator...")
        self.potential_generator = PotentialGenerator(
            grid_size=grid_size,
            domain_size=domain_size,
            device=self.device
        )

        # Initialize feature extractors
        print("Initializing feature extractors...")
        self.temporal_extractor = MNOFeatureExtractor(device=str(device))
        self.initial_extractor = InitialManualExtractor(device=self.device)

        # Probe feature dimensions
        self._probe_feature_dimensions()

    def _probe_feature_dimensions(self):
        """Probe extractors to determine feature dimensions."""
        print("Probing feature dimensions...")

        # Probe temporal extractor with dummy rollout
        dummy_rollout = torch.randn(
            1, self.num_realizations, self.rollout_steps, self.num_channels,
            self.grid_size, self.grid_size, device=self.device
        )
        temporal_result = self.temporal_extractor.extract(dummy_rollout)
        self.temporal_dim = temporal_result['temporal'].shape[-1]
        print(f"  Temporal feature dimension: {self.temporal_dim}")

        # Probe initial extractor
        dummy_ic = torch.randn(
            1, self.num_realizations, self.num_channels,
            self.grid_size, self.grid_size, device=self.device
        )
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
        """Generate complete QBM dataset with checkpointing.

        Args:
            num_rollouts: Total number of rollouts to generate
            batch_size: Batch size for generation (GPU memory dependent)
            output_path: Path to output HDF5 file
            checkpoint_interval: Save checkpoint every N rollouts
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"QBM Dataset Generation")
        print(f"{'='*60}")
        print(f"Rollouts: {num_rollouts}")
        print(f"Batch size: {batch_size}")
        print(f"Realizations per rollout: {self.num_realizations}")
        print(f"Timesteps: {self.rollout_steps}")
        print(f"Grid: {self.grid_size}×{self.grid_size}")
        print(f"Output: {output_path}")
        print(f"Checkpoint interval: {checkpoint_interval}")
        print(f"Store rollouts: {self.store_rollouts}")
        print(f"{'='*60}\n")

        # Create HDF5 file with dataset structure
        with h5py.File(output_path, 'w') as f:
            # Create dataset groups (match SpinlockDataset schema)
            inputs_grp = f.create_group('inputs')
            params_grp = f.create_group('parameters')
            features_grp = f.create_group('features')

            # Feature subgroups
            temporal_grp = features_grp.create_group('temporal')
            initial_grp = features_grp.create_group('initial')
            initial_agg_grp = initial_grp.create_group('aggregated')

            # Create datasets (fixed size)
            fields_ds = inputs_grp.create_dataset(
                'fields',
                shape=(num_rollouts, self.num_realizations, self.num_channels, self.grid_size, self.grid_size),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.num_realizations, self.num_channels, self.grid_size, self.grid_size)
            )

            params_ds = params_grp.create_dataset(
                'params',
                shape=(num_rollouts, self.num_params),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.num_params)
            )

            # Match SpinlockDataset structure: features/temporal/features
            temporal_ds = temporal_grp.create_dataset(
                'features',
                shape=(num_rollouts, self.rollout_steps, self.temporal_dim),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.rollout_steps, self.temporal_dim)
            )

            # Match SpinlockDataset structure: features/initial/aggregated/features
            initial_ds = initial_agg_grp.create_dataset(
                'features',
                shape=(num_rollouts, self.initial_dim),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.initial_dim)
            )

            # Prepare datasets dictionary for async writer
            datasets = {
                'fields': fields_ds,
                'params': params_ds,
                'temporal_features': temporal_ds,
                'initial_features': initial_ds,
            }

            # Optionally store full rollouts
            if self.store_rollouts:
                rollouts_grp = f.create_group('rollouts')
                print("  WARNING: Storing full rollouts (this will use significant disk space)")
                rollouts_ds = rollouts_grp.create_dataset(
                    'qbm',
                    shape=(num_rollouts, self.num_realizations, self.rollout_steps, self.num_channels, self.grid_size, self.grid_size),
                    dtype='float32',
                    compression='lzf',
                    chunks=(batch_size, self.num_realizations, self.rollout_steps, self.num_channels, self.grid_size, self.grid_size)
                )
                datasets['rollouts'] = rollouts_ds
            else:
                print("  Storing features only (not full rollouts)")

            # Generate in batches with async writes
            num_batches = (num_rollouts + batch_size - 1) // batch_size

            with AsyncHDF5Writer(f, datasets, queue_size=4) as writer:
                with tqdm(total=num_rollouts, desc="Generating rollouts") as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_rollouts)
                        current_batch_size = end_idx - start_idx

                        # Generate batch (GPU computation)
                        t_start = time.time()
                        batch_data = self._generate_batch(current_batch_size)
                        t_gen = time.time() - t_start

                        # Submit write (async - happens in background)
                        t_write_start = time.time()
                        if self.profiler:
                            with self.profiler.profile("hdf5_write"):
                                writer.submit(start_idx, end_idx, batch_data)
                        else:
                            writer.submit(start_idx, end_idx, batch_data)
                        t_write = time.time() - t_write_start

                        # Memory cleanup
                        del batch_data
                        gc.collect()
                        if batch_idx % 2 == 0:
                            torch.cuda.empty_cache()

                        pbar.update(current_batch_size)

                        # Periodic checkpoint (flush to disk)
                        if end_idx % checkpoint_interval == 0:
                            writer.flush()
                            print(f"\n  Checkpoint: {end_idx}/{num_rollouts} rollouts saved")

                        # Print profiling summary after first batch
                        if self.profiler and batch_idx == 0:
                            print("\n")
                            self.profiler.print_summary()
                            print("\nContinuing generation...\n")

        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"Saved to: {output_path}")
        print(f"Total rollouts: {num_rollouts}")
        print(f"{'='*60}\n")

    def _generate_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Generate a batch of QBM rollouts with features.

        Args:
            batch_size: Number of rollouts to generate

        Returns:
            Dictionary containing:
                - fields: [B, M, 2, H, W] quantum ICs
                - params: [B, P] QBM parameters
                - temporal_features: [B, T, D_t] temporal features
                - initial_features: [B, D_i] initial features
                - rollouts: [B, M, T, 2, H, W] full trajectories (if store_rollouts=True)
        """
        with torch.no_grad():
            # 1. Sample parameters (Sobol)
            if self.profiler:
                with self.profiler.profile("parameter_sampling"):
                    params = self._sample_parameters(batch_size)
            else:
                params = self._sample_parameters(batch_size)

            # 2. Generate potentials from parameters
            if self.profiler:
                with self.profiler.profile("potential_generation"):
                    potentials = self._generate_potentials(params)
            else:
                potentials = self._generate_potentials(params)

            # 3. Generate quantum ICs
            if self.profiler:
                with self.profiler.profile("ic_generation"):
                    ics = self._generate_initial_conditions(batch_size)
            else:
                ics = self._generate_initial_conditions(batch_size)

            # 4. Run QBM simulations (batched across realizations)
            if self.profiler:
                with self.profiler.profile("qbm_simulation"):
                    rollouts = self._run_qbm_rollouts(ics, potentials, params)
            else:
                rollouts = self._run_qbm_rollouts(ics, potentials, params)

            # 5. Extract temporal features
            if self.profiler:
                with self.profiler.profile("temporal_feature_extraction"):
                    temporal_features = self._extract_temporal_features(rollouts)
            else:
                temporal_features = self._extract_temporal_features(rollouts)

            # 6. Extract initial features
            if self.profiler:
                with self.profiler.profile("initial_feature_extraction"):
                    initial_features = self._extract_initial_features(ics)
            else:
                initial_features = self._extract_initial_features(ics)

            # Return data (async writer handles .cpu().numpy())
            result = {
                'fields': ics,
                'params': params,
                'temporal_features': temporal_features,
                'initial_features': initial_features,
            }

            if self.store_rollouts:
                result['rollouts'] = rollouts

            return result

    def _sample_parameters(self, batch_size: int) -> torch.Tensor:
        """Sample QBM parameters via Sobol sequence.

        Parameter vector structure [B, 12]:
            [0] gamma: Bath coupling strength
            [1] kT: Temperature (energy units)
            [2] mass: Particle mass
            [3] potential_type_idx: Potential type (0-3)
            [4-7] potential_params: Type-specific parameters
            [8-11] Reserved for future extensions

        Args:
            batch_size: Number of parameter vectors to sample

        Returns:
            Parameter tensor [B, 12] in [0, 1] range (unit-scaled)
        """
        # Use Sobol sequence for better parameter space coverage
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.num_params, scramble=True)
        params_unit = sobol_engine.draw(batch_size)  # [B, P] in [0, 1]

        return params_unit.to(self.device)

    def _sample_ic_type(self) -> tuple:
        """Sample an IC type from quantum distribution.

        Returns:
            Tuple of (ic_type, config_dict)
        """
        rand = torch.rand(1).item()

        for threshold, ic_type, config in self.IC_TYPE_CONFIG:
            if rand < threshold:
                return ic_type, config

        # Fallback
        raise RuntimeError("IC type sampling failed - check IC_TYPE_CONFIG")

    def _generate_initial_conditions(self, batch_size: int) -> torch.Tensor:
        """Generate quantum initial conditions.

        Samples from quantum IC distribution:
        - 40% Gaussian wavepackets
        - 30% Coherent states
        - 15% 2-component superpositions
        - 15% 3-component superpositions

        Args:
            batch_size: Number of parameter sets

        Returns:
            Quantum ICs [B, M, 2, H, W]
        """
        total_ics = batch_size * self.num_realizations

        # Sample IC types
        ic_types_and_configs = [self._sample_ic_type() for _ in range(total_ics)]

        # Group by type for batched generation
        type_groups = {}
        for idx, (ic_type, config) in enumerate(ic_types_and_configs):
            config_key = (ic_type, tuple(sorted(config.items())))
            if config_key not in type_groups:
                type_groups[config_key] = []
            type_groups[config_key].append(idx)

        # Generate all ICs in batched calls
        all_ics = [None] * total_ics

        for (ic_type, config_tuple), indices in type_groups.items():
            config = dict(config_tuple)
            group_size = len(indices)

            # Generate batch [group_size, 2, H, W]
            ics_batch = self.ic_generator.generate_batch(
                batch_size=group_size,
                ic_type=ic_type,
                **config
            )

            # Assign to correct positions
            for i, idx in enumerate(indices):
                all_ics[idx] = ics_batch[i]

        # Stack and reshape: [B*M, 2, H, W] -> [B, M, 2, H, W]
        all_ics_tensor = torch.stack(all_ics, dim=0)
        ics = all_ics_tensor.view(batch_size, self.num_realizations, self.num_channels, self.grid_size, self.grid_size)

        return ics

    def _generate_potentials(self, params: torch.Tensor) -> torch.Tensor:
        """Generate potentials from parameter vectors.

        Maps unit-scaled parameters [0,1] to physical ranges and generates
        appropriate potential landscapes.

        Args:
            params: Parameter vectors [B, P] in [0, 1]

        Returns:
            Potentials [B, H, W]
        """
        batch_size = params.shape[0]

        # Decode potential type (categorical from continuous [0,1])
        potential_type_idx = (params[:, 3] * 4).long().clamp(0, 3)  # [B]

        # Map potential types: 0=harmonic, 1=double_well, 2=quartic, 3=random
        potential_types = ['harmonic', 'double_well', 'quartic', 'random']

        # Generate potentials (batch by type for efficiency)
        potentials = torch.zeros(batch_size, self.grid_size, self.grid_size, device=self.device)

        for type_idx, pot_type in enumerate(potential_types):
            # Find samples with this potential type
            mask = (potential_type_idx == type_idx)
            if not mask.any():
                continue

            indices = torch.where(mask)[0]
            type_batch_size = indices.shape[0]

            # Extract and scale potential-specific parameters
            type_params = params[indices, 4:8]  # [type_batch_size, 4]

            # Scale parameters to physical ranges (type-dependent)
            if pot_type == 'harmonic':
                # [omega, center_x, center_y, mass]
                physical_params = torch.stack([
                    type_params[:, 0] * 4.0 + 0.5,  # omega: [0.5, 4.5]
                    (type_params[:, 1] - 0.5) * 2.0,  # center_x: [-1, 1]
                    (type_params[:, 2] - 0.5) * 2.0,  # center_y: [-1, 1]
                    type_params[:, 3] * 9.0 + 1.0,   # mass: [1, 10] (matches params[:, 2])
                ], dim=1)

            elif pot_type == 'double_well':
                # [barrier_height, separation, mass]
                physical_params = torch.stack([
                    type_params[:, 0] * 4.0 + 1.0,   # barrier_height: [1, 5]
                    type_params[:, 1] * 2.0 + 1.0,   # separation: [1, 3]
                    type_params[:, 2] * 9.0 + 1.0,   # mass: [1, 10]
                ], dim=1)

            elif pot_type == 'quartic':
                # [c0, c2, c4]
                physical_params = torch.stack([
                    (type_params[:, 0] - 0.5) * 2.0,  # c0: [-1, 1]
                    type_params[:, 1] * 2.0,          # c2: [0, 2]
                    type_params[:, 2] * 0.1,          # c4: [0, 0.1]
                ], dim=1)

            elif pot_type == 'random':
                # [correlation_length, amplitude]
                physical_params = torch.stack([
                    type_params[:, 0] * 1.5 + 0.3,    # correlation_length: [0.3, 1.8]
                    type_params[:, 1] * 2.0 + 0.5,    # amplitude: [0.5, 2.5]
                ], dim=1)

            # Generate potentials for this type
            V_type = self.potential_generator.from_type_and_params(
                batch_size=type_batch_size,
                potential_type=pot_type,
                params=physical_params
            )

            # Insert into output tensor
            potentials[indices] = V_type

        return potentials

    def _run_qbm_rollouts(
        self,
        ics: torch.Tensor,
        potentials: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """Run QBM simulations for all realizations.

        Args:
            ics: Initial wavefunctions [B, M, 2, H, W]
            potentials: Potential landscapes [B, H, W]
            params: QBM parameters [B, P]

        Returns:
            Rollouts [B, M, T, 2, H, W]
        """
        batch_size = ics.shape[0]

        # Flatten realizations: [B, M, 2, H, W] -> [B*M, 2, H, W]
        ics_flat = ics.view(-1, self.num_channels, self.grid_size, self.grid_size)

        # Replicate potentials: [B, H, W] -> [B*M, H, W]
        potentials_repeated = potentials.unsqueeze(1).repeat(1, self.num_realizations, 1, 1).view(-1, self.grid_size, self.grid_size)

        # Extract and scale physical parameters
        # params: [B, P] -> physical: [B*M, 3] (gamma, kT, mass)
        gamma = params[:, 0] * 0.0999 + 0.0001  # [0.0001, 0.1]
        kT = 10 ** (params[:, 1] * 3 - 2)  # [0.01, 10] log-scale
        mass = params[:, 2] * 9.9 + 0.1  # [0.1, 10]

        # Replicate for realizations
        params_physical = torch.stack([gamma, kT, mass], dim=1)  # [B, 3]
        params_repeated = params_physical.unsqueeze(1).repeat(1, self.num_realizations, 1).view(-1, 3)  # [B*M, 3]

        # Run batched QBM simulation
        rollouts_flat = self.simulator.rollout(
            psi_0=ics_flat,
            potential=potentials_repeated,
            params=params_repeated,
            num_steps=self.rollout_steps,
            return_all_steps=True
        )  # [B*M, T+1, 2, H, W]

        # Remove initial condition (keep only evolved states)
        rollouts_flat = rollouts_flat[:, 1:, ...]  # [B*M, T, 2, H, W]

        # Reshape back: [B*M, T, 2, H, W] -> [B, M, T, 2, H, W]
        rollouts = rollouts_flat.view(batch_size, self.num_realizations, self.rollout_steps, self.num_channels, self.grid_size, self.grid_size)

        return rollouts

    def _extract_temporal_features(self, rollouts: torch.Tensor) -> torch.Tensor:
        """Extract temporal features from QBM rollouts.

        Args:
            rollouts: QBM trajectories [B, M, T, 2, H, W]

        Returns:
            Temporal features [B, T, D_t]
        """
        # MNOFeatureExtractor expects [B, M, T, C, H, W]
        extracted = self.temporal_extractor.extract(rollouts)
        return extracted['temporal']  # [B, T, D_t]

    def _extract_initial_features(self, ics: torch.Tensor) -> torch.Tensor:
        """Extract initial features from quantum ICs.

        Args:
            ics: Initial wavefunctions [B, M, 2, H, W]

        Returns:
            Initial features [B, D_i] (aggregated across M)
        """
        # InitialManualExtractor returns [B, M, D_i]
        features = self.initial_extractor.extract_all(ics)  # [B, M, D_i]

        # Aggregate across realizations (mean)
        features_aggregated = features.mean(dim=1)  # [B, D_i]

        return features_aggregated
