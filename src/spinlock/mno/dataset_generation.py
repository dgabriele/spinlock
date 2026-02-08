"""MNO Rollout Dataset Generation.

Generates HDF5 datasets of MNO rollouts for tokenizer training.

This module handles:
- MNO checkpoint loading
- Sobol parameter sampling across full parameter space
- Batched rollout generation (GPU-accelerated)
- Feature extraction (temporal + initial)
- HDF5 saving with checkpointing for long runs

TODO (Post-Task 4): Refactor to properly reuse DatasetGenerationPipeline components
    - Import SummaryExtractor for features (currently using MNOFeatureExtractor)
    - Import LocalHDF5Backend for storage (currently manual HDF5)
    - Keep InputFieldGenerator for ICs (already done)
    - Ensures 100% identical structure to CNO datasets
    See: /tmp/mno_dataset_refactor_plan.md

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
        rollouts/  # Optional (only if --store-rollouts flag used)
            mno: [N, M, T, C, H, W]  # Full MNO trajectories (WARNING: ~180x more disk space!)

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

import gc
import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import time

from spinlock.mno.validation_utils import load_mno_checkpoint
from spinlock.mno.feature_extraction import MNOFeatureExtractor
from spinlock.features.initial.manual_extractors import InitialManualExtractor
from spinlock.dataset.generators import InputFieldGenerator
from spinlock.utils.profiler import PerformanceProfiler
from spinlock.utils.async_hdf5_writer import AsyncHDF5Writer


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

    IC Type Distribution (matches CNO datasets - cno_50k_v3_1.yaml):
        Ensures apples-to-apples comparison between MNO and CNO tokenizers.

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

    # IC Type Configuration (CNO Distribution - matches cno_50k_v3_1.yaml)
    # DO NOT MODIFY - ensures consistency with CNO datasets for fair comparison
    IC_TYPE_CONFIG = [
        # Gaussian Random Fields: 25% total (5 variance levels @ 5% each)
        (0.05, 'gaussian_random_field', {'length_scale': 0.05, 'variance': 0.25}),
        (0.10, 'gaussian_random_field', {'length_scale': 0.05, 'variance': 0.5}),
        (0.15, 'gaussian_random_field', {'length_scale': 0.05, 'variance': 1.0}),
        (0.20, 'gaussian_random_field', {'length_scale': 0.05, 'variance': 2.0}),
        (0.25, 'gaussian_random_field', {'length_scale': 0.05, 'variance': 4.0}),

        # Band-limited noise: 25% total (3 bands @ ~8.33% each)
        (0.3333, 'multiscale_grf', {'scales': [0.30, 0.35, 0.40], 'variance': 1.0}),
        (0.4166, 'multiscale_grf', {'scales': [0.08, 0.10, 0.12], 'variance': 1.0}),
        (0.50, 'multiscale_grf', {'scales': [0.02, 0.025, 0.03], 'variance': 1.0}),

        # Structured (geometric patterns): 25%
        (0.75, 'structured', {
            'num_structures': 5,
            'structure_types': ["circle", "stripe", "blob"]
        }),

        # Localized (blobs): 25%
        (1.00, 'localized', {
            'num_blobs': 5,
            'min_width': 5.0,
            'max_width': 15.0
        }),
    ]

    def __init__(
        self,
        mno_checkpoint: Path,
        num_realizations: int = 3,
        num_channels: int = 3,
        num_params: int = 14,
        rollout_steps: int = 256,
        device: str = "cuda",
        enable_profiling: bool = False,
        store_rollouts: bool = False
    ):
        """Initialize MNO rollout dataset generator.

        Args:
            mno_checkpoint: Path to trained MNO checkpoint
            num_realizations: Number of realizations per operator (default: 3, matches CNO datasets)
            num_channels: Number of channels in rollouts (default: 3)
            num_params: Number of operator parameters (default: 14)
            rollout_steps: Number of timesteps in rollouts (default: 256)
            device: Torch device for inference
            enable_profiling: Whether to enable performance profiling (default: False)
            store_rollouts: Whether to store full rollouts (default: False, saves ~180x disk space)
        """
        self.device = torch.device(device)
        self.num_realizations = num_realizations
        self.num_channels = num_channels
        self.num_params = num_params
        self.rollout_steps = rollout_steps
        self.store_rollouts = store_rollouts
        self.profiler = PerformanceProfiler() if enable_profiling else None

        # Load MNO checkpoint
        print(f"Loading MNO checkpoint: {mno_checkpoint}")
        self.mno = load_mno_checkpoint(str(mno_checkpoint), device=str(device))
        self.mno.eval()

        # Compile MNO for faster generation (enabled for production runs)
        print("Compiling MNO model (first batch will be slow)...")
        self.mno = torch.compile(self.mno, mode='default')

        # Initialize feature extractors
        print("Initializing feature extractors...")
        self.temporal_extractor = MNOFeatureExtractor(device=str(device))
        self.initial_extractor = InitialManualExtractor(device=self.device)

        # Initialize IC generator (uses same method as CNO datasets)
        self.ic_generator = InputFieldGenerator(
            grid_size=64,
            num_channels=self.num_channels,
            device=self.device
        )

        # Probe feature dimensions
        self._probe_feature_dimensions()

    def _probe_feature_dimensions(self):
        """Probe extractors to determine feature dimensions."""
        print("Probing feature dimensions...")

        # Probe temporal extractor
        dims = self.temporal_extractor.probe_dimensions(
            timesteps=self.rollout_steps,
            channels=self.num_channels,
            height=64,
            width=64,
            batch_size=1
        )
        self.temporal_dim = dims['temporal_dim']
        print(f"  Temporal feature dimension: {self.temporal_dim}")

        # Probe initial extractor
        dummy_ic = torch.randn(1, self.num_realizations, self.num_channels, 64, 64, device=self.device)
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

        # Create HDF5 file with dataset structure (matches CNO structure exactly)
        with h5py.File(output_path, 'w') as f:
            # Create dataset groups (match CNO structure)
            inputs_grp = f.create_group('inputs')
            params_grp = f.create_group('parameters')
            features_grp = f.create_group('features')
            rollouts_grp = f.create_group('rollouts')  # MNO-specific (for alignment analysis)

            # Feature subgroups (match CNO structure)
            temporal_grp = features_grp.create_group('temporal')
            initial_grp = features_grp.create_group('initial')
            initial_agg_grp = initial_grp.create_group('aggregated')

            # NOTE: CNO has features/architecture/aggregated/features for parameter encoding
            # MNO doesn't need this since we store raw parameters in parameters/params

            # Create datasets (fixed size)
            # Use compression level 1 for faster writes (good balance of speed vs size)
            fields_ds = inputs_grp.create_dataset(
                'fields',
                shape=(num_rollouts, self.num_realizations, self.num_channels, 64, 64),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.num_realizations, self.num_channels, 64, 64)
            )

            params_ds = params_grp.create_dataset(
                'params',
                shape=(num_rollouts, self.num_params),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.num_params)
            )

            # Match CNO structure: features/temporal/features (not features/temporal)
            temporal_ds = temporal_grp.create_dataset(
                'features',
                shape=(num_rollouts, self.rollout_steps, self.temporal_dim),
                dtype='float32',
                compression='lzf',
                chunks=(batch_size, self.rollout_steps, self.temporal_dim)
            )

            # Match CNO structure: features/initial/aggregated/features
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

            # Optionally store full rollouts (WARNING: uses ~180x more disk space!)
            # For 100K samples: features-only = ~7GB, with rollouts = ~1.2TB
            # Only enable this if you specifically need raw trajectories for analysis
            if self.store_rollouts:
                print("  WARNING: Storing full rollouts (this will use ~180x more disk space)")
                print("  100K samples: features-only = ~7GB, with rollouts = ~1.2TB")
                rollouts_ds = rollouts_grp.create_dataset(
                    'mno',
                    shape=(num_rollouts, self.num_realizations, self.rollout_steps, self.num_channels, 64, 64),
                    dtype='float32',
                    compression='lzf',
                    chunks=(batch_size, self.num_realizations, self.rollout_steps, self.num_channels, 64, 64)
                )
                datasets['rollouts'] = rollouts_ds
            else:
                print("  Storing features only (not full rollouts) - saves ~180x disk space")

            # Generate in batches with async writes
            num_batches = (num_rollouts + batch_size - 1) // batch_size

            with AsyncHDF5Writer(f, datasets, queue_size=4) as writer:
                with tqdm(total=num_rollouts, desc="Generating rollouts") as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_rollouts)
                        current_batch_size = end_idx - start_idx

                        # DEBUG: Timing
                        t_start = time.time()

                        # Generate batch (GPU computation)
                        t_gen_start = time.time()
                        batch_data = self._generate_batch(current_batch_size)
                        t_gen = time.time() - t_gen_start

                        # Submit write (async - happens in background)
                        t_write_start = time.time()
                        if self.profiler:
                            with self.profiler.profile("7_hdf5_write"):
                                writer.submit(start_idx, end_idx, batch_data)
                        else:
                            writer.submit(start_idx, end_idx, batch_data)
                        t_write = time.time() - t_write_start

                        # Explicit memory cleanup to prevent accumulation
                        t_gc_start = time.time()
                        del batch_data
                        gc.collect()
                        t_gc = time.time() - t_gc_start

                        t_cache_start = time.time()
                        if batch_idx % 2 == 0:
                            torch.cuda.empty_cache()
                        t_cache = time.time() - t_cache_start

                        t_total = time.time() - t_start

                        # Print timing breakdown
                        if batch_idx % 10 == 0:
                            print(f"\nBatch {batch_idx}: gen={t_gen:.2f}s write={t_write:.2f}s gc={t_gc:.2f}s cache={t_cache:.2f}s total={t_total:.2f}s")

                        pbar.update(current_batch_size)

                        # Print profiling summary after first batch
                        if self.profiler and batch_idx == 0:
                            print("\n")
                            self.profiler.print_summary()
                            print("\nContinuing generation...\n")

                        # Periodic checkpoint (flush to disk)
                        if end_idx % checkpoint_interval == 0:
                            writer.flush()
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
        _t_start = time.time()
        with torch.no_grad():
            # Sample parameters via Sobol sequence
            if self.profiler:
                with self.profiler.profile("1_parameter_sampling"):
                    params = self._sample_parameters(batch_size)
            else:
                params = self._sample_parameters(batch_size)

            # Generate initial conditions from parameters
            if self.profiler:
                with self.profiler.profile("2_ic_generation"):
                    ics = self._generate_initial_conditions(params)
            else:
                ics = self._generate_initial_conditions(params)

            # Generate MNO rollouts for all realizations (batched for efficiency)
            # Instead of 3 sequential rollouts, batch all realizations together
            if self.profiler:
                with self.profiler.profile("3_mno_rollout_generation"):
                    # Reshape: [B, M, C, H, W] -> [B*M, C, H, W]
                    ics_flat = ics.view(-1, self.num_channels, 64, 64)

                    # Replicate params for each realization: [B, P] -> [B*M, P]
                    params_repeated = params.unsqueeze(1).repeat(1, self.num_realizations, 1).view(-1, self.num_params)

                    # Single batched rollout for all realizations
                    rollouts_flat = self.mno.rollout(
                        ics_flat,
                        steps=self.rollout_steps,
                        return_all_steps=True,
                        params=params_repeated
                    )[:, 1:, ...]  # [B*M, T, C, H, W]

                    # Reshape back: [B*M, T, C, H, W] -> [B, M, T, C, H, W]
                    rollouts = rollouts_flat.view(batch_size, self.num_realizations, self.rollout_steps, self.num_channels, 64, 64)
            else:
                # Reshape: [B, M, C, H, W] -> [B*M, C, H, W]
                ics_flat = ics.view(-1, self.num_channels, 64, 64)

                # Replicate params for each realization: [B, P] -> [B*M, P]
                params_repeated = params.unsqueeze(1).repeat(1, self.num_realizations, 1).view(-1, self.num_params)

                # Single batched rollout for all realizations
                rollouts_flat = self.mno.rollout(
                    ics_flat,
                    steps=self.rollout_steps,
                    return_all_steps=True,
                    params=params_repeated
                )[:, 1:, ...]  # [B*M, T, C, H, W]

                # Reshape back: [B*M, T, C, H, W] -> [B, M, T, C, H, W]
                rollouts = rollouts_flat.view(batch_size, self.num_realizations, self.rollout_steps, self.num_channels, 64, 64)

            _t_rollout = time.time() - _t_start

            # Extract temporal features
            _t_temp_start = time.time()
            if self.profiler:
                with self.profiler.profile("4_temporal_feature_extraction"):
                    temporal_features = self._extract_temporal_features(rollouts)
            else:
                temporal_features = self._extract_temporal_features(rollouts)

            _t_temp = time.time() - _t_temp_start

            # Extract initial features
            _t_init_start = time.time()
            if self.profiler:
                with self.profiler.profile("5_initial_feature_extraction"):
                    initial_features = self._extract_initial_features(ics)
            else:
                initial_features = self._extract_initial_features(ics)

            _t_init = time.time() - _t_init_start
            _t_total_gen = time.time() - _t_start

            # DEBUG: Print component timing
            if not hasattr(self, '_batch_count'):
                self._batch_count = 0
            if self._batch_count % 5 == 0:
                print(f"  [Batch {self._batch_count}] rollout={_t_rollout:.2f}s temp_features={_t_temp:.2f}s init_features={_t_init:.2f}s total={_t_total_gen:.2f}s")
            self._batch_count += 1

            # Return tensors directly - async writer will handle .cpu().numpy() in background
            # This keeps the main thread on GPU and offloads CPU conversion to writer thread
            result = {
                'fields': ics,
                'params': params,
                'temporal_features': temporal_features,
                'initial_features': initial_features,
            }

            # Only include rollouts if requested (saves ~180x disk space when disabled)
            if self.store_rollouts:
                result['rollouts'] = rollouts

            return result

    def _sample_parameters(self, batch_size: int) -> torch.Tensor:
        """Sample operator parameters via Sobol sequence.

        Args:
            batch_size: Number of parameter vectors to sample

        Returns:
            Parameter tensor [B, num_params] in [0, 1] range (unit-scaled)
        """
        # Use Sobol sequence for better parameter space coverage
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.num_params, scramble=True)
        params_unit = sobol_engine.draw(batch_size)  # [B, num_params] in [0, 1]

        # Return unit-scaled parameters (matches existing dataset format)
        # The MNO model should handle the parameter scaling internally
        return params_unit.to(self.device)

    def _sample_ic_type(self) -> tuple[str, dict]:
        """Sample an IC type from CNO distribution.

        Uses IC_TYPE_CONFIG class constant to ensure consistency with CNO datasets.

        Returns:
            Tuple of (ic_type, config_dict)
        """
        rand = torch.rand(1).item()

        # Weighted sampling from IC_TYPE_CONFIG
        for threshold, ic_type, config in self.IC_TYPE_CONFIG:
            if rand < threshold:
                return ic_type, config

        # Fallback (should never reach here if IC_TYPE_CONFIG is properly configured)
        raise RuntimeError("IC type sampling failed - check IC_TYPE_CONFIG")

    def _generate_single_ic(self) -> torch.Tensor:
        """Generate a single initial condition sampled from CNO distribution.

        Returns:
            Single IC [C, H, W]
        """
        ic_type, config = self._sample_ic_type()

        # Generate IC [1, C, H, W] and remove batch dimension
        ic = self.ic_generator.generate_batch(
            batch_size=1,
            field_type=ic_type,
            **config
        )[0]  # [C, H, W]

        return ic

    def _generate_initial_conditions(self, params: torch.Tensor) -> torch.Tensor:
        """Generate stochastic initial conditions matching CNO dataset distribution.

        Uses exact same IC type distribution as CNO datasets (cno_50k_v3_1.yaml):
        - 25% Gaussian Random Fields (5 variance levels: 0.25, 0.5, 1.0, 2.0, 4.0)
        - 25% Band-limited noise (low/mid/high frequency)
        - 25% Structured (geometric patterns)
        - 25% Localized (blobs)

        Args:
            params: Parameter vectors [B, num_params]

        Returns:
            Initial conditions [B, M, C, H, W]
        """
        batch_size = params.shape[0]
        total_ics = batch_size * self.num_realizations

        # Sample IC types for all ICs at once (more efficient)
        ic_types_and_configs = [self._sample_ic_type() for _ in range(total_ics)]

        # Group by IC type for batched generation
        type_groups = {}
        for idx, (ic_type, config) in enumerate(ic_types_and_configs):
            # Use tuple of config items for hashable key (convert lists to tuples)
            config_items = []
            for k, v in sorted(config.items()):
                if isinstance(v, list):
                    v = tuple(v)  # Convert lists to tuples for hashing
                config_items.append((k, v))
            config_key = (ic_type, tuple(config_items))
            if config_key not in type_groups:
                type_groups[config_key] = []
            type_groups[config_key].append(idx)

        # Generate all ICs in batched calls
        all_ics = [None] * total_ics

        for (ic_type, config_tuple), indices in type_groups.items():
            config = dict(config_tuple)
            group_size = len(indices)

            # Generate batch of same IC type [group_size, C, H, W]
            ics_batch = self.ic_generator.generate_batch(
                batch_size=group_size,
                field_type=ic_type,
                **config
            )

            # Assign to correct positions
            for i, idx in enumerate(indices):
                all_ics[idx] = ics_batch[i]

        # Stack and reshape: [B*M, C, H, W] -> [B, M, C, H, W]
        all_ics_tensor = torch.stack(all_ics, dim=0)  # [B*M, C, H, W]
        ics = all_ics_tensor.view(batch_size, self.num_realizations, self.num_channels, 64, 64)

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
