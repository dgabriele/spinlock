"""
MNO Feature Generation Pipeline with Training Distribution Alignment.

Stage 2 of Independent Optimization Architecture: Generates large-scale feature
datasets from trained MNO that align with its training distribution.

**Training Distribution Alignment (Critical Design Choice):**

MNO is trained on a stratified 2K subsample (stride=50) from the 100K CNO dataset.
To ensure VQ-VAE learns MNO's actual distribution, feature generation uses **denser
stratification within the training span**:

    Training (Stage 1):  2K samples at stride=50 → [0, 50, 100, ..., 99950]
    Generation (Stage 2): 10K samples at stride=10 → [0, 10, 20, ..., 99990]
    Result: 2K exact training points + 8K interpolated points

This ensures:
- ✅ In-distribution: All parameters within MNO's training span [0, 99950]
- ✅ More diversity: 10K unique parameters vs 2K training (5× more)
- ✅ Tests smoothness: 8K interpolations test MNO's local generalization
- ✅ Maintains structure: Denser Sobol sampling preserves low-discrepancy

**Process:**
1. Load MNO checkpoint and extract training configuration (dataset path, n_samples)
2. Calculate training span from checkpoint (e.g., stride=50 → [0, 99950])
3. Generate denser indices within span (e.g., stride=10 → [0, 99990])
4. Load parameter vectors from original dataset at these dense indices
5. Generate diverse ICs for each parameter (25% per IC family)
6. MNO rollout prediction (256 steps, no gradients)
7. Inline feature extraction (GPU-optimized, SUMMARY + INITIAL)
8. Storage to HDF5 (features + parameters, no trajectories → 99.99% savings)

**Key differences from CNO generation:**
- Uses single trained MNO instead of building diverse CNO operators
- Parameters loaded from dataset (not freshly sampled)
- Training distribution alignment (denser stratification within span)
- Feature-only mode (no trajectories stored)

**Documentation:**
- Architecture guide: docs/noa-vqvae-independent.md (Stage 2 section)
- CLI command: src/spinlock/cli/generate_noa_features.py
- Training guide: docs/noa-training-guide.md
"""

import torch
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import time
import h5py

from ..noa.backbone import NOABackbone
from ..dataset.generators import InputFieldGenerator
from ..features.temporal import TemporalFeatureOrchestrator, TemporalFeatureConfig
from ..features.initial import InitialManualExtractor
from ..features.storage import HDF5FeatureWriter


class NOAFeatureGenerationPipeline:
    """
    Generate feature dataset from trained MNO with training distribution alignment.

    Stage 2 of Independent Optimization: Generates large-scale feature datasets
    from trained MNO using denser stratification within the training span.

    **Training Distribution Alignment:**
    Automatically extracts training configuration from MNO checkpoint to determine
    the training span (e.g., 2K samples at stride=50 → [0, 99950]). Then generates
    features with denser stratification (e.g., 10K samples at stride=10) to produce:
    - 2K exact training points (MNO saw these parameters during training)
    - 8K interpolated points (tests MNO's smoothness between training points)

    This ensures VQ-VAE learns MNO's actual distribution, not out-of-distribution
    parameters.

    **Process:**
    1. Load MNO checkpoint and extract training configuration
    2. Calculate dense indices within training span from original dataset
    3. Generate diverse ICs for each parameter (25% per IC family)
    4. Run MNO rollouts (256 steps, fast inference, no gradients)
    5. Extract features inline (GPU-optimized)
    6. Write features + parameters to HDF5 (no trajectories)

    **Usage:**
    ```python
    from spinlock.noa.generation_pipeline import NOAFeatureGenerationPipeline
    from spinlock.config import load_config

    config = load_config("configs/experiments/local_100k_optimized.yaml")
    pipeline = NOAFeatureGenerationPipeline(
        noa_checkpoint=Path("checkpoints/noa/pure_mse_v3_ic_fix/meta_operator_best.pt"),
        config=config,
    )
    pipeline.generate()
    ```

    **CLI:**
    ```bash
    spinlock generate-noa-dataset \\
        --noa-checkpoint checkpoints/noa/pure_mse_v3_ic_fix/meta_operator_best.pt \\
        --output datasets/noa_features/mno_v3_10k.h5 \\
        --n-samples 10000 \\
        --config configs/experiments/local_100k_optimized.yaml \\
        --device cuda --batch-size 16 --num-realizations 3
    ```

    **Documentation:**
    See docs/noa-vqvae-independent.md (Stage 2 section) for complete details.
    """

    def __init__(
        self,
        noa_checkpoint: Path,
        config: Any,
        verbose: bool = False,
        sampling_mode: str = "reference",
        compile_model: bool = False,
    ):
        """
        Initialize NOA feature generation pipeline.

        Args:
            noa_checkpoint: Path to trained NOA checkpoint
            config: SpinlockConfig (for sampling, simulation settings)
            verbose: Print detailed progress information
            sampling_mode: Sampling strategy ('reference' or 'interpolated')
            compile_model: Enable torch.compile() for faster inference (PyTorch 2.0+)
        """
        self.config = config
        self.verbose = verbose
        self.sampling_mode = sampling_mode
        self.compile_model = compile_model

        # Setup device
        device_str = config.simulation.device
        if device_str == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device(device_str)

        # Load NOA directly from checkpoint
        print("Loading trained MNO...")
        checkpoint = torch.load(noa_checkpoint, map_location=self.device, weights_only=False)

        # Validate checkpoint format
        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint: missing 'config' field.\n"
                f"Expected Stage 1 checkpoint format (from train-meta-operator)."
            )

        # Reconstruct MNO from checkpoint config
        model_config = checkpoint["config"]["model"]
        self.noa = NOABackbone(**model_config)
        self.noa.load_state_dict(checkpoint["model_state_dict"])
        self.noa = self.noa.to(self.device).eval()

        # Apply torch.compile() for faster inference (PyTorch 2.0+)
        if compile_model:
            try:
                print("  Compiling MNO with torch.compile()...")

                # Enable TF32 for better performance on Ampere+ GPUs
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    torch.set_float32_matmul_precision('high')
                    print("  ✓ TensorFloat32 enabled for matmul")

                # Use reduce-overhead mode for better performance on repeated calls
                # This mode is optimized for models that run many times
                self.noa = torch.compile(self.noa, mode="reduce-overhead")
                print("  ✓ MNO compiled successfully (mode: reduce-overhead)")
            except Exception as e:
                print(f"  ⚠ torch.compile() failed: {e}")
                print("  Continuing without compilation")

        # Print checkpoint info
        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", "unknown")
        print(f"  ✓ MNO loaded (epoch {epoch}, val_loss {val_loss})")
        print(f"  ✓ Device: {self.device}")

        # Initialize input generator for diverse ICs
        # Extract grid_size from config (or use default)
        grid_size = 64  # Default
        if hasattr(config.parameter_space, 'operator'):
            if hasattr(config.parameter_space.operator, 'grid_size'):
                if hasattr(config.parameter_space.operator.grid_size, 'choices'):
                    grid_size = config.parameter_space.operator.grid_size.choices[0]

        self.input_generator = InputFieldGenerator(
            grid_size=grid_size,
            num_channels=1,  # NOA uses single-channel
            device=self.device,
        )

        # Load training parameters and handle sampling strategy
        print("Calculating parameter sampling strategy...")

        training_n_samples = checkpoint['config']['training']['n_samples']
        training_dataset_path = checkpoint['config']['data']['dataset_path']

        n_generation_samples = config.sampling.total_samples

        with h5py.File(training_dataset_path, 'r') as f:
            total_samples = f['parameters/params'].shape[0]

            if sampling_mode == "reference":
                # Reference mode: Use full reference dataset
                # Validate n_samples doesn't exceed reference size
                if n_generation_samples > total_samples:
                    raise ValueError(
                        f"Requested {n_generation_samples} samples but reference dataset "
                        f"only has {total_samples} samples. Use --n-samples <={total_samples}."
                    )

                # Use first n_generation_samples from reference dataset
                self.generation_indices = np.arange(n_generation_samples)

                # Load parameters from reference dataset
                self.generation_params = torch.from_numpy(
                    f['parameters/params'][self.generation_indices]
                ).float()

                # Store path for batch IC loading (memory optimization)
                self.reference_dataset_path = training_dataset_path
                self.reference_ics = None  # Load on-demand

                # Mark interpolated vs exact training points
                # First training_n_samples are exact (MNO trained on these)
                # Remaining samples are interpolated (MNO generalizing)
                self.is_interpolated_mask = np.ones(len(self.generation_indices), dtype=bool)
                self.is_interpolated_mask[:training_n_samples] = False

                n_exact = (~self.is_interpolated_mask).sum()
                n_interp = self.is_interpolated_mask.sum()

                print(f"  ✓ Reference mode: Using first {n_generation_samples} samples from reference dataset")
                print(f"  ✓ Reference dataset: {training_dataset_path}")
                print(f"  ✓ ICs will be loaded from reference (realization_idx=0)")
                print(f"  ✓ Exact training points (MNO trained): {n_exact} (indices 0-{training_n_samples-1})")
                print(f"  ✓ Interpolated points (MNO generalizing): {n_interp} (indices {training_n_samples}-{n_generation_samples-1})")

            elif sampling_mode == "interpolated":
                # Interpolated mode: Dense stratification within training span (current behavior)
                training_stride = total_samples // training_n_samples
                training_min_idx = 0
                training_max_idx = (training_n_samples - 1) * training_stride

                print(f"  Training span: indices [{training_min_idx}, {training_max_idx}] (stride={training_stride})")

                generation_stride = (training_max_idx + training_stride) // n_generation_samples
                self.generation_indices = np.arange(0, training_max_idx + training_stride, generation_stride)[:n_generation_samples]

                # Load parameter vectors at generation indices
                self.generation_params = torch.from_numpy(
                    f['parameters/params'][self.generation_indices]
                ).float()

                self.reference_dataset_path = None
                self.reference_ics = None

                # Create interpolation mask (current logic)
                training_indices_set = set(np.arange(0, total_samples, training_stride)[:training_n_samples])
                self.is_interpolated_mask = np.ones(len(self.generation_indices), dtype=bool)
                for i, idx in enumerate(self.generation_indices):
                    if idx in training_indices_set:
                        self.is_interpolated_mask[i] = False

                n_exact = (~self.is_interpolated_mask).sum()
                n_interp = self.is_interpolated_mask.sum()

                print(f"  ✓ Interpolated mode: Denser stratification within training span")
                print(f"  ✓ Generation indices: [{self.generation_indices[0]}, ..., {self.generation_indices[-1]}] (stride={generation_stride})")
                print(f"  ✓ Exact training points: {n_exact}")
                print(f"  ✓ Interpolated points: {n_interp}")

            else:
                raise ValueError(f"Unknown sampling_mode: '{sampling_mode}'. Must be 'reference' or 'interpolated'.")

            print(f"  ✓ Total samples: {len(self.generation_indices)}")
            print(f"  ✓ Parameter space: {self.generation_params.shape[1]}D")

        # Initialize feature extractors
        print("Initializing feature extractors...")

        # TEMPORAL features (per-timestep only)
        temporal_config = TemporalFeatureConfig.from_schema_config(config.features.temporal)
        self.temporal_extractor = TemporalFeatureOrchestrator(
            device=self.device,
            config=temporal_config,
        )

        # INITIAL features (manual features from ICs)
        self.initial_extractor = InitialManualExtractor(device=self.device)

        # Check if temporal features are enabled
        self.temporal_enabled = config.features.temporal.enabled
        if not self.temporal_enabled:
            print("  ⚠️  TEMPORAL features DISABLED (per-timestep time series)")
        else:
            print("  ✓ TEMPORAL features ENABLED")

        print("  ✓ SUMMARY features ENABLED")
        print("  ✓ INITIAL features ENABLED")
        print(f"  ✓ Feature extractors ready")

        # Statistics
        self.stats = {
            "total_time": 0.0,
            "generation_time": 0.0,
            "feature_extraction_time": 0.0,
            "storage_time": 0.0,
            "samples_generated": 0,
        }

    def generate(self) -> None:
        """
        Execute complete NOA feature generation pipeline.

        Main entry point that coordinates all stages:
        1. Generate diverse initial conditions
        2. Run MNO rollouts in batches
        3. Extract features inline (GPU-optimized)
        4. Write features to HDF5
        """
        start_time = time.time()

        n_samples = self.config.sampling.total_samples
        batch_size = self.config.sampling.batch_size
        num_realizations = self.config.simulation.num_realizations
        timesteps = self.config.simulation.num_timesteps

        print(f"\n{'='*70}")
        print(f"Generating {n_samples} MNO rollouts with feature extraction")
        print(f"{'='*70}")
        print(f"  Batch size: {batch_size}")
        print(f"  Realizations: {num_realizations}")
        print(f"  Timesteps: {timesteps}")
        print(f"  Output: {self.config.dataset.output_path}")
        print(f"{'='*70}\n")

        # Initialize HDF5 feature writer
        feature_writer = HDF5FeatureWriter(
            dataset_path=Path(self.config.dataset.output_path),
            overwrite=True,  # Create new dataset
        )

        try:
            with feature_writer:
                # Calculate feature dimensions for HDF5 allocation (mirroring standard pipeline)
                registry = self.temporal_extractor.get_feature_registry()

                # Per-timestep (TEMPORAL) categories: spatial, spectral, cross_channel
                if self.temporal_enabled:
                    per_timestep_dim = (
                        len(registry.get_feature_names(category='spatial')) +
                        len(registry.get_feature_names(category='spectral')) +
                        len(registry.get_feature_names(category='cross_channel'))
                    )
                else:
                    per_timestep_dim = 0

                # Per-trajectory categories: temporal, causality, invariant_drift, operator_sensitivity
                per_trajectory_dim = (
                    len(registry.get_feature_names(category='temporal')) +
                    len(registry.get_feature_names(category='causality')) +
                    len(registry.get_feature_names(category='invariant_drift')) +
                    len(registry.get_feature_names(category='operator_sensitivity'))
                )

                # Get INITIAL feature dimension by extracting from a dummy IC
                dummy_grid_size = self.input_generator.grid_size
                dummy_ic = torch.zeros(1, 1, 1, dummy_grid_size, dummy_grid_size, device=self.device)
                dummy_features = self.initial_extractor.extract_all(dummy_ic)
                initial_dim = dummy_features.shape[-1]

                print(f"Feature dimensions calculated from registry:")
                print(f"  INITIAL (manual): {initial_dim}")
                print(f"  Per-timestep (TEMPORAL): {per_timestep_dim}")
                print(f"  Per-trajectory (SUMMARY): {per_trajectory_dim}")
                print(f"  Aggregated: {per_trajectory_dim * 3}\n")

                # Initialize HDF5 structure using create_summary_group
                param_dim = self.config.parameter_space.total_dimensions
                grid_size = self.input_generator.grid_size

                feature_writer.create_summary_group(
                    num_samples=n_samples,
                    num_timesteps=timesteps,
                    num_realizations=num_realizations,
                    registry=registry,
                    config=self.temporal_extractor.config,
                    compression="gzip",
                    compression_opts=4,
                    chunk_size=min(100, n_samples),
                    temporal_enabled=self.temporal_enabled,
                    learned_dim=0,  # No learned features for NOA
                    param_dim=param_dim,  # Operator parameter space dimension
                )

                # Create inputs group for raw ICs (compatible with base operator datasets)
                feature_writer.create_inputs_group(
                    num_samples=n_samples,
                    num_realizations=num_realizations,
                    grid_size=grid_size,
                    compression="gzip",
                    compression_opts=4,
                    chunk_size=min(100, n_samples),
                )

                # Create metadata group for NOA-specific metadata
                feature_writer.create_noa_metadata_group(
                    num_samples=n_samples,
                    compression="gzip",
                    compression_opts=4,
                )

                # Store metadata for reference feature alignment
                # generation_indices: maps to reference dataset indices for feature extraction
                # is_interpolated: identifies interpolated vs exact training points
                if 'metadata' not in feature_writer.file:
                    feature_writer.file.create_group('metadata')

                feature_writer.file['metadata'].create_dataset(
                    'generation_indices',
                    data=self.generation_indices,
                    compression='gzip',
                    compression_opts=4,
                )
                feature_writer.file['metadata/generation_indices'].attrs['description'] = (
                    'Indices into reference dataset (100K) for parameter alignment. '
                    'Used to extract reference features at same parameter points as MNO.'
                )

                feature_writer.file['metadata'].create_dataset(
                    'is_interpolated',
                    data=self.is_interpolated_mask,
                    compression='gzip',
                    compression_opts=4,
                )
                feature_writer.file['metadata/is_interpolated'].attrs['description'] = (
                    'Boolean mask: True for interpolated points (MNO generalizing), '
                    'False for exact training points (MNO learned during training).'
                )

                # Create INITIAL features dataset (manual features from ICs)
                if 'features/initial' not in feature_writer.file:
                    initial_group = feature_writer.file['features'].create_group('initial')
                    initial_agg_group = initial_group.create_group('aggregated')
                    initial_agg_group.create_dataset(
                        'features',
                        shape=(n_samples, initial_dim),
                        dtype='float32',
                        compression='gzip',
                        compression_opts=4,
                        chunks=(min(100, n_samples), initial_dim),
                    )
                    initial_agg_group['features'].attrs['description'] = (
                        f'Manual INITIAL features extracted from raw ICs ({initial_dim}D): '
                        'statistical and geometric properties of initial conditions'
                    )

                print(f"Feature datasets created (SUMMARY + TEMPORAL + INITIAL)\n")

                # Generate in batches
                num_batches = (n_samples + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(num_batches), desc="Generating MNO features"):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_samples)
                    current_batch_size = batch_end - batch_start

                    # Step 1: Get operator parameters from pre-loaded training distribution
                    gen_start = time.time()
                    batch_params = self.generation_params[batch_start:batch_end].to(self.device)  # [B, D] in [0,1]

                    # Step 2: Get initial conditions
                    if self.sampling_mode == "reference":
                        # Reference mode: Load ICs from reference dataset
                        with h5py.File(self.reference_dataset_path, 'r') as f:
                            batch_indices = self.generation_indices[batch_start:batch_end]
                            reference_ics_batch = f['inputs/fields'][batch_indices, 0, :, :]  # [B, H, W]

                        ics = torch.from_numpy(reference_ics_batch).float().unsqueeze(1).to(self.device)  # [B, 1, H, W]

                        # For storage: replicate to M realizations (dataset compatibility)
                        ics_for_storage = ics.unsqueeze(1).repeat(1, num_realizations, 1, 1, 1).squeeze(2)  # [B, M, H, W]

                        # ic_types for metadata
                        ic_types = ["reference_ic"] * current_batch_size

                    elif self.sampling_mode == "interpolated":
                        # Interpolated mode: Generate fresh ICs (current behavior)
                        # Sample IC types for this batch
                        ic_types = [self._sample_ic_type() for _ in range(current_batch_size)]

                        # Group by IC type for efficient batch generation
                        from collections import defaultdict
                        ic_type_to_indices = defaultdict(list)
                        for i, ic_type in enumerate(ic_types):
                            ic_type_to_indices[ic_type].append(i)

                        # Generate ICs per type - create M realizations per sample for dataset compatibility
                        # (MNO uses only the first realization for rollout)
                        all_ics_single = []  # For MNO rollout: [B, 1, H, W]
                        all_ics_multi = []  # For storage: [B, M, H, W]

                        for ic_type, indices in ic_type_to_indices.items():
                            ic_params = self._get_ic_params(ic_type)
                            base_ic_type = self._get_base_ic_type(ic_type)

                            # Generate M realizations per sample
                            # Total: len(indices) samples × M realizations
                            batch_ics_all = self.input_generator.generate_batch(
                                batch_size=len(indices) * num_realizations,
                                field_type=base_ic_type,
                                **ic_params,
                            )  # [B*M, 1, H, W]

                            # Reshape to [B, M, H, W]
                            batch_ics_multi = batch_ics_all.view(len(indices), num_realizations,
                                                                  self.input_generator.grid_size,
                                                                  self.input_generator.grid_size)

                            # Extract first realization for MNO rollout
                            batch_ics_single = batch_ics_multi[:, 0:1]  # [B, 1, H, W]

                            all_ics_single.append((indices, batch_ics_single))
                            all_ics_multi.append((indices, batch_ics_multi))

                        # Assemble ICs for MNO rollout (single realization)
                        ics = torch.zeros(
                            (current_batch_size, 1, self.input_generator.grid_size, self.input_generator.grid_size),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for indices, batch_ics in all_ics_single:
                            for i, idx in enumerate(indices):
                                ics[idx] = batch_ics[i]

                        # Assemble ICs for storage (multiple realizations)
                        ics_for_storage = torch.zeros(
                            (current_batch_size, num_realizations, self.input_generator.grid_size, self.input_generator.grid_size),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        for indices, batch_ics in all_ics_multi:
                            for i, idx in enumerate(indices):
                                ics_for_storage[idx] = batch_ics[i]

                    self.stats["generation_time"] += time.time() - gen_start

                    # Step 3: Generate MNO predictions
                    rollout_start = time.time()
                    with torch.no_grad():
                        trajectories = self.noa(
                            ics,
                            steps=timesteps,
                            return_all_steps=True,
                        )  # [B, T+1, C, H, W] or [B, T, C, H, W]

                    # Remove initial condition if included (depends on NOA implementation)
                    if trajectories.shape[1] == timesteps + 1:
                        trajectories = trajectories[:, 1:]  # Remove IC, keep [B, T, C, H, W]

                    self.stats["generation_time"] += time.time() - rollout_start

                    # Extract features
                    feat_start = time.time()

                    # Extract INITIAL features from ICs (before rollout)
                    # ICs shape: [B, 1, H, W] → add realization dim → [B, 1, 1, H, W]
                    ics_for_initial = ics.unsqueeze(1)  # [B, 1, 1, H, W]
                    initial_features = self.initial_extractor.extract_all(ics_for_initial)  # [B, M, D_initial]
                    # Remove realization dimension (M=1)
                    initial_features = initial_features.squeeze(1)  # [B, D_initial]

                    # Add realization dimension for feature extractor: [B, R, T, C, H, W]
                    trajectories_with_realizations = trajectories.unsqueeze(1)

                    # Extract per-timestep features (TEMPORAL only - 193D)
                    # New per-timestep-only architecture
                    per_timestep = self.temporal_extractor.extract_per_timestep(
                        trajectories_with_realizations
                    )  # [B, T, 193]

                    self.stats["feature_extraction_time"] += time.time() - feat_start

                    # Write to HDF5
                    storage_start = time.time()

                    feature_writer.write_temporal_batch(
                        batch_idx=batch_start,
                        per_timestep=per_timestep.cpu().numpy(),  # [B, T, 193]
                    )

                    # Write INITIAL features
                    initial_features_np = initial_features.cpu().numpy()  # [B, D_initial]
                    feature_writer.file['features/initial/aggregated/features'][batch_start:batch_end] = initial_features_np

                    # Write operator parameters (from training dataset)
                    feature_writer.write_parameters(
                        batch_idx=batch_start,
                        parameters=batch_params.cpu().numpy(),  # [B, D] in [0,1]
                    )

                    # Write initial conditions (multiple realizations for dataset compatibility)
                    feature_writer.write_inputs_batch(
                        batch_idx=batch_start,
                        fields=ics_for_storage.cpu().numpy(),  # [B, M, H, W]
                    )

                    # Write metadata (ic_types, noise_regimes, evolution_policies, grid_sizes)
                    if self.sampling_mode == "reference":
                        # Reference mode: Use fixed metadata
                        noise_regimes = ["reference"] * current_batch_size
                    elif self.sampling_mode == "interpolated":
                        # Interpolated mode: Extract from generated ICs
                        noise_regimes = self._classify_noise_regimes(ic_types)

                    evolution_policies = [self._get_evolution_policy() for _ in range(current_batch_size)]
                    grid_sizes = np.full(current_batch_size, grid_size, dtype=np.int32)

                    feature_writer.write_noa_metadata_batch(
                        batch_idx=batch_start,
                        ic_types=ic_types,
                        noise_regimes=noise_regimes,
                        evolution_policies=evolution_policies,
                        grid_sizes=grid_sizes,
                    )

                    self.stats["storage_time"] += time.time() - storage_start

                    self.stats["samples_generated"] = batch_end

                    # Periodic GPU cache cleanup
                    if self.device.type == "cuda" and (batch_idx + 1) % 10 == 0:
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError during generation: {e}")
            raise
        finally:
            # Cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.stats["total_time"] = time.time() - start_time
        self._print_final_statistics()

    def _sample_ic_type(self) -> str:
        """
        Sample IC type based on configured weights.

        Returns:
            IC type name
        """
        weights = self.config.simulation.input_generation.ic_type_weights
        if not weights:
            return "gaussian_random_field"

        # Normalize weights
        total_weight = sum(weights.values())
        ic_types = list(weights.keys())
        probs = [weights[ic] / total_weight for ic in ic_types]

        # Sample
        import random
        return random.choices(ic_types, weights=probs, k=1)[0]

    def _get_base_ic_type(self, ic_type: str) -> str:
        """
        Get base IC type from alias (e.g., gaussian_random_field_v0 → gaussian_random_field).

        Args:
            ic_type: IC type name (possibly with alias suffix)

        Returns:
            Base IC type name
        """
        import re
        # Strip common alias patterns: _v[0-9], _low, _mid, _high
        base_type = re.sub(r'_(v\d+|low|mid|high)$', '', ic_type)
        return base_type

    def _get_ic_params(self, ic_type: str) -> Dict[str, Any]:
        """
        Get parameters for a specific IC type from config.

        Args:
            ic_type: IC type name (possibly with alias suffix)

        Returns:
            Dict of parameters for this IC type
        """
        config_gen = self.config.simulation.input_generation

        # Try to get parameters directly from the alias name
        if hasattr(config_gen, ic_type):
            params = getattr(config_gen, ic_type)
            if isinstance(params, dict):
                return params
            elif hasattr(params, '__dict__'):
                return vars(params)

        # Fall back to base IC type
        base_type = self._get_base_ic_type(ic_type)
        if hasattr(config_gen, base_type):
            params = getattr(config_gen, base_type)
            if isinstance(params, dict):
                return params
            elif hasattr(params, '__dict__'):
                return vars(params)

        # Default empty params
        return {}

    def _classify_noise_regimes(self, ic_types: list) -> list:
        """
        Classify noise regimes based on IC types.

        Args:
            ic_types: List of IC type names

        Returns:
            List of noise regime classifications
        """
        noise_regimes = []
        for ic_type in ic_types:
            # Classify based on IC type
            if 'gaussian' in ic_type or 'multiscale_grf' in ic_type:
                noise_regimes.append('stochastic')
            elif 'structured' in ic_type:
                noise_regimes.append('deterministic')
            elif 'localized' in ic_type:
                noise_regimes.append('localized')
            else:
                noise_regimes.append('unknown')
        return noise_regimes

    def _get_evolution_policy(self) -> str:
        """
        Get evolution policy from config.

        Returns:
            Evolution policy name
        """
        # Check if update_policy is in config
        if hasattr(self.config.parameter_space, 'update_policy'):
            policy_config = self.config.parameter_space.update_policy
            # If it's a choice parameter, get the default or first choice
            if hasattr(policy_config, 'choices'):
                return policy_config.choices[0]
            elif hasattr(policy_config, 'default'):
                return policy_config.default
            else:
                return 'residual'  # Default fallback
        else:
            return 'residual'  # Default fallback

    def _print_final_statistics(self) -> None:
        """Print final generation statistics."""
        total_time = self.stats["total_time"]
        n_samples = self.stats["samples_generated"]

        print(f"\n{'='*70}")
        print("GENERATION STATISTICS")
        print(f"{'='*70}")
        print(f"  Total samples: {n_samples}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Throughput: {n_samples / total_time:.1f} samples/s")
        print()
        print(f"Time breakdown:")
        print(f"  IC generation + rollouts: {self.stats['generation_time']:.1f}s "
              f"({100 * self.stats['generation_time'] / total_time:.1f}%)")
        print(f"  Feature extraction: {self.stats['feature_extraction_time']:.1f}s "
              f"({100 * self.stats['feature_extraction_time'] / total_time:.1f}%)")
        print(f"  Storage (HDF5): {self.stats['storage_time']:.1f}s "
              f"({100 * self.stats['storage_time'] / total_time:.1f}%)")
        print(f"{'='*70}\n")
