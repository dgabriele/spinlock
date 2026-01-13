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
from ..features.summary import SummaryExtractor, SummaryConfig
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
    ):
        """
        Initialize NOA feature generation pipeline.

        Args:
            noa_checkpoint: Path to trained NOA checkpoint
            config: SpinlockConfig (for sampling, simulation settings)
            verbose: Print detailed progress information
        """
        self.config = config
        self.verbose = verbose

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
        print("Loading trained NOA...")
        checkpoint = torch.load(noa_checkpoint, map_location=self.device, weights_only=False)

        # Validate checkpoint format
        if "config" not in checkpoint:
            raise ValueError(
                f"Invalid checkpoint: missing 'config' field.\n"
                f"Expected Stage 1 checkpoint format (from train-meta-operator)."
            )

        # Reconstruct NOA from checkpoint config
        model_config = checkpoint["config"]["model"]
        self.noa = NOABackbone(**model_config)
        self.noa.load_state_dict(checkpoint["model_state_dict"])
        self.noa = self.noa.to(self.device).eval()

        # Print checkpoint info
        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", "unknown")
        print(f"  ✓ NOA loaded (epoch {epoch}, val_loss {val_loss})")
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

        # Load training parameters for distribution alignment
        # MNO was trained on a stratified 2K subsample - we'll use denser stratification
        # within that training span for feature generation
        print("Calculating parameter sampling strategy...")

        training_n_samples = checkpoint['config']['training']['n_samples']  # 2000
        training_dataset_path = checkpoint['config']['data']['dataset_path']

        # Calculate training span (matching NOAStateDataset stratified sampling)
        with h5py.File(training_dataset_path, 'r') as f:
            total_samples = f['parameters/params'].shape[0]  # 100K

            # Training used stratified sampling: stride = 100K / 2K = 50
            training_stride = total_samples // training_n_samples
            training_min_idx = 0
            training_max_idx = (training_n_samples - 1) * training_stride  # 99950

            print(f"  Training span: indices [{training_min_idx}, {training_max_idx}] (stride={training_stride})")

            # For generation: denser stratification within training span
            # Use 10K samples with stride=10 to get indices [0, 10, 20, ..., 99990]
            # This includes all 2K training points (0, 50, 100, ...) + 8K interpolations
            n_generation_samples = config.sampling.total_samples  # 10K
            generation_stride = (training_max_idx + training_stride) // n_generation_samples  # ~10

            self.generation_indices = np.arange(0, training_max_idx + training_stride, generation_stride)[:n_generation_samples]

            # Load parameter vectors at generation indices
            self.generation_params = torch.from_numpy(
                f['parameters/params'][self.generation_indices]
            ).float()  # [10K, 12]

            # Count how many are exact training points
            training_indices_set = set(np.arange(0, total_samples, training_stride)[:training_n_samples])
            n_exact_training = sum(1 for idx in self.generation_indices if idx in training_indices_set)
            n_interpolated = len(self.generation_indices) - n_exact_training

            # Create interpolation mask for reference feature regularization
            # True = interpolated point (MNO generalizing), False = exact training point (MNO learned)
            self.is_interpolated_mask = np.ones(len(self.generation_indices), dtype=bool)
            for i, idx in enumerate(self.generation_indices):
                if idx in training_indices_set:
                    self.is_interpolated_mask[i] = False

        print(f"  ✓ Generation strategy: Denser stratification within training span")
        print(f"  ✓ Generation indices: [{self.generation_indices[0]}, ..., {self.generation_indices[-1]}] (stride={generation_stride})")
        print(f"  ✓ Total samples: {len(self.generation_indices)}")
        print(f"  ✓ Exact training points: {n_exact_training}")
        print(f"  ✓ Interpolated points: {n_interpolated}")
        print(f"  ✓ Parameter space: {self.generation_params.shape[1]}D")

        # Initialize feature extractors
        print("Initializing feature extractors...")

        # SUMMARY + TEMPORAL features
        summary_config = SummaryConfig.from_schema_config(config.features.summary)
        self.summary_extractor = SummaryExtractor(
            device=self.device,
            config=summary_config,
        )

        # Check if temporal features are enabled
        self.temporal_enabled = config.features.temporal.enabled
        if not self.temporal_enabled:
            print("  ⚠️  TEMPORAL features DISABLED (per-timestep time series)")
        else:
            print("  ✓ TEMPORAL features ENABLED")

        print("  ✓ SUMMARY features ENABLED")
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
        2. Run NOA rollouts in batches
        3. Extract features inline (GPU-optimized)
        4. Write features to HDF5
        """
        start_time = time.time()

        n_samples = self.config.sampling.total_samples
        batch_size = self.config.sampling.batch_size
        num_realizations = self.config.simulation.num_realizations
        timesteps = self.config.simulation.num_timesteps

        print(f"\n{'='*70}")
        print(f"Generating {n_samples} NOA rollouts with feature extraction")
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
                registry = self.summary_extractor.get_feature_registry()

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

                print(f"Feature dimensions calculated from registry:")
                print(f"  Per-timestep (TEMPORAL): {per_timestep_dim}")
                print(f"  Per-trajectory (SUMMARY): {per_trajectory_dim}")
                print(f"  Aggregated: {per_trajectory_dim * 3}\n")

                # Initialize HDF5 structure using create_summary_group
                param_dim = self.config.parameter_space.total_dimensions
                feature_writer.create_summary_group(
                    num_samples=n_samples,
                    num_timesteps=timesteps,
                    num_realizations=num_realizations,
                    registry=registry,
                    config=self.summary_extractor.config,
                    compression="gzip",
                    compression_opts=4,
                    chunk_size=min(100, n_samples),
                    temporal_enabled=self.temporal_enabled,
                    learned_dim=0,  # No learned features for NOA
                    param_dim=param_dim,  # Operator parameter space dimension
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

                print(f"Feature datasets created\n")

                # Generate in batches
                num_batches = (n_samples + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(num_batches), desc="Generating NOA features"):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_samples)
                    current_batch_size = batch_end - batch_start

                    # Step 1: Get operator parameters from pre-loaded training distribution
                    gen_start = time.time()
                    batch_params = self.generation_params[batch_start:batch_end].to(self.device)  # [B, D] in [0,1]

                    # Step 2: Generate initial conditions with diverse IC types
                    # Sample IC types for this batch
                    ic_types = [self._sample_ic_type() for _ in range(current_batch_size)]

                    # Group by IC type for efficient batch generation
                    from collections import defaultdict
                    ic_type_to_indices = defaultdict(list)
                    for i, ic_type in enumerate(ic_types):
                        ic_type_to_indices[ic_type].append(i)

                    # Generate ICs per type
                    all_ics = []
                    for ic_type, indices in ic_type_to_indices.items():
                        ic_params = self._get_ic_params(ic_type)
                        base_ic_type = self._get_base_ic_type(ic_type)

                        # Generate batch for this IC type
                        batch_ics = self.input_generator.generate_batch(
                            batch_size=len(indices),
                            field_type=base_ic_type,
                            **ic_params,
                        )
                        all_ics.append((indices, batch_ics))

                    # Assemble ICs in correct order
                    ics = torch.zeros(
                        (current_batch_size, 1, self.input_generator.grid_size, self.input_generator.grid_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for indices, batch_ics in all_ics:
                        for i, idx in enumerate(indices):
                            ics[idx] = batch_ics[i]

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

                    # Add realization dimension for feature extractor: [B, R, T, C, H, W]
                    trajectories_with_realizations = trajectories.unsqueeze(1)

                    # Extract per-timestep features (TEMPORAL)
                    if self.temporal_enabled:
                        per_timestep = self.summary_extractor.extract_per_timestep(
                            trajectories_with_realizations
                        )  # [B, R, T, D]
                    else:
                        per_timestep = None

                    # Extract per-trajectory features (SUMMARY)
                    per_trajectory = self.summary_extractor.extract_per_trajectory(
                        trajectories_with_realizations
                    )  # [B, M, D]

                    # Aggregate across realizations (mean, std, cv)
                    aggregated_list = []
                    for method in ['mean', 'std', 'cv']:
                        agg = self.summary_extractor.aggregate_realizations(
                            per_trajectory, method=method
                        )
                        aggregated_list.append(agg.cpu())

                    # Concatenate aggregations: [B, D*3]
                    aggregated = torch.cat(aggregated_list, dim=1).numpy()

                    self.stats["feature_extraction_time"] += time.time() - feat_start

                    # Write to HDF5
                    storage_start = time.time()

                    feature_writer.write_summary_batch(
                        batch_idx=batch_start,
                        per_timestep=per_timestep.cpu().numpy() if per_timestep is not None else None,
                        per_trajectory=per_trajectory.cpu().numpy(),  # [B, M, D]
                        aggregated=aggregated,  # Already numpy: [B, D*3]
                    )

                    # Write operator parameters (from training dataset)
                    feature_writer.write_parameters(
                        batch_idx=batch_start,
                        parameters=batch_params.cpu().numpy(),  # [B, D] in [0,1]
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
