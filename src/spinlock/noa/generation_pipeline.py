"""
NOA Feature Generation Pipeline.

Generates large-scale feature datasets from a trained NOA model for VQ-VAE training.
This pipeline is used in the "train tokenizer on simulator's distribution" architecture
where VQ-VAE is trained on NOA's outputs rather than CNO's outputs for optimal alignment.

Key differences from CNO generation:
- Uses single trained NOA instead of building diverse CNO operators
- Diversity comes from diverse initial conditions, not diverse operators
- Simpler pipeline: IC generation → NOA rollout → feature extraction → storage
- Feature-only mode (no trajectories stored to save 99% space)
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
    Generate feature dataset from trained NOA model.

    This pipeline generates a large-scale feature dataset by:
    1. Loading a trained NOA checkpoint
    2. Generating diverse initial conditions
    3. Running NOA rollouts for each IC
    4. Extracting features inline (GPU-optimized)
    5. Writing features to HDF5 (no trajectories stored)

    Example:
        ```python
        from spinlock.noa.generation_pipeline import NOAFeatureGenerationPipeline
        from spinlock.config import load_config

        config = load_config("config.yaml")
        pipeline = NOAFeatureGenerationPipeline(
            noa_checkpoint=Path("checkpoints/noa/best_model.pt"),
            config=config,
        )
        pipeline.generate()
        ```
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
        self.input_generator = InputFieldGenerator(
            grid_size=64,  # TODO: Extract from config or NOA checkpoint
            num_channels=1,  # NOA uses single-channel
            device=self.device,
        )

        # Initialize parameter sampler for operator diversity
        from ..sampling.sobol import StratifiedSobolSampler
        self.parameter_sampler = StratifiedSobolSampler.from_config(
            parameter_space=config.parameter_space,
            config=config.sampling,
        )
        print(f"  ✓ Parameter sampler initialized (D={config.parameter_space.total_dimensions})")

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

                print(f"Feature datasets created\n")

                # Generate in batches
                num_batches = (n_samples + batch_size - 1) // batch_size

                for batch_idx in tqdm(range(num_batches), desc="Generating NOA features"):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, n_samples)
                    current_batch_size = batch_end - batch_start

                    # Step 1: Sample operator parameters via Sobol
                    gen_start = time.time()
                    unit_params = self.parameter_sampler.sample(current_batch_size)  # [B, D] in [0,1]

                    # Step 2: Generate initial conditions for sampled parameters
                    ics = self.input_generator.generate_batch(current_batch_size)  # [B, C, H, W]
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

                    # Write operator parameters (Sobol vectors)
                    feature_writer.write_parameters(
                        batch_idx=batch_start,
                        parameters=unit_params,  # [B, D] in [0,1]
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
