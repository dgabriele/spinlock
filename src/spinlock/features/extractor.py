"""
Main feature extraction orchestrator.

Coordinates the end-to-end feature extraction pipeline:
1. Read rollouts from HDF5 datasets
2. Extract features using family-specific extractors (SDF, etc.)
3. Write features back to HDF5
4. Progress tracking and logging

Example:
    >>> from spinlock.features.extractor import FeatureExtractor
    >>> from spinlock.features.config import FeatureExtractionConfig
    >>> from spinlock.features.summary.config import SummaryConfig
    >>>
    >>> config = FeatureExtractionConfig(
    ...     input_dataset=Path("datasets/benchmark_10k.h5"),
    ...     sdf=SummaryConfig()
    ... )
    >>> extractor = FeatureExtractor(config)
    >>> extractor.extract(verbose=True)
"""

import torch
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, cast
import time
from tqdm import tqdm

from spinlock.features.config import FeatureExtractionConfig
from spinlock.features.storage import HDF5FeatureWriter, HDF5FeatureReader
from spinlock.features.summary.extractors import SummaryExtractor
from spinlock.features.summary.config import SummaryConfig


class FeatureExtractor:
    """
    Main feature extraction orchestrator.

    Manages the end-to-end pipeline for extracting features from datasets.

    Attributes:
        config: Feature extraction configuration
        device: Computation device (cuda or cpu)
        sdf_extractor: SDF feature extractor (if enabled)
    """

    def __init__(self, config: FeatureExtractionConfig):
        """
        Initialize feature extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        # Initialize family-specific extractors
        self.sdf_extractor: Optional[SummaryExtractor] = None

        if config.sdf is not None:
            self.sdf_extractor = SummaryExtractor(
                device=self.device,
                config=config.sdf
            )

    def extract(self, verbose: bool = False) -> None:
        """
        Run feature extraction pipeline.

        Args:
            verbose: Print progress information

        Raises:
            ValueError: If no feature families are enabled
            FileNotFoundError: If input dataset doesn't exist
        """
        # Validate input dataset exists
        if not self.config.input_dataset.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.config.input_dataset}")

        # Check that at least one feature family is enabled
        if self.sdf_extractor is None:
            raise ValueError("No feature families enabled. Set config.sdf to enable SDF extraction.")

        # Determine output path
        output_path = self.config.output_dataset or self.config.input_dataset

        if verbose:
            print(f"Input dataset:  {self.config.input_dataset}")
            print(f"Output dataset: {output_path}")
            print(f"Device:         {self.device}")
            print(f"Batch size:     {self.config.batch_size}")
            print()

        # Read dataset metadata
        with h5py.File(self.config.input_dataset, 'r') as f:
            fields_dataset = cast(h5py.Dataset, f['outputs/fields'])
            shape = fields_dataset.shape
            num_samples_total = shape[0]
            num_realizations = shape[1]

            # Apply max_samples limit if specified
            if self.config.max_samples is not None:
                num_samples = min(num_samples_total, self.config.max_samples)
            else:
                num_samples = num_samples_total

            # Detect shape format:
            # - [N, M, T, C, H, W] (6D) - with timesteps
            # - [N, M, C, H, W] (5D) - single timestep
            if len(shape) == 6:
                num_timesteps = shape[2]
                num_channels = shape[3]
            elif len(shape) == 5:
                num_timesteps = 1  # Single timestep
                num_channels = shape[2]
            else:
                raise ValueError(f"Unexpected dataset shape: {shape}")

            if verbose:
                print(f"Dataset info:")
                print(f"  Total samples: {num_samples_total}")
                if self.config.max_samples is not None:
                    print(f"  Using subset:  {num_samples} (limited by --max-samples)")
                else:
                    print(f"  Samples:       {num_samples}")
                print(f"  Realizations:  {num_realizations}")
                print(f"  Timesteps:     {num_timesteps}")
                print(f"  Channels:      {num_channels}")
                print()

        # Extract SDF features
        if self.sdf_extractor is not None:
            if verbose:
                print("Extracting SDF features...")
                print()

            self._extract_summary(
                num_samples=num_samples,
                num_realizations=num_realizations,
                num_timesteps=num_timesteps,
                output_path=output_path,
                verbose=verbose
            )

        if verbose:
            print("\n✓ Feature extraction complete!")

    def _extract_summary(
        self,
        num_samples: int,
        num_realizations: int,
        num_timesteps: int,
        output_path: Path,
        verbose: bool = False
    ) -> None:
        """
        Extract SDF features from dataset.

        Args:
            num_samples: Number of samples in dataset
            num_realizations: Number of realizations per sample
            num_timesteps: Number of timesteps per trajectory
            output_path: Path to write features
            verbose: Print progress
        """
        assert self.sdf_extractor is not None, "SDF extractor not initialized"

        # Create HDF5 writer
        writer = HDF5FeatureWriter(
            dataset_path=output_path,
            overwrite=self.config.overwrite
        )

        # Get feature registry
        registry = self.sdf_extractor.get_feature_registry()

        if verbose:
            print(f"SDF Registry: {registry.num_features} features")
            for category in registry.categories:
                cat_features = registry.get_features_by_category(category)
                print(f"  - {category}: {len(cat_features)} features")
            print()

        # Create feature groups in HDF5
        with writer.open_for_writing():
            writer.create_sdf_group(
                num_samples=num_samples,
                num_realizations=num_realizations,
                num_timesteps=num_timesteps,
                registry=registry,
                config=self.config.sdf
            )

            # Process in batches
            batch_size = self.config.batch_size
            num_batches = (num_samples + batch_size - 1) // batch_size

            pbar = None
            if verbose:
                pbar = tqdm(total=num_samples, desc="Extracting SDF features")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                current_batch_size = end_idx - start_idx

                # Load batch of rollouts
                rollouts = self._load_rollout_batch(
                    start_idx=start_idx,
                    end_idx=end_idx
                )  # [B, M, T, C, H, W]

                # Move to device
                rollouts = rollouts.to(self.device)

                # Extract features
                start_time = time.time()

                features = self.sdf_extractor.extract_all(rollouts)

                # Move features back to CPU for storage
                per_timestep = features['per_timestep'].cpu().numpy()  # [B, T, D]
                per_trajectory = features['per_trajectory'].cpu().numpy()  # [B, M, D_traj]

                # Combine aggregated features
                aggregated_list = []
                for key in sorted(features.keys()):
                    if key.startswith('aggregated_'):
                        aggregated_list.append(features[key].cpu().numpy())  # [B, D_final]

                aggregated = np.concatenate(aggregated_list, axis=1)  # [B, D_total]

                # Extraction times
                extraction_time = time.time() - start_time
                extraction_times = np.full(current_batch_size, extraction_time / current_batch_size)

                # Write to HDF5
                writer.write_sdf_batch(
                    batch_idx=start_idx,
                    per_timestep=per_timestep,
                    per_trajectory=per_trajectory,
                    aggregated=aggregated,
                    extraction_times=extraction_times
                )

                if verbose and pbar is not None:
                    pbar.update(current_batch_size)

            if verbose and pbar is not None:
                pbar.close()

    def _load_rollout_batch(
        self,
        start_idx: int,
        end_idx: int
    ) -> torch.Tensor:
        """
        Load a batch of rollouts from dataset.

        Args:
            start_idx: Start index
            end_idx: End index (exclusive)

        Returns:
            Rollouts tensor [B, M, T, C, H, W]
        """
        with h5py.File(self.config.input_dataset, 'r') as f:
            # Load outputs (rollouts)
            # Dataset shape: [N, M, C, H, W] (single timestep) or [N, M, T, C, H, W] (multi-timestep)
            fields_dataset = cast(h5py.Dataset, f['outputs/fields'])
            outputs = fields_dataset[start_idx:end_idx]  # Load batch

            # Convert to tensor
            rollouts = torch.from_numpy(outputs).float()

            # Ensure shape is [B, M, T, C, H, W]
            if rollouts.ndim == 5:  # [B, M, C, H, W] - single timestep
                rollouts = rollouts.unsqueeze(2)  # [B, M, 1, C, H, W]
            elif rollouts.ndim == 6:  # [B, M, T, C, H, W] - already correct
                pass
            else:
                raise ValueError(f"Unexpected rollout shape: {rollouts.shape}")

        return rollouts

    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of what will be extracted.

        Returns:
            Dictionary with extraction details
        """
        summary = {
            'input_dataset': str(self.config.input_dataset),
            'output_dataset': str(self.config.output_dataset or self.config.input_dataset),
            'device': str(self.device),
            'batch_size': self.config.batch_size,
            'feature_families': []
        }

        if self.sdf_extractor is not None:
            registry = self.sdf_extractor.get_feature_registry()
            summary['feature_families'].append({
                'name': 'sdf',
                'version': self.sdf_extractor.version,
                'num_features': registry.num_features,
                'categories': {
                    cat: len(registry.get_features_by_category(cat))
                    for cat in registry.categories
                }
            })

        return summary
