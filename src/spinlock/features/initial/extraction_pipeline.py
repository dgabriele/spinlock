"""
Initial feature extraction pipeline.

Provides a unified, configurable interface for extracting initial condition features
from various input formats. Handles shape detection, batch processing, and storage.
"""

from typing import Optional, Literal, Tuple
from pathlib import Path
from enum import Enum

import torch
import numpy as np
import h5py

from .ic_feature_extractors import InitialConditionsFeatureExtractor
from .manual_extractors import InitialManualExtractor


class ExtractorType(str, Enum):
    """Initial feature extractor types."""
    STATISTICAL = "statistical"  # Modern: distributional + energy features
    MANUAL = "manual"  # Legacy: spatial + spectral + information features


class InitialFeatureExtractionPipeline:
    """
    Unified pipeline for extracting initial condition features.

    Handles various input shapes, extractor types, and storage formats.

    Example:
        >>> pipeline = InitialFeatureExtractionPipeline(
        ...     extractor_type="statistical",
        ...     device="cuda",
        ...     batch_size=128
        ... )
        >>> features = pipeline.extract_from_array(ics)  # [N, D]
        >>> pipeline.extract_and_save(dataset_path="data.h5")
    """

    def __init__(
        self,
        extractor_type: ExtractorType = ExtractorType.STATISTICAL,
        device: Optional[str] = None,
        batch_size: int = 100,
        include_spatial: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize extraction pipeline.

        Args:
            extractor_type: Type of feature extractor to use
            device: Computation device ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for processing
            include_spatial: Include spatial features (gradients, Laplacian).
                           Only for statistical extractor. Default False due to
                           collapsed variance on small grids.
            verbose: Print progress information
        """
        self.extractor_type = ExtractorType(extractor_type)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.include_spatial = include_spatial
        self.verbose = verbose

        # Initialize extractor
        if self.extractor_type == ExtractorType.STATISTICAL:
            self.extractor = InitialConditionsFeatureExtractor(
                device=self.device,
                include_spatial=include_spatial
            )
        else:  # MANUAL
            self.extractor = InitialManualExtractor(device=self.device)

    def _normalize_input_shape(
        self,
        ics: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Normalize input tensor to expected shape for each extractor.

        Handles various input formats:
        - [N, H, W] -> add channel dim
        - [N, C, H, W] -> use directly (statistical) or reshape (manual)
        - [N, M, H, W] -> select first realization, add channel
        - [N, M, C, H, W] -> select first realization
        - [N, C, S, H, W] -> flatten channels and species

        Args:
            ics: Input tensor of various shapes

        Returns:
            normalized_tensor: Tensor in expected format
            metadata: Dict with original shape info
        """
        original_shape = ics.shape
        metadata = {'original_shape': original_shape}

        if self.verbose:
            print(f"  Input shape: {original_shape}")

        # Handle different input shapes
        if len(ics.shape) == 3:
            # [N, H, W] -> add channel dimension
            ics = ics.unsqueeze(1)  # [N, 1, H, W]
            metadata['added_channel_dim'] = True

        elif len(ics.shape) == 4:
            # Could be [N, C, H, W] or [N, M, H, W]
            # Assume [N, C, H, W] if C is small (< 10), otherwise [N, M, H, W]
            if ics.shape[1] >= 10:
                # Likely [N, M, H, W] - take first realization
                ics = ics[:, 0]  # [N, H, W]
                ics = ics.unsqueeze(1)  # [N, 1, H, W]
                metadata['selected_first_realization'] = True
            # else: [N, C, H, W] - use as is

        elif len(ics.shape) == 5:
            # Could be [N, M, C, H, W] or [N, C, S, H, W]
            N, dim1, dim2, H, W = ics.shape

            # If dim1 is small (< 10), assume [N, C, S, H, W] (channels, species)
            # Otherwise assume [N, M, C, H, W] (realizations, channels)
            if dim1 < 10:
                # [N, C, S, H, W] -> flatten C and S
                C, S = dim1, dim2
                ics = ics.reshape(N, C * S, H, W)  # [N, C*S, H, W]
                metadata['flattened_channels_species'] = True
                metadata['original_channels'] = C
                metadata['original_species'] = S
                if self.verbose:
                    print(f"  Flattened {C} channels × {S} species -> {C*S} channels")
            else:
                # [N, M, C, H, W] -> take first realization
                ics = ics[:, 0]  # [N, C, H, W]
                metadata['selected_first_realization'] = True

        # Convert to expected format for each extractor
        if self.extractor_type == ExtractorType.STATISTICAL:
            # Statistical extractor expects [N, C, H, W]
            if len(ics.shape) != 4:
                raise ValueError(
                    f"After normalization, expected 4D tensor [N, C, H, W], "
                    f"got {ics.shape}"
                )
            normalized = ics

        else:  # MANUAL
            # Manual extractor expects [N, M, C, H, W] where M is realizations
            if len(ics.shape) == 4:
                # [N, C, H, W] -> add realization dim
                normalized = ics.unsqueeze(1)  # [N, 1, C, H, W]
            elif len(ics.shape) == 5:
                normalized = ics
            else:
                raise ValueError(
                    f"Manual extractor expects 4D or 5D input, got {ics.shape}"
                )

        if self.verbose:
            print(f"  Normalized shape: {normalized.shape}")

        return normalized, metadata

    def extract_from_array(
        self,
        ics: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Extract features from numpy array or torch tensor.

        Args:
            ics: Initial conditions in any supported shape

        Returns:
            features: [N, D] feature array
        """
        # Convert to tensor
        if isinstance(ics, np.ndarray):
            ics = torch.from_numpy(ics).float()

        # Normalize shape
        ics_normalized, metadata = self._normalize_input_shape(ics)

        # Extract features in batches
        if self.verbose:
            print(f"  Extracting {self.extractor_type.value} features...")

        features_list = []
        num_samples = len(ics_normalized)

        for i in range(0, num_samples, self.batch_size):
            batch = ics_normalized[i:i + self.batch_size]

            if self.extractor_type == ExtractorType.STATISTICAL:
                # Statistical extractor: [B, C, H, W] -> [B, D]
                with torch.no_grad():
                    batch_features = self.extractor(batch)
            else:  # MANUAL
                # Manual extractor: [B, M, C, H, W] -> [B, M, D]
                batch_features = self.extractor.extract_all(batch)
                # Remove realization dimension if M=1
                if batch_features.shape[1] == 1:
                    batch_features = batch_features.squeeze(1)

            features_list.append(batch_features)

        features = torch.cat(features_list, dim=0)  # [N, D]

        if self.verbose:
            print(f"  Extracted features shape: {features.shape}")

        return features.cpu().numpy()

    def extract_and_save(
        self,
        dataset_path: str | Path,
        overwrite: bool = True,
        compression: bool = True,
    ) -> None:
        """
        Extract features from HDF5 dataset and save in-place.

        Loads from /inputs/fields and saves to /features/initial/aggregated/features

        Args:
            dataset_path: Path to HDF5 dataset
            overwrite: Overwrite existing features if present
            compression: Use gzip compression for storage

        Raises:
            FileNotFoundError: If dataset doesn't exist
            KeyError: If /inputs/fields not found in dataset
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"EXTRACTING INITIAL FEATURES: {self.extractor_type.value.upper()}")
            print("=" * 60)
            print(f"Dataset: {dataset_path}")

        with h5py.File(dataset_path, 'a') as f:
            # Validate raw ICs exist
            if 'inputs/fields' not in f:
                raise KeyError("No raw ICs found at /inputs/fields")

            ic_dataset = f['inputs/fields']
            num_samples = ic_dataset.shape[0]

            if self.verbose:
                print(f"  Total samples: {num_samples:,}")
                print(f"  IC shape: {ic_dataset.shape}")
                print(f"  Streaming in batches of {self.batch_size}...")

            # Determine feature dimension from a small probe batch
            probe = torch.from_numpy(ic_dataset[:1]).float()
            probe_norm, _ = self._normalize_input_shape(probe)
            if self.extractor_type == ExtractorType.STATISTICAL:
                with torch.no_grad():
                    probe_feat = self.extractor(probe_norm)
            else:
                probe_feat = self.extractor.extract_all(probe_norm)
                if probe_feat.shape[1] == 1:
                    probe_feat = probe_feat.squeeze(1)
            feat_dim = probe_feat.shape[-1]

            # Prepare output group
            if 'features/initial' in f:
                if overwrite:
                    if self.verbose:
                        print("  Overwriting existing initial features...")
                    del f['features/initial']
                else:
                    raise ValueError(
                        "Initial features already exist. Set overwrite=True to replace."
                    )

            init_group = f.create_group('features/initial')
            agg_group = init_group.create_group('aggregated')

            # Pre-allocate output dataset
            ds_kwargs = dict(shape=(num_samples, feat_dim), dtype='float32')
            if compression:
                ds_kwargs.update(compression='gzip', compression_opts=4,
                                 chunks=(min(self.batch_size, num_samples), feat_dim))
            feat_ds = agg_group.create_dataset('features', **ds_kwargs)

            # Stream: read batch from HDF5 → extract → write back
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_np = ic_dataset[start:end]
                batch_t = torch.from_numpy(batch_np).float()
                del batch_np  # free numpy copy immediately

                batch_norm, _ = self._normalize_input_shape(batch_t)
                del batch_t

                if self.extractor_type == ExtractorType.STATISTICAL:
                    with torch.no_grad():
                        batch_feat = self.extractor(batch_norm)
                else:
                    batch_feat = self.extractor.extract_all(batch_norm)
                    if batch_feat.shape[1] == 1:
                        batch_feat = batch_feat.squeeze(1)
                del batch_norm

                feat_ds[start:end] = batch_feat.cpu().numpy()
                del batch_feat

            if self.verbose:
                print(f"  ✓ Saved to /features/initial/aggregated/features [{num_samples}, {feat_dim}]")
                print("=" * 60)

    def get_feature_dimension(self, num_channels: int) -> int:
        """
        Get expected feature dimension for given number of input channels.

        Args:
            num_channels: Number of input channels

        Returns:
            Expected feature dimension
        """
        if self.extractor_type == ExtractorType.STATISTICAL:
            # Distributional: 11 features per channel
            distributional_dim = num_channels * 11

            # Energy: 2 norms per channel + cross-correlations
            energy_dim = num_channels * 2 + (num_channels * (num_channels - 1)) // 2

            # Spatial: 8 features per channel (if enabled)
            spatial_dim = num_channels * 8 if self.include_spatial else 0

            return distributional_dim + energy_dim + spatial_dim
        else:  # MANUAL
            # Manual extractor: 14 features per channel
            return num_channels * 14


def extract_initial_features(
    dataset_path: str | Path,
    extractor_type: ExtractorType | str = ExtractorType.STATISTICAL,
    device: Optional[str] = None,
    batch_size: int = 100,
    include_spatial: bool = False,
    overwrite: bool = True,
    verbose: bool = True,
) -> None:
    """
    Convenience function for extracting initial features.

    Args:
        dataset_path: Path to HDF5 dataset
        extractor_type: Type of extractor ('statistical' or 'manual')
        device: Computation device (None for auto-detect)
        batch_size: Batch size for processing
        include_spatial: Include spatial features (only for statistical)
        overwrite: Overwrite existing features
        verbose: Print progress information

    Example:
        >>> extract_initial_features(
        ...     dataset_path="datasets/my_data.h5",
        ...     extractor_type="statistical"
        ... )
    """
    pipeline = InitialFeatureExtractionPipeline(
        extractor_type=extractor_type,
        device=device,
        batch_size=batch_size,
        include_spatial=include_spatial,
        verbose=verbose,
    )

    pipeline.extract_and_save(
        dataset_path=dataset_path,
        overwrite=overwrite,
    )
