"""
Abstract base classes for feature extraction.

Provides extension points for new feature families. All feature extractors
inherit from FeatureExtractorBase to ensure consistent interfaces.

Design Pattern:
    - FeatureExtractorBase: Abstract extractor interface
    - HDF5FeatureWriter: Abstract HDF5 writer interface

This allows adding new feature families (e.g., temporal_series, spatial_tokens)
while maintaining a consistent API.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.registry import FeatureRegistry


class FeatureExtractorBase(ABC):
    """
    Abstract base class for feature extractors.

    Two sibling feature families:
    - TEMPORAL: Per-timestep time series [N, T, D] (spatial, spectral, cross_channel)
    - SUMMARY: Aggregated scalars [N, D] (temporal dynamics, causality, invariant_drift)

    The extraction pipeline follows a three-stage process:
    1. Per-timestep extraction: Features for each evolution step
    2. Per-trajectory aggregation: Temporal aggregation for each realization
    3. Cross-realization aggregation: Final summary across stochastic realizations

    This design enables both detailed temporal analysis and compact summaries.
    """

    @property
    @abstractmethod
    def family_name(self) -> str:
        """
        Feature family name.

        Returns:
            Family identifier (e.g., 'temporal', 'summary')
        """
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """
        Feature family version.

        Returns:
            Semantic version string (e.g., '1.0.0')
        """
        pass

    @abstractmethod
    def extract_per_timestep(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Extract per-timestep features.

        Computes features for each timestep independently, enabling temporal
        analysis and detection of dynamical transitions.

        Args:
            trajectories: Stochastic trajectories with shape [N, M, T, C, H, W]
                N = batch size (number of samples)
                M = num realizations (stochastic rollouts per sample)
                T = num timesteps (evolution length)
                C = channels (typically 3)
                H, W = spatial dimensions (grid size)
            metadata: Optional per-sample metadata (IC types, evolution policies, etc.)

        Returns:
            Per-timestep features with shape [N, T, D]
                D = number of per-timestep features

            Note: Features are averaged across realizations at this stage
            to produce a single time series per sample.
        """
        pass

    @abstractmethod
    def extract_per_trajectory(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Extract per-trajectory features (aggregated over time).

        Computes trajectory-level summaries that capture overall dynamics
        (growth rates, oscillation periods, stability measures, etc.).

        Args:
            trajectories: Stochastic trajectories [N, M, T, C, H, W]
            metadata: Optional per-sample metadata

        Returns:
            Per-trajectory features with shape [N, M, D_traj]
                D_traj = number of trajectory-level features

            Note: Each realization gets its own feature vector, preserving
            information about stochastic variability.
        """
        pass

    @abstractmethod
    def aggregate_realizations(
        self,
        per_trajectory_features: torch.Tensor,  # [N, M, D_traj]
        method: str = "mean"
    ) -> torch.Tensor:
        """
        Aggregate per-trajectory features across realizations.

        Reduces stochastic realizations to a single summary vector per sample.

        Args:
            per_trajectory_features: Per-realization features [N, M, D_traj]
            method: Aggregation method
                "mean" - Average across realizations
                "std" - Standard deviation across realizations
                "min" - Minimum across realizations
                "max" - Maximum across realizations
                "cv" - Coefficient of variation (std/mean)

        Returns:
            Aggregated features [N, D_final]
                D_final = D_traj (for single aggregation method)
                       or D_traj * num_methods (if combining multiple)
        """
        pass

    @abstractmethod
    def get_feature_registry(self) -> 'FeatureRegistry':
        """
        Get feature registry for this family.

        The registry maps feature names (including dynamic multiscale features)
        to integer indices for efficient extraction and storage.

        Returns:
            FeatureRegistry instance with all features registered
        """
        pass


class HDF5FeatureWriter(ABC):
    """
    Abstract base class for writing features to HDF5.

    Handles schema management, chunked writes, compression, and
    metadata storage for feature datasets.

    Each feature family implements its own writer to customize
    the HDF5 schema (e.g., different temporal granularities,
    different metadata requirements).
    """

    @abstractmethod
    def create_feature_group(
        self,
        h5file,
        family_name: str,
        num_samples: int,
        num_timesteps: int,
        num_realizations: int,
        feature_config: Any
    ) -> None:
        """
        Create HDF5 group structure for feature family.

        Sets up the schema with appropriate datasets, dimensions, chunking,
        and compression based on the feature configuration.

        Args:
            h5file: Open h5py.File handle
            family_name: Feature family identifier (e.g., 'temporal', 'summary')
            num_samples: Number of samples in dataset (N)
            num_timesteps: Number of evolution timesteps (T)
            num_realizations: Number of stochastic realizations (M)
            feature_config: Feature family configuration (family-specific)

        Side Effects:
            Creates /features/{family_name}/ group with subgroups:
            - per_timestep/ (if applicable)
            - per_trajectory/ (if applicable)
            - aggregated/ (always present)
            - metadata/ (always present)
        """
        pass

    @abstractmethod
    def write_features_batch(
        self,
        h5file,
        family_name: str,
        batch_idx: int,
        per_timestep: Optional[np.ndarray],
        per_trajectory: Optional[np.ndarray],
        aggregated: Optional[np.ndarray]
    ) -> None:
        """
        Write a batch of features to HDF5.

        Supports chunked writes for memory efficiency with large datasets.

        Args:
            h5file: Open h5py.File handle
            family_name: Feature family identifier
            batch_idx: Starting index for this batch
            per_timestep: Per-timestep features [B, T, D] or None
            per_trajectory: Per-trajectory features [B, M, D_traj] or None
            aggregated: Aggregated features [B, D_final] or None

        Side Effects:
            Writes features to appropriate datasets in /features/{family_name}/
        """
        pass
