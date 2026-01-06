"""NOA Feature Extraction - Extract SUMMARY and TEMPORAL features from rollouts.

This module provides feature extraction for NOA training on real data.
Uses the same extractors (SummaryExtractor) that generated the dataset features.

Key insight:
- SummaryExtractor produces BOTH summary features AND per-timestep features
- per_timestep output = TEMPORAL features
- aggregated_* outputs = SUMMARY features (after realization aggregation)

ALL DIMENSIONS ARE RESOLVED DYNAMICALLY AT RUNTIME.
No hardcoded feature counts - the system adapts to whatever the extractor produces.

NaN Handling:
    The extractor can optionally use a FeaturePreprocessor to clean NaN features.
    This ensures consistency with the ground-truth features from the dataset.
"""

import torch
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from spinlock.features.preprocessing import FeaturePreprocessor


class NOAFeatureExtractor:
    """Extract SUMMARY and TEMPORAL features from NOA rollouts.

    Uses SummaryExtractor internally - this is the SAME extractor that generated
    the dataset features, ensuring compatibility.

    Architecture:
        NOA rollout [B, T, C, H, W] → SummaryExtractor → {
            'per_timestep': [B, T, D_temporal]   → This IS TEMPORAL
            'per_trajectory': [B, M, D_summary]  → This IS SUMMARY (per realization)
        } → FeaturePreprocessor (if provided) → {
            cleaned features with NaN removed
        }

    All dimensions are determined dynamically at runtime based on extractor output.
    """

    def __init__(
        self,
        device: str = "cuda",
        preprocessor: Optional['FeaturePreprocessor'] = None,
    ):
        """Initialize NOA feature extractor.

        Args:
            device: Computation device
            preprocessor: Optional FeaturePreprocessor to clean NaN features
        """
        from spinlock.features.summary.config import SummaryConfig
        from spinlock.features.summary.extractors import SummaryExtractor

        self.device = torch.device(device)
        self.preprocessor = preprocessor

        # Create SummaryExtractor with default config
        summary_config = SummaryConfig()
        self.summary_extractor = SummaryExtractor(
            device=self.device,
            config=summary_config,
        )

        # Dimensions determined on first extraction
        self._temporal_dim: Optional[int] = None
        self._per_trajectory_dim: Optional[int] = None
        self._initialized = False

    def _initialize_dimensions(self, result: Dict[str, torch.Tensor]):
        """Initialize dimension attributes from first extraction result."""
        if 'per_timestep' in result:
            self._temporal_dim = result['per_timestep'].shape[-1]
        if 'per_trajectory' in result:
            self._per_trajectory_dim = result['per_trajectory'].shape[-1]
        self._initialized = True

    def extract(
        self,
        rollouts: torch.Tensor,
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Extract SUMMARY and TEMPORAL features from NOA rollouts.

        Args:
            rollouts: NOA output [B, T, C, H, W] or [B, M, T, C, H, W] for multi-realization
            return_raw: If True, also return raw extractor outputs

        Returns:
            Dictionary with:
                'summary': [B, D_summary] - per-trajectory SUMMARY features (cleaned if preprocessor)
                'temporal': [B, T, D_temporal] - per-timestep TEMPORAL features (cleaned if preprocessor)
                'raw' (optional): Full extractor output dictionary
        """
        # Handle both single-realization [B, T, C, H, W] and multi-realization [B, M, T, C, H, W]
        if rollouts.dim() == 5:
            B, T, C, H, W = rollouts.shape
            # Add M=1 realization dimension
            trajectories = rollouts.unsqueeze(1)
        elif rollouts.dim() == 6:
            B, M, T, C, H, W = rollouts.shape
            # Already has realization dimension
            trajectories = rollouts
        else:
            raise ValueError(f"Expected 5D or 6D tensor, got {rollouts.dim()}D")

        # Extract features using SummaryExtractor
        result = self.summary_extractor.extract_all(trajectories)

        # Initialize dimensions on first call
        if not self._initialized:
            self._initialize_dimensions(result)

        # TEMPORAL features: per_timestep [B, T, D_temporal]
        temporal_features = result['per_timestep']

        # SUMMARY features: per_trajectory [B, 1, D_summary] → [B, D_summary]
        per_trajectory = result['per_trajectory']
        summary_features = per_trajectory.squeeze(1)

        # Apply preprocessing if provided (clean NaN features)
        if self.preprocessor is not None:
            # Use 'summary_per_trajectory' since NOA extracts per single trajectory (M=1)
            summary_features = self.preprocessor.clean_features(
                summary_features, 'summary_per_trajectory'
            )
            temporal_features = self.preprocessor.clean_features(
                temporal_features, 'temporal'
            )

        # Replace any remaining NaN values with 0
        # This handles cases where std() produces NaN for M=1 realizations
        summary_features = torch.nan_to_num(summary_features, nan=0.0)
        temporal_features = torch.nan_to_num(temporal_features, nan=0.0)

        output = {
            'summary': summary_features,
            'temporal': temporal_features,
        }

        if return_raw:
            output['raw'] = result

        return output

    @property
    def temporal_dim(self) -> Optional[int]:
        """Per-timestep feature dimension (None if not yet determined)."""
        return self._temporal_dim

    @property
    def per_trajectory_dim(self) -> Optional[int]:
        """Per-trajectory summary feature dimension (None if not yet determined)."""
        return self._per_trajectory_dim

    def probe_dimensions(
        self,
        timesteps: int,
        channels: int = 1,
        height: int = 64,
        width: int = 64,
        batch_size: int = 1,
    ) -> Dict[str, int]:
        """Probe extractor to determine output dimensions.

        Runs a dummy extraction to determine feature dimensions without
        requiring actual data.

        Args:
            timesteps: Number of timesteps in rollout
            channels: Number of channels
            height: Grid height
            width: Grid width
            batch_size: Batch size for probe

        Returns:
            Dictionary with dimension info:
                'temporal_dim': Per-timestep feature dimension
                'per_trajectory_dim': Per-trajectory summary dimension
                'timesteps': Number of timesteps
        """
        dummy = torch.randn(
            batch_size, timesteps, channels, height, width,
            device=self.device,
        )
        result = self.extract(dummy)

        return {
            'temporal_dim': result['temporal'].shape[-1],
            'per_trajectory_dim': result['summary'].shape[-1],
            'timesteps': timesteps,
        }
