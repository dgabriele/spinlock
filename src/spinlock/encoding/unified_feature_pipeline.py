"""Unified Feature Pipeline for VQ-VAE Training and Inference.

This module provides a single, DRY implementation of feature extraction,
encoding, and normalization used by both VQ-VAE training and meta-operator training.

Design Principles:
1. Single Source of Truth: One feature extraction implementation everywhere
2. Dimension Consistency: Same dimensions at normalization points
3. Checkpoint Completeness: VQ-VAE checkpoint contains all info for inference
4. Clear Stages: raw → encode → normalize

Usage:
    # During VQ-VAE training:
    pipeline = UnifiedFeaturePipeline.create_for_training(config)
    features = pipeline(trajectories, ics, normalize=False)
    pipeline.compute_normalization_stats([features])
    pipeline.save_to_checkpoint(checkpoint)

    # During meta-operator training (Stage 2):
    pipeline = UnifiedFeaturePipeline.from_checkpoint(vqvae_checkpoint)
    features = pipeline(noa_trajectory, ic, normalize=True)

Documentation:
    - Stage 2 feature processing details: docs/two-stage-curriculum-architecture.md
    - VQ-VAE checkpoint format: docs/vqvae/checkpoint-format.md
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn


class FamilyFeatureExtractor(nn.Module, ABC):
    """Base class for per-family feature extraction.

    Each family (INITIAL, SUMMARY, TEMPORAL) has:
    - Raw extractor: trajectory/IC → raw features
    - Encoder: raw features → encoded features (optional)
    - Normalization stats: mean/std for normalized features
    """

    def __init__(self, family_name: str, encoder: Optional[nn.Module] = None):
        """Initialize family feature extractor.

        Args:
            family_name: Name of feature family ('initial', 'summary', 'temporal')
            encoder: Optional encoder module for dimensionality reduction
        """
        super().__init__()
        self.family_name = family_name
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None
        self.normalization_stats = None  # (mean, std) tensors

    @abstractmethod
    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract raw features from trajectory/IC.

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] initial condition

        Returns:
            Raw features (dimensions depend on family)
        """
        pass

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode raw features (identity if no encoder).

        Args:
            raw_features: Raw features from extract_raw()

        Returns:
            Encoded features
        """
        if self.encoder is None:
            return raw_features
        return self.encoder(raw_features)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using stored stats.

        Args:
            features: Features to normalize

        Returns:
            Normalized features (zero mean, unit variance)
        """
        if self.normalization_stats is None:
            return features

        mean, std = self.normalization_stats

        # Handle empty normalization stats (skip normalization)
        if mean.numel() == 0 or std.numel() == 0:
            return features

        # Ensure stats are on same device as features
        mean = mean.to(features.device)
        std = std.to(features.device)
        return (features - mean) / (std + 1e-8)

    def forward(self, trajectory: torch.Tensor, ic: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Full pipeline: extract → encode → normalize.

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] initial condition
            normalize: Whether to apply normalization

        Returns:
            Processed features
        """
        raw = self.extract_raw(trajectory, ic)
        encoded = self.encode(raw)
        if normalize:
            return self.normalize(encoded)
        return encoded


class InitialFeatureExtractor(FamilyFeatureExtractor):
    """INITIAL family: manual features from IC (per-channel)."""

    def __init__(self, device: str = 'cuda', in_channels: int = 3):
        """Initialize INITIAL feature extractor.

        Args:
            device: Device for computation
            in_channels: Number of input channels (for computing feature dimension)
        """
        super().__init__('initial', encoder=None)  # No encoding for manual features
        from spinlock.features.initial.manual_extractors import InitialManualExtractor
        self.raw_extractor = InitialManualExtractor(device=device)
        self.device = device
        self._in_channels = in_channels  # Store for feature_dim property

    @property
    def feature_dim(self) -> int:
        """Output feature dimension (14 * in_channels)."""
        return 14 * self._in_channels

    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract manual features from IC (per-channel).

        Args:
            trajectory: [B, T, C, H, W] (unused, but kept for interface consistency)
            ic: [B, C, H, W] initial condition

        Returns:
            [B, 14*C] manual features (per-channel)
        """
        # InitialManualExtractor expects [B, M, C, H, W], so add dimension
        ic_with_m = ic.unsqueeze(1)  # [B, 1, C, H, W]
        features = self.raw_extractor.extract_all(ic_with_m)  # [B, 1, 14*C]
        return features.squeeze(1)  # [B, 14*C]


class SummaryFeatureExtractor(FamilyFeatureExtractor):
    """SUMMARY family: 360D raw → 128D encoded."""

    def __init__(self, encoder: nn.Module, device: str = 'cuda'):
        """Initialize SUMMARY feature extractor.

        Args:
            encoder: MLP encoder (360D → 128D)
            device: Device for computation
        """
        super().__init__('summary', encoder=encoder)
        from spinlock.features.temporal.config import SummaryConfig
        from spinlock.features.temporal.extractors import SummaryExtractor

        # Use completely default config to extract all summary features
        # This should produce 360D output matching VQ-VAE training
        config = SummaryConfig()
        self.raw_extractor = SummaryExtractor(device=device, config=config)
        self.device = device
        self.expected_raw_dim = 360  # Expected dimension before encoding

    @property
    def feature_dim(self) -> int:
        """Output feature dimension after encoding."""
        if self.encoder is None:
            return self.expected_raw_dim
        return getattr(self.encoder, 'output_dim', 128)  # Default to 128 if no output_dim

    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract summary features (variable dimension).

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] (unused, but kept for interface consistency)

        Returns:
            [B, D] summary features (D varies based on config)
        """
        # SummaryExtractor expects [B, M, T, C, H, W], so add realization dimension
        traj_with_m = trajectory.unsqueeze(1)  # [B, 1, T, C, H, W]
        result = self.raw_extractor.extract_all(traj_with_m)

        # Extract aggregated features (mean over realizations and time)
        if 'aggregated_mean' in result:
            return result['aggregated_mean']  # [B, D]

        # Fallback: use per_trajectory and squeeze
        return result['per_trajectory'].squeeze(1)

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode raw features with padding/truncation to expected dimension.

        Args:
            raw_features: Raw summary features [B, D]

        Returns:
            Encoded features [B, 128]
        """
        if self.encoder is None:
            return raw_features

        # Pad/truncate to expected dimension (360D) before encoding
        B = raw_features.shape[0]
        if raw_features.shape[1] < self.expected_raw_dim:
            pad = torch.zeros(B, self.expected_raw_dim - raw_features.shape[1],
                              device=raw_features.device, dtype=raw_features.dtype)
            raw_features = torch.cat([raw_features, pad], dim=1)
        elif raw_features.shape[1] > self.expected_raw_dim:
            raw_features = raw_features[:, :self.expected_raw_dim]

        return self.encoder(raw_features)


class TemporalFeatureExtractor(FamilyFeatureExtractor):
    """TEMPORAL family: 63D per-timestep → 128D encoded."""

    def __init__(self, encoder: nn.Module, device: str = 'cuda'):
        """Initialize TEMPORAL feature extractor.

        Args:
            encoder: Temporal CNN encoder (63D×T → 128D)
            device: Device for computation
        """
        super().__init__('temporal', encoder=encoder)
        from spinlock.features.temporal.config import SummaryConfig
        from spinlock.features.temporal.extractors import SummaryExtractor

        # Configuration for per-timestep features
        config = SummaryConfig(
            realization_aggregation=["mean"],
            temporal_aggregation=["mean"],
        )
        self.raw_extractor = SummaryExtractor(device=device, config=config)
        self.device = device
        self.expected_raw_dim = 63  # Expected per-timestep feature dimension

    @property
    def feature_dim(self) -> int:
        """Output feature dimension after encoding."""
        if self.encoder is None:
            return self.expected_raw_dim
        return getattr(self.encoder, 'output_dim', 128)  # Default to 128 if no output_dim

    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep features [B, T, D].

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] (unused, but kept for interface consistency)

        Returns:
            [B, T, D] per-timestep features (D varies based on config)
        """
        # SummaryExtractor expects [B, M, T, C, H, W]
        traj_with_m = trajectory.unsqueeze(1)  # [B, 1, T, C, H, W]
        result = self.raw_extractor.extract_all(traj_with_m)
        return result['per_timestep']  # [B, T, D]

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode per-timestep features with padding/truncation to expected dimension.

        Args:
            raw_features: Per-timestep features [B, T, D]

        Returns:
            Encoded features [B, 128]
        """
        if self.encoder is None:
            return raw_features.mean(dim=1)  # Average over time

        # Pad/truncate feature dimension to expected (63D) before encoding
        B, T, D = raw_features.shape
        if D < self.expected_raw_dim:
            pad = torch.zeros(B, T, self.expected_raw_dim - D,
                              device=raw_features.device, dtype=raw_features.dtype)
            raw_features = torch.cat([raw_features, pad], dim=2)
        elif D > self.expected_raw_dim:
            raw_features = raw_features[:, :, :self.expected_raw_dim]

        return self.encoder(raw_features)


class PyramidTemporalFeatureExtractor(FamilyFeatureExtractor):
    """TEMPORAL family with pyramid multi-resolution encoding.

    Uses PyramidTemporalEncoder to produce multi-scale embeddings from
    per-timestep features. Output dimension depends on encoder level_dims
    (e.g., 320D for [32, 64, 96, 128]).
    """

    def __init__(self, encoder: nn.Module, device: str = 'cuda'):
        """Initialize pyramid temporal feature extractor.

        Args:
            encoder: PyramidTemporalEncoder instance
            device: Device for computation
        """
        super().__init__('temporal', encoder=encoder)
        from spinlock.features.temporal.config import SummaryConfig
        from spinlock.features.temporal.extractors import SummaryExtractor

        config = SummaryConfig(
            realization_aggregation=["mean"],
            temporal_aggregation=["mean"],
        )
        self.raw_extractor = SummaryExtractor(device=device, config=config)
        self.device = device
        self.expected_raw_dim = 63

    @property
    def feature_dim(self) -> int:
        """Output feature dimension after encoding."""
        if self.encoder is None:
            return self.expected_raw_dim
        return getattr(self.encoder, 'output_dim', 320)  # Default to 320 for pyramid

    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract per-timestep features [B, T, D].

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] (unused)

        Returns:
            [B, T, D] per-timestep features
        """
        traj_with_m = trajectory.unsqueeze(1)
        result = self.raw_extractor.extract_all(traj_with_m)
        return result['per_timestep']

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Encode per-timestep features via pyramid encoder.

        Args:
            raw_features: Per-timestep features [B, T, D]

        Returns:
            Multi-scale encoded features [B, sum(level_dims)]
        """
        if self.encoder is None:
            return raw_features.mean(dim=1)

        B, T, D = raw_features.shape
        if D < self.expected_raw_dim:
            pad = torch.zeros(B, T, self.expected_raw_dim - D,
                              device=raw_features.device, dtype=raw_features.dtype)
            raw_features = torch.cat([raw_features, pad], dim=2)
        elif D > self.expected_raw_dim:
            raw_features = raw_features[:, :, :self.expected_raw_dim]

        return self.encoder(raw_features)


class UnifiedFeaturePipeline(nn.Module):
    """Unified feature extraction pipeline for VQ-VAE training and inference.

    This class ensures VQ-VAE training and meta-operator training use
    IDENTICAL feature extraction, encoding, and normalization.

    Features are organized into families:
    - INITIAL: manual features from initial condition (14*C, where C=num_channels)
    - SUMMARY: aggregated features, optionally encoded (optional)
    - TEMPORAL: per-timestep features, encoded via:
        - TemporalCNNEncoder, or
        - PyramidTemporalEncoder

    Total output dimension is computed dynamically by querying extractors' feature_dim properties.
    """

    def __init__(
        self,
        initial_extractor: InitialFeatureExtractor,
        summary_extractor: Optional[SummaryFeatureExtractor],
        temporal_extractor: TemporalFeatureExtractor,
    ):
        """Initialize unified feature pipeline.

        Args:
            initial_extractor: INITIAL family extractor
            summary_extractor: SUMMARY family extractor (optional, can be None for 2-family checkpoints)
            temporal_extractor: TEMPORAL family extractor
        """
        super().__init__()
        self.initial = initial_extractor
        self.summary = summary_extractor
        self.temporal = temporal_extractor

    @property
    def feature_dim(self) -> int:
        """Total feature dimension, computed dynamically from extractors."""
        dim = self.initial.feature_dim  # Query from extractor (14*C for C channels)
        if self.summary is not None:
            dim += self.summary.feature_dim  # Query from extractor
        dim += self.temporal.feature_dim  # Query from extractor
        return dim

    def forward(
        self,
        trajectory: torch.Tensor,
        ic: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """Extract, encode, and optionally normalize features.

        Args:
            trajectory: [B, T, C, H, W] trajectory data
            ic: [B, C, H, W] initial condition
            normalize: Whether to apply normalization

        Returns:
            Features [B, D] where D = initial_dim + temporal_dim (+ summary_dim if enabled)
            Dimension is computed dynamically from extractors.
        """
        # Extract per-family raw features
        initial_raw = self.initial.extract_raw(trajectory, ic)      # [B, initial_dim]
        temporal_raw = self.temporal.extract_raw(trajectory, ic)    # [B, T, 63]

        # Encode per-family (INITIAL has no encoder, returns identity)
        initial_encoded = self.initial.encode(initial_raw)          # [B, initial_dim]
        temporal_encoded = self.temporal.encode(temporal_raw)       # [B, temporal_dim]

        # Normalize per-family (optional)
        if normalize:
            initial_norm = self.initial.normalize(initial_encoded)
            temporal_norm = self.temporal.normalize(temporal_encoded)
        else:
            initial_norm = initial_encoded
            temporal_norm = temporal_encoded

        # Handle summary family (optional for 2-family checkpoints)
        if self.summary is not None:
            summary_raw = self.summary.extract_raw(trajectory, ic)      # [B, raw_dim]
            summary_encoded = self.summary.encode(summary_raw)          # [B, summary_dim]

            if normalize:
                summary_norm = self.summary.normalize(summary_encoded)
            else:
                summary_norm = summary_encoded

            # Concatenate: initial + summary + temporal
            features = torch.cat([initial_norm, summary_norm, temporal_norm], dim=1)
        else:
            # 2-family: initial + temporal
            features = torch.cat([initial_norm, temporal_norm], dim=1)

        return features

    def compute_normalization_stats(self, features_list: List[torch.Tensor]):
        """Compute normalization stats from feature samples.

        This should be called during VQ-VAE training with UNNORMALIZED features
        extracted from the full training set.

        Dimensions are computed dynamically from extractors rather than
        using hardcoded offsets.

        Args:
            features_list: List of [B, D] feature tensors (unnormalized)
        """
        all_features = torch.cat(features_list, dim=0)  # [N, D]

        # Compute offsets dynamically from extractor output dims
        offset = 0

        # INITIAL: query dimension from extractor
        initial_dim = self.initial.feature_dim
        initial_features = all_features[:, offset:offset + initial_dim]
        self.initial.normalization_stats = (
            initial_features.mean(dim=0),
            initial_features.std(dim=0),
        )
        offset += initial_dim

        # SUMMARY (optional): query dimension from extractor
        if self.summary is not None:
            summary_dim = self.summary.feature_dim
            summary_features = all_features[:, offset:offset + summary_dim]
            self.summary.normalization_stats = (
                summary_features.mean(dim=0),
                summary_features.std(dim=0),
            )
            offset += summary_dim

        # TEMPORAL: query dimension from extractor
        temporal_dim = self.temporal.feature_dim
        temporal_features = all_features[:, offset:offset + temporal_dim]
        self.temporal.normalization_stats = (
            temporal_features.mean(dim=0),
            temporal_features.std(dim=0),
        )

    def save_to_checkpoint(self, checkpoint: dict):
        """Save normalization stats to checkpoint.

        Args:
            checkpoint: Checkpoint dictionary to update
        """
        stats = {
            'initial': self.initial.normalization_stats,
            'temporal': self.temporal.normalization_stats,
        }
        if self.summary is not None:
            stats['summary'] = self.summary.normalization_stats
        checkpoint['normalization_stats'] = stats

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cuda') -> 'UnifiedFeaturePipeline':
        """Load pipeline from VQ-VAE checkpoint.

        Loads:
        - Frozen encoders (MLPEncoder, TemporalCNNEncoder)
        - Normalization stats (per-family mean/std)

        Args:
            checkpoint_path: Path to VQ-VAE checkpoint
            device: Device for computation

        Returns:
            UnifiedFeaturePipeline instance with loaded encoders and stats
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load encoders from checkpoint (summary is optional for 2-family checkpoints)
        families = checkpoint['config']['families']

        summary_encoder = None
        if 'summary' in families:
            summary_encoder = cls._load_encoder(checkpoint, 'summary', device)

        temporal_encoder = cls._load_encoder(checkpoint, 'temporal', device)

        # Create extractors
        # Detect in_channels from initial config (default to 3 if not specified)
        initial_config = families.get('initial', {})
        initial_params = initial_config.get('encoder_params', {})
        in_channels = initial_params.get('in_channels', 3)

        initial = InitialFeatureExtractor(device=device, in_channels=in_channels)
        summary = SummaryFeatureExtractor(summary_encoder, device=device) if summary_encoder else None

        # Use pyramid extractor if encoder has per-level dims
        if hasattr(temporal_encoder, 'output_dims_per_level'):
            temporal = PyramidTemporalFeatureExtractor(temporal_encoder, device=device)
        else:
            temporal = TemporalFeatureExtractor(temporal_encoder, device=device)

        # Load normalization stats
        norm_stats = checkpoint.get('normalization_stats', {})
        initial.normalization_stats = cls._load_family_stats(norm_stats, 'initial', device)
        if summary:
            summary.normalization_stats = cls._load_family_stats(norm_stats, 'summary', device)
        temporal.normalization_stats = cls._load_family_stats(norm_stats, 'temporal', device)

        pipeline = cls(initial, summary, temporal)
        return pipeline

    @staticmethod
    def _load_encoder(checkpoint: dict, family_name: str, device: str) -> nn.Module:
        """Load encoder from checkpoint.

        Args:
            checkpoint: VQ-VAE checkpoint
            family_name: 'summary' or 'temporal'
            device: Device for encoder

        Returns:
            Loaded encoder module
        """
        families = checkpoint['config']['families']

        # families is a dict: {'initial': {...}, 'summary': {...}, 'temporal': {...}}
        if family_name not in families:
            raise ValueError(f"Family '{family_name}' not found in checkpoint config")

        family_config = families[family_name]

        encoder_type = family_config.get('encoder')
        encoder_config = family_config.get('encoder_params', {})

        # Add input_dim based on family
        if family_name == 'summary':
            input_dim = 360  # Raw SUMMARY feature dimension
        elif family_name == 'temporal':
            input_dim = 63   # Per-timestep TEMPORAL feature dimension
        else:
            input_dim = encoder_config.get('input_dim', 128)

        if encoder_type in ['mlp', 'MLPEncoder']:
            from spinlock.encoding.encoders.mlp import MLPEncoder
            encoder = MLPEncoder(input_dim=input_dim, **encoder_config)
        elif encoder_type in ['temporal_cnn', 'TemporalCNNEncoder']:
            from spinlock.encoding.encoders.temporal_cnn import TemporalCNNEncoder
            encoder = TemporalCNNEncoder(input_dim=input_dim, **encoder_config)
        elif encoder_type in ['pyramid_temporal', 'PyramidTemporalEncoder']:
            from spinlock.encoding.encoders.pyramid_temporal import PyramidTemporalEncoder
            # Handle backwards compatibility: variable_length -> variable_length_config
            if 'variable_length' in encoder_config:
                encoder_config = encoder_config.copy()
                encoder_config['variable_length_config'] = encoder_config.pop('variable_length')
            encoder = PyramidTemporalEncoder(input_dim=input_dim, **encoder_config)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Load encoder weights from checkpoint
        encoder_state_key = f'family_encoders.{family_name}.encoder'
        if encoder_state_key in checkpoint['model_state_dict']:
            encoder_state = {
                k.replace(f'{encoder_state_key}.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith(encoder_state_key)
            }
            encoder.load_state_dict(encoder_state)

        encoder = encoder.to(device)
        encoder.eval()  # Frozen for inference

        return encoder

    @staticmethod
    def _load_family_stats(
        norm_stats: dict,
        family_name: str,
        device: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Load normalization stats for a family.

        Args:
            norm_stats: Normalization stats dictionary from checkpoint
            family_name: 'initial', 'summary', or 'temporal'
            device: Device for tensors

        Returns:
            (mean, std) tuple or None if not found
        """
        if family_name not in norm_stats:
            return None

        stats = norm_stats[family_name]
        if stats is None:
            return None

        # Stats are stored as (mean_tensor, std_tensor)
        mean, std = stats

        # Convert to tensors if needed
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)

        mean = mean.to(device)
        std = std.to(device)

        return (mean, std)

    @classmethod
    def create_for_training(cls, families_config: dict, device: str = 'cuda') -> 'UnifiedFeaturePipeline':
        """Create pipeline for VQ-VAE training.

        This creates a pipeline with trainable encoders (not frozen).

        Args:
            families_config: Dict with family configs (initial, summary, temporal)
                Format: {family_name: {'encoder': name, 'encoder_params': {...}}}
            device: Device for computation

        Returns:
            UnifiedFeaturePipeline instance with fresh encoders
        """
        from spinlock.encoding.encoders import get_encoder

        # Get encoder configs
        summary_config = families_config.get('summary', {})
        temporal_config = families_config.get('temporal', {})

        # Create encoders
        summary_encoder_name = summary_config.get('encoder', 'MLPEncoder')
        summary_encoder_params = summary_config.get('encoder_params', {})
        summary_encoder = get_encoder(summary_encoder_name, input_dim=360, **summary_encoder_params)
        summary_encoder = summary_encoder.to(device)

        temporal_encoder_name = temporal_config.get('encoder', 'TemporalCNNEncoder')
        temporal_encoder_params = temporal_config.get('encoder_params', {})
        temporal_encoder = get_encoder(temporal_encoder_name, input_dim=63, **temporal_encoder_params)
        temporal_encoder = temporal_encoder.to(device)

        # Create extractors
        # Detect in_channels from initial config (default to 3 if not specified)
        initial_config = families_config.get('initial', {})
        initial_params = initial_config.get('encoder_params', {})
        in_channels = initial_params.get('in_channels', 3)

        initial = InitialFeatureExtractor(device=device, in_channels=in_channels)
        summary = SummaryFeatureExtractor(summary_encoder, device=device)

        # Use pyramid extractor if encoder has per-level dims
        if hasattr(temporal_encoder, 'output_dims_per_level'):
            temporal = PyramidTemporalFeatureExtractor(temporal_encoder, device=device)
        else:
            temporal = TemporalFeatureExtractor(temporal_encoder, device=device)

        return cls(initial, summary, temporal)
