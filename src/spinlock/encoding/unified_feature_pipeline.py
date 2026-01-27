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
    """INITIAL family: 14D manual features from IC."""

    def __init__(self, device: str = 'cuda'):
        """Initialize INITIAL feature extractor.

        Args:
            device: Device for computation
        """
        super().__init__('initial', encoder=None)  # No encoding for manual features
        from spinlock.features.initial.manual_extractors import InitialManualExtractor
        self.raw_extractor = InitialManualExtractor(device=device)
        self.device = device

    def extract_raw(self, trajectory: torch.Tensor, ic: torch.Tensor) -> torch.Tensor:
        """Extract 14D manual features from IC.

        Args:
            trajectory: [B, T, C, H, W] (unused, but kept for interface consistency)
            ic: [B, C, H, W] initial condition

        Returns:
            [B, 14] manual features
        """
        # InitialManualExtractor expects [B, M, C, H, W], so add dimension
        ic_with_m = ic.unsqueeze(1)  # [B, 1, C, H, W]
        features = self.raw_extractor.extract_all(ic_with_m)
        return features.squeeze(1)  # [B, 14]


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


class UnifiedFeaturePipeline(nn.Module):
    """Unified feature extraction pipeline for VQ-VAE training and inference.

    This class ensures VQ-VAE training and meta-operator training use
    IDENTICAL feature extraction, encoding, and normalization.

    Features are organized into 3 families:
    - INITIAL: 14D manual features from initial condition
    - SUMMARY: 360D→128D encoded aggregated features from trajectory
    - TEMPORAL: 63D×T→128D encoded per-timestep features from trajectory

    Total output: 270D = 14 + 128 + 128
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
        """Total feature dimension: 14 + 128 + 128 = 270D (3-family) or 14 + 128 = 142D (2-family)."""
        if self.summary is not None:
            return 270  # 3-family: initial + summary + temporal
        else:
            return 142  # 2-family: initial + temporal

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
            Features [B, 270D] = 14D + 128D + 128D (3-family)
            OR [B, 142D] = 14D + 128D (2-family, no summary)
        """
        # Extract per-family raw features
        initial_raw = self.initial.extract_raw(trajectory, ic)      # [B, 14]
        temporal_raw = self.temporal.extract_raw(trajectory, ic)    # [B, T, 63]

        # Encode per-family (INITIAL has no encoder, returns identity)
        initial_encoded = self.initial.encode(initial_raw)          # [B, 14]
        temporal_encoded = self.temporal.encode(temporal_raw)       # [B, 128]

        # Normalize per-family (optional)
        if normalize:
            initial_norm = self.initial.normalize(initial_encoded)
            temporal_norm = self.temporal.normalize(temporal_encoded)
        else:
            initial_norm = initial_encoded
            temporal_norm = temporal_encoded

        # Handle summary family (optional for 2-family checkpoints)
        if self.summary is not None:
            summary_raw = self.summary.extract_raw(trajectory, ic)      # [B, 360]
            summary_encoded = self.summary.encode(summary_raw)          # [B, 128]

            if normalize:
                summary_norm = self.summary.normalize(summary_encoded)
            else:
                summary_norm = summary_encoded

            # Concatenate: [B, 14] + [B, 128] + [B, 128] = [B, 270]
            features = torch.cat([initial_norm, summary_norm, temporal_norm], dim=1)
        else:
            # 2-family: [B, 14] + [B, 128] = [B, 142]
            features = torch.cat([initial_norm, temporal_norm], dim=1)

        return features

    def compute_normalization_stats(self, features_list: List[torch.Tensor]):
        """Compute normalization stats from feature samples.

        This should be called during VQ-VAE training with UNNORMALIZED features
        extracted from the full training set.

        Args:
            features_list: List of [B, 270] feature tensors (unnormalized)
        """
        # Concatenate all feature samples
        all_features = torch.cat(features_list, dim=0)  # [N, 270]

        # Split into per-family features
        initial_features = all_features[:, :14]          # [N, 14]
        summary_features = all_features[:, 14:142]       # [N, 128]
        temporal_features = all_features[:, 142:270]     # [N, 128]

        # Compute per-family statistics
        self.initial.normalization_stats = (
            initial_features.mean(dim=0),   # [14]
            initial_features.std(dim=0)     # [14]
        )
        self.summary.normalization_stats = (
            summary_features.mean(dim=0),   # [128]
            summary_features.std(dim=0)     # [128]
        )
        self.temporal.normalization_stats = (
            temporal_features.mean(dim=0),  # [128]
            temporal_features.std(dim=0)    # [128]
        )

    def save_to_checkpoint(self, checkpoint: dict):
        """Save normalization stats to checkpoint.

        Args:
            checkpoint: Checkpoint dictionary to update
        """
        checkpoint['normalization_stats'] = {
            'initial': self.initial.normalization_stats,
            'summary': self.summary.normalization_stats,
            'temporal': self.temporal.normalization_stats,
        }

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
        initial = InitialFeatureExtractor(device=device)
        summary = SummaryFeatureExtractor(summary_encoder, device=device) if summary_encoder else None
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
        initial = InitialFeatureExtractor(device=device)
        summary = SummaryFeatureExtractor(summary_encoder, device=device)
        temporal = TemporalFeatureExtractor(temporal_encoder, device=device)

        return cls(initial, summary, temporal)
