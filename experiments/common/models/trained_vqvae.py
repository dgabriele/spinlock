"""Interface for using trained VQ-VAE models in experiments."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List


class TrainedVQVAE:
    """Interface for using trained VQ-VAE models in experiments."""

    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self.normalization_stats = None
        self.feature_config = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load VQ-VAE checkpoint and extract metadata."""
        from spinlock.encoding.models.categorical_vqvae import CategoricalVQVAE

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Load model configuration
        self.config = checkpoint['model_config']
        self.model = CategoricalVQVAE(**self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats
        if 'normalization_stats' in checkpoint:
            self.normalization_stats = checkpoint['normalization_stats']

        # Load feature configuration (CRITICAL: must match training)
        if 'config' in checkpoint:
            self.feature_config = checkpoint['config'].get('features', {})

    def get_feature_families(self) -> List[str]:
        """Get list of feature families used during training."""
        if self.feature_config is None:
            # Default: initial and temporal only (summary deprecated)
            return ['initial', 'temporal']

        families = []
        if self.feature_config.get('initial', {}).get('enabled', True):
            families.append('initial')
        if self.feature_config.get('temporal', {}).get('enabled', True):
            families.append('temporal')
        # Note: summary features are deprecated
        return families

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features to tokens.

        Args:
            features: [batch, feature_dim] tensor

        Returns:
            tokens: [batch, N×L] integer token indices
        """
        with torch.no_grad():
            tokens = self.model.get_tokens(features.to(self.device))
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to features.

        Args:
            tokens: [batch, N×L] integer token indices

        Returns:
            features: [batch, feature_dim] reconstructed features
        """
        with torch.no_grad():
            features_recon = self.model.decode_from_tokens(tokens.to(self.device))
        return features_recon

    def get_category_tokens(self, tokens: torch.Tensor, category: str) -> torch.Tensor:
        """Extract tokens for specific category."""
        with torch.no_grad():
            category_tokens = self.model.get_category_tokens(tokens, category)
        return category_tokens

    @property
    def num_tokens(self) -> int:
        """Total number of tokens (N×L)."""
        return sum(len(vq.embedding.weight) for vq in self.model.vq_layers)

    @property
    def num_categories(self) -> int:
        """Number of feature categories (N)."""
        return len(self.model.vq_layers) // 3  # 3 levels per category

    @property
    def codebook_sizes(self) -> List[int]:
        """Get codebook size for each VQ layer."""
        return [len(vq.embedding.weight) for vq in self.model.vq_layers]
