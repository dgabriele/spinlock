"""Dataset for trajectory completion experiment."""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional

from experiments.common.data.trajectory_loader import TrajectoryDataLoader
from experiments.common.models.trained_vqvae import TrainedVQVAE
from experiments.trajectory_completion.data.masking import TemporalMaskGenerator


class TrajectoryCompletionDataset(Dataset):
    """
    Dataset for trajectory completion experiment.

    Loads features from dataset, tokenizes with VQ-VAE, generates masked examples.
    """

    def __init__(
        self,
        dataset_path: Path,
        vqvae: TrainedVQVAE,
        mask_generator: TemporalMaskGenerator,
        indices: Optional[np.ndarray] = None
    ):
        self.loader = TrajectoryDataLoader(dataset_path)
        self.vqvae = vqvae
        self.mask_generator = mask_generator

        # Get feature families from VQ-VAE training config
        feature_families = self.vqvae.get_feature_families()
        print(f"Using feature families: {feature_families}")

        # Load features (only those used by VQ-VAE)
        print("Loading features...")
        features_dict = self.loader.load_features(
            feature_families=feature_families,
            indices=indices
        )

        # Concatenate features to match VQ-VAE input format
        self.features = self._concatenate_features(features_dict, feature_families)

        print(f"Tokenizing {len(self.features)} samples...")
        self.tokens = self.vqvae.encode(self.features)

        print(f"Dataset initialized: {len(self)} samples")

    def _concatenate_features(
        self,
        features_dict: Dict[str, torch.Tensor],
        feature_families: list
    ) -> torch.Tensor:
        """
        Concatenate features to match VQ-VAE input format.

        Must match exact format used during VQ-VAE training.
        """
        feature_parts = []

        # Add features in order (order matters!)
        if 'initial' in feature_families and 'initial' in features_dict:
            feature_parts.append(features_dict['initial'])

        if 'temporal' in feature_families and 'temporal' in features_dict:
            # Temporal: [B, T, D] → aggregate to [B, D]
            temporal = features_dict['temporal']
            temporal_agg = temporal.mean(dim=1)  # Average over time
            feature_parts.append(temporal_agg)

        if not feature_parts:
            raise ValueError("No valid features found")

        # Concatenate all feature families
        features = torch.cat(feature_parts, dim=1)
        return features

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get completion example.

        Returns:
            {
                'tokens_full': [N×L] Full token sequence (ground truth)
                'tokens_observed': [N×L] Partially observed tokens
                'mask_observed': [N×L] bool mask (True = observed)
                'mask_target': [N×L] bool mask (True = predict)
                'features_full': [D] Full features (ground truth)
            }
        """
        tokens_full = self.tokens[idx]
        features_full = self.features[idx]

        # Generate mask
        num_categories = len(tokens_full) // 3  # 3 levels per category
        mask_observed, mask_target = self.mask_generator.generate_token_mask(
            num_categories=num_categories,
            num_levels=3
        )

        # Create observed tokens (masked tokens set to padding index 0)
        tokens_observed = tokens_full.clone()
        tokens_observed[mask_target] = 0  # Mask with padding

        return {
            'tokens_full': tokens_full,
            'tokens_observed': tokens_observed,
            'mask_observed': mask_observed,
            'mask_target': mask_target,
            'features_full': features_full
        }
