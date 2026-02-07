"""Metrics computation for MNO-VQ-VAE validation."""

from typing import Dict, Union
import torch
import numpy as np


class ValidationMetrics:
    """Compute validation metrics for MNO-VQ-VAE alignment."""

    @staticmethod
    def compute(
        features_original: torch.Tensor,
        features_reconstructed: torch.Tensor,
        tokens: Union[torch.Tensor, Dict[str, torch.Tensor]],
        cno_baseline_error: float,
        config
    ) -> Dict[str, float]:
        """
        Compute comprehensive validation metrics.

        Args:
            features_original: [N, D] Original MNO features
            features_reconstructed: [N, D] Reconstructed features from tokens
            tokens: Token indices - either [N, T] tensor (V1) or dict of tensors (V2)
            cno_baseline_error: CNO reconstruction error from VQ-VAE training
            config: ValidationConfig

        Returns:
            Dictionary of metrics including:
            - reconstruction_mse: Mean squared error
            - reconstruction_mae: Mean absolute error
            - relative_l2: Relative L2 error
            - reconstruction_ratio: MNO/CNO error ratio (primary metric)
            - mean_correlation: Average per-dimension correlation
            - median_correlation: Median per-dimension correlation
            - unique_tokens: Number of unique tokens used
            - token_entropy: Shannon entropy of token distribution
        """
        metrics = {}

        # Reconstruction error (MSE)
        metrics['reconstruction_mse'] = torch.nn.functional.mse_loss(
            features_reconstructed, features_original
        ).item()

        # Reconstruction error (MAE)
        metrics['reconstruction_mae'] = torch.nn.functional.l1_loss(
            features_reconstructed, features_original
        ).item()

        # Relative L2 error
        error_norm = torch.norm(features_reconstructed - features_original, dim=1).mean()
        signal_norm = torch.norm(features_original, dim=1).mean()
        metrics['relative_l2'] = (error_norm / signal_norm).item()

        # Reconstruction ratio (primary metric)
        metrics['reconstruction_ratio'] = metrics['reconstruction_mse'] / cno_baseline_error

        # Per-dimension correlation
        correlations = []
        for dim in range(features_original.shape[1]):
            corr = np.corrcoef(
                features_original[:, dim].cpu().numpy(),
                features_reconstructed[:, dim].cpu().numpy()
            )[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        metrics['mean_correlation'] = np.mean(correlations) if correlations else 0.0
        metrics['median_correlation'] = np.median(correlations) if correlations else 0.0

        # Token statistics (handle both V1 tensor and V2 dict formats)
        if isinstance(tokens, dict):
            # V2 format: concatenate all token indices
            all_tokens = torch.cat([t.flatten() for t in tokens.values()])
            metrics['unique_tokens'] = len(torch.unique(all_tokens))
            metrics['token_entropy'] = ValidationMetrics._compute_token_entropy(all_tokens)
        else:
            # V1 format: single tensor
            metrics['unique_tokens'] = len(torch.unique(tokens))
            metrics['token_entropy'] = ValidationMetrics._compute_token_entropy(tokens)

        # Per-level token accuracy (placeholder for future ground truth comparison)
        metrics['per_level_token_accuracy'] = {}

        return metrics

    @staticmethod
    def _compute_token_entropy(tokens: torch.Tensor) -> float:
        """
        Compute Shannon entropy of token distribution.

        Args:
            tokens: Token indices (already flattened or [N, T] to be flattened)

        Returns:
            Shannon entropy in bits
        """
        # Flatten if needed
        if tokens.ndim > 1:
            tokens = tokens.flatten()

        # Count occurrences
        unique, counts = torch.unique(tokens, return_counts=True)
        probs = counts.float() / counts.sum()

        # Shannon entropy
        entropy = -(probs * torch.log2(probs + 1e-10)).sum()

        return entropy.item()
