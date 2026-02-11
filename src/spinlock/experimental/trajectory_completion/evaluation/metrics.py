"""Compute comprehensive completion metrics."""

import torch
import numpy as np
from typing import Dict
from scipy.stats import pearsonr


def compute_completion_metrics(
    features_pred: torch.Tensor,
    features_true: torch.Tensor,
    tokens_pred: torch.Tensor,
    tokens_true: torch.Tensor,
    mask_target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute comprehensive completion metrics.

    Args:
        features_pred: [batch, D] Decoded features from completed tokens
        features_true: [batch, D] Ground truth features
        tokens_pred: [batch, N×L] Predicted tokens
        tokens_true: [batch, N×L] Ground truth tokens
        mask_target: [batch, N×L] Mask indicating predicted positions

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Token-level accuracy (on masked positions only)
    correct_tokens = (tokens_pred == tokens_true) & mask_target
    metrics['token_accuracy'] = correct_tokens.float().mean().item()

    # Per-level token accuracy
    num_levels = 3
    for level in range(num_levels):
        level_mask = mask_target[:, level::num_levels]
        level_correct = (tokens_pred[:, level::num_levels] == tokens_true[:, level::num_levels]) & level_mask
        if level_mask.any():
            metrics[f'token_accuracy_L{level}'] = level_correct.float().sum().item() / level_mask.sum().item()
        else:
            metrics[f'token_accuracy_L{level}'] = 0.0

    # Feature reconstruction error
    metrics['mse'] = torch.nn.functional.mse_loss(features_pred, features_true).item()
    metrics['mae'] = torch.nn.functional.l1_loss(features_pred, features_true).item()

    # Relative error
    feature_norm = torch.norm(features_true, dim=1).mean()
    error_norm = torch.norm(features_pred - features_true, dim=1).mean()
    metrics['relative_error'] = (error_norm / feature_norm).item()

    # Per-dimension correlation
    correlations = []
    for dim in range(features_pred.shape[1]):
        pred_dim = features_pred[:, dim].cpu().numpy()
        true_dim = features_true[:, dim].cpu().numpy()
        if pred_dim.std() > 0 and true_dim.std() > 0:
            corr, _ = pearsonr(pred_dim, true_dim)
            correlations.append(corr)

    if correlations:
        metrics['mean_correlation'] = np.mean(correlations)
        metrics['median_correlation'] = np.median(correlations)
    else:
        metrics['mean_correlation'] = 0.0
        metrics['median_correlation'] = 0.0

    return metrics
