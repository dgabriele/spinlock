"""Loss functions for learnable category assignments."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def soft_orthogonality_loss(
    features: torch.Tensor,
    soft_assignments: torch.Tensor
) -> torch.Tensor:
    """Minimize inter-category correlation.

    Args:
        features: [B, D] feature matrix (batch)
        soft_assignments: [D, K] soft assignment probabilities

    Returns:
        loss: Scalar orthogonality loss
    """
    # Compute category-weighted features
    # For each category k, weight features by assignment probs: features * soft_assign[:, k]
    K = soft_assignments.shape[1]
    D = soft_assignments.shape[0]

    # Expand for batch processing: [B, D] @ [D, K] -> [B, K]
    # This gives us the effective "category activations" for each sample
    category_activations = features @ soft_assignments  # [B, K]

    # Compute correlation between category activations across the batch
    correlations = []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            act_k1 = category_activations[:, k1]  # [B]
            act_k2 = category_activations[:, k2]  # [B]

            # Normalize
            act_k1_norm = F.normalize(act_k1.unsqueeze(0), dim=1).squeeze(0)
            act_k2_norm = F.normalize(act_k2.unsqueeze(0), dim=1).squeeze(0)

            # Correlation
            corr = (act_k1_norm * act_k2_norm).sum()
            correlations.append(corr.abs())

    if correlations:
        return torch.stack(correlations).mean()
    else:
        return torch.tensor(0.0, device=features.device)


def soft_balance_loss(soft_assignments: torch.Tensor) -> torch.Tensor:
    """Variance of category sizes (lower = more balanced).

    Args:
        soft_assignments: [D, K] soft assignment probabilities

    Returns:
        loss: Scalar balance loss (variance of category sizes)
    """
    # Category sizes are sum of assignment probabilities
    category_sizes = soft_assignments.sum(dim=0)  # [K]

    # Variance of sizes
    mean_size = category_sizes.mean()
    variance = ((category_sizes - mean_size) ** 2).mean()

    return variance


def family_constraint_loss(
    soft_assignments: torch.Tensor,
    feature_families: Dict[str, list],
    category_offsets: Dict[str, int],
    categories_per_family: Dict[str, int]
) -> torch.Tensor:
    """Penalty for cross-family assignments.

    Args:
        soft_assignments: [D, K] soft assignment probabilities
        feature_families: {family_name: [feature_indices]}
        category_offsets: {family_name: start_category_idx}
        categories_per_family: {family_name: num_categories}

    Returns:
        loss: Scalar penalty for cross-family assignments
    """
    total_loss = 0.0
    device = soft_assignments.device

    for family_name, feature_indices in feature_families.items():
        cat_start = category_offsets[family_name]
        cat_end = cat_start + categories_per_family[family_name]

        # Get assignments for this family's features
        family_assignments = soft_assignments[feature_indices]  # [num_feats, K]

        # Penalty is sum of probabilities assigned to wrong categories
        # (i.e., categories outside [cat_start, cat_end))
        wrong_cats_mask = torch.ones(soft_assignments.shape[1], device=device)
        wrong_cats_mask[cat_start:cat_end] = 0.0

        penalty = (family_assignments * wrong_cats_mask.unsqueeze(0)).sum()
        total_loss += penalty

    return total_loss / len(feature_families)
