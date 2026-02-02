"""Learnable category assignment matrices for VQ-VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class LearnableAssignmentConfig:
    """Configuration for learnable category assignments."""

    init_method: str = "clustering"
    temperature_start: float = 1.0
    temperature_end: float = 0.1
    temperature_schedule: str = "linear"
    orthogonality_weight: float = 0.1
    balance_weight: float = 0.05
    family_constraint_weight: float = 1.0
    assignment_lr: float = 0.001
    freeze_after_epochs: Optional[int] = None


class SoftAssignmentMatrix(nn.Module):
    """Learnable [D, K] assignment probabilities with Gumbel-Softmax."""

    def __init__(
        self,
        num_features: int,
        num_categories: int,
        init_logits: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_features = num_features
        self.num_categories = num_categories

        # Learnable logits [D, K]
        if init_logits is not None:
            assert init_logits.shape == (num_features, num_categories)
            self.logits = nn.Parameter(init_logits.clone())
        else:
            self.logits = nn.Parameter(torch.randn(num_features, num_categories) * 0.1)

    def forward(self, temperature: float = 1.0) -> torch.Tensor:
        """Returns soft assignment probabilities [D, K]."""
        return F.softmax(self.logits / temperature, dim=1)

    def get_hard_assignments(self) -> torch.Tensor:
        """Returns hard assignments [D] via argmax."""
        return torch.argmax(self.logits, dim=1)


class PerFamilyAssignmentMatrix(nn.Module):
    """Per-family assignment matrix with block-diagonal constraints."""

    def __init__(
        self,
        feature_families: Dict[str, List[int]],
        categories_per_family: Dict[str, int],
        init_logits: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.feature_families = feature_families
        self.categories_per_family = categories_per_family

        # Calculate total features and categories
        self.num_features = sum(len(features) for features in feature_families.values())
        self.num_categories = sum(categories_per_family.values())

        # Create per-family assignment matrices
        self.family_matrices = nn.ModuleDict()
        for family_name, feature_indices in feature_families.items():
            num_feats = len(feature_indices)
            num_cats = categories_per_family[family_name]

            if init_logits and family_name in init_logits:
                init = init_logits[family_name]
            else:
                init = None

            self.family_matrices[family_name] = SoftAssignmentMatrix(
                num_feats, num_cats, init
            )

        # Build feature index mapping
        self.feature_to_family = {}
        self.feature_to_local_idx = {}
        for family_name, feature_indices in feature_families.items():
            for local_idx, global_idx in enumerate(feature_indices):
                self.feature_to_family[global_idx] = family_name
                self.feature_to_local_idx[global_idx] = local_idx

        # Build category offset mapping
        self.category_offsets = {}
        offset = 0
        for family_name in sorted(feature_families.keys()):
            self.category_offsets[family_name] = offset
            offset += categories_per_family[family_name]

    def forward(self, temperature: float = 1.0) -> torch.Tensor:
        """Returns block-diagonal soft assignments [D, K]."""
        device = next(self.parameters()).device
        assignments = torch.zeros(self.num_features, self.num_categories, device=device)

        # Fill in block-diagonal structure
        for family_name, feature_indices in self.feature_families.items():
            family_assign = self.family_matrices[family_name](temperature)
            cat_offset = self.category_offsets[family_name]
            num_cats = self.categories_per_family[family_name]

            for local_idx, global_idx in enumerate(feature_indices):
                assignments[global_idx, cat_offset:cat_offset + num_cats] = family_assign[local_idx]

        return assignments

    def get_hard_assignments(self) -> Dict[str, Dict[int, int]]:
        """Returns hard assignments as {family: {feature_idx: category_idx}}."""
        assignments = {}
        for family_name in self.feature_families.keys():
            hard_assign = self.family_matrices[family_name].get_hard_assignments()
            cat_offset = self.category_offsets[family_name]

            assignments[family_name] = {
                global_idx: cat_offset + hard_assign[local_idx].item()
                for local_idx, global_idx in enumerate(self.feature_families[family_name])
            }

        return assignments


def initialize_from_clustering(
    group_indices: Dict[str, List[int]],
    num_features: int,
    per_family: bool,
    feature_families: Optional[Dict[str, List[int]]] = None
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Convert existing group assignments to high-confidence logits.

    NOTE: Clustering has already been done at this point (group_indices exist).
    This just converts those assignments to initialization logits.

    Args:
        group_indices: Category assignments {category_name: [feature_indices]}
        num_features: Total number of features
        per_family: Whether using per-family mode
        feature_families: Optional family groupings

    Returns:
        init_logits: Initialization logits (high confidence)
        categories_per_family: Category counts per family (if per_family)
    """
    # Create mapping from feature index to category index
    feature_to_category = {}
    category_names = sorted(group_indices.keys())
    for cat_idx, cat_name in enumerate(category_names):
        for feat_idx in group_indices[cat_name]:
            feature_to_category[feat_idx] = cat_idx

    num_categories = len(category_names)

    if per_family:
        # Per-family initialization
        init_logits = {}
        categories_per_family = {}

        for family_name, family_feat_indices in feature_families.items():
            # Find categories for this family
            family_categories = set()
            for feat_idx in family_feat_indices:
                if feat_idx in feature_to_category:
                    family_categories.add(feature_to_category[feat_idx])

            num_family_cats = len(family_categories)
            categories_per_family[family_name] = num_family_cats

            # Create category mapping for this family
            family_cat_list = sorted(family_categories)
            family_cat_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(family_cat_list)}

            # Create high-confidence logits
            family_logits = torch.full((len(family_feat_indices), num_family_cats), -5.0)
            for local_idx, global_feat_idx in enumerate(family_feat_indices):
                if global_feat_idx in feature_to_category:
                    global_cat_idx = feature_to_category[global_feat_idx]
                    local_cat_idx = family_cat_mapping[global_cat_idx]
                    family_logits[local_idx, local_cat_idx] = 5.0

            init_logits[family_name] = family_logits

        return init_logits, categories_per_family
    else:
        # Global initialization
        init_logits = torch.full((num_features, num_categories), -5.0)

        for feat_idx, cat_idx in feature_to_category.items():
            init_logits[feat_idx, cat_idx] = 5.0

        return init_logits, {"global": num_categories}
