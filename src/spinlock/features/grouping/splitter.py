"""Recursive mega-group splitting."""

from typing import Dict, List
import numpy as np

from .models import SplittingParams, ClusteringParams


class RecursiveSplitter:
    """
    Recursive splitting of oversized groups.

    Breaks large groups into smaller sub-groups via hierarchical
    clustering with depth limits.
    """

    def __init__(self, config: SplittingParams):
        self.config = config

    def split(
        self,
        groups: Dict[str, List[int]],
        features: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, List[int]]:
        """
        Recursively split oversized groups.

        Args:
            groups: Initial groups
            features: Normalized features [N, D]
            feature_names: Feature names

        Returns:
            Groups with mega-groups split
        """
        if not self.config.enabled:
            return groups

        result = {}

        for group_name, indices in groups.items():
            if len(indices) <= self.config.max_group_size:
                # Small enough, keep as-is
                result[group_name] = indices
            else:
                # Too large, split recursively
                sub_groups = self._recursive_split(
                    group_name=group_name,
                    indices=indices,
                    features=features,
                    feature_names=feature_names,
                    current_depth=0,
                )
                result.update(sub_groups)

        return result

    def _recursive_split(
        self,
        group_name: str,
        indices: List[int],
        features: np.ndarray,
        feature_names: List[str],
        current_depth: int,
    ) -> Dict[str, List[int]]:
        """
        Recursively split a single group.

        Returns:
            Dict of sub-groups
        """
        # Base case 1: Small enough
        if len(indices) <= self.config.max_group_size:
            return {group_name: indices}

        # Base case 2: Max depth reached
        if current_depth >= self.config.max_recursion_depth:
            print(f"Warning: Group '{group_name}' still oversized ({len(indices)} features) "
                  f"but max recursion depth reached")
            return {group_name: indices}

        # Base case 3: Too few to split
        if len(indices) < 2 * self.config.min_features_per_group:
            return {group_name: indices}

        # Recursive case: Split this group
        sub_features = features[:, indices]
        sub_feature_names = [feature_names[i] for i in indices]

        # Cluster the subset
        from .clustering import ClusteringEngine
        clustering_engine = ClusteringEngine(ClusteringParams(
            min_groups=2,
            max_groups=min(8, len(indices) // self.config.min_features_per_group),
        ))

        try:
            sub_groups = clustering_engine.cluster(sub_features, sub_feature_names)
        except Exception as e:
            print(f"Warning: Failed to split group '{group_name}': {e}")
            return {group_name: indices}

        # Map back to global indices and recurse
        result = {}
        for sub_idx, (sub_name, local_indices) in enumerate(sub_groups.items()):
            global_indices = [indices[i] for i in local_indices]
            split_name = f"{group_name}_split_{sub_idx}"

            # Recurse on this sub-group
            sub_result = self._recursive_split(
                group_name=split_name,
                indices=global_indices,
                features=features,
                feature_names=feature_names,
                current_depth=current_depth + 1,
            )
            result.update(sub_result)

        return result
