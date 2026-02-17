"""PCA + striped-assignment feature grouper.

This is Stage 1 of the OPQ-based grouping pipeline (Ge et al. 2013).
PCA rotation sorts features by variance; striped assignment distributes
variance evenly across groups so every VQ codebook faces a genuinely
multi-dimensional quantization problem.

Reference:
    Jégou et al. (2011). "Product Quantization for Nearest Neighbor Search."
    Ge et al. (2013). "Optimized Product Quantization."
"""

import logging
from typing import List

import numpy as np
from sklearn.decomposition import PCA

from spinlock.encoding.normalization import LinearTransform

from .base import FeatureGrouper
from .models import GroupingConfig, GroupingResult

logger = logging.getLogger(__name__)


class PCAGrouper(FeatureGrouper):
    """Feature grouper using PCA rotation + striped (round-robin) group assignment.

    Pipeline:
        1. Fit full PCA on the temporal feature matrix [N, D] — no truncation,
           pure rotation into a variance-sorted orthonormal basis.
        2. Assign PC i → group (i % M), so each group receives:
               one high-variance PC,  one medium-variance PC,  several low-variance PCs.
           Every group has the same number of dimensions (D // M or D // M + 1).
        3. Store the PCA rotation as a LinearTransform so it can be applied at
           inference time before passing features to per-group encoders.

    This eliminates the correlation-collapse problem of Ward clustering:
    Ward groups *similar* features together → each codebook sees near-1D input
    → only 2-4 codes used. PCA + striped ensures each codebook faces a diverse,
    multi-dimensional subspace.

    Args:
        config: GroupingConfig with clustering.num_groups set explicitly.
                The gradient refinement and splitting pipelines are bypassed
                (not applicable to a rotation-based method).

    Raises:
        ValueError: If clustering.num_groups is not set.
    """

    def get_default_config(self) -> GroupingConfig:
        return GroupingConfig(method="pca_striped", skip_gradient_refinement=True)

    def validate_features(self, features: np.ndarray, feature_names: List[str]) -> None:
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features [N, D], got shape {features.shape}")
        N, D = features.shape
        if N < self.config.min_samples_required:
            raise ValueError(
                f"Need >= {self.config.min_samples_required} samples, got {N}"
            )
        if D < self.config.min_features_required:
            raise ValueError(
                f"Need >= {self.config.min_features_required} features, got {D}"
            )
        if len(feature_names) != D:
            raise ValueError(
                f"feature_names length {len(feature_names)} != feature dim {D}"
            )
        if self.config.clustering.num_groups is None:
            raise ValueError(
                "PCAGrouper requires clustering.num_groups to be set explicitly. "
                "There is no silhouette search for rotation-based methods."
            )

    def group_features(self, features: np.ndarray, feature_names: List[str]) -> GroupingResult:
        """Fit PCA and assign PCs to groups via striped (round-robin) assignment.

        Args:
            features: Feature matrix [N, D] (time-averaged temporal features).
            feature_names: Names for each of the D features.

        Returns:
            GroupingResult with linear_transform set (LinearTransform containing
            PCA mean and components). Feature names in the result reflect the
            rotated PCA space (pc_0, pc_1, ..., pc_{D-1}).
        """
        self.validate_features(features, feature_names)

        N, D = features.shape
        M = self.config.clustering.num_groups

        logger.info(
            f"PCAGrouper: fitting PCA on [{N}, {D}] features, "
            f"assigning to {M} groups via striped assignment."
        )

        # ── 1. Full PCA (no truncation: pure rotation + variance sorting) ──────
        seed = self.config.random_seed
        pca = PCA(n_components=D, random_state=seed, svd_solver="full")
        pca.fit(features)

        transform = LinearTransform(
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),  # [D, D], rows = PCs
        )

        explained = pca.explained_variance_ratio_
        logger.info(
            f"PCA: top-5 explained variance = {explained[:5].round(4).tolist()}, "
            f"cumulative={explained.sum():.4f}"
        )

        # ── 2. Striped assignment: PC i → group (i % M) ──────────────────────
        # PC names reflect the rotated space; the per-group MLP will learn to
        # encode these into latent vectors for VQ.
        pc_names = [f"pc_{i}" for i in range(D)]
        group_dict: dict[str, list[int]] = {f"group_{g}": [] for g in range(M)}
        for pc_idx in range(D):
            group_dict[f"group_{pc_idx % M}"].append(pc_idx)

        # Log group sizes for diagnostic clarity
        sizes = [len(v) for v in group_dict.values()]
        logger.info(
            f"Group sizes (min={min(sizes)}, max={max(sizes)}, "
            f"mean={np.mean(sizes):.1f}): {sizes[:10]}{'...' if M > 10 else ''}"
        )

        # ── 3. Build result with rotation attached ────────────────────────────
        result = self._to_result(group_dict, pc_names)
        result.linear_transform = transform

        logger.info(
            f"PCAGrouper done: {result.num_groups} groups, "
            f"linear_transform stored (shape {transform.components.shape})."
        )
        return result
