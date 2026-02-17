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
from sklearn.preprocessing import StandardScaler

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
        """Dispatch to pca_striped or pca_raw based on config.method.

        Args:
            features: Feature matrix [N, D] (time-averaged temporal features).
            feature_names: Names for each of the D features.

        Returns:
            GroupingResult. For pca_striped: linear_transform is set (PCA rotation
            must be applied at inference). For pca_raw: linear_transform is None
            (no rotation at inference — indices are raw feature positions).
        """
        if self.config.method == "pca_raw":
            return self._group_features_raw(features, feature_names)
        return self._group_features_striped(features, feature_names)

    def _group_features_raw(
        self, features: np.ndarray, feature_names: List[str]
    ) -> GroupingResult:
        """Use PCA loadings to assign each RAW feature to a group by dominant PC.

        PCA is fit on cat([standardised_mean, std_proxy]) of the input features
        to capture both level and dynamical amplitude in the loading structure.
        Each raw feature j is then assigned to group ``dominant_pc[j] % M`` where
        dominant_pc[j] = argmax_k ( |C[k,j]| + |C[k,j+D]| ).

        Outcome: GroupingResult with raw feature indices (0..D-1), linear_transform=None.
        The model slices temporal[:, :, raw_indices] directly — no rotation at inference.

        Args:
            features: Feature matrix [N, D] (time-averaged temporal features).
            feature_names: Semantic names for each of the D raw features.

        Returns:
            GroupingResult with linear_transform=None and raw feature indices.
        """
        self.validate_features(features, feature_names)

        N, D = features.shape
        M = self.config.clustering.num_groups
        seed = self.config.random_seed

        logger.info(
            f"PCAGrouper (pca_raw): fitting PCA on [{N}, {2 * D}] agg features, "
            f"assigning {D} raw features to {M} groups via dominant-PC assignment."
        )

        # Standardise then form a mean+std-proxy concatenation for PCA fitting.
        # std_proxy captures per-feature variance across samples (amplitude signal).
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)               # [N, D]
        std_proxy = np.abs(features - features.mean(axis=0))        # [N, D]
        agg = np.concatenate([features_std, std_proxy], axis=1)     # [N, 2D]

        agg_scaler = StandardScaler()
        agg = agg_scaler.fit_transform(agg)                         # [N, 2D]

        n_components = min(2 * D, N - 1)
        pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
        pca.fit(agg)
        components = pca.components_                                 # [n_pcs, 2D]

        explained = pca.explained_variance_ratio_
        logger.info(
            f"PCA (pca_raw): top-5 explained variance = {explained[:5].round(4).tolist()}, "
            f"cumulative = {explained.sum():.4f}"
        )

        # For each raw feature j: dominant_pc = argmax_k ( |C[k,j]| + |C[k,j+D]| )
        # This sums the absolute loadings of the mean and std contributions for feature j.
        loadings = np.abs(components[:, :D]) + np.abs(components[:, D:])  # [n_pcs, D]
        dominant_pc = np.argmax(loadings, axis=0)                          # [D]

        # Assign feature j → group (dominant_pc[j] % M)
        # Striped modulo distributes high-variance PCs (low k) across different groups.
        group_dict: dict[str, list[int]] = {f"group_{g}": [] for g in range(M)}
        for j, pc_k in enumerate(dominant_pc):
            group_dict[f"group_{int(pc_k) % M}"].append(j)

        sizes = [len(v) for v in group_dict.values()]
        logger.info(
            f"pca_raw group sizes (min={min(sizes)}, max={max(sizes)}, "
            f"mean={np.mean(sizes):.1f}): {sizes[:10]}{'...' if M > 10 else ''}"
        )

        # Build result with RAW feature names; no rotation transform stored.
        result = self._to_result(group_dict, feature_names)
        result.linear_transform = None  # explicit: no rotation at inference

        logger.info(
            f"PCAGrouper (pca_raw) done: {result.num_groups} groups, "
            f"linear_transform=None (raw feature slicing at inference)."
        )
        return result

    def _group_features_striped(
        self, features: np.ndarray, feature_names: List[str]
    ) -> GroupingResult:
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

        # ── 1. Standardize then PCA (no truncation: pure rotation + variance sort) ──
        # Standardize first (zero mean, unit variance per feature) so that features
        # with large absolute variance don't dominate PC0. Without this, a single
        # high-scale feature can claim 99.98% of explained variance, reducing
        # effective rank to 1 for all 30 groups.
        #
        # The scale is folded into LinearTransform.components so that
        #   apply(x) = (x - mean) @ components.T
        # transparently implements the full standardize-then-rotate pipeline:
        #   x_rot = ((x - mu_std) / sigma_std) @ R.T
        #         = (x - mu_std) @ (R * (1/sigma_std)[None, :]).T
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)  # [N, D], zero-mean, unit-var

        seed = self.config.random_seed
        pca = PCA(n_components=D, random_state=seed, svd_solver="full")
        pca.fit(features_std)

        # Fold scaler into components: apply() = (x - mean) @ components.T
        # PCA.mean_ ≈ 0 (features_std is already centered), so we use scaler.mean_.
        inv_scale = (1.0 / scaler.scale_).astype(np.float32)   # [D]
        # pca.components_ shape: [D, D], rows = principal components
        components_folded = pca.components_.astype(np.float32) * inv_scale[None, :]

        transform = LinearTransform(
            mean=scaler.mean_.astype(np.float32),
            components=components_folded,  # [D, D], rows = PCs (with scale folded in)
        )

        explained = pca.explained_variance_ratio_
        pcs_90 = int(np.searchsorted(np.cumsum(explained), 0.90)) + 1
        logger.info(
            f"PCA (standardized): top-5 explained variance = {explained[:5].round(4).tolist()}, "
            f"PCs for 90%={pcs_90}, cumulative={explained.sum():.4f}"
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
