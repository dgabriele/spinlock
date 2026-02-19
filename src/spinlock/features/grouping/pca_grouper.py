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
        config: GroupingConfig with clustering.num_groups (fixed K) or
                clustering.variance_threshold (data-driven K from PCA spectrum),
                or both (num_groups then acts as an upper cap on K_auto).
                The gradient refinement and splitting pipelines are bypassed
                (not applicable to a rotation-based method).

    Raises:
        ValueError: If neither clustering.num_groups nor clustering.variance_threshold is set.
        Note: variance_threshold is most effective for pca_striped on concentrated spectra.
              For pca_raw (augmented [N, 2D] PCA), high-dimensional data will give K_auto >> D,
              so num_groups as a direct fixed count is usually the right choice.
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
        c = self.config.clustering
        if c.num_groups is None and c.variance_threshold is None and c.max_groups is None:
            raise ValueError(
                "PCAGrouper requires at least one of: clustering.num_groups (fixed K), "
                "clustering.variance_threshold (data-driven K from PCA spectrum), or "
                "clustering.max_groups (upper cap; variance_threshold selects within it)."
            )

    def _determine_k(self, explained_variance_ratio: np.ndarray) -> int:
        """Return effective group count K from config and the PCA spectrum.

        Priority:
        1. variance_threshold set → K = min PCs s.t. cumvar >= threshold.
        2. num_groups or max_groups set → applied as an upper cap on K_auto.
        3. Only num_groups set (no variance_threshold) → K = num_groups (fixed).
        4. Only max_groups set (no variance_threshold) → K = max_groups (fixed cap).
        """
        params = self.config.clustering
        n_pcs = len(explained_variance_ratio)

        if params.variance_threshold is not None:
            cumvar = np.cumsum(explained_variance_ratio)
            idx = int(np.searchsorted(cumvar, params.variance_threshold, side="left"))
            K = min(idx + 1, n_pcs)
            K = max(K, 1)
            # Apply caps: num_groups and max_groups both act as upper bounds
            cap = None
            if params.num_groups is not None:
                cap = params.num_groups if cap is None else min(cap, params.num_groups)
            if params.max_groups is not None:
                cap = params.max_groups if cap is None else min(cap, params.max_groups)
            if cap is not None:
                K = min(K, cap)
            actual_var = float(cumvar[K - 1])
            logger.info(
                f"variance_threshold={params.variance_threshold:.6f} → K={K} groups "
                f"(cumulative variance={actual_var:.6f})"
                + (f"; capped at {cap}" if cap is not None else "")
            )
        elif params.num_groups is not None:
            K = params.num_groups
        else:
            K = params.max_groups   # validated non-None above
        return K

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
        Each raw feature j is assigned to the group whose PC index has the highest
        absolute loading for that feature, constrained to the top-K significant PCs:
        dominant_pc[j] = argmax_{k<K} ( |C[k,j]| + |C[k,j+D]| ).
        K is determined by _determine_k() from clustering.num_groups / variance_threshold.

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
        seed = self.config.random_seed

        logger.info(
            f"PCAGrouper (pca_raw): fitting PCA on [{N}, {2 * D}] agg features."
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

        # Determine K (number of groups) from config + PCA spectrum.
        K = self._determine_k(explained)
        n_pcs = len(explained)

        # For each raw feature j: loading = |C[k,j]| + |C[k,j+D]|
        # Sums absolute loadings of the mean and std contributions for feature j.
        loadings = np.abs(components[:, :D]) + np.abs(components[:, D:])  # [n_pcs, D]

        # Count features that would fall in the noise tail under unconstrained argmax.
        dominant_pc_unconstrained = np.argmax(loadings, axis=0)            # [D]
        n_reallocated = int(np.sum(dominant_pc_unconstrained >= K))

        # Constrain argmax to top-K PCs only.
        # Features preferring a noise PC (>= K) are assigned to the significant PC
        # (< K) with the highest absolute loading for them.
        dominant_pc = np.argmax(loadings[:K, :], axis=0)                   # [D], 0..K-1

        if n_reallocated > 0:
            logger.info(
                f"pca_raw: {n_reallocated}/{D} features re-allocated from noise PCs "
                f"(dominant PC >= {K}) to their closest significant PC by loading magnitude."
            )

        # Direct assignment — group index IS the dominant significant-PC index.
        # No modulo: each of the K significant PCs becomes exactly one group.
        group_dict: dict[str, list[int]] = {}
        for j, pc_k in enumerate(dominant_pc):
            group_dict.setdefault(f"group_{int(pc_k)}", []).append(j)

        # Defensive filter: remove empty groups (not expected for well-formed data).
        empty = [k for k, v in group_dict.items() if not v]
        if empty:
            logger.warning(f"pca_raw: removing {len(empty)} empty groups: {empty}")
        group_dict = {k: v for k, v in group_dict.items() if v}

        sizes = [len(v) for v in group_dict.values()]
        logger.info(
            f"pca_raw done: K={K} groups (data-driven from {n_pcs} PCs), "
            f"sizes (min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}): "
            f"{sizes[:10]}{'...' if K > 10 else ''}"
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

        logger.info(
            f"PCAGrouper: fitting PCA on [{N}, {D}] features."
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

        explained = pca.explained_variance_ratio_
        pcs_90 = int(np.searchsorted(np.cumsum(explained), 0.90)) + 1
        logger.info(
            f"PCA (standardized): top-5 explained variance = {explained[:5].round(4).tolist()}, "
            f"PCs for 90%={pcs_90}, cumulative={explained.sum():.4f}"
        )

        # Determine K from config + PCA spectrum.
        K = self._determine_k(explained)

        # Fold scaler into components, then truncate to top-K PCs.
        # apply(x) = (x - mean) @ components.T  where components is [K, D].
        # Dropping noise PCs (K..D-1) reduces the output to K dimensions —
        # only the variance-bearing subspace is passed to per-group encoders.
        inv_scale = (1.0 / scaler.scale_).astype(np.float32)              # [D]
        components_folded = pca.components_.astype(np.float32) * inv_scale[None, :]  # [D, D]
        components_folded_k = components_folded[:K, :]                     # [K, D]

        transform = LinearTransform(
            mean=scaler.mean_.astype(np.float32),
            components=components_folded_k,  # [K, D], rows = top-K PCs (scale folded in)
        )

        logger.info(
            f"pca_striped: truncated to K={K} PCs (dropped {D - K} noise PCs), "
            f"assigning to {K} groups via striped assignment."
        )

        # ── 2. Striped assignment over K groups ───────────────────────────────
        pc_names = [f"pc_{i}" for i in range(K)]
        group_dict: dict[str, list[int]] = {f"group_{g}": [] for g in range(K)}
        for pc_idx in range(K):
            group_dict[f"group_{pc_idx % K}"].append(pc_idx)

        # Log group sizes for diagnostic clarity
        sizes = [len(v) for v in group_dict.values()]
        logger.info(
            f"Group sizes (min={min(sizes)}, max={max(sizes)}, "
            f"mean={np.mean(sizes):.1f}): {sizes[:10]}{'...' if K > 10 else ''}"
        )

        # ── 3. Build result with rotation attached ────────────────────────────
        result = self._to_result(group_dict, pc_names)
        result.linear_transform = transform

        logger.info(
            f"PCAGrouper done: {result.num_groups} groups, "
            f"linear_transform stored (shape {transform.components.shape})."
        )
        return result
