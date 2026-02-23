"""Optimized Product Quantization (OPQ) for feature grouping.

Implements Ge et al. 2013 — alternating optimization of an orthogonal rotation
matrix R and per-subspace codebooks.  The rotation decorrelates features and
equalizes per-group reconstruction error, which is the gold-standard approach
for product quantization.

Two entry points:
  - ``compute_opq_rotation(X, M, d_sub, ...)`` — core algorithm, returns
    ``LinearTransform(mean, R)``
  - ``OPQGrouper`` — wraps the algorithm in the ``FeatureGrouper`` interface
    (for manual-mode compatibility with the grouping factory)

Dependencies: numpy, sklearn.cluster.MiniBatchKMeans (both already in project).
No faiss required.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from spinlock.encoding.normalization import LinearTransform
from .models import GroupingConfig, GroupingResult, FeatureGroup

logger = logging.getLogger(__name__)


def compute_opq_rotation(
    X: np.ndarray,
    M: int,
    d_sub: int,
    n_iter: int = 20,
    n_codes: int = 32,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute an OPQ rotation matrix via alternating optimization.

    Algorithm:
      1. Center X, initialize R with PCA (SVD of centered X).
      2. For each iteration:
         a. Fix R, solve codebooks: rotate data, run MiniBatchKMeans per
            group slice.
         b. Reconstruct: assign each sample to nearest centroid per group,
            build X_recon.
         c. Fix codebooks, solve R: Procrustes — SVD(X.T @ X_recon) → R = U @ Vt.
      3. Return R ([D, D] orthogonal) and mean ([D]).

    Args:
        X:      Calibration data [N, D].
        M:      Number of groups.
        d_sub:  Dimension per group (D // M).
        n_iter: Alternating optimization iterations.
        n_codes: Proxy codebook size per subspace.
        random_state: Random seed for reproducibility.

    Returns:
        (R, mean) where R is [D, D] orthogonal rotation matrix and
        mean is [D] centering vector.
    """
    N, D = X.shape
    assert D == M * d_sub, f"D={D} != M*d_sub={M}*{d_sub}={M * d_sub}"

    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Initialize R with PCA (top-D eigenvectors of covariance)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    R = Vt  # [D, D] — rows are principal components

    rng = np.random.RandomState(random_state)

    for iteration in range(n_iter):
        # Step A: Fix R, rotate data, fit per-group codebooks
        X_rot = X_centered @ R.T  # [N, D]

        codebooks = []
        X_recon_rot = np.zeros_like(X_rot)

        for g in range(M):
            start = g * d_sub
            end = (g + 1) * d_sub
            X_g = X_rot[:, start:end]  # [N, d_sub]

            # Fit per-group codebook
            actual_codes = min(n_codes, N)
            kmeans = MiniBatchKMeans(
                n_clusters=actual_codes,
                batch_size=min(1024, N),
                n_init=1,
                random_state=rng.randint(0, 2**31),
            )
            kmeans.fit(X_g)
            codebooks.append(kmeans)

            # Reconstruct: assign to nearest centroid
            labels = kmeans.predict(X_g)
            X_recon_rot[:, start:end] = kmeans.cluster_centers_[labels]

        # Step B: Fix codebooks, solve R via Procrustes
        # We want R that minimizes ||X_centered @ R.T - X_recon_rot||_F
        # Equivalently: maximize trace(R @ X_centered.T @ X_recon_rot)
        # Solution: SVD of X_centered.T @ X_recon_rot = U_p @ S_p @ Vt_p
        #           R = Vt_p.T @ U_p.T = (U_p @ Vt_p).T
        C = X_centered.T @ X_recon_rot  # [D, D]
        U_p, S_p, Vt_p = np.linalg.svd(C, full_matrices=True)
        R = (U_p @ Vt_p)  # [D, D] orthogonal

        # Log per-group variance of rotated data for monitoring
        if iteration == 0 or iteration == n_iter - 1:
            X_rot_final = X_centered @ R.T
            group_vars = [
                X_rot_final[:, g * d_sub:(g + 1) * d_sub].var()
                for g in range(M)
            ]
            var_ratio = max(group_vars) / max(min(group_vars), 1e-12)
            logger.info(
                "OPQ iter %d/%d: per-group var range [%.4f, %.4f], ratio=%.1f",
                iteration + 1, n_iter,
                min(group_vars), max(group_vars), var_ratio,
            )

    return R, mean


class OPQGrouper:
    """Wraps compute_opq_rotation in the FeatureGrouper interface.

    Intended for manual-mode compatibility with the grouping factory.
    For learned-mode, the trainer calls compute_opq_rotation directly.
    """

    def __init__(self, config: GroupingConfig):
        self.config = config

    def group_features(
        self,
        features: np.ndarray,
        feature_names: list,
        num_groups: Optional[int] = None,
    ) -> GroupingResult:
        """Run OPQ and return groups with sequential d_sub-sized slices.

        Args:
            features: [N, D] calibration features.
            feature_names: List of D feature names.
            num_groups: Override for number of groups (default from config).

        Returns:
            GroupingResult with linear_transform set.
        """
        N, D = features.shape
        M = num_groups or self.config.clustering.num_groups or 30
        if D % M != 0:
            raise ValueError(f"D={D} not divisible by num_groups={M}")
        d_sub = D // M

        R, mean = compute_opq_rotation(
            features, M=M, d_sub=d_sub,
            random_state=self.config.random_seed,
        )
        transform = LinearTransform(mean=mean, components=R)

        # Groups are sequential slices of the rotated space
        groups = {}
        for g in range(M):
            start = g * d_sub
            indices = list(range(start, start + d_sub))
            name = f"temporal_group_{g}"
            groups[name] = FeatureGroup(
                name=name,
                feature_indices=indices,
                feature_names=[f"opq_dim_{i}" for i in indices],
                size=d_sub,
            )

        return GroupingResult(
            groups=groups,
            num_groups=M,
            total_features=D,
            config=self.config,
            linear_transform=transform,
        )
