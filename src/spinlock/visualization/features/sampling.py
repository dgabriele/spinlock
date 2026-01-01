"""
Diversity-based operator sampling for visualization.

Implements farthest point sampling in joint parameter-feature space
to select maximally diverse operators for showcasing.
"""

import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def select_diverse_operators(
    parameters: np.ndarray,
    features: np.ndarray,
    n_select: int = 2,
    alpha: float = 0.5,
    seed: int = 42
) -> List[int]:
    """
    Select operators maximizing diversity in joint parameter-feature space.

    Uses two-stage farthest point sampling to find operators that are
    diverse in both their architectural parameters (what they ARE) and
    their behavioral features (what they DO).

    Algorithm:
        1. Normalize parameters to [0, 1] (min-max scaling)
        2. Aggregate features across timesteps: [N, T, D] → [N, D]
        3. Standardize features: (x - mean) / std
        4. Compute joint distance: D = alpha * D_param + (1-alpha) * D_feat
        5. Farthest point sampling: Greedily select most distant points

    Args:
        parameters: Operator parameters [N, P]
            N = number of operators
            P = parameter dimension
        features: Per-timestep features [N, T, D]
            T = number of timesteps
            D = feature dimension
        n_select: Number of operators to select (default: 2)
        alpha: Weight for parameter vs feature diversity, in [0, 1]
            - alpha=1.0: Pure parameter diversity (architectural)
            - alpha=0.0: Pure feature diversity (behavioral)
            - alpha=0.5: Equal weight (recommended)
        seed: Random seed for reproducibility

    Returns:
        List of n_select operator indices sorted in selection order

    Complexity:
        O(N²×D) for distance computation
        O(N²×k) for farthest point sampling
        Acceptable for N ≤ 10,000

    Example:
        >>> params = np.random.rand(100, 10)  # 100 operators, 10 parameters
        >>> feats = np.random.rand(100, 200, 50)  # 100 ops, 200 steps, 50 features
        >>> indices = select_diverse_operators(params, feats, n_select=5)
        >>> print(f"Selected operators: {indices}")
        Selected operators: [42, 7, 91, 23, 66]
    """
    rng = np.random.default_rng(seed)
    N = len(parameters)

    if n_select > N:
        raise ValueError(f"Cannot select {n_select} operators from {N} total")

    if n_select == N:
        return list(range(N))

    # Stage 1: Normalize parameters
    # Min-max scaling to [0, 1] per dimension
    param_scaler = MinMaxScaler()
    params_norm = param_scaler.fit_transform(parameters)  # [N, P]

    # Stage 2: Aggregate and standardize features
    # Mean across timesteps: [N, T, D] → [N, D]
    features_agg = features.mean(axis=1)  # [N, D]

    # Standardize: (x - mean) / std per dimension
    feat_scaler = StandardScaler()
    feats_norm = feat_scaler.fit_transform(features_agg)  # [N, D]

    # Stage 3: Compute distance matrices
    # Euclidean distance in normalized spaces
    D_param = _pairwise_euclidean(params_norm)  # [N, N]
    D_feat = _pairwise_euclidean(feats_norm)    # [N, N]

    # Joint distance: weighted combination
    D_joint = alpha * D_param + (1 - alpha) * D_feat  # [N, N]

    # Stage 4: Farthest point sampling
    selected_indices = _farthest_point_sampling(D_joint, n_select, rng)

    return selected_indices


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances.

    Args:
        X: Data matrix [N, D]

    Returns:
        Distance matrix [N, N] where D[i, j] = ||X[i] - X[j]||₂

    Note:
        Uses efficient vectorized computation:
        ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
    """
    # Compute squared norms: [N]
    norms_sq = (X ** 2).sum(axis=1)

    # Compute dot products: [N, N]
    dot_products = X @ X.T

    # Distance squared: [N, N]
    # D²[i,j] = ||x_i||² + ||x_j||² - 2⟨x_i, x_j⟩
    dist_sq = norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - 2 * dot_products

    # Handle numerical errors (negative values due to floating point)
    dist_sq = np.maximum(dist_sq, 0.0)

    # Return distance (not squared)
    return np.sqrt(dist_sq)


def _farthest_point_sampling(
    distances: np.ndarray,
    n_select: int,
    rng: np.random.Generator
) -> List[int]:
    """
    Farthest point sampling for diversity maximization.

    Greedy algorithm that iteratively selects the point farthest from
    all previously selected points.

    Args:
        distances: Pairwise distance matrix [N, N]
        n_select: Number of points to select
        rng: NumPy random number generator

    Returns:
        List of n_select indices in selection order

    Algorithm:
        1. Start with random point
        2. For each iteration:
           - Compute minimum distance from each point to selected set
           - Select point with maximum minimum distance
        3. Repeat until n_select points selected

    Complexity:
        O(N² × k) where k = n_select
        For k=2, N=1000: ~2M operations (fast)
    """
    N = len(distances)
    selected = []

    # Start with random point
    first_idx = rng.integers(0, N)
    selected.append(first_idx)

    # Iteratively select farthest points
    for _ in range(n_select - 1):
        # Compute minimum distance from each point to selected set
        # min_dists[i] = min distance from point i to any selected point
        dists_to_selected = distances[selected, :]  # [len(selected), N]
        min_dists = dists_to_selected.min(axis=0)   # [N]

        # Select point with maximum minimum distance (farthest from set)
        # Don't select already selected points (set their distance to -inf)
        min_dists[selected] = -np.inf
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return selected
