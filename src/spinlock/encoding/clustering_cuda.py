"""CUDA-accelerated correlation and clustering for dynamic category assignment.

Implements efficient GPU kernels for:
1. Pearson correlation matrix computation
2. Distance matrix computation from correlations
3. Hierarchical clustering preparation

Designed to handle large datasets efficiently (10K+ samples, 300+ features).

Ported from unisim.system.models.clustering_cuda (2025-12-30).
"""

from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.debug("CuPy not available - CUDA acceleration disabled for clustering")


def compute_correlation_matrix_cuda(
    features: np.ndarray, subsample_size: Optional[int] = None
) -> np.ndarray:
    """Compute feature correlation matrix using GPU acceleration.

    Uses MAD (Median Absolute Deviation) normalization before computing
    Pearson correlation for robustness to outliers.

    Implementation: MAD-normalize data on GPU, then compute Pearson correlation.

    Args:
        features: [N_samples, N_features] feature data (numpy array)
        subsample_size: If provided, randomly subsample this many rows

    Returns:
        correlation_matrix: [N_features, N_features] numpy array
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot use CUDA correlation")

    N_samples, N_features = features.shape

    # Subsample if requested
    if subsample_size is not None and subsample_size < N_samples:
        indices = np.random.choice(N_samples, subsample_size, replace=False)
        features = features[indices]
        N_samples = subsample_size

    # Transfer to GPU
    features_gpu = cp.asarray(features, dtype=cp.float32)

    # MAD-normalize each feature column for robustness to outliers
    normalized_gpu = cp.zeros_like(features_gpu)
    for j in range(N_features):
        col = features_gpu[:, j]
        # Compute median
        median = cp.median(col)
        # Compute MAD (Median Absolute Deviation)
        mad = cp.median(cp.abs(col - median)) * 1.4826
        mad = cp.maximum(mad, 1e-8)  # Avoid division by zero
        # Normalize
        normalized_gpu[:, j] = (col - median) / mad

    del features_gpu  # Free original data

    # Now compute Pearson correlation on MAD-normalized features
    # Center each feature (zero mean)
    mean = cp.mean(normalized_gpu, axis=0, keepdims=True)
    normalized_gpu -= mean
    del mean

    # Compute standard deviations
    std = cp.std(normalized_gpu, axis=0)
    std = cp.where(std < 1e-8, 1.0, std)

    # Normalize by std
    normalized_gpu /= std[cp.newaxis, :]
    del std

    # Correlation matrix via matrix multiplication: (1/N) * X^T @ X
    # where X is now normalized (centered + standardized)
    corr_matrix = (1.0 / (N_samples - 1)) * cp.dot(normalized_gpu.T, normalized_gpu)
    del normalized_gpu

    # Ensure diagonal is exactly 1.0 and clamp to [-1, 1]
    cp.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = cp.clip(corr_matrix, -1.0, 1.0)

    # Transfer back to CPU
    result = cp.asnumpy(corr_matrix).astype(np.float64)
    del corr_matrix

    return result


def compute_correlation_distance_cuda(corr_matrix: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix for clustering.

    Distance = 1 - |correlation|

    Args:
        corr_matrix: [N_features, N_features] correlation matrix

    Returns:
        distance_matrix: [N_features, N_features] distance matrix
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available - cannot use CUDA distance")

    corr_gpu = cp.asarray(corr_matrix, dtype=cp.float32)

    # Distance = 1 - |correlation|
    dist_gpu = 1.0 - cp.abs(corr_gpu)

    # Ensure diagonal is exactly 0
    cp.fill_diagonal(dist_gpu, 0.0)

    return cp.asnumpy(dist_gpu).astype(np.float64)


def prepare_condensed_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    """Convert square distance matrix to condensed form for scipy linkage.

    Args:
        distance_matrix: [N, N] square symmetric distance matrix

    Returns:
        condensed_distances: [N*(N-1)/2] upper triangular distances
    """
    from scipy.spatial.distance import squareform

    # Extract upper triangular (excluding diagonal)
    return squareform(distance_matrix, checks=False)


def compute_correlation_and_distance_cuda(
    features: np.ndarray, subsample_size: Optional[int] = None, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end GPU pipeline: features → correlation → distance.

    Args:
        features: [N_samples, N_features] feature data
        subsample_size: Optional subsampling for large datasets
        verbose: Print diagnostics

    Returns:
        corr_matrix: [N_features, N_features] correlation matrix
        condensed_dist: [N_features*(N_features-1)/2] condensed distance matrix
    """
    N_samples, N_features = features.shape

    if verbose:
        logger.info("\n=== CUDA Correlation & Distance Computation ===")
        logger.info(f"Input: {N_samples:,} samples × {N_features} features")
        if subsample_size is not None and subsample_size < N_samples:
            logger.info(
                f"Subsampling: {subsample_size:,} / {N_samples:,} samples ({100*subsample_size/N_samples:.1f}%)"
            )

    # Step 1: Correlation matrix (GPU)
    corr_matrix = compute_correlation_matrix_cuda(features, subsample_size)

    if verbose:
        off_diag = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        logger.info("Correlation stats:")
        logger.info(f"  Mean: {off_diag.mean():.3f}")
        logger.info(f"  Median: {np.median(off_diag):.3f}")
        logger.info(f"  Std: {off_diag.std():.3f}")
        logger.info(f"  Min: {off_diag.min():.3f}")
        logger.info(f"  Max: {off_diag.max():.3f}")

    # Step 2: Distance matrix (GPU)
    dist_matrix = compute_correlation_distance_cuda(corr_matrix)

    # Step 3: Condensed form for scipy linkage (CPU)
    condensed_dist = prepare_condensed_distance_matrix(dist_matrix)

    if verbose:
        logger.info("Output:")
        logger.info(f"  Correlation matrix: {corr_matrix.shape}")
        logger.info(f"  Condensed distance: {condensed_dist.shape}")
        logger.info("===")

    # Explicitly free GPU memory used by CuPy
    # This is critical for hybrid CUDA/PyTorch workflows where PyTorch
    # will need GPU memory after CuPy correlation computation
    if CUPY_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        if verbose:
            logger.info("GPU memory freed (CuPy)")

    return corr_matrix, condensed_dist


def benchmark_correlation_methods(
    features: np.ndarray, subsample_size: Optional[int] = None, n_trials: int = 3
) -> None:
    """Benchmark CPU vs GPU correlation computation.

    Args:
        features: Test features
        subsample_size: Optional subsampling
        n_trials: Number of trials for timing
    """
    import time
    from .clustering_assignment import compute_correlation_matrix_cpu

    N_samples, N_features = features.shape

    if subsample_size is not None and subsample_size < N_samples:
        indices = np.random.choice(N_samples, subsample_size, replace=False)
        features_test = features[indices]
    else:
        features_test = features

    logger.info("\n=== Correlation Benchmark ===")
    logger.info(f"Features: {features_test.shape}")

    # CPU baseline
    times_cpu = []
    for _ in range(n_trials):
        t0 = time.time()
        _ = compute_correlation_matrix_cpu(features_test)
        times_cpu.append(time.time() - t0)
    cpu_mean = np.mean(times_cpu)

    # GPU implementation
    if CUPY_AVAILABLE:
        times_gpu = []
        for _ in range(n_trials):
            t0 = time.time()
            _ = compute_correlation_matrix_cuda(features_test)
            times_gpu.append(time.time() - t0)
        gpu_mean = np.mean(times_gpu)

        logger.info(f"CPU: {cpu_mean:.3f}s (baseline)")
        logger.info(f"GPU: {gpu_mean:.3f}s ({cpu_mean/gpu_mean:.1f}× speedup)")
    else:
        logger.info(f"CPU: {cpu_mean:.3f}s (CuPy not available)")
    logger.info("===")
