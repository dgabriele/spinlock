"""
Validation metrics for sample quality assessment.

Pure functions for computing:
- Discrepancy (star, centered, wrap-around)
- Pairwise correlations
- Coverage uniformity

Design principles:
- Pure functions (no side effects)
- Reusable across samplers
- Clear numerical targets

References:
- Sobol, I. M. (1967). On the distribution of points in a cube
- Joe, S., & Kuo, F. Y. (2008). Constructing Sobol sequences with better two-dimensional projections
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import discrepancy
from typing import Dict, Any, Literal

from ..operators.parameters import SamplingMetrics


def compute_discrepancy(
    samples: NDArray[np.float64], method: Literal["CD", "WD", "MD", "L2-star"] = "CD"
) -> float:
    """
    Compute discrepancy of sample set.

    Lower discrepancy indicates more uniform distribution.
    Target for quasi-Monte Carlo: <0.01 for 10k samples.

    Args:
        samples: Sample array in [0,1]^d, shape (n_samples, dimensionality)
        method: Discrepancy measure
                - 'CD': Centered discrepancy (default, balanced)
                - 'WD': Wrap-around discrepancy (periodic domains)
                - 'MD': Modified L2 discrepancy
                - 'L2-star': L2-star discrepancy

    Returns:
        Discrepancy value (lower is better)

    Example:
        ```python
        samples = sobol_sampler.sample(10000)
        disc = compute_discrepancy(samples)
        print(f"Discrepancy: {disc:.6f} (target: <0.01)")
        ```
    """
    return float(discrepancy(samples, method=method))


def compute_pairwise_correlations(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute correlation matrix for sample dimensions.

    Low pairwise correlations indicate good decorrelation.
    Target: max absolute correlation <0.05

    Args:
        samples: Sample array of shape (n_samples, dimensionality)

    Returns:
        Correlation matrix of shape (dimensionality, dimensionality)

    Example:
        ```python
        samples = sampler.sample(1000)
        corr = compute_pairwise_correlations(samples)
        max_corr = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)]).max()
        print(f"Max correlation: {max_corr:.6f}")
        ```
    """
    return np.corrcoef(samples.T)


def compute_max_correlation(samples: NDArray[np.float64]) -> float:
    """
    Compute maximum absolute pairwise correlation (excluding diagonal).

    Convenience function for common use case.

    Args:
        samples: Sample array

    Returns:
        Maximum absolute correlation

    Example:
        ```python
        max_corr = compute_max_correlation(samples)
        assert max_corr < 0.05, "Samples too correlated!"
        ```
    """
    corr_matrix = compute_pairwise_correlations(samples)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    return float(np.abs(corr_matrix[mask]).max()) if corr_matrix.size > 1 else 0.0


def compute_coverage_uniformity(samples: NDArray[np.float64], num_bins_per_dim: int = 10) -> float:
    """
    Compute uniformity of sample coverage using chi-squared statistic.

    Lower values indicate more uniform coverage.

    Args:
        samples: Sample array
        num_bins_per_dim: Number of bins per dimension for histogram

    Returns:
        Chi-squared statistic (lower is more uniform)

    Example:
        ```python
        chi_sq = compute_coverage_uniformity(samples, num_bins_per_dim=5)
        print(f"Coverage chi-squared: {chi_sq:.2f}")
        ```
    """
    n_samples, dimensionality = samples.shape
    expected_per_bin = n_samples / (num_bins_per_dim**dimensionality)

    # Compute multi-dimensional histogram
    bins = [np.linspace(0, 1, num_bins_per_dim + 1) for _ in range(dimensionality)]
    hist, _ = np.histogramdd(samples, bins=bins)

    # Chi-squared statistic
    chi_squared = ((hist - expected_per_bin) ** 2 / expected_per_bin).sum()

    return float(chi_squared)


def compute_min_distance(samples: NDArray[np.float64]) -> float:
    """
    Compute minimum Euclidean distance between any two samples.

    Higher values indicate better space-filling properties.

    Args:
        samples: Sample array

    Returns:
        Minimum pairwise distance

    Note:
        O(n^2) complexity - use only for small to medium sample sets.

    Example:
        ```python
        min_dist = compute_min_distance(samples)
        print(f"Minimum separation: {min_dist:.6f}")
        ```
    """
    from scipy.spatial.distance import pdist

    if len(samples) < 2:
        return np.inf

    distances = pdist(samples)
    return float(np.min(distances))


def validate_sample_quality(
    samples: NDArray[np.float64], targets: Dict[str, float]
) -> SamplingMetrics:
    """
    Comprehensive sample quality validation.

    DRY: Compose individual metrics into a validation pipeline.

    Args:
        samples: Sample array to validate
        targets: Dict of metric_name -> target_value
                Example: {"max_discrepancy": 0.01, "max_correlation": 0.05}

    Returns:
        SamplingMetrics dataclass with computed metrics and pass/fail status

    Example:
        ```python
        targets = {
            "max_discrepancy": 0.01,
            "max_correlation": 0.05,
        }
        metrics = validate_sample_quality(samples, targets)

        if metrics.discrepancy_pass and metrics.correlation_pass:
            print("✓ Sample quality meets targets!")
        else:
            print(f"✗ Failed: discrepancy={metrics.discrepancy:.6f}")
        ```
    """
    # Compute required metrics
    disc = compute_discrepancy(samples, method="CD")
    max_corr = compute_max_correlation(samples)

    # Check against targets
    disc_pass = disc < targets.get("max_discrepancy", float("inf"))
    corr_pass = max_corr < targets.get("max_correlation", float("inf"))

    # Compute optional detailed metrics
    corr_matrix = compute_pairwise_correlations(samples)
    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
    if corr_matrix.size > 1:
        min_corr = float(np.abs(corr_matrix[mask]).min())
        mean_corr = float(np.abs(corr_matrix[mask]).mean())
    else:
        min_corr = None
        mean_corr = None

    # Create dataclass
    return SamplingMetrics(
        discrepancy=disc,
        max_correlation=max_corr,
        discrepancy_pass=disc_pass,
        correlation_pass=corr_pass,
        min_correlation=min_corr,
        mean_correlation=mean_corr,
        coverage_uniformity=None,  # Can be added if needed
    )


def print_sample_quality_report(metrics: SamplingMetrics) -> None:
    """
    Pretty-print sample quality metrics.

    Args:
        metrics: SamplingMetrics dataclass from validate_sample_quality()

    Example:
        ```python
        metrics = validate_sample_quality(samples, targets)
        print_sample_quality_report(metrics)
        ```
    """
    print("=" * 60)
    print("SAMPLE QUALITY REPORT")
    print("=" * 60)

    # Core metrics (always present)
    status = "✓" if metrics.discrepancy_pass else "✗"
    print(f"{status} Discrepancy: {metrics.discrepancy:.6f}")

    status = "✓" if metrics.correlation_pass else "✗"
    print(f"{status} Max correlation: {metrics.max_correlation:.6f}")

    # Optional detailed metrics
    if metrics.min_correlation is not None:
        print(f"  Min correlation: {metrics.min_correlation:.6f}")

    if metrics.mean_correlation is not None:
        print(f"  Mean correlation: {metrics.mean_correlation:.6f}")

    if metrics.coverage_uniformity is not None:
        print(f"  Coverage uniformity: {metrics.coverage_uniformity:.2f}")

    print("=" * 60)
    if metrics.discrepancy_pass and metrics.correlation_pass:
        print("✓ ALL CORE METRICS PASSED")
    else:
        failed = []
        if not metrics.discrepancy_pass:
            failed.append("discrepancy")
        if not metrics.correlation_pass:
            failed.append("correlation")
        print(f"✗ FAILED METRICS: {', '.join(failed)}")
    print("=" * 60)
