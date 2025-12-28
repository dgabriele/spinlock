"""
Trajectory metrics for temporal evolution analysis.

Computes diagnostic metrics for evolved trajectories:
- Energy: L2 norm (field magnitude)
- Entropy: Shannon entropy of field distribution
- Autocorrelation: Temporal correlation between consecutive states
- Variance: Spatial variance
- Mean magnitude: Average absolute value

All computations are GPU-accelerated using PyTorch operations.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryMetrics:
    """
    Container for trajectory diagnostics.

    Provides type-safe access to computed metrics with clear semantics.
    All metrics are single scalar values representing aggregate properties
    of the field at a given timestep.

    Example:
        ```python
        metrics = TrajectoryMetrics(
            energy=42.5,
            entropy=3.14,
            autocorrelation=0.95,
            variance=1.2,
            mean_magnitude=2.1
        )

        # Type-safe access
        print(f"Energy: {metrics.energy:.2f}")
        print(f"Entropy: {metrics.entropy:.2f}")
        ```
    """

    energy: float              # ||X||^2 (L2 energy)
    entropy: float             # Shannon entropy of field values
    autocorrelation: float     # Correlation with previous timestep
    variance: float            # Spatial variance
    mean_magnitude: float      # Mean absolute value


class MetricsComputer:
    """
    GPU-accelerated metrics computation for trajectories.

    All methods are static and operate on PyTorch tensors, maintaining
    GPU residency for performance. Computations use stable numerical
    algorithms suitable for large-scale data.

    Example:
        ```python
        computer = MetricsComputer()

        # Compute individual metrics
        energy = computer.compute_energy(X_t)
        entropy = computer.compute_entropy(X_t)

        # Compute all metrics at once
        metrics = computer.compute_all(X_t, X_prev)
        ```
    """

    @staticmethod
    def compute_energy(X: torch.Tensor) -> float:
        """
        Compute L2 energy: ||X||^2

        Measures total field magnitude. Useful for detecting
        blow-up, decay, or equilibrium states.

        Args:
            X: Field tensor [B, C, H, W] or [C, H, W]

        Returns:
            Scalar energy value

        Example:
            ```python
            energy = MetricsComputer.compute_energy(X)
            # energy = sum(X**2)
            ```
        """
        return float(torch.sum(X ** 2).item())

    @staticmethod
    def compute_entropy(X: torch.Tensor, num_bins: int = 50) -> float:
        """
        Compute Shannon entropy of field value distribution.

        Estimates information content by binning field values
        and computing entropy of the histogram. Higher entropy
        indicates more diverse/complex distributions.

        Args:
            X: Field tensor [B, C, H, W] or [C, H, W]
            num_bins: Number of bins for histogram (default: 50)

        Returns:
            Entropy in nats (natural logarithm base)

        Example:
            ```python
            entropy = MetricsComputer.compute_entropy(X)
            # High entropy = complex distribution
            # Low entropy = concentrated distribution
            ```
        """
        X_flat = X.flatten()

        # Compute histogram
        hist = torch.histc(X_flat, bins=num_bins)

        # Normalize to probability distribution
        hist = hist / hist.sum()

        # Remove zero bins (would cause log(0))
        hist = hist[hist > 0]

        # Shannon entropy: -sum(p * log(p))
        entropy = -torch.sum(hist * torch.log(hist))

        return float(entropy.item())

    @staticmethod
    def compute_autocorrelation(X_t: torch.Tensor, X_t_minus_1: torch.Tensor) -> float:
        """
        Compute temporal autocorrelation between consecutive states.

        Measures correlation between X_t and X_{t-1}. Values near 1
        indicate strong persistence, values near 0 indicate decorrelation,
        negative values indicate anti-correlation.

        Args:
            X_t: Current state [B, C, H, W] or [C, H, W]
            X_t_minus_1: Previous state (same shape)

        Returns:
            Correlation coefficient in [-1, 1]

        Example:
            ```python
            corr = MetricsComputer.compute_autocorrelation(X_t, X_prev)
            # corr ≈ 1: strongly persistent
            # corr ≈ 0: decorrelated
            # corr ≈ -1: anti-correlated
            ```
        """
        # Flatten both tensors
        X_t_flat = X_t.flatten()
        X_prev_flat = X_t_minus_1.flatten()

        # Stack and compute correlation matrix
        stacked = torch.stack([X_t_flat, X_prev_flat])
        corr_matrix = torch.corrcoef(stacked)

        # Extract correlation coefficient (off-diagonal element)
        correlation = corr_matrix[0, 1]

        return float(correlation.item())

    @staticmethod
    def compute_variance(X: torch.Tensor) -> float:
        """
        Compute spatial variance of field.

        Measures spread of field values around the mean.
        Useful for detecting stability, concentration, or dispersion.

        Args:
            X: Field tensor [B, C, H, W] or [C, H, W]

        Returns:
            Variance (squared standard deviation)

        Example:
            ```python
            var = MetricsComputer.compute_variance(X)
            # High variance = dispersed values
            # Low variance = concentrated values
            ```
        """
        return float(torch.var(X).item())

    @staticmethod
    def compute_mean_magnitude(X: torch.Tensor) -> float:
        """
        Compute mean absolute value of field.

        Measures average field magnitude ignoring sign.
        Useful for tracking overall activity level.

        Args:
            X: Field tensor [B, C, H, W] or [C, H, W]

        Returns:
            Mean absolute value

        Example:
            ```python
            mag = MetricsComputer.compute_mean_magnitude(X)
            # mag = mean(|X|)
            ```
        """
        return float(torch.mean(torch.abs(X)).item())

    @staticmethod
    def compute_all(
        X_t: torch.Tensor,
        X_t_minus_1: Optional[torch.Tensor] = None
    ) -> TrajectoryMetrics:
        """
        Compute all trajectory metrics for a given state.

        Convenient method for computing all metrics in one call.
        If previous state is not provided, autocorrelation is set to 0.

        Args:
            X_t: Current state [B, C, H, W] or [C, H, W]
            X_t_minus_1: Previous state (optional, for autocorrelation)

        Returns:
            TrajectoryMetrics dataclass with all metrics

        Example:
            ```python
            # First timestep (no previous state)
            metrics_0 = MetricsComputer.compute_all(X_0)

            # Later timesteps (with previous state)
            metrics_t = MetricsComputer.compute_all(X_t, X_{t-1})
            ```
        """
        energy = MetricsComputer.compute_energy(X_t)
        entropy = MetricsComputer.compute_entropy(X_t)
        variance = MetricsComputer.compute_variance(X_t)
        mean_mag = MetricsComputer.compute_mean_magnitude(X_t)

        # Autocorrelation requires previous state
        autocorr = 0.0
        if X_t_minus_1 is not None:
            autocorr = MetricsComputer.compute_autocorrelation(X_t, X_t_minus_1)

        return TrajectoryMetrics(
            energy=energy,
            entropy=entropy,
            autocorrelation=autocorr,
            variance=variance,
            mean_magnitude=mean_mag
        )
