"""
Abstract base classes for parameter space samplers.

Provides interfaces that enable:
- Swappable sampling strategies (Sobol, LHS, Halton, etc.)
- Consistent validation across samplers
- Easy testing via mocking

Design principles:
- Abstract base classes define contracts
- Pure functions for metrics
- Composition over inheritance
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any


class BaseSampler(ABC):
    """
    Abstract base class for parameter space samplers.

    All samplers must implement:
    1. sample(n_samples) -> samples in [0,1]^d
    2. validate(samples) -> quality metrics dict

    Example:
        ```python
        class MySampler(BaseSampler):
            def sample(self, n_samples: int) -> NDArray[np.float64]:
                return np.random.rand(n_samples, self.dimensionality)

            def validate(self, samples: NDArray[np.float64]) -> Dict[str, float]:
                return {"metric": 0.5}
        ```
    """

    def __init__(self, dimensionality: int, seed: int = 42):
        """
        Initialize base sampler.

        Args:
            dimensionality: Number of dimensions in parameter space
            seed: Random seed for reproducibility
        """
        if dimensionality < 1:
            raise ValueError(f"Dimensionality must be ≥ 1, got {dimensionality}")

        self.dimensionality = dimensionality
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """
        Generate n_samples from parameter space.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, dimensionality) with values in [0, 1]^d

        Raises:
            ValueError: If n_samples < 1
        """
        pass

    @abstractmethod
    def validate(self, samples: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Compute validation metrics for sample quality.

        Args:
            samples: Sample array of shape (n_samples, dimensionality)

        Returns:
            Dictionary with metrics (discrepancy, correlations, coverage, etc.)

        Example:
            ```python
            samples = sampler.sample(1000)
            metrics = sampler.validate(samples)
            print(f"Discrepancy: {metrics['discrepancy']:.6f}")
            ```
        """
        pass

    def _validate_sample_array(self, samples: NDArray[np.float64]) -> None:
        """
        Shared validation for sample arrays (DRY).

        Args:
            samples: Sample array to validate

        Raises:
            ValueError: If array has wrong shape or values outside [0,1]
        """
        if samples.ndim != 2:
            raise ValueError(f"Samples must be 2D array, got shape {samples.shape}")

        if samples.shape[1] != self.dimensionality:
            raise ValueError(
                f"Samples must have {self.dimensionality} dimensions, " f"got {samples.shape[1]}"
            )

        if not np.all((samples >= 0) & (samples <= 1)):
            raise ValueError(
                f"Sample values must be in [0,1], "
                f"got range [{samples.min():.4f}, {samples.max():.4f}]"
            )


class NoOpSampler(BaseSampler):
    """
    Null object pattern: sampler that does nothing.

    Useful for testing or as a default when stratification is disabled.
    """

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        return self._rng.rand(n_samples, self.dimensionality)

    def validate(self, samples: NDArray[np.float64]) -> Dict[str, Any]:
        return {}
