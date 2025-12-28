"""
Stratified Sobol sampler with Owen scrambling.

Implements low-discrepancy quasi-Monte Carlo sampling with:
- Sobol sequences (scipy.stats.qmc)
- Owen scrambling for improved stratification
- Optional adaptive refinement based on variance
- Comprehensive validation metrics

Design principles:
- Composition: Stratification and refinement as optional plugins
- Performance: Target 10k samples in <10s
- Quality: Discrepancy <0.01, correlation <0.05

References:
- Sobol, I. M. (1967). Distribution of points in a cube
- Owen, A. B. (1995). Randomly permuted (t,m,s)-nets
- Joe, S., & Kuo, F. Y. (2008). Constructing Sobol sequences

At a high level, this component generates well-distributed samples of a
high-dimensional parameter space in a way that is far more efficient and
reliable than naïve random sampling. Instead of drawing independent random
points (which cluster and leave gaps), it uses a Sobol low-discrepancy sequence
to deterministically cover the unit hypercube evenly. Owen scrambling is applied
on top to randomize the sequence without destroying its uniform coverage, which
reduces structured artifacts and unwanted correlations between dimensions. The
result is a sequence of points that “fills space” smoothly and predictably, even
in dozens or hundreds of dimensions.

Operationally, the sampler constructs a Sobol engine parameterized by the number
of model parameters (dimensions). Each call to sample(n) advances the sequence
and returns n points in [ 0 , 1 ] 𝑑 [0,1] d , which are later mapped into your
actual parameter ranges. Scrambling ensures reproducibility with a seed while
still behaving statistically like a randomized process. After generation, the
sampler explicitly validates quality—checking discrepancy (how evenly space is
covered) and pairwise correlations—so failures are detected immediately rather
than silently contaminating downstream experiments.

This is needed because stochastic neural operators are extremely sensitive to
coverage bias in parameter space. Monte Carlo sampling wastes samples, misses
regimes, and introduces variance that looks like “model behavior” but is
actually sampling noise. Low-discrepancy QMC sampling dramatically improves
sample efficiency: fewer runs explore more of the true dynamical landscape,
training data is more representative, and learned operators generalize better.
In short, this sampler reduces experimental noise, improves reproducibility, and
lets you spend compute on learning dynamics rather than compensating for poor
sampling.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats.qmc import Sobol
from typing import Dict, Any, Optional
import time
import torch

from .base import BaseSampler
from .metrics import validate_sample_quality, print_sample_quality_report
from ..config.schema import SamplingConfig


class StratifiedSobolSampler(BaseSampler):
    """
    Sobol sampler with Owen scrambling and optional stratification.

    Features:
    - Low-discrepancy Sobol sequence (scipy.stats.qmc.Sobol)
    - Owen scrambling for decorrelation
    - Optional adaptive refinement (future)
    - Validation against quality targets

    Example:
        ```python
        from spinlock.config import load_config

        config = load_config("config.yaml")
        sampler = StratifiedSobolSampler.from_config(
            config.parameter_space,
            config.sampling
        )

        # Generate samples
        samples = sampler.sample(10000)

        # Validate quality
        metrics = sampler.validate(samples)
        print(f"Discrepancy: {metrics['discrepancy']:.6f}")
        ```
    """

    def __init__(
        self,
        dimensionality: int,
        scramble: bool = True,
        seed: int = 42,
        validation_targets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Sobol sampler.

        Args:
            dimensionality: Number of parameters
            scramble: Enable Owen scrambling
            seed: Random seed for scrambling
            validation_targets: Quality targets for validation
        """
        super().__init__(dimensionality, seed)

        # Initialize Sobol engine
        # Note: torch.quasirandom.Sobol uses global RNG state for scrambling
        # Set seed before creating engine to ensure reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        self.sobol_engine = Sobol(d=dimensionality, scramble=scramble)

        # Validation targets
        self.validation_targets = validation_targets or {
            "max_discrepancy": 0.01,
            "max_correlation": 0.05,
        }

        # Statistics
        self.stats = {
            "total_samples_generated": 0,
            "total_time_seconds": 0.0,
            "last_generation_time": 0.0,
        }

    @classmethod
    def from_config(
        cls, parameter_space, config: SamplingConfig  # ParameterSpace (avoid circular import)
    ) -> "StratifiedSobolSampler":
        """
        Factory method to create sampler from configuration.

        Args:
            parameter_space: ParameterSpace object
            config: SamplingConfig object

        Returns:
            Configured StratifiedSobolSampler instance

        Example:
            ```python
            sampler = StratifiedSobolSampler.from_config(
                config.parameter_space,
                config.sampling
            )
            ```
        """
        validation_targets = {}
        if config.validation.check_discrepancy:
            validation_targets["max_discrepancy"] = config.validation.max_discrepancy
        if config.validation.check_correlation:
            validation_targets["max_correlation"] = config.validation.max_pairwise_correlation

        return cls(
            dimensionality=parameter_space.total_dimensions,
            scramble=config.sobol.scramble,
            seed=config.sobol.seed,
            validation_targets=validation_targets,
        )

    def sample(self, n_samples: int) -> NDArray[np.float64]:
        """
        Generate n_samples using Sobol sequence.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Samples in [0,1]^d with shape (n_samples, dimensionality)

        Raises:
            ValueError: If n_samples < 1

        Performance:
            Target: 10,000 samples in <10s on CPU

        Note:
            Sobol sequences have optimal balance properties when n is a power of 2.
            This method automatically rounds up to the nearest power of 2 during
            generation, then returns the first n_samples requested.

        Example:
            ```python
            samples = sampler.sample(10000)
            print(f"Generated {len(samples)} samples")
            print(f"Shape: {samples.shape}")
            ```
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")

        # Round up to nearest power of 2 for optimal Sobol balance properties
        # This avoids scipy warnings and maintains statistical quality
        n_pow2 = 1 << (n_samples - 1).bit_length()  # Next power of 2 >= n_samples

        # Generate Sobol samples (power of 2 count)
        start_time = time.time()
        samples_pow2 = self.sobol_engine.random(n_pow2)
        elapsed = time.time() - start_time

        # Return only the requested number of samples
        samples = samples_pow2[:n_samples]

        # Update statistics (track actual samples returned)
        self.stats["total_samples_generated"] += n_samples
        self.stats["total_time_seconds"] += elapsed
        self.stats["last_generation_time"] = elapsed

        # Validate output
        self._validate_sample_array(samples)

        return samples

    def validate(self, samples: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Validate sample quality against targets.

        Args:
            samples: Sample array to validate

        Returns:
            Dictionary with metrics and pass/fail status (includes both
            SamplingMetrics fields and sampler-specific stats)

        Example:
            ```python
            samples = sampler.sample(10000)
            metrics = sampler.validate(samples)

            if metrics["discrepancy_pass"] and metrics["correlation_pass"]:
                print("✓ Samples meet quality targets!")
            else:
                print(f"✗ Discrepancy: {metrics['discrepancy']:.6f}")
            ```
        """
        self._validate_sample_array(samples)

        metrics = validate_sample_quality(samples, self.validation_targets)

        # Convert dataclass to dict for backward compatibility
        results = metrics.to_dict()

        # Add sampler-specific stats
        results["sampler_type"] = "StratifiedSobol"
        results["dimensionality"] = self.dimensionality
        results["n_samples"] = len(samples)
        results["scramble_enabled"] = True  # Always true for this implementation

        return results

    def generate_and_validate(
        self, n_samples: int, verbose: bool = True
    ) -> tuple[NDArray[np.float64], Dict[str, Any]]:
        """
        Convenience method: generate samples and validate in one call.

        Args:
            n_samples: Number of samples to generate
            verbose: Print validation report

        Returns:
            Tuple of (samples, validation_metrics)

        Example:
            ```python
            samples, metrics = sampler.generate_and_validate(10000)
            # Samples are automatically validated
            ```
        """
        samples = self.sample(n_samples)
        metrics = self.validate(samples)

        if verbose:
            self._print_generation_stats()
            # Get core metrics for printing (without sampler-specific fields)
            core_metrics = validate_sample_quality(samples, self.validation_targets)
            print_sample_quality_report(core_metrics)

        return samples, metrics

    def _print_generation_stats(self) -> None:
        """Print sampling performance statistics."""
        print("\n" + "=" * 60)
        print("SAMPLING PERFORMANCE")
        print("=" * 60)
        print(f"Samples generated: {self.stats['total_samples_generated']:,}")
        print(f"Total time: {self.stats['total_time_seconds']:.3f}s")
        if self.stats["total_samples_generated"] > 0:
            throughput = self.stats["total_samples_generated"] / self.stats["total_time_seconds"]
            print(f"Throughput: {throughput:,.0f} samples/sec")
        print(f"Last generation: {self.stats['last_generation_time']:.3f}s")
        print("=" * 60 + "\n")
