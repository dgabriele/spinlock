"""
Trajectory length sampling for variable-length VQ-VAE training.

This module provides utilities for sampling trajectory lengths during training,
enabling the model to learn from variable-length temporal sequences.
"""

from typing import List, Optional, Dict, Any
import torch
import numpy as np


class TrajectoryLengthSampler:
    """Sample trajectory lengths for variable-length training.

    Supports multiple sampling strategies:
    - uniform: Sample uniformly from [min_T, max_T]
    - geometric: Sample from geometric distribution (favors shorter or longer)
    - fixed_bins: Sample uniformly from predefined length bins

    Args:
        min_timesteps: Minimum trajectory length (inclusive)
        max_timesteps: Maximum trajectory length (inclusive)
        strategy: Sampling strategy ("uniform", "geometric", "fixed_bins")
        **kwargs: Additional strategy-specific parameters
            - geometric_decay: For "geometric" strategy (default: 0.01)
            - length_bins: For "fixed_bins" strategy (list of int)

    Examples:
        >>> # Uniform sampling
        >>> sampler = TrajectoryLengthSampler(min_timesteps=8, max_timesteps=500)
        >>> lengths = sampler.sample_lengths(batch_size=32)

        >>> # Geometric sampling (favor shorter trajectories)
        >>> sampler = TrajectoryLengthSampler(
        ...     min_timesteps=8, max_timesteps=500,
        ...     strategy="geometric", geometric_decay=0.02
        ... )
        >>> lengths = sampler.sample_lengths(batch_size=32)

        >>> # Fixed bins
        >>> sampler = TrajectoryLengthSampler(
        ...     min_timesteps=100, max_timesteps=500,
        ...     strategy="fixed_bins",
        ...     length_bins=[100, 200, 300, 400, 500]
        ... )
        >>> lengths = sampler.sample_lengths(batch_size=32)
    """

    def __init__(
        self,
        min_timesteps: int,
        max_timesteps: int,
        strategy: str = "uniform",
        **kwargs
    ):
        self.min_T = min_timesteps
        self.max_T = max_timesteps
        self.strategy = strategy

        # Validate basic constraints
        if min_timesteps < 1:
            raise ValueError(f"min_timesteps must be >= 1, got {min_timesteps}")

        if min_timesteps >= max_timesteps:
            raise ValueError(
                f"min_timesteps ({min_timesteps}) must be < max_timesteps ({max_timesteps})"
            )

        # Strategy-specific initialization
        if strategy == "geometric":
            self.decay = kwargs.get("geometric_decay", 0.01)
            if self.decay <= 0:
                raise ValueError(f"geometric_decay must be > 0, got {self.decay}")

        elif strategy == "fixed_bins":
            if "length_bins" not in kwargs:
                raise ValueError("fixed_bins strategy requires 'length_bins' parameter")

            self.bins = kwargs["length_bins"]

            if not self.bins:
                raise ValueError("length_bins cannot be empty")

            # Validate bins are within range
            for bin_length in self.bins:
                if bin_length < min_timesteps or bin_length > max_timesteps:
                    raise ValueError(
                        f"Bin length {bin_length} outside valid range "
                        f"[{min_timesteps}, {max_timesteps}]"
                    )

            # Weighted curriculum: per-bin probability weights (None = uniform)
            self._bin_weights: Optional[torch.Tensor] = None

        elif strategy == "uniform":
            # No additional parameters needed
            pass

        else:
            raise ValueError(
                f"Unknown sampling strategy: {strategy}. "
                f"Choose from: uniform, geometric, fixed_bins"
            )

    def sample_lengths(self, batch_size: int) -> torch.Tensor:
        """Sample trajectory lengths for a batch.

        Args:
            batch_size: Number of lengths to sample

        Returns:
            Tensor of shape [batch_size] with integer lengths in [min_T, max_T]

        Examples:
            >>> sampler = TrajectoryLengthSampler(8, 500)
            >>> lengths = sampler.sample_lengths(32)
            >>> assert lengths.shape == (32,)
            >>> assert (lengths >= 8).all() and (lengths <= 500).all()
        """
        if self.strategy == "uniform":
            return self._sample_uniform(batch_size)
        elif self.strategy == "geometric":
            return self._sample_geometric(batch_size)
        elif self.strategy == "fixed_bins":
            return self._sample_fixed_bins(batch_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _sample_uniform(self, batch_size: int) -> torch.Tensor:
        """Sample uniformly from [min_T, max_T]."""
        return torch.randint(
            self.min_T,
            self.max_T + 1,  # randint is exclusive of upper bound
            (batch_size,)
        )

    def _sample_geometric(self, batch_size: int) -> torch.Tensor:
        """Sample from geometric distribution.

        Uses exponential decay: p(t) ∝ exp(-decay * (t - min_T))

        With decay > 0:
        - Higher decay → more probability mass on shorter trajectories
        - Lower decay → more uniform distribution

        Uses inverse transform sampling from the exponential CDF.
        """
        # Generate uniform [0, 1) samples
        u = torch.rand(batch_size)

        # Inverse CDF of exponential distribution
        # F^{-1}(u) = min_T - (1/decay) * log(1 - u * (1 - exp(-decay * range)))
        range_size = self.max_T - self.min_T

        # Compute normalizing constant
        Z = 1.0 - torch.exp(torch.tensor(-self.decay * range_size))

        # Inverse transform
        lengths = self.min_T - (1.0 / self.decay) * torch.log(1.0 - u * Z)

        # Clamp and convert to integers
        return lengths.long().clamp(self.min_T, self.max_T)

    def set_bin_weights(self, weights: List[float]) -> None:
        """Set per-bin sampling probability weights.

        Args:
            weights: One weight per bin. Normalized to sum=1 internally.
                     All values must be >= 0 with at least one > 0.
        """
        if len(weights) != len(self.bins):
            raise ValueError(
                f"weights length ({len(weights)}) must match bins length ({len(self.bins)})"
            )
        w = torch.tensor(weights, dtype=torch.float32)
        total = w.sum()
        if total <= 0:
            raise ValueError("At least one weight must be > 0")
        self._bin_weights = w / total

    def _sample_fixed_bins(self, batch_size: int) -> torch.Tensor:
        """Sample from predefined bins, optionally weighted.

        When _bin_weights is set (weighted curriculum), uses torch.multinomial
        to sample bin indices with per-bin probabilities. Otherwise falls back
        to min_T-filtered uniform sampling (legacy curriculum).
        """
        if self._bin_weights is not None:
            # Weighted curriculum: all bins always accessible
            indices = torch.multinomial(self._bin_weights, batch_size, replacement=True)
            bins_tensor = torch.tensor(self.bins, dtype=torch.long)
            return bins_tensor[indices]

        # Legacy: uniform over bins >= min_T
        valid_bins = [b for b in self.bins if b >= self.min_T]
        if not valid_bins:
            # All bins below min_T — fall back to min_T itself
            return torch.full((batch_size,), self.min_T, dtype=torch.long)
        indices = torch.randint(0, len(valid_bins), (batch_size,))
        return torch.tensor([valid_bins[i] for i in indices], dtype=torch.long)

    def create_mask(
        self,
        lengths: torch.Tensor,
        max_T: Optional[int] = None
    ) -> torch.Tensor:
        """Create boolean mask from trajectory lengths.

        Args:
            lengths: Tensor of shape [B] with integer lengths
            max_T: Maximum sequence length. If None, uses max(lengths)

        Returns:
            Boolean mask of shape [B, max_T] where mask[i, t] = (t < lengths[i])

        Examples:
            >>> sampler = TrajectoryLengthSampler(8, 500)
            >>> lengths = torch.tensor([10, 20, 15])
            >>> mask = sampler.create_mask(lengths, max_T=25)
            >>> assert mask.shape == (3, 25)
            >>> assert mask[0, :10].all() and not mask[0, 10:].any()
            >>> assert mask[1, :20].all() and not mask[1, 20:].any()
        """
        B = lengths.shape[0]

        if max_T is None:
            max_T = int(lengths.max().item())

        # Create position indices [1, max_T]
        positions = torch.arange(max_T, device=lengths.device).unsqueeze(0)  # [1, max_T]

        # Broadcast comparison: [B, 1] vs [1, max_T] -> [B, max_T]
        mask = positions < lengths.unsqueeze(1)  # [B, max_T]

        return mask

    def get_stats(self, num_samples: int = 10000) -> Dict[str, float]:
        """Compute statistics of the length distribution.

        Args:
            num_samples: Number of samples to draw for statistics

        Returns:
            Dictionary with mean, std, min, max, median
        """
        lengths = self.sample_lengths(num_samples).float()

        return {
            "mean": lengths.mean().item(),
            "std": lengths.std().item(),
            "min": lengths.min().item(),
            "max": lengths.max().item(),
            "median": lengths.median().item(),
        }

    def __repr__(self) -> str:
        """String representation."""
        base = (
            f"TrajectoryLengthSampler("
            f"min_T={self.min_T}, max_T={self.max_T}, "
            f"strategy='{self.strategy}'"
        )

        if self.strategy == "geometric":
            base += f", decay={self.decay}"
        elif self.strategy == "fixed_bins":
            base += f", bins={self.bins}"

        base += ")"
        return base


class LengthCurriculumScheduler:
    """Gradually decrease min_timesteps over training epochs.

    Implements curriculum learning by starting with longer trajectories
    and gradually allowing shorter ones as training progresses.

    Args:
        start_min: Initial minimum length
        end_min: Final minimum length
        total_epochs: Number of epochs for curriculum
        schedule: Schedule type ("linear", "cosine", "exponential")

    Examples:
        >>> # Linear curriculum: 400 -> 8 over 200 epochs
        >>> scheduler = LengthCurriculumScheduler(
        ...     start_min=400, end_min=8, total_epochs=200
        ... )
        >>> scheduler.get_min_timesteps(epoch=0)   # 400
        >>> scheduler.get_min_timesteps(epoch=100) # ~200
        >>> scheduler.get_min_timesteps(epoch=200) # 8
    """

    def __init__(
        self,
        start_min: int,
        end_min: int,
        total_epochs: int,
        schedule: str = "linear"
    ):
        self.start_min = start_min
        self.end_min = end_min
        self.total_epochs = total_epochs
        self.schedule = schedule

        if start_min <= end_min:
            raise ValueError(
                f"start_min ({start_min}) must be > end_min ({end_min})"
            )

        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")

    def get_min_timesteps(self, epoch: int) -> int:
        """Get minimum timesteps for current epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Minimum timesteps for this epoch
        """
        if epoch >= self.total_epochs:
            return self.end_min

        # Compute progress [0, 1]
        progress = epoch / self.total_epochs

        if self.schedule == "linear":
            alpha = progress
        elif self.schedule == "cosine":
            # Cosine annealing
            alpha = (1 - np.cos(np.pi * progress)) / 2
        elif self.schedule == "exponential":
            # Exponential decay
            alpha = 1 - np.exp(-5 * progress)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        # Interpolate
        min_T = self.start_min - alpha * (self.start_min - self.end_min)

        return int(min_T)

    def __repr__(self) -> str:
        return (
            f"LengthCurriculumScheduler("
            f"{self.start_min} -> {self.end_min} over {self.total_epochs} epochs, "
            f"schedule='{self.schedule}')"
        )
