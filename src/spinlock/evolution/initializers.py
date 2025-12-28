"""
Initial condition sampling for temporal evolution.

Provides multiple strategies for generating initial states:
- Dataset: Load from existing HDF5 dataset inputs
- GRF: Generate Gaussian Random Fields
- Zeros: Zero initialization

Design follows the existing InputFieldGenerator pattern for consistency.
"""

import torch
from typing import Literal, Optional
from pathlib import Path


class InitialConditionSampler:
    """
    Sample initial conditions for temporal evolution.

    Supports multiple initialization strategies to enable different
    experimental workflows. All methods return GPU-resident tensors.

    Example:
        ```python
        # Initialize from dataset
        sampler = InitialConditionSampler(
            method="dataset",
            dataset_path=Path("dataset.h5")
        )
        X0 = sampler.sample(batch_size=10)  # [10, C, H, W]

        # Initialize from GRF
        sampler = InitialConditionSampler(
            method="grf",
            grid_size=64,
            num_channels=3
        )
        X0 = sampler.sample(batch_size=10, length_scale=0.1)
        ```
    """

    def __init__(
        self,
        method: Literal["dataset", "grf", "zeros"] = "grf",
        dataset_path: Optional[Path] = None,
        grid_size: int = 64,
        num_channels: int = 3,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize initial condition sampler.

        Args:
            method: Sampling method ("dataset", "grf", "zeros")
            dataset_path: Path to HDF5 dataset (required for "dataset" method)
            grid_size: Spatial grid size (H=W=grid_size)
            num_channels: Number of channels
            device: Torch device (cuda or cpu)

        Raises:
            ValueError: If method="dataset" but dataset_path is None
        """
        self.method = method
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.device = device

        if method == "dataset":
            if dataset_path is None:
                raise ValueError("dataset_path required for 'dataset' method")

            # Import here to avoid circular dependency
            from ..dataset.storage import HDF5DatasetReader

            self.dataset_reader = HDF5DatasetReader(dataset_path)
            self.dataset_reader.__enter__()

        elif method == "grf":
            # Import here to avoid circular dependency
            from ..dataset.generators import InputFieldGenerator

            self.generator = InputFieldGenerator(grid_size, num_channels, device)

    def sample(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample initial conditions.

        Args:
            batch_size: Number of initial conditions to sample
            seed: Random seed for reproducibility
            **kwargs: Method-specific parameters
                - For "grf": length_scale, variance
                - For "dataset": indices (optional)

        Returns:
            Initial conditions [B, C, H, W] on device

        Example:
            ```python
            # Sample from dataset
            X0 = sampler.sample(batch_size=10)

            # Sample GRF with specific parameters
            X0 = sampler.sample(
                batch_size=10,
                length_scale=0.15,
                variance=1.5,
                seed=42
            )

            # Zero initialization
            X0 = sampler.sample(batch_size=10)
            ```
        """
        if self.method == "dataset":
            return self._sample_from_dataset(batch_size, **kwargs)

        elif self.method == "grf":
            return self._sample_grf(batch_size, seed=seed, **kwargs)

        elif self.method == "zeros":
            return self._sample_zeros(batch_size)

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _sample_from_dataset(
        self,
        batch_size: int,
        indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample initial conditions from dataset inputs.

        Args:
            batch_size: Number of samples
            indices: Specific indices to sample (optional)

        Returns:
            Initial conditions [B, C, H, W]
        """
        if indices is None:
            # Sample random indices from dataset
            num_samples = self.dataset_reader.num_samples
            indices = torch.randint(0, num_samples, (batch_size,))

        # Read from dataset
        inputs = self.dataset_reader.get_inputs(indices.cpu().numpy())

        # Convert to tensor and move to device
        return torch.from_numpy(inputs).to(self.device).float()

    def _sample_grf(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        length_scale: float = 0.1,
        variance: float = 1.0
    ) -> torch.Tensor:
        """
        Generate Gaussian Random Fields as initial conditions.

        Args:
            batch_size: Number of GRF samples
            seed: Random seed
            length_scale: Correlation length scale
            variance: Field variance

        Returns:
            GRF samples [B, C, H, W]
        """
        return self.generator.generate_batch(
            batch_size=batch_size,
            field_type="gaussian_random_field",
            seed=seed,
            length_scale=length_scale,
            variance=variance
        )

    def _sample_zeros(self, batch_size: int) -> torch.Tensor:
        """
        Zero initialization.

        Args:
            batch_size: Number of samples

        Returns:
            Zero tensor [B, C, H, W]
        """
        return torch.zeros(
            batch_size,
            self.num_channels,
            self.grid_size,
            self.grid_size,
            device=self.device,
            dtype=torch.float32
        )

    def cleanup(self):
        """Release resources (e.g., close dataset reader)."""
        if hasattr(self, 'dataset_reader'):
            self.dataset_reader.__exit__(None, None, None)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
