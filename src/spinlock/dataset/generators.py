"""
Input field generators for neural operator datasets.

Provides synthetic input generation methods:
- Gaussian Random Fields (GRF) via spectral method
- Structured patterns (circles, stripes, blobs)
- Mixed generation strategies

Design principles:
- GPU-accelerated generation
- Reproducible with seeds
- Flexible parameterization
"""

from typing import Literal, Optional, Tuple

import torch
import numpy as np


class InputFieldGenerator:
    """
    Generate synthetic input fields for neural operator training.

    Supports multiple generation methods for diverse inputs.

    Example:
        ```python
        generator = InputFieldGenerator(
            grid_size=64,
            num_channels=3,
            device=torch.device("cuda")
        )

        # Generate Gaussian random fields
        fields = generator.generate_batch(
            batch_size=16,
            field_type="gaussian_random_field",
            length_scale=0.1
        )
        print(fields.shape)  # (16, 3, 64, 64)
        ```
    """

    def __init__(
        self, grid_size: int, num_channels: int, device: torch.device = torch.device("cpu")
    ):
        """
        Initialize input field generator.

        Args:
            grid_size: Spatial grid dimension (square grids)
            num_channels: Number of channels per field
            device: Torch device for generation
        """
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.device = device

        # Pre-compute k-space grid for GRF generation
        self._setup_fourier_grid()

    def _setup_fourier_grid(self) -> None:
        """Pre-compute Fourier space grid for efficient GRF generation."""
        # Frequency grid
        k = torch.fft.fftfreq(self.grid_size, d=1.0 / self.grid_size, device=self.device)
        kx, ky = torch.meshgrid(k, k, indexing="ij")
        self.k_squared = kx**2 + ky**2

    def generate_batch(
        self,
        batch_size: int,
        field_type: Literal[
            "gaussian_random_field", "random", "structured", "mixed"
        ] = "gaussian_random_field",
        seed: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a batch of input fields.

        Args:
            batch_size: Number of samples to generate
            field_type: Type of field to generate
                - "gaussian_random_field" or "random": GRF via spectral method
                - "structured": Geometric patterns (circles, stripes, blobs)
                - "mixed": Combination of GRF and structured
            seed: Random seed for reproducibility
            **kwargs: Additional parameters for specific generators

        Returns:
            Tensor of shape [B, C, H, W]

        Example:
            ```python
            fields = generator.generate_batch(
                batch_size=32,
                field_type="gaussian_random_field",
                length_scale=0.15,
                variance=1.5
            )
            ```
        """
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        # Normalize field type (accept "random" as alias for "gaussian_random_field")
        if field_type in ("gaussian_random_field", "random"):
            return self._generate_grf_batch(batch_size, **kwargs)
        elif field_type == "structured":
            return self._generate_structured_batch(batch_size, **kwargs)
        elif field_type == "mixed":
            return self._generate_mixed_batch(batch_size, **kwargs)
        else:
            raise ValueError(
                f"Unknown field type: {field_type}. "
                f"Must be one of: 'gaussian_random_field', 'random', 'structured', 'mixed'"
            )

    def _generate_grf_batch(
        self, batch_size: int, length_scale: float = 0.1, variance: float = 1.0
    ) -> torch.Tensor:
        """
        Generate Gaussian Random Fields using spectral method.

        GRFs have controlled correlation structure, useful for representing
        spatial processes with known length scales.

        Args:
            batch_size: Number of samples
            length_scale: Correlation length (0-1, fraction of domain)
            variance: Field variance

        Returns:
            Tensor [B, C, H, W]

        Algorithm:
            1. Generate random Fourier modes
            2. Apply power spectrum (Gaussian kernel in k-space)
            3. Inverse FFT to get real-space field

        Example:
            ```python
            # Short correlation length (rough fields)
            fields = generator._generate_grf_batch(16, length_scale=0.05)

            # Long correlation length (smooth fields)
            fields = generator._generate_grf_batch(16, length_scale=0.3)
            ```
        """
        # Power spectrum: Gaussian kernel in k-space
        # P(k) = variance * exp(-k^2 * length_scale^2 / 2)
        power_spectrum = variance * torch.exp(-self.k_squared * length_scale**2 / 2)

        fields = []
        for _ in range(batch_size):
            # Random complex Fourier modes for each channel
            # Real and imaginary parts are independent Gaussians
            fourier_modes = torch.randn(
                self.num_channels,
                self.grid_size,
                self.grid_size,
                device=self.device,
                dtype=torch.complex64,
            ) * torch.sqrt(power_spectrum).unsqueeze(0)

            # Inverse FFT to get real-space field
            field = torch.fft.ifft2(fourier_modes).real

            fields.append(field)

        return torch.stack(fields, dim=0)  # [B, C, H, W]

    def _generate_structured_batch(
        self, batch_size: int, num_structures: int = 5, structure_types: Optional[list] = None
    ) -> torch.Tensor:
        """
        Generate fields with structured geometric patterns.

        Useful for testing operators on non-random inputs.

        Args:
            batch_size: Number of samples
            num_structures: Number of geometric structures per field
            structure_types: List of structure types to use

        Returns:
            Tensor [B, C, H, W]

        Example:
            ```python
            fields = generator._generate_structured_batch(
                batch_size=8,
                num_structures=10,
                structure_types=["circle", "stripe"]
            )
            ```
        """
        if structure_types is None:
            structure_types = ["circle", "stripe", "blob"]

        fields = []

        for _ in range(batch_size):
            field = torch.zeros(
                self.num_channels, self.grid_size, self.grid_size, device=self.device
            )

            # Add random geometric structures
            for _ in range(num_structures):
                structure_type = np.random.choice(structure_types)

                if structure_type == "circle":
                    # Random circle
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size
                    radius = torch.rand(1, device=self.device) * self.grid_size * 0.2

                    # Distance grid
                    y, x = torch.meshgrid(
                        torch.arange(self.grid_size, device=self.device),
                        torch.arange(self.grid_size, device=self.device),
                        indexing="ij",
                    )
                    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    mask = (dist < radius).float()

                    # Add to random channel
                    channel = np.random.randint(0, self.num_channels)
                    field[channel] += mask * (torch.randn(1, device=self.device) * 2)

                elif structure_type == "stripe":
                    # Random stripe
                    angle = torch.rand(1, device=self.device) * 2 * np.pi
                    frequency = torch.rand(1, device=self.device) * 5 + 1

                    y, x = torch.meshgrid(
                        torch.arange(self.grid_size, device=self.device),
                        torch.arange(self.grid_size, device=self.device),
                        indexing="ij",
                    )
                    stripe = torch.sin(
                        (x * torch.cos(angle) + y * torch.sin(angle))
                        * frequency
                        * 2
                        * np.pi
                        / self.grid_size
                    )

                    channel = np.random.randint(0, self.num_channels)
                    field[channel] += stripe

                elif structure_type == "blob":
                    # Random blob (Gaussian)
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size
                    sigma = torch.rand(1, device=self.device) * self.grid_size * 0.15

                    y, x = torch.meshgrid(
                        torch.arange(self.grid_size, device=self.device),
                        torch.arange(self.grid_size, device=self.device),
                        indexing="ij",
                    )
                    blob = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

                    channel = np.random.randint(0, self.num_channels)
                    field[channel] += blob * (torch.randn(1, device=self.device) * 3)

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_mixed_batch(self, batch_size: int, **kwargs) -> torch.Tensor:
        """
        Mix of GRF and structured fields.

        Args:
            batch_size: Number of samples
            **kwargs: Parameters for sub-generators

        Returns:
            Tensor [B, C, H, W]
        """
        half = batch_size // 2

        grf = self._generate_grf_batch(half, **kwargs)
        structured = self._generate_structured_batch(batch_size - half, **kwargs)

        return torch.cat([grf, structured], dim=0)
