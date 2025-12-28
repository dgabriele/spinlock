"""
Input field generators for neural operator datasets.

Provides diverse synthetic input generation methods:
- Gaussian Random Fields (GRF) via spectral method
- Multi-scale GRF (superposition of multiple length scales)
- Localized features (sparse Gaussian blobs)
- Composite fields (structured patterns + noise perturbations)
- Heavy-tailed distributions (power-law spectrum)
- Structured patterns (circles, stripes, blobs)
- Mixed generation strategies

Design principles:
- GPU-accelerated generation
- Reproducible with seeds
- Flexible parameterization
- Discovery-focused diversity (not benchmarking)
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
            "gaussian_random_field", "random", "structured", "mixed",
            "multiscale_grf", "localized", "composite", "heavy_tailed"  # NEW
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
        elif field_type == "multiscale_grf":
            return self._generate_multiscale_grf(batch_size, **kwargs)
        elif field_type == "localized":
            return self._generate_localized_features(batch_size, **kwargs)
        elif field_type == "composite":
            return self._generate_composite_field(batch_size, **kwargs)
        elif field_type == "heavy_tailed":
            return self._generate_heavy_tailed(batch_size, **kwargs)
        else:
            raise ValueError(
                f"Unknown field type: {field_type}. "
                f"Must be one of: 'gaussian_random_field', 'random', 'structured', 'mixed', "
                f"'multiscale_grf', 'localized', 'composite', 'heavy_tailed'"
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

    def _generate_multiscale_grf(
        self,
        batch_size: int,
        scales: Optional[list] = None,
        variance: float = 1.0
    ) -> torch.Tensor:
        """
        Generate multi-scale Gaussian Random Fields.

        Superposition of GRFs at different length scales to create
        rich multi-scale spatial structure.

        Args:
            batch_size: Number of samples
            scales: List of length scales (default: [0.02, 0.05, 0.1, 0.2, 0.4])
            variance: Total variance (distributed across scales)

        Returns:
            Tensor [B, C, H, W]

        Example:
            ```python
            # 5 scales from fine to coarse
            fields = generator._generate_multiscale_grf(
                batch_size=16,
                scales=[0.02, 0.05, 0.1, 0.2, 0.4]
            )
            ```
        """
        if scales is None:
            scales = [0.02, 0.05, 0.1, 0.2, 0.4]

        # Distribute variance equally across scales
        scale_variance = variance / len(scales)

        # Initialize accumulator
        field = torch.zeros(
            batch_size, self.num_channels, self.grid_size, self.grid_size,
            device=self.device
        )

        # Add contribution from each scale
        for scale in scales:
            field += self._generate_grf_batch(
                batch_size,
                length_scale=scale,
                variance=scale_variance
            )

        return field

    def _generate_localized_features(
        self,
        batch_size: int,
        num_blobs: int = 5,
        min_width: float = 5.0,
        max_width: float = 15.0
    ) -> torch.Tensor:
        """
        Generate fields with localized Gaussian blobs.

        Creates sparse activation patterns with randomly placed
        Gaussian features. Useful for testing locality assumptions.

        Args:
            batch_size: Number of samples
            num_blobs: Number of Gaussian blobs per field
            min_width: Minimum blob width (pixels)
            max_width: Maximum blob width (pixels)

        Returns:
            Tensor [B, C, H, W]

        Example:
            ```python
            # Sparse localized features
            fields = generator._generate_localized_features(
                batch_size=16,
                num_blobs=3,
                min_width=8.0,
                max_width=20.0
            )
            ```
        """
        fields = []

        for _ in range(batch_size):
            field = torch.zeros(
                self.num_channels, self.grid_size, self.grid_size,
                device=self.device
            )

            # Create coordinate grids (cached for efficiency)
            y, x = torch.meshgrid(
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                indexing="ij"
            )

            # Add random Gaussian blobs
            for _ in range(num_blobs):
                # Random position (periodic boundary conditions)
                cx = torch.rand(1, device=self.device) * self.grid_size
                cy = torch.rand(1, device=self.device) * self.grid_size

                # Random width
                sigma = (
                    torch.rand(1, device=self.device) * (max_width - min_width) + min_width
                )

                # Random amplitude (per channel)
                amp = torch.randn(self.num_channels, 1, 1, device=self.device) * 2.0

                # Gaussian blob
                blob = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

                # Add to all channels with different amplitudes
                field += amp * blob.unsqueeze(0)

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_composite_field(
        self,
        batch_size: int,
        pattern: str = "waves",
        noise_level: float = 0.1,
        grf_length_scale: float = 0.05
    ) -> torch.Tensor:
        """
        Generate composite fields: structured background + random perturbations.

        Combines deterministic structure with stochastic noise to create
        order-to-chaos transitions and symmetry-breaking dynamics.

        Args:
            batch_size: Number of samples
            pattern: Background pattern type ("waves", "grid", "radial")
            noise_level: GRF noise amplitude (fraction of pattern amplitude)
            grf_length_scale: Length scale for noise perturbations

        Returns:
            Tensor [B, C, H, W]

        Example:
            ```python
            # Sine waves + small perturbations
            fields = generator._generate_composite_field(
                batch_size=16,
                pattern="waves",
                noise_level=0.15
            )
            ```
        """
        # Generate structured background
        if pattern == "waves":
            # Sine wave pattern with random orientation
            y, x = torch.meshgrid(
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                indexing="ij"
            )

            base_patterns = []
            for _ in range(batch_size):
                # Random wave direction and frequency
                angle = torch.rand(1, device=self.device) * 2 * np.pi
                freq = torch.rand(1, device=self.device) * 4 + 2  # frequency 2-6

                pattern_field = torch.sin(
                    (x * torch.cos(angle) + y * torch.sin(angle))
                    * freq * 2 * np.pi / self.grid_size
                )

                # Replicate across channels
                base_patterns.append(pattern_field.unsqueeze(0).repeat(self.num_channels, 1, 1))

            base = torch.stack(base_patterns, dim=0)

        elif pattern == "grid":
            # Grid pattern
            base_patterns = []
            for _ in range(batch_size):
                y, x = torch.meshgrid(
                    torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                    torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                    indexing="ij"
                )
                freq = torch.rand(1, device=self.device) * 3 + 3  # 3-6 grid cells
                pattern_field = (
                    torch.sin(x * freq * 2 * np.pi / self.grid_size) *
                    torch.sin(y * freq * 2 * np.pi / self.grid_size)
                )
                base_patterns.append(pattern_field.unsqueeze(0).repeat(self.num_channels, 1, 1))

            base = torch.stack(base_patterns, dim=0)

        elif pattern == "radial":
            # Radial pattern from center
            y, x = torch.meshgrid(
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
                indexing="ij"
            )
            cx, cy = self.grid_size / 2, self.grid_size / 2
            r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            base_patterns = []
            for _ in range(batch_size):
                freq = torch.rand(1, device=self.device) * 4 + 2
                pattern_field = torch.sin(r * freq * 2 * np.pi / self.grid_size)
                base_patterns.append(pattern_field.unsqueeze(0).repeat(self.num_channels, 1, 1))

            base = torch.stack(base_patterns, dim=0)

        else:
            # Default to waves
            base = self._generate_composite_field(
                batch_size, pattern="waves", noise_level=noise_level
            )

        # Add GRF noise perturbation
        noise = self._generate_grf_batch(
            batch_size,
            length_scale=grf_length_scale,
            variance=noise_level ** 2
        )

        return base + noise

    def _generate_heavy_tailed(
        self,
        batch_size: int,
        alpha: float = 1.5,
        variance: float = 1.0
    ) -> torch.Tensor:
        """
        Generate heavy-tailed random fields using power-law spectrum.

        Creates fields with extreme values and long-range correlations,
        useful for testing robustness to non-Gaussian statistics.

        Args:
            batch_size: Number of samples
            alpha: Power-law exponent (smaller = heavier tails)
                   typical range: [1.0, 2.0]
            variance: Field variance

        Returns:
            Tensor [B, C, H, W]

        Example:
            ```python
            # Very heavy tails (alpha=1.2)
            fields = generator._generate_heavy_tailed(
                batch_size=16,
                alpha=1.2
            )
            ```

        Algorithm:
            Uses power-law spectrum P(k) ∝ k^(-alpha) in Fourier space,
            which produces heavy-tailed spatial statistics via inverse FFT.
        """
        # Power-law spectrum: P(k) = variance * k^(-alpha)
        # Avoid division by zero at k=0
        k_magnitude = torch.sqrt(self.k_squared + 1e-10)
        power_spectrum = variance * torch.pow(k_magnitude, -alpha)

        # Normalize to preserve variance
        power_spectrum = power_spectrum / power_spectrum.sum() * (self.grid_size ** 2) * variance

        fields = []
        for _ in range(batch_size):
            # Random complex Fourier modes
            fourier_modes = torch.randn(
                self.num_channels,
                self.grid_size,
                self.grid_size,
                device=self.device,
                dtype=torch.complex64
            ) * torch.sqrt(power_spectrum).unsqueeze(0)

            # Inverse FFT to get real-space field
            field = torch.fft.ifft2(fourier_modes).real

            fields.append(field)

        return torch.stack(fields, dim=0)
