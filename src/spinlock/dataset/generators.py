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

from typing import Literal, Optional, Tuple, List

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

        # OPTIMIZATION: Pre-compute coordinate grids (cached once for all generations)
        self._setup_coordinate_grids()

        # Pre-compute k-space grid for GRF generation
        self._setup_fourier_grid()

    def _setup_coordinate_grids(self) -> None:
        """Pre-compute spatial coordinate grids for efficient field generation."""
        # Cartesian coordinates: x, y in [0, grid_size-1]
        y, x = torch.meshgrid(
            torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
            torch.arange(self.grid_size, device=self.device, dtype=torch.float32),
            indexing="ij"
        )
        self._cached_x = x
        self._cached_y = y

        # Center coordinates (origin at grid center)
        center = (self.grid_size - 1) / 2.0
        x_centered = x - center
        y_centered = y - center

        # Polar coordinates (relative to grid center)
        self._cached_r = torch.sqrt(x_centered**2 + y_centered**2)
        self._cached_theta = torch.atan2(y_centered, x_centered)

        # Cache centered coordinates for convenience
        self._cached_x_centered = x_centered
        self._cached_y_centered = y_centered

    def _setup_fourier_grid(self) -> None:
        """Pre-compute Fourier space grid for efficient GRF generation."""
        # Frequency grids for spectral methods
        kx = torch.fft.fftfreq(self.grid_size, d=1.0, device=self.device)
        ky = torch.fft.fftfreq(self.grid_size, d=1.0, device=self.device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        # Cache individual frequency grids
        self._cached_kx = kx_grid
        self._cached_ky = ky_grid
        self._cached_k = torch.sqrt(kx_grid**2 + ky_grid**2)

        # Legacy compatibility: k_squared for old code
        self.k_squared = kx_grid**2 + ky_grid**2

    def generate_batch(
        self,
        batch_size: int,
        field_type: Literal[
            "gaussian_random_field", "random", "structured", "mixed",
            "multiscale_grf", "localized", "composite", "heavy_tailed",
            # Tier 1 domain-specific ICs
            "quantum_wave_packet", "turing_pattern", "thermal_gradient",
            "morphogen_gradient", "reaction_front",
            # Tier 2 domain-specific ICs
            "light_cone", "critical_fluctuation", "phase_boundary",
            "bz_reaction", "shannon_entropy",
            # Tier 3 domain-specific ICs
            "interference_pattern", "cell_population", "chromatin_domain",
            "shock_front", "gene_expression",
            # Tier 4 research frontiers ICs
            "coherent_state", "relativistic_wave_packet", "mutual_information",
            "regulatory_network", "dla_cluster", "error_correcting_code"
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
        # Tier 1 domain-specific ICs
        elif field_type == "quantum_wave_packet":
            return self._generate_quantum_wave_packet(batch_size, **kwargs)
        elif field_type == "turing_pattern":
            return self._generate_turing_pattern(batch_size, **kwargs)
        elif field_type == "thermal_gradient":
            return self._generate_thermal_gradient(batch_size, **kwargs)
        elif field_type == "morphogen_gradient":
            return self._generate_morphogen_gradient(batch_size, **kwargs)
        elif field_type == "reaction_front":
            return self._generate_reaction_front(batch_size, **kwargs)
        # Tier 2 domain-specific ICs
        elif field_type == "light_cone":
            return self._generate_light_cone(batch_size, **kwargs)
        elif field_type == "critical_fluctuation":
            return self._generate_critical_fluctuation(batch_size, **kwargs)
        elif field_type == "phase_boundary":
            return self._generate_phase_boundary(batch_size, **kwargs)
        elif field_type == "bz_reaction":
            return self._generate_bz_reaction(batch_size, **kwargs)
        elif field_type == "shannon_entropy":
            return self._generate_shannon_entropy(batch_size, **kwargs)
        # Tier 3 domain-specific ICs
        elif field_type == "interference_pattern":
            return self._generate_interference_pattern(batch_size, **kwargs)
        elif field_type == "cell_population":
            return self._generate_cell_population(batch_size, **kwargs)
        elif field_type == "chromatin_domain":
            return self._generate_chromatin_domain(batch_size, **kwargs)
        elif field_type == "shock_front":
            return self._generate_shock_front(batch_size, **kwargs)
        elif field_type == "gene_expression":
            return self._generate_gene_expression(batch_size, **kwargs)
        # Tier 4 research frontiers ICs
        elif field_type == "coherent_state":
            return self._generate_coherent_state(batch_size, **kwargs)
        elif field_type == "relativistic_wave_packet":
            return self._generate_relativistic_wave_packet(batch_size, **kwargs)
        elif field_type == "mutual_information":
            return self._generate_mutual_information(batch_size, **kwargs)
        elif field_type == "regulatory_network":
            return self._generate_regulatory_network(batch_size, **kwargs)
        elif field_type == "dla_cluster":
            return self._generate_dla_cluster(batch_size, **kwargs)
        elif field_type == "error_correcting_code":
            return self._generate_error_correcting_code(batch_size, **kwargs)
        else:
            raise ValueError(
                f"Unknown field type: {field_type}. "
                f"Must be one of: 'gaussian_random_field', 'random', 'structured', 'mixed', "
                f"'multiscale_grf', 'localized', 'composite', 'heavy_tailed', "
                f"'quantum_wave_packet', 'turing_pattern', 'thermal_gradient', "
                f"'morphogen_gradient', 'reaction_front', 'light_cone', 'critical_fluctuation', "
                f"'phase_boundary', 'bz_reaction', 'shannon_entropy', 'interference_pattern', "
                f"'cell_population', 'chromatin_domain', 'shock_front', 'gene_expression', "
                f"'coherent_state', 'relativistic_wave_packet', 'mutual_information', "
                f"'regulatory_network', 'dla_cluster', 'error_correcting_code'"
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
                    # OPTIMIZATION: Use cached coordinate grids
                    x = self._cached_x
                    y = self._cached_y
                    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    mask = (dist < radius).float()

                    # Add to random channel
                    channel = np.random.randint(0, self.num_channels)
                    field[channel] += mask * (torch.randn(1, device=self.device) * 2)

                elif structure_type == "stripe":
                    # Random stripe
                    angle = torch.rand(1, device=self.device) * 2 * np.pi
                    frequency = torch.rand(1, device=self.device) * 5 + 1

                    # OPTIMIZATION: Use cached coordinate grids
                    x = self._cached_x
                    y = self._cached_y
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

                    # OPTIMIZATION: Use cached coordinate grids
                    x = self._cached_x
                    y = self._cached_y
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

            # OPTIMIZATION: Use cached coordinate grids
            x = self._cached_x
            y = self._cached_y

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
            # OPTIMIZATION: Use cached coordinate grids
            x = self._cached_x
            y = self._cached_y

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
            # OPTIMIZATION: Use cached coordinate grids (moved outside loop)
            x = self._cached_x
            y = self._cached_y
            base_patterns = []
            for _ in range(batch_size):
                freq = torch.rand(1, device=self.device) * 3 + 3  # 3-6 grid cells
                pattern_field = (
                    torch.sin(x * freq * 2 * np.pi / self.grid_size) *
                    torch.sin(y * freq * 2 * np.pi / self.grid_size)
                )
                base_patterns.append(pattern_field.unsqueeze(0).repeat(self.num_channels, 1, 1))

            base = torch.stack(base_patterns, dim=0)

        elif pattern == "radial":
            # Radial pattern from center
            # OPTIMIZATION: Use cached coordinate grids
            x = self._cached_x
            y = self._cached_y
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

    # ========================================================================
    # DOMAIN-SPECIFIC INITIAL CONDITIONS (Tier 1)
    # ========================================================================

    def _generate_quantum_wave_packet(
        self,
        batch_size: int,
        sigma: float = 10.0,
        momentum_range: Tuple[float, float] = (0.1, 0.5),
        num_packets: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate quantum wave packet initial conditions.

        Creates minimal-uncertainty Gaussian wave packets fundamental to
        quantum mechanics. Useful for testing wave propagation, dispersion,
        and interference dynamics.

        Physics: Gaussian envelope with plane wave modulation
            psi(x,y) = exp(i*k·r) * exp(-r^2/(4*sigma^2))

        Args:
            batch_size: Number of samples
            sigma: Wave packet width (position uncertainty, in pixels)
            momentum_range: (k_min, k_max) wave vector magnitude range
            num_packets: Number of independent packets per field

        Returns:
            Tensor [B, 3, H, W] with channels:
                - Channel 0: Real part of wavefunction
                - Channel 1: Imaginary part of wavefunction
                - Channel 2: Probability density |psi|^2

        Expected dynamics: Spreading, interference, dispersion
        Cross-domain: Optical wave packets, information encoding

        Example:
            ```python
            # Single wave packet with moderate momentum
            fields = generator._generate_quantum_wave_packet(
                batch_size=16,
                sigma=12.0,
                momentum_range=(0.2, 0.3),
                num_packets=1
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Initialize field (3 channels: real, imag, |psi|^2)
            field = torch.zeros(3, self.grid_size, self.grid_size, device=self.device)

            for _ in range(num_packets):
                # Random wave packet center
                cx = torch.rand(1, device=self.device) * self.grid_size
                cy = torch.rand(1, device=self.device) * self.grid_size

                # Random momentum (wave vector)
                k_magnitude = (
                    torch.rand(1, device=self.device) * (momentum_range[1] - momentum_range[0])
                    + momentum_range[0]
                )
                k_angle = torch.rand(1, device=self.device) * 2 * np.pi
                k_x = k_magnitude * torch.cos(k_angle)
                k_y = k_magnitude * torch.sin(k_angle)

                # Gaussian envelope
                r_squared = (x - cx) ** 2 + (y - cy) ** 2
                envelope = torch.exp(-r_squared / (4 * sigma ** 2))

                # Phase: k·r
                phase = k_x * (x - cx) + k_y * (y - cy)

                # Wave packet: psi = envelope * exp(i * phase)
                real_part = envelope * torch.cos(phase)
                imag_part = envelope * torch.sin(phase)
                prob_density = envelope ** 2  # |psi|^2

                # Accumulate (for multiple packets)
                field[0] += real_part
                field[1] += imag_part
                field[2] += prob_density

            # Normalize probability density to [0, 1]
            if field[2].max() > 0:
                field[2] = field[2] / field[2].max()

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_turing_pattern(
        self,
        batch_size: int,
        pattern_type: str = "spots",
        wavelength: float = 16.0,
        perturbation_amplitude: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate Turing pattern initial conditions.

        Creates reaction-diffusion patterns via spectral method. Turing patterns
        arise from instabilities in activator-inhibitor systems and appear in
        morphogenesis (animal coat patterns, skin pigmentation).

        Physics: Wavelength-selected patterns from linear stability analysis
            - Spots: Hexagonal lattice (3 wave vectors at 60°)
            - Stripes: Single wave vector
            - Labyrinth: Multiple interfering wave vectors

        Args:
            batch_size: Number of samples
            pattern_type: "spots", "stripes", "labyrinth", or "mixed"
            wavelength: Pattern wavelength (pixels, corresponds to 2π/k_c)
            perturbation_amplitude: Initial noise level

        Returns:
            Tensor [B, C, H, W] with pattern seeded at critical wavelength

        Expected dynamics: Pattern formation, selection, defect dynamics
        Cross-domain: BZ reactions (chemistry), convection patterns

        Example:
            ```python
            # Hexagonal spot patterns
            fields = generator._generate_turing_pattern(
                batch_size=16,
                pattern_type="spots",
                wavelength=20.0
            )
            ```
        """
        fields = []

        # Critical wave number
        k_c = 2 * np.pi / wavelength

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Determine pattern type for this sample
            if pattern_type == "mixed":
                sample_pattern = np.random.choice(["spots", "stripes", "labyrinth"])
            else:
                sample_pattern = pattern_type

            if sample_pattern == "spots":
                # Hexagonal spots: 3 wave vectors at 60° apart
                angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
                pattern_field = torch.zeros(self.grid_size, self.grid_size, device=self.device)

                for angle in angles:
                    k_x = k_c * np.cos(angle)
                    k_y = k_c * np.sin(angle)
                    pattern_field += torch.cos(k_x * x + k_y * y)

                # Normalize
                pattern_field = pattern_field / 3.0

            elif sample_pattern == "stripes":
                # Single wave vector with random orientation
                angle = torch.rand(1, device=self.device).item() * 2 * np.pi
                k_x = k_c * np.cos(angle)
                k_y = k_c * np.sin(angle)
                pattern_field = torch.cos(k_x * x + k_y * y)

            elif sample_pattern == "labyrinth":
                # Multiple interfering wave vectors (2-4 directions)
                num_waves = np.random.randint(2, 5)
                angles = torch.rand(num_waves, device=self.device) * 2 * np.pi
                pattern_field = torch.zeros(self.grid_size, self.grid_size, device=self.device)

                for angle in angles:
                    k_x = k_c * torch.cos(angle)
                    k_y = k_c * torch.sin(angle)
                    pattern_field += torch.cos(k_x * x + k_y * y)

                # Normalize
                pattern_field = pattern_field / num_waves

            else:
                raise ValueError(f"Unknown pattern_type: {sample_pattern}")

            # Add small perturbations (GRF noise)
            noise = self._generate_grf_batch(
                batch_size=1,
                length_scale=wavelength / self.grid_size,
                variance=perturbation_amplitude ** 2
            )[0]  # [C, H, W]

            # Combine pattern + noise for all channels
            field = pattern_field.unsqueeze(0).repeat(self.num_channels, 1, 1) + noise

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_thermal_gradient(
        self,
        batch_size: int,
        gradient_direction: str = "random",
        temperature_range: Tuple[float, float] = (0.0, 1.0),
        beta: float = 1.0,
        thermal_noise_amplitude: float = 0.1,
        thermal_length_scale: float = 5.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate thermal gradient initial conditions.

        Creates temperature gradients that drive heat flow via Fourier's law.
        Useful for testing diffusion dynamics and equilibration.

        Physics: Linear or power-law gradient + thermal fluctuations
            T(x) = T_min + (T_max - T_min) * (x/L)^beta + fluctuations

        Args:
            batch_size: Number of samples
            gradient_direction: "x", "y", "diagonal", or "random"
            temperature_range: (T_min, T_max) temperature bounds
            beta: Gradient exponent (1=linear, <1=concave, >1=convex)
            thermal_noise_amplitude: Fluctuation strength
            thermal_length_scale: Correlation length of fluctuations (pixels)

        Returns:
            Tensor [B, C, H, W] with temperature gradient

        Expected dynamics: Diffusion toward equilibrium, heat flow patterns
        Cross-domain: Concentration gradients (chemistry), morphogen gradients (biology)

        Example:
            ```python
            # Linear gradient along x with small fluctuations
            fields = generator._generate_thermal_gradient(
                batch_size=16,
                gradient_direction="x",
                beta=1.0,
                thermal_noise_amplitude=0.05
            )
            ```
        """
        T_min, T_max = temperature_range
        fields = []

        # OPTIMIZATION: Use cached coordinate grids (normalized to [0, 1])
        x = self._cached_x / self.grid_size
        y = self._cached_y / self.grid_size

        for _ in range(batch_size):
            # Determine gradient direction
            if gradient_direction == "random":
                direction = np.random.choice(["x", "y", "diagonal"])
            else:
                direction = gradient_direction

            # Create base gradient
            if direction == "x":
                gradient = x ** beta
            elif direction == "y":
                gradient = y ** beta
            elif direction == "diagonal":
                gradient = ((x + y) / 2) ** beta
            else:
                raise ValueError(f"Unknown gradient_direction: {direction}")

            # Scale to temperature range
            temperature = T_min + (T_max - T_min) * gradient

            # Add thermal fluctuations (GRF)
            noise = self._generate_grf_batch(
                batch_size=1,
                length_scale=thermal_length_scale / self.grid_size,
                variance=thermal_noise_amplitude ** 2
            )[0]  # [C, H, W]

            # Combine gradient + noise
            field = temperature.unsqueeze(0).repeat(self.num_channels, 1, 1) + noise

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_morphogen_gradient(
        self,
        batch_size: int,
        decay_length: float = 20.0,
        num_sources: int = 1,
        noise_level: float = 0.05,
        gradient_type: str = "exponential",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate morphogen gradient initial conditions.

        Creates signaling molecule gradients that specify cell fate in
        developmental biology. Morphogens diffuse from source and decay,
        forming concentration gradients.

        Physics: Exponential decay from point sources
            C(r) = C_0 * exp(-|r - r_source| / lambda) + noise
            where lambda = sqrt(D/k) (diffusion/degradation)

        Args:
            batch_size: Number of samples
            decay_length: Characteristic decay length lambda (pixels)
            num_sources: Number of morphogen sources (1-4)
            noise_level: Stochastic fluctuation amplitude
            gradient_type: "exponential", "power_law", or "sigmoidal"

        Returns:
            Tensor [B, C, H, W] with morphogen concentration

        Expected dynamics: Gradient maintenance, boundary refinement, noise filtering
        Cross-domain: Thermal gradients, chemical gradients, information channels

        Example:
            ```python
            # Single exponential gradient from random source
            fields = generator._generate_morphogen_gradient(
                batch_size=16,
                decay_length=25.0,
                num_sources=1,
                gradient_type="exponential"
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Initialize concentration field
            concentration = torch.zeros(self.grid_size, self.grid_size, device=self.device)

            # Random number of sources (if specified as range)
            n_sources = num_sources if isinstance(num_sources, int) else np.random.randint(1, num_sources + 1)

            for _ in range(n_sources):
                # Random source position
                source_x = torch.rand(1, device=self.device) * self.grid_size
                source_y = torch.rand(1, device=self.device) * self.grid_size

                # Distance from source
                r = torch.sqrt((x - source_x) ** 2 + (y - source_y) ** 2)

                # Random source strength
                C_0 = torch.rand(1, device=self.device) * 0.5 + 0.5  # [0.5, 1.0]

                # Apply gradient profile
                if gradient_type == "exponential":
                    profile = C_0 * torch.exp(-r / decay_length)
                elif gradient_type == "power_law":
                    # Avoid singularity at source
                    profile = C_0 / (1 + (r / decay_length) ** 2)
                elif gradient_type == "sigmoidal":
                    # Smooth step function
                    profile = C_0 / (1 + torch.exp((r - decay_length) / (decay_length * 0.2)))
                else:
                    raise ValueError(f"Unknown gradient_type: {gradient_type}")

                # Accumulate sources
                concentration += profile

            # Add noise fluctuations
            noise = self._generate_grf_batch(
                batch_size=1,
                length_scale=0.05,
                variance=noise_level ** 2
            )[0]  # [C, H, W]

            # Replicate concentration across channels and add noise
            field = concentration.unsqueeze(0).repeat(self.num_channels, 1, 1) + noise

            # Normalize to [0, 1]
            field = (field - field.min()) / (field.max() - field.min() + 1e-10)

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_reaction_front(
        self,
        batch_size: int,
        front_shape: str = "planar",
        front_width: float = 3.0,
        num_fronts: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate reaction front initial conditions.

        Creates propagating fronts in autocatalytic reactions (Fisher-KPP equation).
        Fronts separate reacted from unreacted regions and propagate at constant speed.

        Physics: Tanh/sigmoid profile
            C(x) = 1 / (1 + exp((x - x_front) / delta))

        Args:
            batch_size: Number of samples
            front_shape: "planar", "circular", or "irregular"
            front_width: Front thickness delta (pixels)
            num_fronts: Number of independent fronts (1-3)

        Returns:
            Tensor [B, C, H, W] with reaction front(s)

        Expected dynamics: Front propagation, front interactions, speed selection
        Cross-domain: Phase boundaries, shock waves, biological invasions

        Example:
            ```python
            # Single planar front
            fields = generator._generate_reaction_front(
                batch_size=16,
                front_shape="planar",
                front_width=4.0,
                num_fronts=1
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Initialize field
            field_val = torch.zeros(self.grid_size, self.grid_size, device=self.device)

            for _ in range(num_fronts):
                if front_shape == "planar":
                    # Random orientation
                    angle = torch.rand(1, device=self.device) * 2 * np.pi
                    # Random position along perpendicular axis
                    offset = (torch.rand(1, device=self.device) - 0.5) * self.grid_size * 0.5

                    # Distance along normal direction
                    dist = x * torch.cos(angle) + y * torch.sin(angle) - offset

                    # Tanh front profile
                    front = 0.5 * (1 + torch.tanh(dist / front_width))

                elif front_shape == "circular":
                    # Random center
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size

                    # Random radius
                    radius = (torch.rand(1, device=self.device) * 0.3 + 0.2) * self.grid_size

                    # Radial distance
                    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                    # Circular front (inside=1, outside=0)
                    front = 0.5 * (1 + torch.tanh((radius - r) / front_width))

                elif front_shape == "irregular":
                    # Irregular front via low-frequency GRF isoline
                    grf = self._generate_grf_batch(
                        batch_size=1,
                        length_scale=0.2,
                        variance=1.0
                    )[0, 0]  # [H, W]

                    # Threshold GRF to create irregular boundary
                    threshold = torch.rand(1, device=self.device) * 2 - 1  # [-1, 1]
                    front = 0.5 * (1 + torch.tanh((grf - threshold) / front_width))

                else:
                    raise ValueError(f"Unknown front_shape: {front_shape}")

                # Accumulate fronts (max to avoid overlap issues)
                field_val = torch.maximum(field_val, front)

            # Replicate across channels
            field = field_val.unsqueeze(0).repeat(self.num_channels, 1, 1)

            fields.append(field)

        return torch.stack(fields, dim=0)

    # ========================================================================
    # DOMAIN-SPECIFIC INITIAL CONDITIONS (Tier 2)
    # ========================================================================

    def _generate_light_cone(
        self,
        batch_size: int,
        cone_radius: float = 20.0,
        smoothing: float = 2.0,
        interior_length_scale: float = 5.0,
        num_cones: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate light cone initial conditions.

        Creates causal structure from special relativity - separates timelike
        from spacelike regions. Signals confined within light cone boundaries.

        Physics: Sigmoid-masked interior with GRF
            mask = sigmoid((cone_radius - r) / smoothing)
            field = mask * GRF(interior_length_scale)

        Args:
            batch_size: Number of samples
            cone_radius: Opening size (physical: c*t, in pixels)
            smoothing: Edge sharpness (pixels)
            interior_length_scale: GRF correlation length inside cone
            num_cones: Number of independent cones (1-3)

        Returns:
            Tensor [B, C, H, W] with cone-confined fields

        Expected dynamics: Operators preserve causal structure, signals don't escape cones
        Cross-domain: Morphogen diffusion boundaries, information propagation limits

        Example:
            ```python
            # Single light cone with sharp boundary
            fields = generator._generate_light_cone(
                batch_size=16,
                cone_radius=25.0,
                smoothing=1.5,
                num_cones=1
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Initialize field
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            for _ in range(num_cones):
                # Random cone center
                cx = torch.rand(1, device=self.device) * self.grid_size
                cy = torch.rand(1, device=self.device) * self.grid_size

                # Distance from center
                r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

                # Smooth cone mask (sigmoid transition)
                cone_mask = torch.sigmoid((cone_radius - r) / smoothing)

                # Generate GRF for interior
                interior = self._generate_grf_batch(
                    batch_size=1,
                    length_scale=interior_length_scale / self.grid_size,
                    variance=1.0
                )[0]  # [C, H, W]

                # Apply mask to confine field to cone
                masked_interior = cone_mask.unsqueeze(0) * interior

                # Accumulate cones
                field += masked_interior

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_critical_fluctuation(
        self,
        batch_size: int,
        correlation_length: float = 15.0,
        eta: float = 0.04,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate critical fluctuation initial conditions.

        Creates scale-free fluctuations near critical points via Ornstein-Zernike
        spectrum. Models universality and critical phenomena.

        Physics: Ornstein-Zernike power spectrum
            P(k) = A / (k^2 + xi^(-2))^(eta/2)
            where xi is correlation length, eta is anomalous dimension

        Args:
            batch_size: Number of samples
            correlation_length: Correlation length xi (pixels)
            eta: Anomalous dimension (0.04 for 3D Ising universality class)

        Returns:
            Tensor [B, C, H, W] with critical fluctuations

        Expected dynamics: Scale-invariant dynamics, critical slowing down
        Cross-domain: Power-law correlations in turbulence, 1/f noise, biological networks

        Example:
            ```python
            # Critical fluctuations with 3D Ising exponent
            fields = generator._generate_critical_fluctuation(
                batch_size=16,
                correlation_length=20.0,
                eta=0.04
            )
            ```
        """
        # Ornstein-Zernike spectrum: P(k) = A / (k^2 + xi^(-2))^(eta/2)
        xi_inv_sq = (1.0 / correlation_length) ** 2
        power_spectrum = 1.0 / torch.pow(self.k_squared + xi_inv_sq, eta / 2)

        # Normalize to preserve variance
        power_spectrum = power_spectrum / power_spectrum.sum() * (self.grid_size ** 2)

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

    def _generate_phase_boundary(
        self,
        batch_size: int,
        interface_width: float = 3.0,
        interface_angle: Optional[float] = None,
        fluctuation_amplitude: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate phase boundary initial conditions.

        Creates interfaces between phases (solid/liquid, ordered/disordered)
        via Allen-Cahn order parameter with fluctuations.

        Physics: Allen-Cahn order parameter
            phi = tanh((x - x_interface) / (sqrt(2) * xi)) + fluctuations

        Args:
            batch_size: Number of samples
            interface_width: Interface width xi (pixels)
            interface_angle: Interface orientation (radians, None=random)
            fluctuation_amplitude: Interface roughness amplitude

        Returns:
            Tensor [B, C, H, W] with phase boundaries

        Expected dynamics: Interface motion, coarsening, curvature-driven flow
        Cross-domain: Reaction fronts, cell membrane boundaries

        Example:
            ```python
            # Phase boundary with random orientation
            fields = generator._generate_phase_boundary(
                batch_size=16,
                interface_width=4.0,
                fluctuation_amplitude=0.15
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Random interface orientation
            if interface_angle is None:
                angle = torch.rand(1, device=self.device) * 2 * np.pi
            else:
                angle = torch.tensor(interface_angle, device=self.device)

            # Random interface position
            offset = (torch.rand(1, device=self.device) - 0.5) * self.grid_size * 0.5

            # Distance along normal to interface
            dist = x * torch.cos(angle) + y * torch.sin(angle) - offset

            # Base Allen-Cahn profile
            base_profile = torch.tanh(dist / (np.sqrt(2) * interface_width))

            # Add interface fluctuations (low-frequency GRF)
            fluctuations = self._generate_grf_batch(
                batch_size=1,
                length_scale=0.1,
                variance=fluctuation_amplitude ** 2
            )[0]  # [C, H, W]

            # Combine profile + fluctuations
            field = base_profile.unsqueeze(0).repeat(self.num_channels, 1, 1) + fluctuations

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_bz_reaction(
        self,
        batch_size: int,
        pattern_type: str = "spiral",
        num_spirals: int = 1,
        wavelength: float = 16.0,
        phase_offset: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate Belousov-Zhabotinsky (BZ) reaction initial conditions.

        Creates oscillatory chemical reaction patterns showing spiral waves.
        Paradigmatic example of non-equilibrium chemistry and excitable media.

        Physics: Spiral wave seeds in polar coordinates
            u = u_ss + A * cos(theta - k*r)
            v = v_ss + A * sin(theta - k*r + delta)

        Args:
            batch_size: Number of samples
            pattern_type: "spiral", "target", or "turbulent"
            num_spirals: Number of spiral cores (1-5)
            wavelength: Spiral wavelength (pixels)
            phase_offset: Phase difference between u and v fields

        Returns:
            Tensor [B, C, H, W] with BZ reaction patterns

        Expected dynamics: Spiral tip meandering, target waves, chemical turbulence
        Cross-domain: Cardiac spiral waves, excitable media in biology

        Example:
            ```python
            # Single spiral wave
            fields = generator._generate_bz_reaction(
                batch_size=16,
                pattern_type="spiral",
                num_spirals=1,
                wavelength=20.0
            )
            ```
        """
        fields = []

        # Wave number
        k = 2 * np.pi / wavelength

        # OPTIMIZATION: Use cached coordinate grids
        x_grid = self._cached_x
        y_grid = self._cached_y

        for _ in range(batch_size):
            # Initialize field
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            if pattern_type == "spiral":
                # Create spiral patterns
                for _ in range(num_spirals):
                    # Random spiral center
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size

                    # Polar coordinates from center
                    dx = x_grid - cx
                    dy = y_grid - cy
                    r = torch.sqrt(dx ** 2 + dy ** 2)
                    theta = torch.atan2(dy, dx)

                    # Random spiral handedness
                    handedness = 1.0 if torch.rand(1).item() > 0.5 else -1.0

                    # Spiral pattern: phase = theta - k*r
                    spiral_phase = handedness * theta - k * r

                    # Create u and v fields (activator and inhibitor)
                    u_spiral = torch.cos(spiral_phase)
                    v_spiral = torch.sin(spiral_phase + phase_offset)

                    # Add to channels (distribute across channels)
                    field[0] += u_spiral
                    if self.num_channels > 1:
                        field[1] += v_spiral
                    if self.num_channels > 2:
                        field[2] += (u_spiral + v_spiral) / 2

            elif pattern_type == "target":
                # Concentric target waves
                for _ in range(num_spirals):
                    # Random center
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size

                    # Radial distance
                    r = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

                    # Target pattern: pure radial dependence
                    target_pattern = torch.cos(k * r)

                    # Add to channels
                    field[0] += target_pattern
                    if self.num_channels > 1:
                        field[1] += torch.sin(k * r + phase_offset)
                    if self.num_channels > 2:
                        field[2] += target_pattern

            elif pattern_type == "turbulent":
                # Chemical turbulence: multiple interfering spirals
                num_spirals_turbulent = np.random.randint(3, 8)
                for _ in range(num_spirals_turbulent):
                    cx = torch.rand(1, device=self.device) * self.grid_size
                    cy = torch.rand(1, device=self.device) * self.grid_size

                    dx = x_grid - cx
                    dy = y_grid - cy
                    r = torch.sqrt(dx ** 2 + dy ** 2)
                    theta = torch.atan2(dy, dx)

                    handedness = 1.0 if torch.rand(1).item() > 0.5 else -1.0
                    spiral_phase = handedness * theta - k * r

                    field[0] += torch.cos(spiral_phase) / num_spirals_turbulent
                    if self.num_channels > 1:
                        field[1] += torch.sin(spiral_phase) / num_spirals_turbulent

            else:
                raise ValueError(f"Unknown pattern_type: {pattern_type}")

            # Normalize to reasonable range
            field = field / (field.abs().max() + 1e-10)

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_shannon_entropy(
        self,
        batch_size: int,
        entropy_pattern: str = "gradient",
        entropy_range: Tuple[float, float] = (0.1, 1.0),
        patch_size: int = 8,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate Shannon entropy field initial conditions.

        Creates spatially-varying noise levels where local entropy quantifies
        information content. High entropy = unpredictable, low entropy = ordered.

        Physics: Spatially-varying noise
            field = mu(x,y) + sigma(x,y) * noise
            where H ~ log(sigma) encodes local Shannon entropy

        Args:
            batch_size: Number of samples
            entropy_pattern: "gradient", "patchy", or "random"
            entropy_range: (low, high) entropy bounds (in bits)
            patch_size: Spatial scale for entropy variation (pixels)

        Returns:
            Tensor [B, C, H, W] with spatially-varying entropy

        Expected dynamics: Entropy transport, information loss/preservation
        Cross-domain: Thermodynamic entropy, compression quality

        Example:
            ```python
            # Entropy gradient from ordered to chaotic
            fields = generator._generate_shannon_entropy(
                batch_size=16,
                entropy_pattern="gradient",
                entropy_range=(0.1, 2.0)
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids (normalized to [0, 1])
        x = self._cached_x / self.grid_size
        y = self._cached_y / self.grid_size

        for _ in range(batch_size):
            # Create entropy pattern (controls local noise amplitude)
            if entropy_pattern == "gradient":
                # Linear gradient in entropy
                direction = np.random.choice(["x", "y", "diagonal"])
                if direction == "x":
                    entropy_map = x
                elif direction == "y":
                    entropy_map = y
                else:
                    entropy_map = (x + y) / 2

            elif entropy_pattern == "patchy":
                # Patchy entropy via low-frequency GRF
                entropy_base = self._generate_grf_batch(
                    batch_size=1,
                    length_scale=patch_size / self.grid_size,
                    variance=1.0
                )[0, 0]  # [H, W]
                # Normalize to [0, 1]
                entropy_map = (entropy_base - entropy_base.min()) / (entropy_base.max() - entropy_base.min() + 1e-10)

            elif entropy_pattern == "random":
                # Random entropy (nearly uniform)
                entropy_map = torch.rand(self.grid_size, self.grid_size, device=self.device)

            else:
                raise ValueError(f"Unknown entropy_pattern: {entropy_pattern}")

            # Scale to entropy range
            entropy_low, entropy_high = entropy_range
            local_entropy = entropy_low + (entropy_high - entropy_low) * entropy_map

            # Convert entropy to noise amplitude: sigma = exp(H)
            # (Higher entropy = larger sigma = more unpredictable)
            noise_amplitude = torch.exp(local_entropy)

            # Generate base signal (low-frequency structure)
            base_signal = self._generate_grf_batch(
                batch_size=1,
                length_scale=0.2,
                variance=0.5
            )[0]  # [C, H, W]

            # Generate spatially-varying noise
            noise = torch.randn(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            # Modulate noise by local entropy
            modulated_noise = noise * noise_amplitude.unsqueeze(0)

            # Combine base signal + spatially-varying noise
            field = base_signal + modulated_noise

            fields.append(field)

        return torch.stack(fields, dim=0)

    # =========================================================================
    # Tier 3 Domain-Specific Initial Conditions
    # =========================================================================

    def _generate_interference_pattern(
        self,
        batch_size: int,
        num_sources: int = 2,
        wavelength: float = 16.0,
        coherence_length: float = 100.0,
        source_spacing: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate interference pattern initial conditions.

        Creates wave interference patterns (double-slit, multi-slit) showing
        wave-particle duality. Used in quantum optics, acoustics, water waves.

        Physics: Multi-source interference
            field = sum_j cos(k * sqrt((x-x_j)^2 + (y-y_j)^2) + phase_j) * envelope
            where k = 2π/λ

        Args:
            batch_size: Number of samples
            num_sources: Number of coherent sources (2 = double-slit, etc.)
            wavelength: Wave wavelength (pixels)
            coherence_length: Coherence envelope scale
            source_spacing: Distance between sources (None = random)

        Returns:
            Tensor [B, C, H, W] showing interference fringes

        Expected dynamics: Fringe evolution, coherence decay, pattern stability
        Cross-domain: Acoustic interference, Moiré patterns in materials

        Example:
            ```python
            # Double-slit interference
            fields = generator._generate_interference_pattern(
                batch_size=16,
                num_sources=2,
                wavelength=20.0
            )
            ```
        """
        fields = []
        k = 2 * np.pi / wavelength

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            # Position sources
            if num_sources == 2 and source_spacing is not None:
                # Standard double-slit: centered, horizontal spacing
                center_x = self.grid_size / 2
                center_y = self.grid_size / 2
                source_positions = [
                    (center_x - source_spacing / 2, center_y),
                    (center_x + source_spacing / 2, center_y)
                ]
            else:
                # Random source positions
                source_positions = [
                    (
                        torch.rand(1, device=self.device).item() * self.grid_size * 0.8 + self.grid_size * 0.1,
                        torch.rand(1, device=self.device).item() * self.grid_size * 0.8 + self.grid_size * 0.1
                    )
                    for _ in range(num_sources)
                ]

            # Random phase offsets for each source
            phase_offsets = torch.rand(num_sources, device=self.device) * 2 * np.pi

            # Compute interference pattern
            for source_idx, (sx, sy) in enumerate(source_positions):
                # Distance from source
                r = torch.sqrt((x - sx) ** 2 + (y - sy) ** 2)

                # Spherical wave with coherence envelope
                wave = torch.cos(k * r + phase_offsets[source_idx])
                envelope = torch.exp(-r ** 2 / (2 * coherence_length ** 2))

                # Add to all channels (can have different phases per channel)
                for c in range(self.num_channels):
                    channel_phase = torch.rand(1, device=self.device).item() * 2 * np.pi
                    field[c] += wave * np.cos(channel_phase) * envelope

            # Normalize
            field = field / (field.abs().max() + 1e-10)

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_cell_population(
        self,
        batch_size: int,
        num_cells: int = 50,
        cell_radius: float = 3.0,
        clustering_strength: float = 0.5,
        kernel_width: float = 5.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate cell population initial conditions.

        Creates spatial cell distributions via point processes with kernel
        density estimation. Models epithelia, tumors, bacterial colonies.

        Physics: Point process → density field
            rho = sum_i K((r - r_i)/h) / h^2
            where K is a smoothing kernel

        Args:
            batch_size: Number of samples
            num_cells: Number of cells in the population
            cell_radius: Minimum cell separation (hard-core)
            clustering_strength: 0 = random, 1 = highly clustered
            kernel_width: KDE bandwidth (pixels)

        Returns:
            Tensor [B, C, H, W] representing cell density

        Expected dynamics: Migration, proliferation, collective behavior
        Cross-domain: Galaxy distributions, molecular positions in materials

        Example:
            ```python
            # Random cell distribution
            fields = generator._generate_cell_population(
                batch_size=16,
                num_cells=100,
                clustering_strength=0.2
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Generate cell positions via hard-core point process
            cell_positions = []
            attempts = 0
            max_attempts = num_cells * 100

            # If clustering, create cluster centers
            if clustering_strength > 0:
                num_clusters = max(1, int(num_cells * 0.1 * (1 - clustering_strength)))
                cluster_centers = [
                    (
                        torch.rand(1, device=self.device).item() * self.grid_size,
                        torch.rand(1, device=self.device).item() * self.grid_size
                    )
                    for _ in range(num_clusters)
                ]
            else:
                cluster_centers = [(self.grid_size / 2, self.grid_size / 2)]

            while len(cell_positions) < num_cells and attempts < max_attempts:
                # Sample position (clustered or random)
                if clustering_strength > 0 and torch.rand(1).item() < clustering_strength:
                    # Sample near a cluster center
                    center = cluster_centers[torch.randint(0, len(cluster_centers), (1,)).item()]
                    cluster_radius = self.grid_size * 0.3 * (1 - clustering_strength)
                    angle = torch.rand(1, device=self.device).item() * 2 * np.pi
                    radius = torch.rand(1, device=self.device).item() * cluster_radius
                    cx = (center[0] + radius * np.cos(angle)) % self.grid_size
                    cy = (center[1] + radius * np.sin(angle)) % self.grid_size
                else:
                    # Random position
                    cx = torch.rand(1, device=self.device).item() * self.grid_size
                    cy = torch.rand(1, device=self.device).item() * self.grid_size

                # Check hard-core constraint
                valid = True
                for (ex_x, ex_y) in cell_positions:
                    dist = np.sqrt((cx - ex_x) ** 2 + (cy - ex_y) ** 2)
                    if dist < cell_radius * 2:
                        valid = False
                        break

                if valid:
                    cell_positions.append((cx, cy))

                attempts += 1

            # Convert to density field via kernel density estimation
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            for (cx, cy) in cell_positions:
                # Gaussian kernel centered at cell position
                kernel = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * kernel_width ** 2))

                # Add to all channels (can represent different cell types)
                for c in range(self.num_channels):
                    cell_type_weight = torch.rand(1, device=self.device).item()
                    field[c] += kernel * cell_type_weight

            # Normalize
            if field.abs().max() > 0:
                field = field / field.abs().max()

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_chromatin_domain(
        self,
        batch_size: int,
        num_domains: int = 5,
        domain_size_range: Tuple[int, int] = (10, 30),
        boundary_sharpness: float = 3.0,
        compartment_type: str = "TAD",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate chromatin domain initial conditions.

        Creates chromosome organization patterns: topologically associated
        domains (TADs) or A/B compartments. Models 3D genome structure.

        Physics: Block-diagonal domain structure (2D projection)
            field = sum_d indicator(x in domain_d) * indicator(y in domain_d) * interaction_d
            Or checkerboard: field = sign(sin(k*x) * sin(k*y)) * amplitude

        Args:
            batch_size: Number of samples
            num_domains: Number of TADs or compartments
            domain_size_range: (min, max) domain size in pixels
            boundary_sharpness: Domain boundary width
            compartment_type: "TAD" (block-diagonal) or "AB" (checkerboard)

        Returns:
            Tensor [B, C, H, W] representing chromosome interaction map

        Expected dynamics: Domain dynamics, boundary strengthening/weakening
        Cross-domain: Image segmentation, phase separation in materials

        Example:
            ```python
            # TAD organization
            fields = generator._generate_chromatin_domain(
                batch_size=16,
                num_domains=8,
                compartment_type="TAD"
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            if compartment_type == "TAD":
                # Generate TADs as block-diagonal structure
                domains = []
                current_pos = 0

                for _ in range(num_domains):
                    domain_size = np.random.randint(domain_size_range[0], domain_size_range[1] + 1)
                    if current_pos + domain_size > self.grid_size:
                        domain_size = self.grid_size - current_pos

                    domains.append({
                        'start': current_pos,
                        'end': current_pos + domain_size,
                        'interaction': torch.rand(1, device=self.device).item() * 2 - 1
                    })

                    current_pos += domain_size
                    if current_pos >= self.grid_size:
                        break

                # Create block-diagonal interaction pattern
                for domain in domains:
                    start, end = domain['start'], domain['end']
                    interaction = domain['interaction']

                    # Block region in 2D interaction map
                    x_mask = torch.sigmoid((x - start) / boundary_sharpness) * torch.sigmoid((end - x) / boundary_sharpness)
                    y_mask = torch.sigmoid((y - start) / boundary_sharpness) * torch.sigmoid((end - y) / boundary_sharpness)

                    block = x_mask * y_mask * interaction

                    # Add to all channels
                    for c in range(self.num_channels):
                        field[c] += block

            elif compartment_type == "AB":
                # Generate A/B compartments as checkerboard pattern
                # Wavelength varies to create irregular compartments
                wavelength = np.random.uniform(15, 40)
                k = 2 * np.pi / wavelength

                # Random phase and orientation
                phase_x = torch.rand(1, device=self.device).item() * 2 * np.pi
                phase_y = torch.rand(1, device=self.device).item() * 2 * np.pi

                # Checkerboard: product of sine waves
                pattern = torch.sin(k * x + phase_x) * torch.sin(k * y + phase_y)

                # Threshold to create binary A/B compartments
                compartment_map = torch.sign(pattern)

                # Add to all channels with different interaction strengths
                for c in range(self.num_channels):
                    interaction_A = torch.rand(1, device=self.device).item()
                    interaction_B = torch.rand(1, device=self.device).item()
                    field[c] = torch.where(compartment_map > 0, interaction_A, interaction_B)

            else:
                raise ValueError(f"Unknown compartment_type: {compartment_type}. Use 'TAD' or 'AB'.")

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_shock_front(
        self,
        batch_size: int,
        shock_position: Optional[float] = None,
        shock_angle: Optional[float] = None,
        shock_width: float = 2.0,
        amplitude_ratio: float = 2.0,
        fluctuation_amplitude: float = 0.05,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate shock front initial conditions.

        Creates discontinuous fronts modeling relativistic shocks in
        astrophysical jets, plasma physics, supersonic flows.

        Physics: Tanh shock profile
            field = A1*(1-tanh((x-x_shock)/delta))/2 + A2*tanh((x-x_shock)/delta)/2 + noise

        Args:
            batch_size: Number of samples
            shock_position: Position along gradient (None = random)
            shock_angle: Shock orientation in radians (None = random)
            shock_width: Shock thickness (pixels)
            amplitude_ratio: Ratio of post-shock to pre-shock amplitude
            fluctuation_amplitude: Small-scale noise level

        Returns:
            Tensor [B, C, H, W] with discontinuous shock fronts

        Expected dynamics: Shock stability, numerical diffusion, gradient handling
        Cross-domain: Same as phase boundaries, reaction fronts

        Example:
            ```python
            # Oblique shock front
            fields = generator._generate_shock_front(
                batch_size=16,
                shock_angle=0.5,
                amplitude_ratio=3.0
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids (normalized to [0, 1])
        x_grid = self._cached_x / self.grid_size
        y_grid = self._cached_y / self.grid_size

        for _ in range(batch_size):
            # Random shock position if not specified
            if shock_position is None:
                shock_pos = torch.rand(1, device=self.device).item() * 0.6 + 0.2  # [0.2, 0.8]
            else:
                shock_pos = shock_position

            # Random shock angle if not specified
            if shock_angle is None:
                angle = torch.rand(1, device=self.device).item() * 2 * np.pi
            else:
                angle = shock_angle

            # Rotated coordinate for shock front
            x_rot = x_grid * np.cos(angle) + y_grid * np.sin(angle)

            # Tanh shock profile
            shock_profile = torch.tanh((x_rot - shock_pos) / (shock_width / self.grid_size))

            # Pre-shock and post-shock amplitudes
            A_pre = 1.0
            A_post = amplitude_ratio

            # Shock field: transition from A_pre to A_post
            shock_field = A_pre * (1 - shock_profile) / 2 + A_post * (1 + shock_profile) / 2

            # Add small-scale fluctuations
            fluctuations = self._generate_grf_batch(
                batch_size=1,
                length_scale=0.05,
                variance=fluctuation_amplitude ** 2
            )[0]

            # Combine shock + fluctuations
            field = shock_field.unsqueeze(0) + fluctuations

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_gene_expression(
        self,
        batch_size: int,
        num_genes: int = 3,
        expression_patterns: Optional[List[str]] = None,
        correlation_strength: float = 0.3,
        expression_range: Tuple[float, float] = (0.0, 1.0),
        **kwargs
    ) -> torch.Tensor:
        """
        Generate gene expression pattern initial conditions.

        Creates spatial transcriptomics data: gene expression levels across
        tissue space. Models developmental biology, tumor heterogeneity.

        Physics: Pattern composition
            E_g = sum_i w_{g,i} * basis_i + noise
            where basis_i are spatial patterns (gradients, stripes, spots)

        Args:
            batch_size: Number of samples
            num_genes: Number of genes (uses channels if num_genes <= num_channels)
            expression_patterns: List of pattern types per gene (None = random)
            correlation_strength: Gene-gene correlation (0 = independent, 1 = identical)
            expression_range: (min, max) expression levels

        Returns:
            Tensor [B, C, H, W] where channels represent genes

        Expected dynamics: Regulatory dynamics, pattern sharpening, noise filtering
        Cross-domain: Image segmentation, multi-channel signal processing

        Example:
            ```python
            # 3-gene expression with gradients and spots
            fields = generator._generate_gene_expression(
                batch_size=16,
                num_genes=3,
                expression_patterns=["gradient", "spots", "stripes"]
            )
            ```
        """
        fields = []

        # Available pattern types
        pattern_types = ["gradient", "spots", "stripes", "radial", "uniform"]

        # OPTIMIZATION: Use cached coordinate grids (normalized to [0, 1])
        x = self._cached_x / self.grid_size
        y = self._cached_y / self.grid_size

        for _ in range(batch_size):
            # Use channels to represent genes
            genes_to_use = min(num_genes, self.num_channels)

            # Generate shared latent factor for correlation
            if correlation_strength > 0:
                shared_pattern = self._generate_grf_batch(
                    batch_size=1,
                    length_scale=0.15,
                    variance=1.0
                )[0, 0]  # [H, W]
            else:
                shared_pattern = torch.zeros(self.grid_size, self.grid_size, device=self.device)

            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            for gene_idx in range(genes_to_use):
                # Select pattern type for this gene
                if expression_patterns is not None and gene_idx < len(expression_patterns):
                    pattern_type = expression_patterns[gene_idx]
                else:
                    pattern_type = np.random.choice(pattern_types)

                # Generate base pattern
                if pattern_type == "gradient":
                    direction = np.random.choice(["x", "y", "diagonal"])
                    if direction == "x":
                        base_pattern = x
                    elif direction == "y":
                        base_pattern = y
                    else:
                        base_pattern = (x + y) / 2

                elif pattern_type == "spots":
                    # Turing-like spots
                    wavelength = np.random.uniform(0.1, 0.3)
                    k = 2 * np.pi / wavelength
                    angles = [0, 60, 120]  # Hexagonal
                    base_pattern = torch.zeros(self.grid_size, self.grid_size, device=self.device)
                    for angle_deg in angles:
                        angle_rad = np.deg2rad(angle_deg)
                        kx = k * np.cos(angle_rad)
                        ky = k * np.sin(angle_rad)
                        base_pattern += torch.cos(kx * x * self.grid_size + ky * y * self.grid_size)
                    base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min() + 1e-10)

                elif pattern_type == "stripes":
                    wavelength = np.random.uniform(0.1, 0.3)
                    k = 2 * np.pi / wavelength
                    angle = torch.rand(1).item() * 2 * np.pi
                    x_rot = x * np.cos(angle) + y * np.sin(angle)
                    base_pattern = torch.cos(k * x_rot * self.grid_size)
                    base_pattern = (base_pattern + 1) / 2  # [0, 1]

                elif pattern_type == "radial":
                    cx = torch.rand(1, device=self.device).item()
                    cy = torch.rand(1, device=self.device).item()
                    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    base_pattern = torch.exp(-r ** 2 / 0.1)

                elif pattern_type == "uniform":
                    base_pattern = torch.ones(self.grid_size, self.grid_size, device=self.device) * 0.5

                else:
                    raise ValueError(f"Unknown pattern_type: {pattern_type}")

                # Add correlation with shared pattern
                if correlation_strength > 0:
                    # Normalize shared pattern to [0, 1]
                    shared_norm = (shared_pattern - shared_pattern.min()) / (shared_pattern.max() - shared_pattern.min() + 1e-10)
                    correlated_pattern = (
                        correlation_strength * shared_norm +
                        (1 - correlation_strength) * base_pattern
                    )
                else:
                    correlated_pattern = base_pattern

                # Add noise
                noise = torch.randn(self.grid_size, self.grid_size, device=self.device) * 0.1

                # Combine and scale to expression range
                expr_min, expr_max = expression_range
                gene_expression = correlated_pattern + noise
                gene_expression = expr_min + (expr_max - expr_min) * (
                    (gene_expression - gene_expression.min()) /
                    (gene_expression.max() - gene_expression.min() + 1e-10)
                )

                field[gene_idx] = gene_expression

            # If num_genes < num_channels, fill remaining channels with copies or zeros
            if genes_to_use < self.num_channels:
                # Copy first gene to remaining channels with noise
                for c in range(genes_to_use, self.num_channels):
                    field[c] = field[0] + torch.randn(self.grid_size, self.grid_size, device=self.device) * 0.1

            fields.append(field)

        return torch.stack(fields, dim=0)

    # =========================================================================
    # Tier 4 Research Frontiers Initial Conditions
    # =========================================================================

    def _generate_coherent_state(
        self,
        batch_size: int,
        alpha: float = 1.0,
        sigma: float = 10.0,
        oscillation_frequency: float = 0.2,
        num_modes: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate coherent state initial conditions.

        Creates "most classical" quantum states used in quantum optics,
        superconductivity. Minimal spreading, quasi-classical oscillations.

        Physics: Gaussian envelope + oscillation + random background
            field = GRF(length_scale) + amplitude * Gaussian(center, sigma) * cos(omega*x)

        Args:
            batch_size: Number of samples
            alpha: Complex amplitude (coherent state parameter)
            sigma: Gaussian envelope width (pixels)
            oscillation_frequency: Spatial oscillation frequency
            num_modes: Number of independent coherent modes

        Returns:
            Tensor [B, C, H, W] representing coherent state

        Expected dynamics: Stable oscillations, minimal spreading (quasi-classical)
        Cross-domain: Laser beam profiles, Gaussian beam optics

        Example:
            ```python
            # Single coherent mode
            fields = generator._generate_coherent_state(
                batch_size=16,
                alpha=1.5,
                sigma=15.0
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached coordinate grids
        x = self._cached_x
        y = self._cached_y

        for _ in range(batch_size):
            # Start with random GRF background (vacuum fluctuations)
            field = self._generate_grf_batch(
                batch_size=1,
                length_scale=0.1,
                variance=0.5
            )[0]

            # Add coherent modes
            for _ in range(num_modes):
                # Random center position
                cx = torch.rand(1, device=self.device).item() * self.grid_size
                cy = torch.rand(1, device=self.device).item() * self.grid_size

                # Gaussian envelope
                r_sq = (x - cx) ** 2 + (y - cy) ** 2
                envelope = torch.exp(-r_sq / (2 * sigma ** 2))

                # Random oscillation direction
                angle = torch.rand(1, device=self.device).item() * 2 * np.pi
                k_x = oscillation_frequency * np.cos(angle)
                k_y = oscillation_frequency * np.sin(angle)

                # Coherent state pattern
                oscillation = torch.cos(k_x * x + k_y * y)

                # Add to all channels
                for c in range(self.num_channels):
                    field[c] += alpha * envelope * oscillation

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_relativistic_wave_packet(
        self,
        batch_size: int,
        velocity: float = 0.5,
        sigma: float = 10.0,
        wavenumber_range: Tuple[float, float] = (0.1, 0.5),
        **kwargs
    ) -> torch.Tensor:
        """
        Generate relativistic wave packet initial conditions.

        Creates Lorentz-contracted structures moving at relativistic speeds.
        Tests translational equivariance and special relativity signatures.

        Physics: Lorentz-contracted Gaussian wave packet
            gamma = 1/sqrt(1 - v^2/c^2)
            field = exp(-(gamma^2*(x-x0)^2 + (y-y0)^2)/(2*sigma^2)) * cos(k*x + phase)

        Args:
            batch_size: Number of samples
            velocity: Velocity as fraction of c (0-1)
            sigma: Packet width (before contraction)
            wavenumber_range: (min, max) wavenumber for oscillation

        Returns:
            Tensor [B, C, H, W] with Lorentz-contracted packets

        Expected dynamics: Evolve with Lorentz contraction, test translational equivariance
        Cross-domain: Optical wave packets, information encoding

        Example:
            ```python
            # Fast-moving relativistic packet
            fields = generator._generate_relativistic_wave_packet(
                batch_size=16,
                velocity=0.8,
                sigma=12.0
            )
            ```
        """
        fields = []

        # OPTIMIZATION: Use cached centered coordinate grids
        x = self._cached_x_centered
        y = self._cached_y_centered

        for _ in range(batch_size):
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            # Random initial position
            x0 = (torch.rand(1, device=self.device).item() - 0.5) * self.grid_size * 0.5
            y0 = (torch.rand(1, device=self.device).item() - 0.5) * self.grid_size * 0.5

            # Lorentz factor
            gamma = 1.0 / np.sqrt(1 - velocity ** 2)

            # Random velocity direction
            direction_angle = torch.rand(1, device=self.device).item() * 2 * np.pi
            v_x = velocity * np.cos(direction_angle)
            v_y = velocity * np.sin(direction_angle)

            # Rotated coordinates (align x' with velocity direction)
            x_rot = (x - x0) * np.cos(direction_angle) + (y - y0) * np.sin(direction_angle)
            y_rot = -(x - x0) * np.sin(direction_angle) + (y - y0) * np.cos(direction_angle)

            # Lorentz-contracted Gaussian
            contracted_gaussian = torch.exp(
                -(gamma ** 2 * x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2)
            )

            # Random wavenumber
            k_min, k_max = wavenumber_range
            k = torch.rand(1, device=self.device).item() * (k_max - k_min) + k_min

            # Random phase
            phase = torch.rand(1, device=self.device).item() * 2 * np.pi

            # Wave packet
            wave = torch.cos(k * x_rot + phase)

            # Combine
            packet = contracted_gaussian * wave

            # Add to all channels
            for c in range(self.num_channels):
                field[c] = packet

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_mutual_information(
        self,
        batch_size: int,
        correlation_pattern: str = "local",
        latent_length_scale: float = 0.3,
        local_noise_scale: float = 0.5,
        num_channels_correlated: int = 2,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate mutual information pattern initial conditions.

        Creates fields with controlled mutual information between spatial
        regions or channels. Tests information flow and correlation propagation.

        Physics: Shared latent variable model
            z = GRF(long_length_scale)
            field_i = alpha_i * z + sqrt(1 - alpha_i^2) * noise_i
            where alpha_i controls local MI with latent variable

        Args:
            batch_size: Number of samples
            correlation_pattern: "local", "long_range", or "hierarchical"
            latent_length_scale: Correlation length of shared latent variable
            local_noise_scale: Independent noise level (0 = fully correlated, 1 = independent)
            num_channels_correlated: Number of channels to correlate (max: num_channels)

        Returns:
            Tensor [B, C, H, W] with controlled MI structure

        Expected dynamics: Correlation propagation, information flow
        Cross-domain: Neural coding, entanglement structure

        Example:
            ```python
            # Long-range correlated fields
            fields = generator._generate_mutual_information(
                batch_size=16,
                correlation_pattern="long_range",
                latent_length_scale=0.5
            )
            ```
        """
        fields = []

        for _ in range(batch_size):
            # Generate shared latent variable
            if correlation_pattern == "local":
                latent_scale = latent_length_scale * 0.5
            elif correlation_pattern == "long_range":
                latent_scale = latent_length_scale * 2.0
            elif correlation_pattern == "hierarchical":
                # Multi-scale latent
                latent = self._generate_multiscale_grf(
                    batch_size=1,
                    num_scales=3,
                    base_length_scale=latent_length_scale
                )[0, 0]  # [H, W]
            else:
                latent_scale = latent_length_scale

            if correlation_pattern != "hierarchical":
                latent = self._generate_grf_batch(
                    batch_size=1,
                    length_scale=latent_scale,
                    variance=1.0
                )[0, 0]  # [H, W]

            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            # Determine which channels are correlated
            channels_to_correlate = min(num_channels_correlated, self.num_channels)

            for c in range(self.num_channels):
                if c < channels_to_correlate:
                    # Correlated channels: mix latent + independent noise
                    # alpha controls correlation strength
                    alpha = 1.0 - local_noise_scale
                    independent_noise = torch.randn(
                        self.grid_size, self.grid_size, device=self.device
                    )
                    field[c] = alpha * latent + np.sqrt(1 - alpha ** 2 + 1e-10) * independent_noise
                else:
                    # Uncorrelated channels: pure noise
                    field[c] = torch.randn(
                        self.grid_size, self.grid_size, device=self.device
                    )

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_regulatory_network(
        self,
        batch_size: int,
        network_topology: str = "scale_free",
        network_size: int = 5,
        connection_probability: float = 0.3,
        spatial_decay: float = 0.2,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate regulatory network initial conditions.

        Creates gene regulatory networks with spatial embedding. Models
        bistability, oscillations, cascade propagation.

        Physics: Network-constrained expression + spatial embedding
            x_i(t+1) = f(sum_j W_ij * x_j(t)) + noise
            field = sum_i x_i * psi_i(x, y)  (spatial embedding)

        Args:
            batch_size: Number of samples
            network_topology: "scale_free", "random", or "hub"
            network_size: Number of genes in network
            connection_probability: Edge probability (for random networks)
            spatial_decay: Spatial localization scale for genes

        Returns:
            Tensor [B, C, H, W] representing spatially-embedded network state

        Expected dynamics: Bistability, oscillations, cascade propagation
        Cross-domain: Neural networks, reaction networks

        Example:
            ```python
            # Scale-free regulatory network
            fields = generator._generate_regulatory_network(
                batch_size=16,
                network_topology="scale_free",
                network_size=8
            )
            ```
        """
        fields = []

        for _ in range(batch_size):
            # Generate network adjacency matrix
            W = torch.zeros(network_size, network_size, device=self.device)

            if network_topology == "random":
                # Random Erdős-Rényi network
                W = (torch.rand(network_size, network_size, device=self.device) < connection_probability).float()
            elif network_topology == "scale_free":
                # Simple preferential attachment (approximate scale-free)
                for i in range(1, network_size):
                    # New node connects to existing nodes with probability proportional to degree
                    degrees = W.sum(dim=1)[:i] + 1  # Add 1 to avoid zero degree
                    probs = degrees / degrees.sum()
                    num_connections = min(i, max(1, int(connection_probability * i)))
                    connected = torch.multinomial(probs, num_connections, replacement=False)
                    W[i, connected] = 1.0
                    W[connected, i] = 1.0
            elif network_topology == "hub":
                # Hub network: one central node connected to all others
                W[0, 1:] = 1.0
                W[1:, 0] = 1.0
            else:
                raise ValueError(f"Unknown network_topology: {network_topology}")

            # Random interaction strengths
            W = W * (torch.rand(network_size, network_size, device=self.device) * 2 - 1)

            # Simulate network dynamics for a few steps
            x = torch.rand(network_size, device=self.device)  # Initial state
            for _ in range(5):
                x_new = torch.tanh(torch.matmul(W, x) + torch.randn(network_size, device=self.device) * 0.1)
                x = x_new

            # Spatial embedding: place each gene at a random location
            gene_positions = []
            for _ in range(network_size):
                gene_positions.append((
                    torch.rand(1, device=self.device).item() * self.grid_size,
                    torch.rand(1, device=self.device).item() * self.grid_size
                ))

            # Create spatial field from network state
            # OPTIMIZATION: Use cached coordinate grids
            x_grid = self._cached_x
            y_grid = self._cached_y

            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            for gene_idx, (gx, gy) in enumerate(gene_positions):
                # Spatial basis function for this gene
                r_sq = (x_grid - gx) ** 2 + (y_grid - gy) ** 2
                spatial_basis = torch.exp(-r_sq / (2 * (self.grid_size * spatial_decay) ** 2))

                # Add to field weighted by gene expression
                expression = x[gene_idx].item()
                for c in range(self.num_channels):
                    field[c] += expression * spatial_basis

            # Normalize
            if field.abs().max() > 0:
                field = field / field.abs().max()

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_dla_cluster(
        self,
        batch_size: int,
        cluster_size: int = 500,
        seed_type: str = "point",
        stickiness: float = 1.0,
        field_type_dla: str = "distance_transform",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate diffusion-limited aggregation (DLA) cluster initial conditions.

        Creates fractal growth patterns: electrodeposition, crystal growth,
        bacterial colonies. Highly complex, computationally expensive.

        Physics: Random walk with sticking
            - Particle random walks from far field
            - Sticks when touching cluster (with probability = stickiness)
            - Converts to field via distance transform or indicator

        Args:
            batch_size: Number of samples
            cluster_size: Number of particles in cluster
            seed_type: "point" or "line"
            stickiness: Probability of sticking (0-1)
            field_type_dla: "distance_transform" or "indicator"

        Returns:
            Tensor [B, C, H, W] representing DLA cluster

        Expected dynamics: Growth dynamics, screening, branch competition
        Cross-domain: Dendritic solidification, vascular networks

        Example:
            ```python
            # Point-seeded DLA cluster
            fields = generator._generate_dla_cluster(
                batch_size=16,
                cluster_size=300,
                seed_type="point"
            )
            ```
        """
        fields = []

        for _ in range(batch_size):
            # Initialize cluster
            cluster = set()

            # Seed
            if seed_type == "point":
                cluster.add((self.grid_size // 2, self.grid_size // 2))
            elif seed_type == "line":
                for x in range(self.grid_size // 2 - 10, self.grid_size // 2 + 10):
                    cluster.add((x, self.grid_size // 2))
            else:
                raise ValueError(f"Unknown seed_type: {seed_type}")

            # Grow cluster via random walk (simplified DLA)
            attempts = 0
            max_attempts = cluster_size * 1000

            while len(cluster) < cluster_size and attempts < max_attempts:
                attempts += 1

                # Start particle at random edge
                edge = np.random.randint(0, 4)
                if edge == 0:  # Top
                    px, py = np.random.randint(0, self.grid_size), 0
                elif edge == 1:  # Right
                    px, py = self.grid_size - 1, np.random.randint(0, self.grid_size)
                elif edge == 2:  # Bottom
                    px, py = np.random.randint(0, self.grid_size), self.grid_size - 1
                else:  # Left
                    px, py = 0, np.random.randint(0, self.grid_size)

                # Random walk until stick or escape
                for _ in range(self.grid_size * 2):
                    # Check if adjacent to cluster
                    neighbors = [
                        (px + 1, py), (px - 1, py),
                        (px, py + 1), (px, py - 1)
                    ]
                    if any((nx, ny) in cluster for nx, ny in neighbors if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                        # Stick with probability = stickiness
                        if np.random.rand() < stickiness:
                            cluster.add((px, py))
                            break

                    # Random step
                    dx, dy = np.random.choice([-1, 0, 1], size=2)
                    px, py = px + dx, py + dy

                    # Check bounds
                    if not (0 <= px < self.grid_size and 0 <= py < self.grid_size):
                        break  # Escaped

            # Convert cluster to field
            cluster_array = torch.zeros(self.grid_size, self.grid_size, device=self.device)
            for (cx, cy) in cluster:
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    cluster_array[cy, cx] = 1.0

            if field_type_dla == "distance_transform":
                # Smooth distance field (approximate)
                # OPTIMIZATION: Use cached coordinate grids
                x_grid = self._cached_x
                y_grid = self._cached_y
                field_dla = torch.zeros(self.grid_size, self.grid_size, device=self.device)
                for (cx, cy) in cluster:
                    r = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
                    field_dla = torch.maximum(field_dla, torch.exp(-r / 5.0))
            else:  # indicator
                field_dla = cluster_array

            # Replicate to all channels
            field = field_dla.unsqueeze(0).repeat(self.num_channels, 1, 1)

            # Normalize
            if field.abs().max() > 0:
                field = field / field.abs().max()

            fields.append(field)

        return torch.stack(fields, dim=0)

    def _generate_error_correcting_code(
        self,
        batch_size: int,
        code_type: str = "repetition",
        code_rate: float = 0.5,
        block_size: int = 8,
        error_rate: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate error-correcting code initial conditions.

        Creates structured redundancy enabling error correction. Tests
        whether operators exploit redundancy for robustness.

        Physics: Encode message into redundant codeword
            message = random_bits(k)
            codeword = encoder(message)  # n > k bits
            field = sum_j codeword[j] * basis_j(x, y)

        Args:
            batch_size: Number of samples
            code_type: "repetition", "parity", or "random"
            code_rate: k/n (information rate)
            block_size: Spatial block size for encoding bits
            error_rate: Fraction of bits to flip (noise)

        Returns:
            Tensor [B, C, H, W] with error-correcting code structure

        Expected dynamics: Error propagation, redundancy exploitation
        Cross-domain: DNA error correction, neural redundancy

        Example:
            ```python
            # Repetition code with noise
            fields = generator._generate_error_correcting_code(
                batch_size=16,
                code_type="repetition",
                error_rate=0.15
            )
            ```
        """
        fields = []

        for _ in range(batch_size):
            # Determine message and codeword lengths
            num_blocks = (self.grid_size // block_size) ** 2
            k = max(1, int(num_blocks * code_rate))  # Message bits
            n = num_blocks  # Codeword bits

            # Generate random message
            message = torch.randint(0, 2, (k,), device=self.device).float()

            # Encode
            codeword = torch.zeros(n, device=self.device)

            if code_type == "repetition":
                # Simple repetition code: repeat each message bit
                repetitions = n // k
                for i in range(k):
                    codeword[i * repetitions:(i + 1) * repetitions] = message[i]
            elif code_type == "parity":
                # Parity check code
                codeword[:k] = message
                codeword[k:] = message.sum() % 2  # Parity bit(s)
            elif code_type == "random":
                # Random linear code (generator matrix)
                G = torch.randint(0, 2, (k, n), device=self.device).float()
                codeword = torch.matmul(message, G) % 2
            else:
                raise ValueError(f"Unknown code_type: {code_type}")

            # Add errors
            error_mask = torch.rand(n, device=self.device) < error_rate
            codeword = (codeword + error_mask.float()) % 2

            # Map to spatial field
            field = torch.zeros(self.num_channels, self.grid_size, self.grid_size, device=self.device)

            block_idx = 0
            for by in range(self.grid_size // block_size):
                for bx in range(self.grid_size // block_size):
                    if block_idx < n:
                        # Fill block with bit value
                        value = codeword[block_idx].item() * 2 - 1  # Map {0,1} → {-1,1}
                        for c in range(self.num_channels):
                            field[
                                c,
                                by * block_size:(by + 1) * block_size,
                                bx * block_size:(bx + 1) * block_size
                            ] = value
                        block_idx += 1

            fields.append(field)

        return torch.stack(fields, dim=0)
