"""Potential Energy Landscape Generators for QBM.

Provides GPU-accelerated generation of potential energy landscapes V(x,y):
- Harmonic (parabolic wells)
- Double-well (symmetry breaking)
- Quartic (anharmonic)
- Random disorder (Gaussian random fields)

All generators produce potentials [B, H, W] parameterized from config vectors.

Physics:
    The potential energy V(x,y) appears in the Hamiltonian:
    H = p²/(2m) + V(x,y)

    Different potential shapes lead to different quantum dynamics:
    - Harmonic: Coherent oscillations, closed orbits
    - Double-well: Tunneling, symmetry breaking
    - Quartic: Anharmonic oscillations, chaotic classical limit
    - Random: Anderson localization, disorder effects

References:
    - Sakurai, J. J. (2017). Modern Quantum Mechanics
    - Anderson, P. W. (1958). Absence of Diffusion in Certain Random Lattices
"""

import torch
import numpy as np
from typing import Optional


class PotentialGenerator:
    """Generate potential energy landscapes on GPU.

    All methods produce potentials [B, H, W] where values represent
    potential energy at each spatial grid point.

    Example:
        ```python
        generator = PotentialGenerator(
            grid_size=64,
            domain_size=10.0,
            device="cuda"
        )

        # Harmonic trap
        V = generator.harmonic_2d(
            batch_size=16,
            omega=torch.ones(16),
            center_x=torch.zeros(16),
            center_y=torch.zeros(16)
        )
        # Shape: [16, 64, 64]
        ```

    Args:
        grid_size: Number of grid points per dimension (default: 64)
        domain_size: Physical size of domain (default: 10.0)
        device: Torch device
    """

    def __init__(
        self,
        grid_size: int = 64,
        domain_size: float = 10.0,
        device: torch.device = torch.device("cuda")
    ):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.device = device

        # Spatial grid spacing
        self.dx = domain_size / grid_size

        # Pre-compute spatial grids
        self._setup_grids()

    def _setup_grids(self):
        """Pre-compute spatial coordinate grids."""
        # Spatial coordinates centered at origin
        x = torch.linspace(
            -self.domain_size/2, self.domain_size/2, self.grid_size, device=self.device
        )
        y = torch.linspace(
            -self.domain_size/2, self.domain_size/2, self.grid_size, device=self.device
        )

        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        self.x_grid = x_grid  # [H, W]
        self.y_grid = y_grid  # [H, W]
        self.r_squared = x_grid**2 + y_grid**2  # [H, W]

        # K-space grid for random potentials
        kx = torch.fft.fftfreq(self.grid_size, d=self.dx, device=self.device)
        ky = torch.fft.fftfreq(self.grid_size, d=self.dx, device=self.device)
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')
        self.k_squared = kx_grid**2 + ky_grid**2  # [H, W]

    def harmonic_2d(
        self,
        batch_size: int,
        omega: torch.Tensor,
        center_x: Optional[torch.Tensor] = None,
        center_y: Optional[torch.Tensor] = None,
        mass: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate 2D harmonic oscillator potential.

        V(x,y) = ½ m ω² [(x-x₀)² + (y-y₀)²]

        This is the most fundamental confining potential in quantum mechanics.

        Args:
            batch_size: Number of potentials
            omega: Angular frequency [B]
            center_x: Trap center x-coordinate [B] (default: 0)
            center_y: Trap center y-coordinate [B] (default: 0)
            mass: Particle mass [B] (default: 1)

        Returns:
            Potential [B, H, W]
        """
        # Default parameters
        if center_x is None:
            center_x = torch.zeros(batch_size, device=self.device)
        if center_y is None:
            center_y = torch.zeros(batch_size, device=self.device)
        if mass is None:
            mass = torch.ones(batch_size, device=self.device)

        # Ensure tensors are on correct device
        if not isinstance(omega, torch.Tensor):
            omega = torch.tensor(omega, device=self.device)
        if not isinstance(center_x, torch.Tensor):
            center_x = torch.tensor(center_x, device=self.device)
        if not isinstance(center_y, torch.Tensor):
            center_y = torch.tensor(center_y, device=self.device)
        if not isinstance(mass, torch.Tensor):
            mass = torch.tensor(mass, device=self.device)

        # Reshape for broadcasting: [B] -> [B, 1, 1]
        omega = omega.view(batch_size, 1, 1)
        center_x = center_x.view(batch_size, 1, 1)
        center_y = center_y.view(batch_size, 1, 1)
        mass = mass.view(batch_size, 1, 1)

        # Displacement from center
        dx = self.x_grid.unsqueeze(0) - center_x  # [B, H, W]
        dy = self.y_grid.unsqueeze(0) - center_y  # [B, H, W]
        r_squared = dx**2 + dy**2

        # Harmonic potential: V = ½ m ω² r²
        potential = 0.5 * mass * omega**2 * r_squared

        return potential

    def double_well(
        self,
        batch_size: int,
        barrier_height: torch.Tensor,
        separation: torch.Tensor,
        mass: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate double-well potential (quartic).

        V(x,y) = -½ m ω² (x² + y²) + λ (x⁴ + y⁴)

        where parameters are chosen to create two wells separated by a barrier.

        This potential is fundamental for studying:
        - Quantum tunneling
        - Symmetry breaking
        - Bistability

        Args:
            batch_size: Number of potentials
            barrier_height: Height of central barrier [B]
            separation: Distance between wells [B]
            mass: Particle mass [B] (default: 1)

        Returns:
            Potential [B, H, W]
        """
        # Default parameters
        if mass is None:
            mass = torch.ones(batch_size, device=self.device)

        # Ensure tensors are on correct device
        if not isinstance(barrier_height, torch.Tensor):
            barrier_height = torch.tensor(barrier_height, device=self.device)
        if not isinstance(separation, torch.Tensor):
            separation = torch.tensor(separation, device=self.device)
        if not isinstance(mass, torch.Tensor):
            mass = torch.tensor(mass, device=self.device)

        # Reshape for broadcasting
        barrier_height = barrier_height.view(batch_size, 1, 1)
        separation = separation.view(batch_size, 1, 1)
        mass = mass.view(batch_size, 1, 1)

        # Compute quartic coefficients
        # At minima (x = ±a): V'(a) = 0 → -ma²ω² + 4λa³ = 0 → λ = mω²/(4a)
        # Barrier height: V(0) - V(a) = ma⁴λ
        # Solve for omega_squared and lambda
        a = separation / 2  # Well position
        lambda_coeff = barrier_height / (mass * a**4)
        omega_squared = 4 * lambda_coeff * a

        # Get spatial grids
        x = self.x_grid.unsqueeze(0)  # [1, H, W]
        y = self.y_grid.unsqueeze(0)  # [1, H, W]

        # Double-well potential (1D along x, harmonic in y)
        # V = -½ m ω² x² + λ x⁴ + ½ m ω_y² y²
        omega_y_squared = 0.5 * omega_squared  # Weaker confinement in y

        potential = (
            -0.5 * mass * omega_squared * x**2
            + lambda_coeff * x**4
            + 0.5 * mass * omega_y_squared * y**2
        )

        return potential

    def quartic(
        self,
        batch_size: int,
        c0: torch.Tensor,
        c2: torch.Tensor,
        c4: torch.Tensor
    ) -> torch.Tensor:
        """Generate quartic (anharmonic) potential.

        V(r) = c₀ + c₂ r² + c₄ r⁴

        where r² = x² + y² is the radial coordinate.

        Anharmonic potentials lead to rich dynamics:
        - Non-uniform energy level spacing
        - Classical chaos in high-energy limit

        Args:
            batch_size: Number of potentials
            c0: Constant offset [B]
            c2: Quadratic coefficient [B]
            c4: Quartic coefficient [B]

        Returns:
            Potential [B, H, W]
        """
        # Ensure tensors are on correct device
        if not isinstance(c0, torch.Tensor):
            c0 = torch.tensor(c0, device=self.device)
        if not isinstance(c2, torch.Tensor):
            c2 = torch.tensor(c2, device=self.device)
        if not isinstance(c4, torch.Tensor):
            c4 = torch.tensor(c4, device=self.device)

        # Reshape for broadcasting: [B] -> [B, 1, 1]
        c0 = c0.view(batch_size, 1, 1)
        c2 = c2.view(batch_size, 1, 1)
        c4 = c4.view(batch_size, 1, 1)

        # Radial distance squared
        r_squared = self.r_squared.unsqueeze(0)  # [1, H, W]

        # Quartic potential
        potential = c0 + c2 * r_squared + c4 * r_squared**2

        return potential

    def random_potential(
        self,
        batch_size: int,
        correlation_length: float = 0.5,
        amplitude: float = 1.0
    ) -> torch.Tensor:
        """Generate random disorder potential via Gaussian Random Field.

        Creates spatially correlated random potentials modeling:
        - Surface roughness
        - Material impurities
        - Anderson localization

        Uses spectral method:
        1. Generate random Fourier modes
        2. Apply power spectrum P(k) ∝ exp(-k² ℓ²/2)
        3. Inverse FFT to real space

        Args:
            batch_size: Number of potentials
            correlation_length: Correlation length (default: 0.5)
            amplitude: RMS amplitude of disorder (default: 1.0)

        Returns:
            Potential [B, H, W]
        """
        # Power spectrum: Gaussian in k-space
        power_spectrum = amplitude**2 * torch.exp(
            -self.k_squared * correlation_length**2 / 2
        )

        # Generate random Fourier modes: [B, H, W]
        fourier_modes = torch.randn(
            batch_size, self.grid_size, self.grid_size,
            device=self.device, dtype=torch.complex64
        ) * torch.sqrt(power_spectrum).unsqueeze(0)

        # Inverse FFT to real space
        potential = torch.fft.ifft2(fourier_modes).real

        return potential

    def from_type_and_params(
        self,
        batch_size: int,
        potential_type: str,
        params: torch.Tensor
    ) -> torch.Tensor:
        """Generate potential from type string and parameter tensor.

        This is the main interface used by the dataset generation pipeline.

        Args:
            batch_size: Number of potentials
            potential_type: Type of potential:
                - "harmonic": Harmonic oscillator
                - "double_well": Double-well (quartic)
                - "quartic": General quartic
                - "random": Random disorder
            params: Parameter tensor [B, P] containing potential-specific parameters

        Returns:
            Potential [B, H, W]
        """
        if potential_type == "harmonic":
            # params: [omega, center_x, center_y, mass]
            return self.harmonic_2d(
                batch_size=batch_size,
                omega=params[:, 0],
                center_x=params[:, 1] if params.shape[1] > 1 else None,
                center_y=params[:, 2] if params.shape[1] > 2 else None,
                mass=params[:, 3] if params.shape[1] > 3 else None
            )

        elif potential_type == "double_well":
            # params: [barrier_height, separation, mass]
            return self.double_well(
                batch_size=batch_size,
                barrier_height=params[:, 0],
                separation=params[:, 1],
                mass=params[:, 2] if params.shape[1] > 2 else None
            )

        elif potential_type == "quartic":
            # params: [c0, c2, c4]
            return self.quartic(
                batch_size=batch_size,
                c0=params[:, 0],
                c2=params[:, 1],
                c4=params[:, 2]
            )

        elif potential_type == "random":
            # params: [correlation_length, amplitude]
            # Note: Can't batch this directly, so generate one at a time
            potentials = []
            for i in range(batch_size):
                V = self.random_potential(
                    batch_size=1,
                    correlation_length=params[i, 0].item(),
                    amplitude=params[i, 1].item() if params.shape[1] > 1 else 1.0
                )
                potentials.append(V[0])
            return torch.stack(potentials, dim=0)

        else:
            raise ValueError(
                f"Unknown potential type: {potential_type}. "
                f"Must be one of: 'harmonic', 'double_well', 'quartic', 'random'"
            )
