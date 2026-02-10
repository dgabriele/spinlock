"""Quantum Initial Condition Generators.

Provides GPU-accelerated generation of quantum wavefunctions for QBM simulations:
- Gaussian wavepackets (localized states with definite momentum)
- Coherent states (displaced harmonic oscillator ground states)
- Superposition states (quantum superpositions of multiple wavepackets)

All generators produce properly normalized wavefunctions satisfying ∫|ψ|²dx = 1.

Physics:
    Quantum states are represented as complex wavefunctions ψ(x,y) stored as
    2-channel tensors [Re(ψ), Im(ψ)].

    For Gaussian wavepackets:
        ψ(r) = N exp(i k·r) exp(-|r-r₀|²/(4σ²))

    where N is normalization constant, k is momentum, r₀ is center, σ is width.

References:
    - Griffiths, D. J. (2018). Introduction to Quantum Mechanics
    - Sakurai, J. J. (2017). Modern Quantum Mechanics
"""

import torch
import numpy as np
from typing import Tuple, Optional


class QuantumICGenerator:
    """Generate quantum initial conditions on GPU.

    All methods produce normalized wavefunctions [B, 2, H, W] where:
    - Channel 0: Real part of ψ
    - Channel 1: Imaginary part of ψ
    - ∫|ψ|² dx dy = 1 (probability normalization)

    Example:
        ```python
        generator = QuantumICGenerator(
            grid_size=64,
            domain_size=10.0,
            device="cuda"
        )

        # Generate Gaussian wavepackets
        psi = generator.generate_gaussian_wavepacket(
            batch_size=16,
            center_x=torch.zeros(16),
            center_y=torch.zeros(16),
            momentum_x=torch.ones(16),
            momentum_y=torch.zeros(16),
            sigma=1.0
        )
        # Shape: [16, 2, 64, 64]
        ```

    Args:
        grid_size: Number of grid points per dimension (default: 64)
        domain_size: Physical size of domain (default: 10.0)
        hbar: Reduced Planck constant (default: 1.0 in natural units)
        device: Torch device
    """

    def __init__(
        self,
        grid_size: int = 64,
        domain_size: float = 10.0,
        hbar: float = 1.0,
        device: torch.device = torch.device("cuda")
    ):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.hbar = hbar
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

    def _normalize_wavefunction(self, psi: torch.Tensor) -> torch.Tensor:
        """Normalize wavefunction to satisfy ∫|ψ|²dx = 1.

        Args:
            psi: Wavefunction [B, 2, H, W]

        Returns:
            Normalized wavefunction [B, 2, H, W]
        """
        # Compute probability density: |ψ|² = Re² + Im²
        prob_density = psi[:, 0]**2 + psi[:, 1]**2  # [B, H, W]

        # Integrate over space: ∫|ψ|² dx dy
        norm_squared = prob_density.sum(dim=(1, 2)) * self.dx**2  # [B]

        # Normalization factor
        norm = torch.sqrt(norm_squared).view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # Prevent division by zero
        norm = torch.clamp(norm, min=1e-10)

        return psi / norm

    def generate_gaussian_wavepacket(
        self,
        batch_size: int,
        center_x: Optional[torch.Tensor] = None,
        center_y: Optional[torch.Tensor] = None,
        momentum_x: Optional[torch.Tensor] = None,
        momentum_y: Optional[torch.Tensor] = None,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Generate Gaussian wavepackets with definite momentum.

        Physics:
            ψ(x,y) = N exp(i(kₓx + k_yy)) exp(-[(x-x₀)² + (y-y₀)²]/(4σ²))

        where k = p/ℏ is the wave vector.

        Args:
            batch_size: Number of wavepackets to generate
            center_x: Packet center x-coordinates [B] (default: random in [-2, 2])
            center_y: Packet center y-coordinates [B] (default: random in [-2, 2])
            momentum_x: x-momentum [B] (default: random in [-1, 1])
            momentum_y: y-momentum [B] (default: random in [-1, 1])
            sigma: Spatial width (default: 1.0)

        Returns:
            Normalized wavefunction [B, 2, H, W]
        """
        # Default parameters: random sampling
        if center_x is None:
            center_x = (torch.rand(batch_size, device=self.device) - 0.5) * 4  # [-2, 2]
        if center_y is None:
            center_y = (torch.rand(batch_size, device=self.device) - 0.5) * 4  # [-2, 2]
        if momentum_x is None:
            momentum_x = (torch.rand(batch_size, device=self.device) - 0.5) * 2  # [-1, 1]
        if momentum_y is None:
            momentum_y = (torch.rand(batch_size, device=self.device) - 0.5) * 2  # [-1, 1]

        # Ensure tensors are on correct device
        if not isinstance(center_x, torch.Tensor):
            center_x = torch.tensor(center_x, device=self.device)
        if not isinstance(center_y, torch.Tensor):
            center_y = torch.tensor(center_y, device=self.device)
        if not isinstance(momentum_x, torch.Tensor):
            momentum_x = torch.tensor(momentum_x, device=self.device)
        if not isinstance(momentum_y, torch.Tensor):
            momentum_y = torch.tensor(momentum_y, device=self.device)

        # Reshape for broadcasting: [B] -> [B, 1, 1]
        center_x = center_x.view(batch_size, 1, 1)
        center_y = center_y.view(batch_size, 1, 1)
        momentum_x = momentum_x.view(batch_size, 1, 1)
        momentum_y = momentum_y.view(batch_size, 1, 1)

        # Displacement from center
        dx = self.x_grid.unsqueeze(0) - center_x  # [B, H, W]
        dy = self.y_grid.unsqueeze(0) - center_y  # [B, H, W]
        r_squared = dx**2 + dy**2

        # Gaussian envelope: exp(-r²/(4σ²))
        envelope = torch.exp(-r_squared / (4 * sigma**2))

        # Phase: exp(i k·r) = exp(i(kₓx + k_yy))
        # k = p/ℏ
        kx = momentum_x / self.hbar
        ky = momentum_y / self.hbar
        phase = kx * self.x_grid.unsqueeze(0) + ky * self.y_grid.unsqueeze(0)  # [B, H, W]

        # Complex wavefunction: ψ = envelope * exp(i phase)
        psi_real = envelope * torch.cos(phase)  # [B, H, W]
        psi_imag = envelope * torch.sin(phase)  # [B, H, W]

        # Stack into [B, 2, H, W]
        psi = torch.stack([psi_real, psi_imag], dim=1)

        # Normalize
        psi = self._normalize_wavefunction(psi)

        return psi

    def generate_coherent_state(
        self,
        batch_size: int,
        alpha_real: Optional[torch.Tensor] = None,
        alpha_imag: Optional[torch.Tensor] = None,
        omega: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate coherent states (displaced harmonic oscillator ground states).

        Physics:
            Coherent states are minimum uncertainty states that remain Gaussian
            under harmonic evolution:

            |α⟩ = D(α)|0⟩ = exp(αa† - α*a)|0⟩

        where D(α) is the displacement operator.

        For a 2D harmonic oscillator:
            ψ(x,y) = (ω/(πℏ))^(1/2) exp(-ω(x² + y²)/(2ℏ)) exp(i·Im[α·r])

        displaced by Re[α].

        Args:
            batch_size: Number of coherent states
            alpha_real: Real part of displacement [B] (default: random in [-1.5, 1.5])
            alpha_imag: Imaginary part of displacement [B] (default: random in [-1.5, 1.5])
            omega: Oscillator frequency [B] (default: 1.0)

        Returns:
            Normalized wavefunction [B, 2, H, W]
        """
        # Default parameters
        if alpha_real is None:
            alpha_real = (torch.rand(batch_size, device=self.device) - 0.5) * 3  # [-1.5, 1.5]
        if alpha_imag is None:
            alpha_imag = (torch.rand(batch_size, device=self.device) - 0.5) * 3  # [-1.5, 1.5]
        if omega is None:
            omega = torch.ones(batch_size, device=self.device)

        # Ensure tensors are on correct device
        if not isinstance(alpha_real, torch.Tensor):
            alpha_real = torch.tensor(alpha_real, device=self.device)
        if not isinstance(alpha_imag, torch.Tensor):
            alpha_imag = torch.tensor(alpha_imag, device=self.device)
        if not isinstance(omega, torch.Tensor):
            omega = torch.tensor(omega, device=self.device)

        # Reshape for broadcasting
        alpha_real = alpha_real.view(batch_size, 1, 1)
        alpha_imag = alpha_imag.view(batch_size, 1, 1)
        omega = omega.view(batch_size, 1, 1)

        # Coherent state center (spatial displacement)
        x0 = alpha_real * torch.sqrt(self.hbar / omega)
        y0 = alpha_imag * torch.sqrt(self.hbar / omega)

        # Displacement from center
        dx = self.x_grid.unsqueeze(0) - x0  # [B, H, W]
        dy = self.y_grid.unsqueeze(0) - y0  # [B, H, W]
        r_squared = dx**2 + dy**2

        # Gaussian envelope (ground state width)
        sigma_squared = self.hbar / omega
        envelope = torch.exp(-r_squared / (2 * sigma_squared))

        # Phase contribution from momentum
        # For coherent state: p = Im[α] * √(mωℏ)
        # Simplify with m=1: p ≈ Im[α] * √(ωℏ)
        px = alpha_imag * torch.sqrt(omega * self.hbar)
        py = torch.zeros_like(px)  # No y-momentum for simplicity

        phase = (px * self.x_grid.unsqueeze(0) + py * self.y_grid.unsqueeze(0)) / self.hbar

        # Complex wavefunction
        psi_real = envelope * torch.cos(phase)
        psi_imag = envelope * torch.sin(phase)

        # Stack and normalize
        psi = torch.stack([psi_real, psi_imag], dim=1)
        psi = self._normalize_wavefunction(psi)

        return psi

    def generate_superposition(
        self,
        batch_size: int,
        num_components: int = 2,
        sigma: float = 0.8,
        separation: float = 2.0
    ) -> torch.Tensor:
        """Generate superposition states (quantum superpositions of wavepackets).

        Physics:
            ψ = Σᵢ cᵢ·ψᵢ  where Σᵢ|cᵢ|² = 1

        Creates quantum superpositions of multiple Gaussian wavepackets with
        different spatial locations, demonstrating quantum interference.

        Args:
            batch_size: Number of superposition states
            num_components: Number of wavepackets in superposition (default: 2)
            sigma: Width of each wavepacket (default: 0.8)
            separation: Typical separation between components (default: 2.0)

        Returns:
            Normalized wavefunction [B, 2, H, W]
        """
        # Initialize empty wavefunction
        psi = torch.zeros(
            batch_size, 2, self.grid_size, self.grid_size,
            device=self.device
        )

        # Generate random coefficients (complex amplitudes)
        # Uniform random phases
        phases = torch.rand(batch_size, num_components, device=self.device) * 2 * np.pi
        # Equal amplitudes (will normalize at end)
        amplitudes = torch.ones(batch_size, num_components, device=self.device) / np.sqrt(num_components)

        # Complex coefficients
        coeff_real = amplitudes * torch.cos(phases)  # [B, N]
        coeff_imag = amplitudes * torch.sin(phases)  # [B, N]

        # Generate components
        for i in range(num_components):
            # Random position for each component
            # Arrange in a circle for aesthetic interference patterns
            angle = 2 * np.pi * i / num_components
            base_x = separation * np.cos(angle)
            base_y = separation * np.sin(angle)

            # Add small random perturbation
            center_x = base_x + (torch.rand(batch_size, device=self.device) - 0.5) * 0.5
            center_y = base_y + (torch.rand(batch_size, device=self.device) - 0.5) * 0.5

            # Random momentum for each component
            momentum_x = (torch.rand(batch_size, device=self.device) - 0.5) * 0.5
            momentum_y = (torch.rand(batch_size, device=self.device) - 0.5) * 0.5

            # Generate component wavepacket
            psi_component = self.generate_gaussian_wavepacket(
                batch_size=batch_size,
                center_x=center_x,
                center_y=center_y,
                momentum_x=momentum_x,
                momentum_y=momentum_y,
                sigma=sigma
            )

            # Multiply by complex coefficient: c * ψ
            # Real part: c_r * ψ_r - c_i * ψ_i
            # Imag part: c_r * ψ_i + c_i * ψ_r
            c_r = coeff_real[:, i].view(batch_size, 1, 1)
            c_i = coeff_imag[:, i].view(batch_size, 1, 1)

            psi_real_weighted = c_r * psi_component[:, 0] - c_i * psi_component[:, 1]
            psi_imag_weighted = c_r * psi_component[:, 1] + c_i * psi_component[:, 0]

            # Add to superposition
            psi[:, 0] += psi_real_weighted
            psi[:, 1] += psi_imag_weighted

        # Normalize final superposition
        psi = self._normalize_wavefunction(psi)

        return psi

    def generate_batch(
        self,
        batch_size: int,
        ic_type: str,
        **kwargs
    ) -> torch.Tensor:
        """Generate batch of quantum initial conditions by type.

        Args:
            batch_size: Number of samples
            ic_type: Type of initial condition:
                - "quantum_gaussian_wavepacket": Gaussian wavepacket
                - "quantum_coherent_state": Coherent state
                - "quantum_superposition_2": 2-component superposition
                - "quantum_superposition_3": 3-component superposition
            **kwargs: Additional parameters for specific generators

        Returns:
            Wavefunction [B, 2, H, W]
        """
        if ic_type == "quantum_gaussian_wavepacket":
            return self.generate_gaussian_wavepacket(batch_size, **kwargs)
        elif ic_type == "quantum_coherent_state":
            return self.generate_coherent_state(batch_size, **kwargs)
        elif ic_type == "quantum_superposition_2":
            return self.generate_superposition(batch_size, num_components=2, **kwargs)
        elif ic_type == "quantum_superposition_3":
            return self.generate_superposition(batch_size, num_components=3, **kwargs)
        else:
            raise ValueError(
                f"Unknown quantum IC type: {ic_type}. "
                f"Must be one of: 'quantum_gaussian_wavepacket', 'quantum_coherent_state', "
                f"'quantum_superposition_2', 'quantum_superposition_3'"
            )
