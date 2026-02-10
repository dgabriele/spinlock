"""Quantum Brownian Motion Simulator.

Implements the split-operator method for quantum evolution with Lindblad dissipation.
All computations are GPU-accelerated for efficient batch processing.

Physics:
    The Caldeira-Leggett model describes a quantum particle in a potential V(x,y)
    coupled to a thermal bath:

    H = p²/(2m) + V(x,y)

    Time evolution uses the split-operator method:
    U(dt) ≈ exp[-iH_kin dt/(2ℏ)] exp[-iH_pot dt/ℏ] exp[-iH_kin dt/(2ℏ)] exp[-L dt]

    where L is the Lindblad dissipator for environmental coupling.

Method:
    1. Kinetic evolution (half-step) in momentum space via FFT
    2. Potential evolution (full step) in real space
    3. Kinetic evolution (half-step) in momentum space
    4. Dissipation via Lindblad operators
    5. Renormalization to preserve probability

References:
    - Breuer & Petruccione (2002). The Theory of Open Quantum Systems
    - Gardiner & Zoller (2004). Quantum Noise
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class QuantumBrownianSimulator:
    """GPU-accelerated quantum Brownian motion simulator.

    Uses split-operator method with FFT for kinetic evolution and
    Lindblad master equation for environmental decoherence.

    Example:
        ```python
        sim = QuantumBrownianSimulator(
            grid_size=64,
            domain_size=10.0,
            device="cuda"
        )

        # Initial Gaussian wavepacket [B, 2, H, W]
        psi_0 = ...

        # Harmonic potential [B, H, W]
        potential = ...

        # Parameters [B, P]: gamma, kT, omega_c, mass, ...
        params = ...

        # Evolve for 256 steps
        trajectory = sim.rollout(psi_0, potential, params, num_steps=256)
        # trajectory shape: [B, T+1, 2, H, W]
        ```

    Args:
        grid_size: Number of grid points per dimension (default: 64)
        domain_size: Physical size of domain (default: 10.0)
        hbar: Reduced Planck constant (default: 1.0 in natural units)
        device: Torch device ("cuda" or "cpu")
    """

    def __init__(
        self,
        grid_size: int = 64,
        domain_size: float = 10.0,
        hbar: float = 1.0,
        device: str = "cuda"
    ):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.hbar = hbar
        self.device = torch.device(device)

        # Spatial grid spacing
        self.dx = domain_size / grid_size

        # Pre-compute momentum grid (k-space)
        self._setup_momentum_grid()

        # Pre-compute spatial grid for position operator
        self._setup_spatial_grid()

    def _setup_momentum_grid(self):
        """Pre-compute momentum space grid for FFT-based kinetic evolution."""
        # Frequency grid (FFT convention)
        kx = torch.fft.fftfreq(self.grid_size, d=self.dx, device=self.device)
        ky = torch.fft.fftfreq(self.grid_size, d=self.dx, device=self.device)

        # 2D momentum grids
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing='ij')

        # Convert frequency to momentum: p = ℏk
        self.px_grid = self.hbar * 2 * np.pi * kx_grid  # [H, W]
        self.py_grid = self.hbar * 2 * np.pi * ky_grid  # [H, W]

        # Kinetic energy operator: p²/(2m) = (px² + py²)/(2m)
        # We'll apply mass-dependent factor during evolution
        self.p_squared = self.px_grid**2 + self.py_grid**2  # [H, W]

    def _setup_spatial_grid(self):
        """Pre-compute spatial coordinate grids."""
        # Spatial coordinates centered at origin
        x = torch.linspace(-self.domain_size/2, self.domain_size/2, self.grid_size, device=self.device)
        y = torch.linspace(-self.domain_size/2, self.domain_size/2, self.grid_size, device=self.device)

        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        self.x_grid = x_grid  # [H, W]
        self.y_grid = y_grid  # [H, W]
        self.r_squared = x_grid**2 + y_grid**2  # [H, W]

    def _complex_wavefunction(self, psi: torch.Tensor) -> torch.Tensor:
        """Convert real-valued [B, 2, H, W] to complex [B, H, W].

        Args:
            psi: Wavefunction with shape [B, 2, H, W] where channel 0 is real, channel 1 is imaginary

        Returns:
            Complex wavefunction [B, H, W]
        """
        return torch.complex(psi[:, 0], psi[:, 1])

    def _real_valued_wavefunction(self, psi_complex: torch.Tensor) -> torch.Tensor:
        """Convert complex [B, H, W] to real-valued [B, 2, H, W].

        Args:
            psi_complex: Complex wavefunction [B, H, W]

        Returns:
            Real-valued wavefunction [B, 2, H, W]
        """
        return torch.stack([psi_complex.real, psi_complex.imag], dim=1)

    def _kinetic_evolution(
        self,
        psi: torch.Tensor,
        mass: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Apply kinetic evolution operator in momentum space.

        exp[-i p²/(2m) dt/ℏ]

        Args:
            psi: Wavefunction [B, 2, H, W]
            mass: Particle mass [B] or [B, 1]
            dt: Time step

        Returns:
            Evolved wavefunction [B, 2, H, W]
        """
        batch_size = psi.shape[0]

        # Convert to complex
        psi_complex = self._complex_wavefunction(psi)  # [B, H, W]

        # FFT to momentum space
        psi_k = torch.fft.fft2(psi_complex)  # [B, H, W]

        # Kinetic energy phase factor: exp[-i p²/(2m) dt/ℏ]
        # Reshape mass for broadcasting: [B] -> [B, 1, 1]
        if mass.dim() == 1:
            mass = mass.view(batch_size, 1, 1)
        elif mass.dim() == 2:
            mass = mass.view(batch_size, 1, 1)

        # p_squared is [H, W], need to broadcast with [B, 1, 1]
        kinetic_phase = -1j * self.p_squared.unsqueeze(0) / (2 * mass * self.hbar) * dt
        kinetic_operator = torch.exp(kinetic_phase)  # [B, H, W]

        # Apply kinetic operator
        psi_k_evolved = psi_k * kinetic_operator

        # IFFT back to real space
        psi_evolved = torch.fft.ifft2(psi_k_evolved)  # [B, H, W]

        # Convert back to real-valued representation
        return self._real_valued_wavefunction(psi_evolved)

    def _potential_evolution(
        self,
        psi: torch.Tensor,
        potential: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Apply potential evolution operator in real space.

        exp[-i V(x,y) dt/ℏ]

        Args:
            psi: Wavefunction [B, 2, H, W]
            potential: Potential energy [B, H, W]
            dt: Time step

        Returns:
            Evolved wavefunction [B, 2, H, W]
        """
        # Convert to complex
        psi_complex = self._complex_wavefunction(psi)  # [B, H, W]

        # Potential phase factor: exp[-i V dt/ℏ]
        potential_phase = -1j * potential / self.hbar * dt
        potential_operator = torch.exp(potential_phase)  # [B, H, W]

        # Apply potential operator
        psi_evolved = psi_complex * potential_operator

        # Convert back to real-valued representation
        return self._real_valued_wavefunction(psi_evolved)

    def _apply_dissipation(
        self,
        psi: torch.Tensor,
        gamma: torch.Tensor,
        kT: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Apply Lindblad dissipation for environmental coupling.

        This implements a simplified high-temperature Lindblad equation:
        dρ/dt = -γ/2 [x, [x, ρ]] - γ/(2β) [p, [p, ρ]]

        For wavefunctions, we approximate this with damping + thermal noise.

        Args:
            psi: Wavefunction [B, 2, H, W]
            gamma: Bath coupling strength [B]
            kT: Temperature (energy units) [B]
            dt: Time step

        Returns:
            Wavefunction with dissipation [B, 2, H, W]
        """
        batch_size = psi.shape[0]

        # Reshape gamma and kT for broadcasting
        if gamma.dim() == 1:
            gamma = gamma.view(batch_size, 1, 1, 1)
        if kT.dim() == 1:
            kT = kT.view(batch_size, 1, 1, 1)

        # Damping factor: exp(-γ dt / 2)
        # This models decoherence without full density matrix evolution
        damping = torch.exp(-gamma * dt / 2)

        # Apply damping to wavefunction
        psi_damped = psi * damping

        # Thermal noise (Gaussian random field added to wavefunction)
        # Variance scales with √(γ kT dt)
        noise_amplitude = torch.sqrt(gamma * kT * dt / self.hbar)

        # Add noise independently to real and imaginary parts
        noise = torch.randn_like(psi) * noise_amplitude

        psi_dissipated = psi_damped + noise

        return psi_dissipated

    def _renormalize(self, psi: torch.Tensor) -> torch.Tensor:
        """Renormalize wavefunction to preserve probability.

        ∫|ψ|² dx dy = 1

        Args:
            psi: Wavefunction [B, 2, H, W]

        Returns:
            Normalized wavefunction [B, 2, H, W]
        """
        # Compute norm: ∫|ψ|² dx
        # |ψ|² = Re² + Im²
        prob_density = psi[:, 0]**2 + psi[:, 1]**2  # [B, H, W]

        # Integrate over space (sum * dx²)
        norm_squared = prob_density.sum(dim=(1, 2)) * self.dx**2  # [B]

        # Normalization factor
        norm = torch.sqrt(norm_squared).view(-1, 1, 1, 1)  # [B, 1, 1, 1]

        # Prevent division by zero
        norm = torch.clamp(norm, min=1e-10)

        return psi / norm

    def evolve_step(
        self,
        psi: torch.Tensor,
        potential: torch.Tensor,
        params: torch.Tensor,
        dt: float = 0.01
    ) -> torch.Tensor:
        """Single time step evolution using split-operator method.

        Args:
            psi: Wavefunction [B, 2, H, W] (Re, Im)
            potential: Potential energy [B, H, W]
            params: Physical parameters [B, P] where:
                    params[:, 0] = gamma (bath coupling)
                    params[:, 1] = kT (temperature)
                    params[:, 2] = mass (particle mass)
                    Additional params depend on configuration
            dt: Time step (default: 0.01)

        Returns:
            Evolved wavefunction [B, 2, H, W]
        """
        # Extract parameters
        gamma = params[:, 0]  # Bath coupling
        kT = params[:, 1]     # Temperature
        mass = params[:, 2]   # Particle mass

        # Split-operator sequence:
        # 1. Kinetic evolution (half-step)
        psi = self._kinetic_evolution(psi, mass, dt / 2)

        # 2. Potential evolution (full step)
        psi = self._potential_evolution(psi, potential, dt)

        # 3. Kinetic evolution (half-step)
        psi = self._kinetic_evolution(psi, mass, dt / 2)

        # 4. Dissipation (Lindblad)
        psi = self._apply_dissipation(psi, gamma, kT, dt)

        # 5. Renormalize
        psi = self._renormalize(psi)

        return psi

    def rollout(
        self,
        psi_0: torch.Tensor,
        potential: torch.Tensor,
        params: torch.Tensor,
        num_steps: int = 256,
        dt: float = 0.01,
        return_all_steps: bool = True
    ) -> torch.Tensor:
        """Full trajectory rollout.

        Args:
            psi_0: Initial wavefunction [B, 2, H, W]
            potential: Potential energy [B, H, W]
            params: Physical parameters [B, P]
            num_steps: Number of time steps (default: 256)
            dt: Time step (default: 0.01)
            return_all_steps: Return full trajectory (default: True)

        Returns:
            If return_all_steps=True:
                Trajectory [B, T+1, 2, H, W] (includes initial condition)
            If return_all_steps=False:
                Final state [B, 2, H, W]
        """
        batch_size = psi_0.shape[0]
        psi = psi_0

        if return_all_steps:
            # Pre-allocate trajectory array
            trajectory = torch.zeros(
                batch_size, num_steps + 1, 2, self.grid_size, self.grid_size,
                device=self.device, dtype=psi_0.dtype
            )
            trajectory[:, 0] = psi_0

            # Time evolution loop
            for t in range(num_steps):
                psi = self.evolve_step(psi, potential, params, dt)
                trajectory[:, t + 1] = psi

            return trajectory
        else:
            # Just return final state
            for t in range(num_steps):
                psi = self.evolve_step(psi, potential, params, dt)

            return psi

    def compute_energy(
        self,
        psi: torch.Tensor,
        potential: torch.Tensor,
        mass: torch.Tensor
    ) -> torch.Tensor:
        """Compute expectation value of energy: E = ⟨ψ|H|ψ⟩.

        H = p²/(2m) + V(x,y)

        Args:
            psi: Wavefunction [B, 2, H, W]
            potential: Potential energy [B, H, W]
            mass: Particle mass [B]

        Returns:
            Energy expectation value [B]
        """
        batch_size = psi.shape[0]

        # Convert to complex
        psi_complex = self._complex_wavefunction(psi)  # [B, H, W]

        # Kinetic energy: ⟨p²/(2m)⟩
        # FFT to momentum space
        psi_k = torch.fft.fft2(psi_complex)

        # |ψ(k)|² * p²/(2m) integrated over k-space
        if mass.dim() == 1:
            mass = mass.view(batch_size, 1, 1)

        kinetic_density = (psi_k.abs()**2) * self.p_squared.unsqueeze(0) / (2 * mass)
        kinetic_energy = kinetic_density.sum(dim=(1, 2)) * self.dx**2  # [B]

        # Potential energy: ⟨V⟩
        prob_density = psi[:, 0]**2 + psi[:, 1]**2  # [B, H, W]
        potential_density = prob_density * potential
        potential_energy = potential_density.sum(dim=(1, 2)) * self.dx**2  # [B]

        # Total energy
        total_energy = kinetic_energy + potential_energy

        return total_energy

    def compute_position_uncertainty(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute position uncertainties Δx and Δy.

        Args:
            psi: Wavefunction [B, 2, H, W]

        Returns:
            Tuple of (Δx, Δy) each with shape [B]
        """
        # Probability density
        prob_density = psi[:, 0]**2 + psi[:, 1]**2  # [B, H, W]

        # Mean position: ⟨x⟩, ⟨y⟩
        x_mean = (prob_density * self.x_grid.unsqueeze(0)).sum(dim=(1, 2)) * self.dx**2  # [B]
        y_mean = (prob_density * self.y_grid.unsqueeze(0)).sum(dim=(1, 2)) * self.dx**2  # [B]

        # Mean square: ⟨x²⟩, ⟨y²⟩
        x_squared_mean = (prob_density * self.x_grid.unsqueeze(0)**2).sum(dim=(1, 2)) * self.dx**2
        y_squared_mean = (prob_density * self.y_grid.unsqueeze(0)**2).sum(dim=(1, 2)) * self.dx**2

        # Uncertainty: Δx = √(⟨x²⟩ - ⟨x⟩²)
        delta_x = torch.sqrt(torch.clamp(x_squared_mean - x_mean**2, min=0))
        delta_y = torch.sqrt(torch.clamp(y_squared_mean - y_mean**2, min=0))

        return delta_x, delta_y

    def compute_momentum_uncertainty(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute momentum uncertainties Δpx and Δpy.

        Args:
            psi: Wavefunction [B, 2, H, W]

        Returns:
            Tuple of (Δpx, Δpy) each with shape [B]
        """
        # Convert to complex and FFT
        psi_complex = self._complex_wavefunction(psi)
        psi_k = torch.fft.fft2(psi_complex)

        # Momentum space probability density
        prob_density_k = psi_k.abs()**2  # [B, H, W]

        # Normalize (probability in k-space)
        norm_k = prob_density_k.sum(dim=(1, 2), keepdim=True)
        prob_density_k = prob_density_k / torch.clamp(norm_k, min=1e-10)

        # Mean momentum: ⟨px⟩, ⟨py⟩
        px_mean = (prob_density_k * self.px_grid.unsqueeze(0)).sum(dim=(1, 2))
        py_mean = (prob_density_k * self.py_grid.unsqueeze(0)).sum(dim=(1, 2))

        # Mean square: ⟨px²⟩, ⟨py²⟩
        px_squared_mean = (prob_density_k * self.px_grid.unsqueeze(0)**2).sum(dim=(1, 2))
        py_squared_mean = (prob_density_k * self.py_grid.unsqueeze(0)**2).sum(dim=(1, 2))

        # Uncertainty: Δp = √(⟨p²⟩ - ⟨p⟩²)
        delta_px = torch.sqrt(torch.clamp(px_squared_mean - px_mean**2, min=0))
        delta_py = torch.sqrt(torch.clamp(py_squared_mean - py_mean**2, min=0))

        return delta_px, delta_py
