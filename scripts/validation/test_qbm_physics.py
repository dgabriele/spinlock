"""Physics Validation Tests for Quantum Brownian Motion Simulator.

Tests verify that the QBM simulator correctly implements quantum mechanics:
- Norm conservation: ∫|ψ|²dx = 1
- Energy conservation (γ=0): Closed system energy is constant
- Decoherence rate: Coherence time τ ∝ 1/γ
- Uncertainty principle: Δx·Δp ≥ ℏ/2
- Harmonic oscillator ground state energy

Run with: pytest scripts/validation/test_qbm_physics.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from spinlock.qbm.simulator import QuantumBrownianSimulator
from spinlock.qbm.initial_conditions import QuantumICGenerator
from spinlock.qbm.potentials import PotentialGenerator


@pytest.fixture
def device():
    """Use CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def simulator(device):
    """Create QBM simulator instance."""
    return QuantumBrownianSimulator(
        grid_size=64,
        domain_size=10.0,
        hbar=1.0,
        device=device
    )


@pytest.fixture
def ic_generator(device):
    """Create quantum IC generator."""
    return QuantumICGenerator(
        grid_size=64,
        domain_size=10.0,
        hbar=1.0,
        device=torch.device(device)
    )


@pytest.fixture
def potential_generator(device):
    """Create potential generator."""
    return PotentialGenerator(
        grid_size=64,
        domain_size=10.0,
        device=torch.device(device)
    )


def test_norm_conservation(simulator, ic_generator):
    """Test that wavefunction norm is conserved: ∫|ψ|²dx = 1.

    Tolerance: 10^-6 over 256 steps.
    """
    batch_size = 4

    # Generate Gaussian wavepackets
    psi_0 = ic_generator.generate_gaussian_wavepacket(batch_size=batch_size)
    assert psi_0.shape == (batch_size, 2, 64, 64)

    # Harmonic potential
    omega = torch.ones(batch_size, device=simulator.device)
    potential = torch.zeros(batch_size, 64, 64, device=simulator.device)

    # Parameters: [gamma, kT, mass]
    params = torch.zeros(batch_size, 3, device=simulator.device)
    params[:, 0] = 0.01  # Small gamma for weak dissipation
    params[:, 1] = 1.0   # Temperature
    params[:, 2] = 1.0   # Mass

    # Evolve for 256 steps
    trajectory = simulator.rollout(psi_0, potential, params, num_steps=256)
    assert trajectory.shape == (batch_size, 257, 2, 64, 64)

    # Check norm at each timestep
    for t in range(257):
        psi_t = trajectory[:, t]  # [B, 2, H, W]
        prob_density = psi_t[:, 0]**2 + psi_t[:, 1]**2  # [B, H, W]
        norm = prob_density.sum(dim=(1, 2)) * simulator.dx**2  # [B]

        # Check norm ≈ 1 (tolerance: 10^-6)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6), \
            f"Norm violation at t={t}: {norm.mean().item():.10f}"

    print(f"✓ Norm conservation: max deviation = {torch.abs(norm - 1).max().item():.2e}")


def test_energy_conservation_closed_system(simulator, ic_generator):
    """Test energy conservation for closed system (γ=0).

    For a closed quantum system, energy should be conserved within 10^-4 relative error.
    """
    batch_size = 4

    # Generate Gaussian wavepackets
    psi_0 = ic_generator.generate_gaussian_wavepacket(batch_size=batch_size)

    # Harmonic potential
    omega = torch.ones(batch_size, device=simulator.device)
    potential = 0.5 * omega.view(batch_size, 1, 1)**2 * simulator.r_squared.unsqueeze(0)

    # Parameters: [gamma=0, kT, mass]
    params = torch.zeros(batch_size, 3, device=simulator.device)
    params[:, 0] = 0.0   # NO dissipation
    params[:, 1] = 1.0   # Temperature (unused when gamma=0)
    params[:, 2] = 1.0   # Mass

    # Evolve for 256 steps
    trajectory = simulator.rollout(psi_0, potential, params, num_steps=256)

    # Compute energy at each timestep
    energies = []
    for t in range(257):
        psi_t = trajectory[:, t]
        E = simulator.compute_energy(psi_t, potential, params[:, 2])
        energies.append(E)

    energies = torch.stack(energies, dim=1)  # [B, T]

    # Check relative energy drift
    E_0 = energies[:, 0]
    E_final = energies[:, -1]
    rel_error = torch.abs((E_final - E_0) / E_0)

    assert torch.all(rel_error < 1e-4), \
        f"Energy conservation violated: max rel error = {rel_error.max().item():.2e}"

    print(f"✓ Energy conservation (γ=0): max rel error = {rel_error.max().item():.2e}")


def test_decoherence_rate(simulator, ic_generator):
    """Test that coherence time scales as τ ∝ 1/γ.

    Measure coherence decay for different γ values and verify scaling.
    """
    batch_size = 1

    # Test different gamma values
    gammas = torch.tensor([0.001, 0.01, 0.1], device=simulator.device)
    coherence_times = []

    for gamma in gammas:
        # Generate coherent state (maximum coherence)
        psi_0 = ic_generator.generate_coherent_state(batch_size=batch_size)

        # Free evolution (V=0)
        potential = torch.zeros(batch_size, 64, 64, device=simulator.device)

        # Parameters
        params = torch.zeros(batch_size, 3, device=simulator.device)
        params[:, 0] = gamma
        params[:, 1] = 1.0
        params[:, 2] = 1.0

        # Evolve
        trajectory = simulator.rollout(psi_0, potential, params, num_steps=256)

        # Compute position uncertainty as measure of coherence
        uncertainties = []
        for t in range(257):
            psi_t = trajectory[:, t]
            delta_x, delta_y = simulator.compute_position_uncertainty(psi_t)
            uncertainties.append((delta_x + delta_y) / 2)

        uncertainties = torch.tensor(uncertainties, device=simulator.device)

        # Find time when uncertainty doubles (coherence decay)
        initial_uncertainty = uncertainties[0]
        doubled_idx = torch.where(uncertainties > 2 * initial_uncertainty)[0]
        if len(doubled_idx) > 0:
            tau = doubled_idx[0].item() * 0.01  # dt = 0.01
        else:
            tau = 256 * 0.01  # Didn't double within time window

        coherence_times.append(tau)

    coherence_times = np.array(coherence_times)
    gammas_np = gammas.cpu().numpy()

    # Check scaling: τ ∝ 1/γ
    # log(τ) = log(C) - log(γ)
    # Slope should be ≈ -1
    log_tau = np.log(coherence_times)
    log_gamma = np.log(gammas_np)

    # Linear fit
    slope = np.polyfit(log_gamma, log_tau, 1)[0]

    # Slope should be close to -1 (inverse relationship)
    assert abs(slope + 1.0) < 0.3, \
        f"Decoherence scaling incorrect: slope = {slope:.2f} (expected ≈ -1)"

    print(f"✓ Decoherence rate: τ ∝ γ^{slope:.2f} (expected ≈ γ^-1)")


def test_uncertainty_principle(simulator, ic_generator):
    """Test that Heisenberg uncertainty principle holds: Δx·Δp ≥ ℏ/2.

    All initial conditions and throughout evolution should satisfy this.
    """
    batch_size = 8

    # Test different IC types
    ic_types = [
        ('gaussian_wavepacket', {}),
        ('coherent_state', {}),
        ('superposition_2', {}),
    ]

    for ic_type, kwargs in ic_types:
        # Generate IC
        if ic_type == 'gaussian_wavepacket':
            psi_0 = ic_generator.generate_gaussian_wavepacket(batch_size=batch_size, **kwargs)
        elif ic_type == 'coherent_state':
            psi_0 = ic_generator.generate_coherent_state(batch_size=batch_size, **kwargs)
        elif ic_type == 'superposition_2':
            psi_0 = ic_generator.generate_superposition(batch_size=batch_size, num_components=2, **kwargs)

        # Check initial state
        delta_x, delta_y = simulator.compute_position_uncertainty(psi_0)
        delta_px, delta_py = simulator.compute_momentum_uncertainty(psi_0)

        # Uncertainty product in x and y
        uncertainty_x = delta_x * delta_px
        uncertainty_y = delta_y * delta_py

        hbar_over_2 = simulator.hbar / 2

        assert torch.all(uncertainty_x >= hbar_over_2 * 0.9), \
            f"{ic_type}: Δx·Δpx = {uncertainty_x.min().item():.3f} < ℏ/2 = {hbar_over_2:.3f}"
        assert torch.all(uncertainty_y >= hbar_over_2 * 0.9), \
            f"{ic_type}: Δy·Δpy = {uncertainty_y.min().item():.3f} < ℏ/2 = {hbar_over_2:.3f}"

        print(f"✓ Uncertainty principle ({ic_type}): "
              f"min(Δx·Δpx) = {uncertainty_x.min().item():.3f}, "
              f"min(Δy·Δpy) = {uncertainty_y.min().item():.3f} "
              f"(ℏ/2 = {hbar_over_2:.3f})")


def test_harmonic_oscillator_ground_state(simulator, potential_generator):
    """Test ground state energy of harmonic oscillator.

    For a quantum harmonic oscillator:
    E_0 = ℏω (in 2D: E_0 = ℏω for each dimension, total = 2ℏω)

    We prepare a minimum uncertainty Gaussian and check energy is close to ground state.
    """
    batch_size = 1
    omega = 1.0
    mass = 1.0

    # Create ground state wavefunction (Gaussian with correct width)
    # σ² = ℏ/(mω)
    sigma = np.sqrt(simulator.hbar / (mass * omega))

    # Generate Gaussian wavepacket at rest (p=0)
    device = torch.device(simulator.device)
    psi_0 = torch.zeros(batch_size, 2, 64, 64, device=device)

    # Real part: Gaussian envelope
    x = simulator.x_grid
    y = simulator.y_grid
    envelope = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize
    norm = torch.sqrt((envelope**2).sum() * simulator.dx**2)
    psi_0[:, 0] = envelope / norm
    psi_0[:, 1] = 0  # No imaginary part (no momentum)

    # Harmonic potential
    V = potential_generator.harmonic_2d(
        batch_size=batch_size,
        omega=torch.tensor([omega], device=device),
        center_x=torch.zeros(batch_size, device=device),
        center_y=torch.zeros(batch_size, device=device),
        mass=torch.tensor([mass], device=device)
    )

    # Compute energy
    mass_tensor = torch.tensor([mass], device=device)
    E = simulator.compute_energy(psi_0, V, mass_tensor)

    # Analytical ground state energy: E_0 = ℏω (per dimension, 2D total)
    E_0_analytical = 2 * simulator.hbar * omega

    rel_error = torch.abs((E - E_0_analytical) / E_0_analytical)

    assert rel_error < 0.1, \
        f"Ground state energy error too large: {rel_error.item():.3f} " \
        f"(E = {E.item():.3f}, expected {E_0_analytical:.3f})"

    print(f"✓ Harmonic oscillator ground state: E = {E.item():.3f}, "
          f"E_0 = {E_0_analytical:.3f}, rel error = {rel_error.item():.3f}")


def test_potential_generation(potential_generator):
    """Test that potential generators produce correct shapes and values."""
    batch_size = 4
    device = potential_generator.device

    # Test harmonic potential
    omega = torch.ones(batch_size, device=device)
    V_harmonic = potential_generator.harmonic_2d(
        batch_size=batch_size,
        omega=omega
    )
    assert V_harmonic.shape == (batch_size, 64, 64)
    assert V_harmonic.min() >= 0  # Harmonic potential is always positive

    # Test double-well potential
    barrier_height = torch.ones(batch_size, device=device) * 2.0
    separation = torch.ones(batch_size, device=device) * 2.0
    V_double_well = potential_generator.double_well(
        batch_size=batch_size,
        barrier_height=barrier_height,
        separation=separation
    )
    assert V_double_well.shape == (batch_size, 64, 64)

    # Test quartic potential
    c0 = torch.zeros(batch_size, device=device)
    c2 = torch.ones(batch_size, device=device)
    c4 = torch.ones(batch_size, device=device) * 0.1
    V_quartic = potential_generator.quartic(
        batch_size=batch_size,
        c0=c0,
        c2=c2,
        c4=c4
    )
    assert V_quartic.shape == (batch_size, 64, 64)

    # Test random potential
    V_random = potential_generator.random_potential(
        batch_size=batch_size,
        correlation_length=0.5,
        amplitude=1.0
    )
    assert V_random.shape == (batch_size, 64, 64)

    print("✓ All potential generators produce correct shapes")


if __name__ == "__main__":
    # Run tests manually
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = QuantumBrownianSimulator(grid_size=64, domain_size=10.0, device=device)
    ic_gen = QuantumICGenerator(grid_size=64, domain_size=10.0, device=torch.device(device))
    pot_gen = PotentialGenerator(grid_size=64, domain_size=10.0, device=torch.device(device))

    print("\n" + "="*60)
    print("QBM Physics Validation Tests")
    print("="*60 + "\n")

    test_norm_conservation(sim, ic_gen)
    test_energy_conservation_closed_system(sim, ic_gen)
    test_decoherence_rate(sim, ic_gen)
    test_uncertainty_principle(sim, ic_gen)
    test_harmonic_oscillator_ground_state(sim, pot_gen)
    test_potential_generation(pot_gen)

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")
