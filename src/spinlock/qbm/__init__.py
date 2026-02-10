"""Quantum Brownian Motion (Caldeira-Leggett Model) Module.

This module implements quantum Brownian motion simulations for open quantum systems
that exhibit decoherence and the quantum-to-classical transition.

Components:
    - QuantumBrownianSimulator: Core physics engine with split-operator method
    - QuantumICGenerator: Quantum initial condition generators
    - PotentialGenerator: Potential energy landscape generators
    - QBMDatasetGenerator: Full dataset generation pipeline

Physics:
    The Caldeira-Leggett model describes a quantum particle coupled to a thermal bath:

    H = p²/(2m) + V(x,y)  (System Hamiltonian)

    Master equation:
    dρ/dt = -i[H, ρ]/ℏ + L[ρ]  (Lindblad dissipator)

    where L[ρ] describes environmental decoherence.

References:
    - Caldeira & Leggett (1983). Path integral approach to quantum Brownian motion
    - Breuer & Petruccione (2002). The Theory of Open Quantum Systems
"""

from spinlock.qbm.simulator import QuantumBrownianSimulator
from spinlock.qbm.initial_conditions import QuantumICGenerator
from spinlock.qbm.potentials import PotentialGenerator

__all__ = [
    "QuantumBrownianSimulator",
    "QuantumICGenerator",
    "PotentialGenerator",
]
