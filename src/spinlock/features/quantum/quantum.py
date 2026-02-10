"""Main quantum feature extractor"""

import torch
from typing import Optional
from spinlock.features.registry import FeatureRegistry
from . import density_matrix, purity, entropy, coherence, variance, decoherence
from .config import QuantumConfig


class QuantumFeatureExtractor:
    """Extract quantum-specific temporal features.

    Computes per-timestep features for quantum systems:
    - Purity, linear entropy
    - Coherence measure
    - Position/momentum uncertainties
    - Von Neumann entropy (approximate)
    - Decoherence rate (optional, computed over full trace)

    All operations are GPU-accelerated and batched for efficiency.
    """

    def __init__(
        self,
        device: torch.device,
        config: Optional[QuantumConfig] = None,
        grid_size: int = 64,
    ):
        """Initialize quantum feature extractor.

        Args:
            device: PyTorch device (cuda/cpu)
            config: Configuration for feature extraction
            grid_size: Spatial grid resolution (H = W = grid_size)
        """
        self.device = device
        self.config = config or QuantumConfig()
        self.grid_size = grid_size

        # Pre-compute spatial and momentum grids
        self._setup_grids()

        # Build feature registry
        self.registry = FeatureRegistry(family_name="quantum")
        self._register_features()

    def _setup_grids(self):
        """Pre-compute coordinate grids for position/momentum operators."""
        L = self.config.domain_size
        N = self.grid_size

        # Spatial grid: [-L/2, L/2] × [-L/2, L/2]
        x = torch.linspace(-L / 2, L / 2, N, device=self.device)
        y = torch.linspace(-L / 2, L / 2, N, device=self.device)
        self.y_grid, self.x_grid = torch.meshgrid(y, x, indexing="ij")

        # Momentum grid (FFT frequencies)
        dx = L / N
        kx = torch.fft.fftfreq(N, d=dx, device=self.device)
        ky = torch.fft.fftfreq(N, d=dx, device=self.device)
        self.ky_grid, self.kx_grid = torch.meshgrid(ky, kx, indexing="ij")

        # Convert to momentum: p = ℏ·2π·k
        self.px_grid = self.config.hbar * 2 * torch.pi * self.kx_grid
        self.py_grid = self.config.hbar * 2 * torch.pi * self.ky_grid

    def _register_features(self):
        """Register all quantum features in the registry."""
        if self.config.include_purity:
            self.registry.register("purity", "quantum_state")
            self.registry.register("linear_entropy", "quantum_state")

        if self.config.include_entropy:
            self.registry.register("von_neumann_entropy_approx", "quantum_info")

        if self.config.include_coherence:
            self.registry.register("coherence_measure", "quantum_coherence")

        if self.config.include_variance:
            self.registry.register("position_uncertainty_x", "quantum_uncertainty")
            self.registry.register("position_uncertainty_y", "quantum_uncertainty")
            self.registry.register("momentum_uncertainty_px", "quantum_uncertainty")
            self.registry.register("momentum_uncertainty_py", "quantum_uncertainty")
            self.registry.register("uncertainty_product_x", "quantum_uncertainty")
            self.registry.register("uncertainty_product_y", "quantum_uncertainty")

        if self.config.include_decoherence:
            self.registry.register("decoherence_rate", "quantum_dynamics")

    def extract(self, fields: torch.Tensor) -> torch.Tensor:
        """Extract quantum features from trajectories.

        Args:
            fields: [N, T, C, H, W] where C=2 (Re, Im channels)

        Returns:
            [N, T, D_quantum] quantum features

        Raises:
            AssertionError: If input shape is incompatible
        """
        N, T, C, H, W = fields.shape

        assert C == 2, f"Quantum features require 2 channels (Re, Im), got {C}"
        assert (
            H == W == self.grid_size
        ), f"Grid size mismatch: {H}×{W} vs {self.grid_size}"

        # Convert to proper shape: [N, T, 2, H, W]
        psi = fields

        # Flatten spatial dimensions for density matrix ops
        psi_flat = density_matrix.wavefunction_to_density_matrix(psi)  # [N, T, D]

        features = []

        # Purity & linear entropy
        if self.config.include_purity:
            pur = purity.compute_purity(psi_flat)  # [N, T]
            lin_ent = purity.compute_linear_entropy(pur, dimension=H * W)  # [N, T]
            features.extend([pur.unsqueeze(-1), lin_ent.unsqueeze(-1)])

        # Von Neumann entropy (approximate)
        if self.config.include_entropy:
            prob = density_matrix.compute_diagonal_elements(psi_flat)
            vn_ent = entropy.compute_von_neumann_entropy_approximate(prob)  # [N, T]
            features.append(vn_ent.unsqueeze(-1))

        # Coherence
        if self.config.include_coherence:
            coh = coherence.compute_coherence_measure(psi_flat)  # [N, T]
            features.append(coh.unsqueeze(-1))

        # Position & momentum uncertainties
        if self.config.include_variance:
            delta_x, delta_y = variance.compute_position_variance(
                psi, self.x_grid, self.y_grid
            )
            delta_px, delta_py = variance.compute_momentum_variance(
                psi, self.px_grid, self.py_grid, self.config.hbar
            )

            # Uncertainty products (should satisfy Δx·Δp ≥ ℏ/2)
            prod_x = delta_x * delta_px
            prod_y = delta_y * delta_py

            features.extend(
                [
                    delta_x.unsqueeze(-1),
                    delta_y.unsqueeze(-1),
                    delta_px.unsqueeze(-1),
                    delta_py.unsqueeze(-1),
                    prod_x.unsqueeze(-1),
                    prod_y.unsqueeze(-1),
                ]
            )

        # Concatenate all features: [N, T, D_quantum]
        quantum_features = torch.cat(features, dim=-1)

        # Optional: Decoherence rate (summary statistic, replicated across T)
        if self.config.include_decoherence:
            # Use coherence trace for decoherence rate estimation
            if self.config.include_coherence:
                coh_trace = coh  # [N, T]
                gamma = decoherence.estimate_decoherence_rate(coh_trace)  # [N]
                # Replicate across time: [N] → [N, T, 1]
                gamma_expanded = gamma.unsqueeze(1).unsqueeze(2).expand(N, T, 1)
                quantum_features = torch.cat([quantum_features, gamma_expanded], dim=-1)

        return quantum_features

    def get_feature_registry(self) -> FeatureRegistry:
        """Get feature registry for HDF5 storage."""
        return self.registry
