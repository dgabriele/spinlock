"""QBM Replayer for reconstructing quantum trajectories from stored parameters.

Enables efficient replay of QBM simulations for:
- MNO training supervision (state-level ground truth)
- Validation (compare MNO predictions to QBM ground truth)
- Dataset augmentation (generate additional samples)

Unlike CNOReplayer which caches trained operators, QBMReplayer directly runs
the physics simulator using stored Sobol parameter vectors.

Example:
    ```python
    replayer = QBMReplayer.from_config("configs/qbm/substrate.yaml")

    # Replay from stored parameters
    params = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1, 0.9])  # [9,] Sobol vector
    trajectory = replayer.rollout(params, num_realizations=3, num_timesteps=256, seed=42)
    # Shape: [3, 256, 2, 64, 64] (M realizations, T steps, 2 channels Re/Im, 64x64 grid)

    # Batch rollout
    batch_params = np.random.rand(16, 9)
    batch_trajectory = replayer.rollout_batch(batch_params, num_realizations=3, num_timesteps=256)
    # Shape: [16, 3, 256, 2, 64, 64]
    ```

Physics:
    QBM simulations use the Caldeira-Leggett model for open quantum systems:
    H = p²/(2m) + V(x,y)

    Parameters ([9,] Sobol vector in [0,1]):
    - gamma: Bath coupling strength
    - kT: Temperature
    - mass: Particle mass
    - potential_type: Discrete choice (harmonic, double_well, quartic, random)
    - potential_strength: Energy scale
    - potential_width: Characteristic length
    - [Additional potential-specific parameters]
"""

import torch
import numpy as np
import yaml
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from numpy.typing import NDArray

from spinlock.qbm.simulator import QuantumBrownianSimulator
from spinlock.qbm.potentials import PotentialGenerator
from spinlock.qbm.initial_conditions import QuantumICGenerator


class QBMReplayer:
    """Reconstruct and rollout QBM simulations from stored parameter vectors.

    Maps Sobol [0,1]^9 parameter vectors → physical parameters → QBM trajectories.

    Args:
        parameter_space: Parameter space configuration dict with:
            - operator: Parameter definitions (gamma, kT, mass, potential params)
        grid_size: Spatial resolution (default: 64)
        domain_size: Physical domain size (default: 10.0)
        hbar: Reduced Planck constant (default: 1.0)
        dt: Time step (default: 0.01)
        device: Device for computation (default: "cuda")
        ic_distribution: IC type probabilities (default: matches dataset generation)
            - quantum_gaussian_wavepacket: 0.40
            - quantum_coherent_state: 0.30
            - quantum_superposition_2: 0.15
            - quantum_superposition_3: 0.15
    """

    def __init__(
        self,
        parameter_space: Dict[str, Dict[str, Any]],
        grid_size: int = 64,
        domain_size: float = 10.0,
        hbar: float = 1.0,
        dt: float = 0.01,
        device: str = "cuda",
        ic_distribution: Optional[Dict[str, float]] = None,
    ):
        self.parameter_space = parameter_space
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.hbar = hbar
        self.dt = dt
        self.device = torch.device(device)

        # Default IC distribution (matches dataset generation)
        self.ic_distribution = ic_distribution or {
            "quantum_gaussian_wavepacket": 0.40,
            "quantum_coherent_state": 0.30,
            "quantum_superposition_2": 0.15,
            "quantum_superposition_3": 0.15,
        }

        # Initialize QBM components
        self.simulator = QuantumBrownianSimulator(
            grid_size=grid_size,
            domain_size=domain_size,
            hbar=hbar,
            device=str(self.device)
        )

        self.potential_generator = PotentialGenerator(
            grid_size=grid_size,
            domain_size=domain_size,
            device=self.device
        )

        self.ic_generator = QuantumICGenerator(
            grid_size=grid_size,
            domain_size=domain_size,
            hbar=hbar,
            device=self.device
        )

        # Parse parameter space to understand mapping
        self._parse_parameter_space()

    def _parse_parameter_space(self):
        """Parse parameter space config to understand Sobol → physical parameter mapping."""
        # Extract parameter definitions
        operator_params = self.parameter_space.get("operator", {})

        # Store parameter metadata
        self.param_names = list(operator_params.keys())
        self.param_configs = operator_params

        # Expected parameter order in Sobol vector
        expected_params = ["gamma", "kT", "mass", "potential_type", "potential_strength", "potential_width"]

        # Validate critical parameters exist
        for param in expected_params[:6]:  # First 6 are critical
            if param not in self.param_names:
                raise ValueError(
                    f"Parameter space missing required parameter: '{param}'. "
                    f"Found: {self.param_names}"
                )

    def _sobol_to_physical(self, sobol_vector: NDArray[np.float64]) -> Dict[str, Any]:
        """Convert Sobol [0,1]^9 vector to physical parameters.

        Args:
            sobol_vector: Parameter vector in [0,1] (shape: [9,])

        Returns:
            Dict of physical parameters
        """
        params = {}

        for i, param_name in enumerate(self.param_names):
            if i >= len(sobol_vector):
                break

            param_config = self.param_configs[param_name]
            param_type = param_config.get("type")
            sobol_value = sobol_vector[i]

            if param_type == "continuous":
                # Continuous parameter: map [0,1] → [min, max]
                distribution = param_config.get("distribution", "uniform")
                min_val = param_config["min"]
                max_val = param_config["max"]

                if distribution == "log_uniform":
                    # Log-uniform sampling: log-space interpolation
                    log_min = np.log(min_val)
                    log_max = np.log(max_val)
                    physical_value = np.exp(log_min + sobol_value * (log_max - log_min))
                else:
                    # Uniform sampling: linear interpolation
                    physical_value = min_val + sobol_value * (max_val - min_val)

                params[param_name] = float(physical_value)

            elif param_type == "discrete":
                # Discrete parameter: map [0,1] → categorical choice
                values = param_config["values"]
                probabilities = param_config.get("probabilities")

                if probabilities is None:
                    # Uniform distribution over values
                    idx = int(sobol_value * len(values))
                    idx = min(idx, len(values) - 1)  # Clamp to valid range
                else:
                    # Sample according to cumulative probabilities
                    cumsum = np.cumsum(probabilities)
                    idx = np.searchsorted(cumsum, sobol_value)
                    idx = min(idx, len(values) - 1)

                params[param_name] = values[idx]

        return params

    def _generate_potential(self, params: Dict[str, Any], batch_size: int = 1) -> torch.Tensor:
        """Generate potential V(x,y) from physical parameters.

        Args:
            params: Physical parameters dict
            batch_size: Number of potentials to generate

        Returns:
            Potential tensor [B, H, W]
        """
        potential_type = params.get("potential_type", "harmonic")
        strength = params.get("potential_strength", 1.0)
        width = params.get("potential_width", 1.0)

        # Convert scalars to batch tensors
        omega_batch = torch.full((batch_size,), strength, device=self.device)
        width_batch = torch.full((batch_size,), width, device=self.device)

        if potential_type == "harmonic":
            V = self.potential_generator.harmonic_2d(
                batch_size=batch_size,
                omega=omega_batch,
                center_x=torch.zeros(batch_size, device=self.device),
                center_y=torch.zeros(batch_size, device=self.device)
            )
        elif potential_type == "double_well":
            V = self.potential_generator.double_well(
                batch_size=batch_size,
                barrier_height=omega_batch,
                separation=width_batch
            )
        elif potential_type == "quartic":
            # Quartic: V(r) = c0 + c2*r^2 + c4*r^4
            # Map strength/width to coefficients
            V = self.potential_generator.quartic(
                batch_size=batch_size,
                c0=torch.zeros(batch_size, device=self.device),  # No offset
                c2=width_batch,  # Quadratic term from width
                c4=omega_batch   # Quartic term from strength
            )
        elif potential_type == "random":
            V = self.potential_generator.random_potential(
                batch_size=batch_size,
                correlation_length=width_batch.item() if batch_size == 1 else 0.5,
                amplitude=omega_batch.item() if batch_size == 1 else 1.0
            )
        else:
            raise ValueError(f"Unknown potential type: {potential_type}")

        return V

    def _generate_initial_condition(
        self,
        num_realizations: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """Generate quantum initial condition following dataset IC distribution.

        Args:
            num_realizations: Number of IC realizations (M)
            seed: Random seed for reproducibility

        Returns:
            Initial wavefunction [M, 2, H, W]
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Sample IC type according to distribution
        ic_types = list(self.ic_distribution.keys())
        ic_probs = list(self.ic_distribution.values())
        ic_type = rng.choice(ic_types, p=ic_probs)

        # Generate IC based on type
        if ic_type == "quantum_gaussian_wavepacket":
            # Gaussian wavepacket with random center and momentum
            center_x = rng.uniform(-2.0, 2.0, size=(num_realizations,))
            center_y = rng.uniform(-2.0, 2.0, size=(num_realizations,))
            momentum_x = rng.uniform(-1.0, 1.0, size=(num_realizations,))
            momentum_y = rng.uniform(-1.0, 1.0, size=(num_realizations,))

            psi = self.ic_generator.generate_gaussian_wavepacket(
                batch_size=num_realizations,
                center_x=torch.tensor(center_x, device=self.device),
                center_y=torch.tensor(center_y, device=self.device),
                momentum_x=torch.tensor(momentum_x, device=self.device),
                momentum_y=torch.tensor(momentum_y, device=self.device),
                sigma=1.0
            )

        elif ic_type == "quantum_coherent_state":
            # Coherent state (displaced harmonic oscillator ground state)
            # Parameters: alpha_real, alpha_imag control displacement
            alpha_real = rng.uniform(-1.5, 1.5, size=(num_realizations,))
            alpha_imag = rng.uniform(-1.5, 1.5, size=(num_realizations,))

            psi = self.ic_generator.generate_coherent_state(
                batch_size=num_realizations,
                alpha_real=torch.tensor(alpha_real, device=self.device),
                alpha_imag=torch.tensor(alpha_imag, device=self.device),
                omega=torch.ones(num_realizations, device=self.device)
            )

        elif ic_type == "quantum_superposition_2":
            # Superposition of 2 Gaussian wavepackets
            separation = float(rng.uniform(1.5, 3.0))
            psi = self.ic_generator.generate_superposition(
                batch_size=num_realizations,
                num_components=2,
                sigma=0.8,
                separation=separation
            )

        elif ic_type == "quantum_superposition_3":
            # Superposition of 3 Gaussian wavepackets
            separation = float(rng.uniform(1.5, 3.0))
            psi = self.ic_generator.generate_superposition(
                batch_size=num_realizations,
                num_components=3,
                sigma=0.8,
                separation=separation
            )

        else:
            raise ValueError(f"Unknown IC type: {ic_type}")

        return psi

    @torch.no_grad()
    def rollout(
        self,
        params_vector: NDArray[np.float64],
        num_realizations: int = 3,
        num_timesteps: int = 256,
        seed: Optional[int] = None,
        # Compatibility with CNOReplayer interface
        ic: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        return_all_steps: bool = True,
    ) -> torch.Tensor:
        """Roll out QBM simulation from stored parameters.

        Args:
            params_vector: Sobol parameter vector [9,] in [0,1]
            num_realizations: Number of stochastic realizations (M)
            num_timesteps: Number of timesteps to simulate (T)
            seed: Random seed for reproducible ICs
            ic: Initial condition (ignored - QBM generates ICs internally). For API compatibility with CNOReplayer.
            timesteps: Alias for num_timesteps (for CNOReplayer compatibility)
            return_all_steps: Always True for QBM (for CNOReplayer compatibility)

        Returns:
            Trajectory [M, T, 2, H, W] (M realizations, T timesteps, 2 channels Re/Im)

        Note:
            QBM generates initial conditions internally from the IC distribution
            (quantum_gaussian_wavepacket, quantum_coherent_state, etc.).
            The `ic` parameter is ignored to maintain API compatibility with CNOReplayer.
        """
        # Handle parameter aliases for CNOReplayer compatibility
        if timesteps is not None:
            num_timesteps = timesteps
        # Convert Sobol vector to physical parameters
        params = self._sobol_to_physical(params_vector)

        # Generate potential V(x,y) [1, H, W]
        V = self._generate_potential(params, batch_size=1)

        # Generate initial conditions [M, 2, H, W]
        psi_0 = self._generate_initial_condition(num_realizations, seed=seed)

        # Expand potential to match realizations [M, H, W]
        V_expanded = V.expand(num_realizations, -1, -1)

        # Create parameter tensor for simulator [M, 3] (gamma, kT, mass)
        gamma = params.get("gamma", 0.01)
        kT = params.get("kT", 1.0)
        mass = params.get("mass", 1.0)

        sim_params = torch.tensor(
            [[gamma, kT, mass]] * num_realizations,
            device=self.device,
            dtype=torch.float32
        )

        # Run QBM simulation [M, T+1, 2, H, W]
        trajectory = self.simulator.rollout(
            psi_0=psi_0,
            potential=V_expanded,
            params=sim_params,
            num_steps=num_timesteps,
            dt=self.dt
        )

        # Return trajectory excluding initial condition [M, T, 2, H, W]
        # Ensure float32 for PyTorch compatibility
        return trajectory[:, 1:, :, :, :].to(torch.float32)

    @torch.no_grad()
    def rollout_batch(
        self,
        params_batch: NDArray[np.float64],
        num_realizations: int = 3,
        num_timesteps: int = 256,
        seed: Optional[int] = None,
        # Compatibility with CNOReplayer interface
        timesteps: Optional[int] = None,
        return_all_steps: bool = True,
        return_ics: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """Batch rollout for efficiency.

        Args:
            params_batch: Batch of Sobol vectors [B, 9]
            num_realizations: Realizations per sample (M)
            num_timesteps: Timesteps to simulate (T)
            seed: Base random seed (incremented for each batch element)
            timesteps: Alias for num_timesteps (for CNOReplayer compatibility)
            return_all_steps: Always True for QBM (for CNOReplayer compatibility)
            return_ics: If True, also return initial conditions [B, M, 2, H, W]

        Returns:
            trajectories: [B, M, T, 2, H, W]
            ics (only if return_ics=True): [B, M, 2, H, W]
        """
        # Handle parameter alias for CNOReplayer compatibility
        if timesteps is not None:
            num_timesteps = timesteps

        batch_size = params_batch.shape[0]

        # Convert all parameters to physical parameters
        all_params = [self._sobol_to_physical(params_batch[i]) for i in range(batch_size)]

        # Collect parameters for batch generation
        potential_types = [p.get("potential_type", "harmonic") for p in all_params]
        strengths = torch.tensor([p.get("potential_strength", 1.0) for p in all_params], device=self.device)
        widths = torch.tensor([p.get("potential_width", 1.0) for p in all_params], device=self.device)
        gammas = torch.tensor([p.get("gamma", 0.01) for p in all_params], device=self.device)
        kTs = torch.tensor([p.get("kT", 1.0) for p in all_params], device=self.device)
        masses = torch.tensor([p.get("mass", 1.0) for p in all_params], device=self.device)

        # Check if all samples use the same potential type (common case)
        if len(set(potential_types)) == 1:
            # All same type - can batch everything efficiently!
            ptype = potential_types[0]

            if ptype == "harmonic":
                V_batch = self.potential_generator.harmonic_2d(
                    batch_size=batch_size,
                    omega=strengths,
                    center_x=torch.zeros(batch_size, device=self.device),
                    center_y=torch.zeros(batch_size, device=self.device)
                )
            elif ptype == "double_well":
                V_batch = self.potential_generator.double_well(
                    batch_size=batch_size,
                    barrier_height=strengths,
                    separation=widths
                )
            elif ptype == "quartic":
                V_batch = self.potential_generator.quartic(
                    batch_size=batch_size,
                    c0=torch.zeros(batch_size, device=self.device),
                    c2=widths,
                    c4=strengths
                )
            elif ptype == "random":
                # Random requires sequential generation
                V_batch = torch.stack([
                    self.potential_generator.random_potential(
                        batch_size=1,
                        correlation_length=widths[i].item(),
                        amplitude=strengths[i].item()
                    ).squeeze(0) for i in range(batch_size)
                ], dim=0)
            else:
                raise ValueError(f"Unknown potential type: {ptype}")

        else:
            # Mixed types - generate sequentially (rare case)
            V_list = []
            for i, ptype in enumerate(potential_types):
                V = self._generate_potential(all_params[i], batch_size=1)
                V_list.append(V.squeeze(0))
            V_batch = torch.stack(V_list, dim=0)

        # Generate ICs for all batch*realizations at once
        # Shape: [B*M, 2, H, W]
        total_sims = batch_size * num_realizations
        psi_all = self._generate_initial_condition(total_sims, seed=seed)

        # Prepare for batched simulation:
        # Expand each potential for M realizations: [B, H, W] -> [B*M, H, W]
        V_expanded = V_batch.repeat_interleave(num_realizations, dim=0)

        # Repeat sim params for M realizations: [B, 3] -> [B*M, 3]
        sim_params_batch = torch.stack([gammas, kTs, masses], dim=1)  # [B, 3]
        sim_params_expanded = sim_params_batch.repeat_interleave(num_realizations, dim=0)  # [B*M, 3]

        # Single batched QBM simulation call! [B*M, T+1, 2, H, W]
        trajectory_flat = self.simulator.rollout(
            psi_0=psi_all,  # [B*M, 2, H, W]
            potential=V_expanded,  # [B*M, H, W]
            params=sim_params_expanded,  # [B*M, 3]
            num_steps=num_timesteps,
            dt=self.dt,
            return_all_steps=True
        )

        # Exclude IC and convert to float32: [B*M, T, 2, H, W]
        trajectory_flat = trajectory_flat[:, 1:, :, :, :].to(torch.float32)

        # Reshape to [B, M, T, 2, H, W]
        trajectories = trajectory_flat.view(batch_size, num_realizations, num_timesteps, 2, self.grid_size, self.grid_size)

        if return_ics:
            # Reshape ICs to [B, M, 2, H, W]
            ics = psi_all.to(torch.float32).view(
                batch_size, num_realizations, 2, self.grid_size, self.grid_size,
            )
            return trajectories, ics

        return trajectories

    def rollout_batch_with_explicit_ic(
        self,
        params_batch: NDArray[np.float64],
        psi_0: "torch.Tensor",
        num_timesteps: int = 256,
    ) -> "torch.Tensor":
        """Rollout with explicitly provided initial conditions.

        Unlike rollout_batch() which generates random ICs, this method uses
        the provided psi_0 directly. Used by Mode B of the roundtrip
        consistency loss: decoded ICs from the VQ pipeline are fed back
        into the simulator for ground-truth trajectory generation.

        Args:
            params_batch: Sobol vectors [B, 9] in [0,1]
            psi_0: Explicit initial conditions [B, 2, H, W]
            num_timesteps: Number of simulation timesteps

        Returns:
            trajectories: [B, T, 2, H, W] float32
        """
        batch_size = params_batch.shape[0]

        # Convert Sobol → physical parameters → potential
        all_params = [self._sobol_to_physical(params_batch[i]) for i in range(batch_size)]

        potential_types = [p.get("potential_type", "harmonic") for p in all_params]
        strengths = torch.tensor([p.get("potential_strength", 1.0) for p in all_params], device=self.device)
        widths = torch.tensor([p.get("potential_width", 1.0) for p in all_params], device=self.device)
        gammas = torch.tensor([p.get("gamma", 0.01) for p in all_params], device=self.device)
        kTs = torch.tensor([p.get("kT", 1.0) for p in all_params], device=self.device)
        masses = torch.tensor([p.get("mass", 1.0) for p in all_params], device=self.device)

        # Generate potentials (same logic as rollout_batch)
        if len(set(potential_types)) == 1:
            ptype = potential_types[0]
            if ptype == "harmonic":
                V_batch = self.potential_generator.harmonic_2d(
                    batch_size=batch_size,
                    omega=strengths,
                    center_x=torch.zeros(batch_size, device=self.device),
                    center_y=torch.zeros(batch_size, device=self.device)
                )
            elif ptype == "double_well":
                V_batch = self.potential_generator.double_well(
                    batch_size=batch_size,
                    barrier_height=strengths,
                    separation=widths
                )
            elif ptype == "quartic":
                V_batch = self.potential_generator.quartic(
                    batch_size=batch_size,
                    c0=torch.zeros(batch_size, device=self.device),
                    c2=widths,
                    c4=strengths
                )
            elif ptype == "random":
                V_batch = torch.stack([
                    self.potential_generator.random_potential(
                        batch_size=1,
                        correlation_length=widths[i].item(),
                        amplitude=strengths[i].item()
                    ).squeeze(0) for i in range(batch_size)
                ], dim=0)
            else:
                raise ValueError(f"Unknown potential type: {ptype}")
        else:
            V_list = []
            for i, ptype in enumerate(potential_types):
                V = self._generate_potential(all_params[i], batch_size=1)
                V_list.append(V.squeeze(0))
            V_batch = torch.stack(V_list, dim=0)

        # Use provided ICs directly (move to device if needed)
        psi_0_device = psi_0.to(self.device)

        # Simulation parameters
        sim_params = torch.stack([gammas, kTs, masses], dim=1)  # [B, 3]

        # Single batched simulation
        trajectory = self.simulator.rollout(
            psi_0=psi_0_device,
            potential=V_batch,
            params=sim_params,
            num_steps=num_timesteps,
            dt=self.dt,
            return_all_steps=True,
        )

        # Exclude IC step: [B, T+1, 2, H, W] -> [B, T, 2, H, W]
        return trajectory[:, 1:, :, :, :].to(torch.float32)

    @classmethod
    def from_config(cls, config_path: str, device: str = "cuda") -> "QBMReplayer":
        """Create replayer from YAML config file.

        Args:
            config_path: Path to QBM substrate config (configs/qbm/substrate.yaml)
            device: Device for computation

        Returns:
            Configured QBMReplayer instance

        Example:
            >>> replayer = QBMReplayer.from_config("configs/qbm/substrate.yaml")
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract parameter space
        parameter_space = config.get("parameter_space", {})

        # Extract simulation settings
        sim_config = config.get("simulation", {})
        grid_size = sim_config.get("grid_size", 64)
        domain_size = sim_config.get("domain_size", 10.0)
        hbar = sim_config.get("hbar", 1.0)
        dt = sim_config.get("dt", 0.01)

        # Extract IC distribution
        ic_distribution = config.get("ic_distribution", None)

        return cls(
            parameter_space=parameter_space,
            grid_size=grid_size,
            domain_size=domain_size,
            hbar=hbar,
            dt=dt,
            device=device,
            ic_distribution=ic_distribution
        )

    def __repr__(self) -> str:
        return (
            f"QBMReplayer(grid_size={self.grid_size}, "
            f"domain_size={self.domain_size}, device={self.device})"
        )
