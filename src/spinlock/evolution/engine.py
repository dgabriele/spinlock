"""
Operator rollout for neural operators.

Core orchestrator that integrates update policies, metrics computation,
and trajectory storage into a unified GPU-accelerated pipeline.

"Rollout" is the standard ML term for autoregressively applying a model
over time (common in neural ODEs, model-based RL, sequence modeling).

Designed for reusability across multiple use cases:
- Visualization
- Feature extraction
- Scientific analysis
- Model debugging and replay
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Literal
from pathlib import Path
from tqdm import tqdm

from .policies import UpdatePolicy, create_update_policy
from .metrics import MetricsComputer, TrajectoryMetrics
from .trajectory import TrajectoryWriter


class OperatorRollout:
    """
    Executes temporal evolution of neural operators.

    Orchestrates the complete evolution pipeline:
    1. Initialize state X_0
    2. For t = 1 to T:
       - Apply operator: O_θ(X_{t-1})
       - Apply update policy: X_t = policy.update(...)
       - Apply post-processing (normalization, clamping)
       - Compute metrics
       - Store state
    3. Return trajectory and metrics

    Features:
    - Configurable update policies
    - Multiple stochastic realizations
    - Optional normalization/clamping
    - GPU memory management
    - Progress tracking

    Example:
        ```python
        from spinlock.evolution import TemporalEvolutionEngine
        from spinlock.operators import NeuralOperator

        engine = TemporalEvolutionEngine(
            policy="convex",
            alpha=0.7,
            num_timesteps=100,
            device=torch.device("cuda")
        )

        trajectories, metrics = engine.evolve_operator(
            operator=neural_operator,
            initial_condition=X0,
            num_realizations=10
        )
        ```
    """

    def __init__(
        self,
        policy: Optional[UpdatePolicy] = None,
        policy_type: str = "convex",
        num_timesteps: int = 100,
        dt: float = 0.01,
        alpha: float = 0.5,
        normalization: Optional[Literal["minmax", "zscore"]] = None,
        clamp_range: Optional[Tuple[float, float]] = None,
        compute_metrics: bool = True,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize operator rollout.

        Args:
            policy: UpdatePolicy instance (overrides policy_type)
            policy_type: Policy type if policy is None ("autoregressive", "residual", "convex")
            num_timesteps: Number of timesteps to evolve
            dt: Step size for residual policy
            alpha: Mixing parameter for convex policy
            normalization: Post-update normalization ("minmax", "zscore", None)
            clamp_range: Optional (min, max) clamping range
            compute_metrics: Whether to compute trajectory metrics
            device: Torch device (cuda or cpu)
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.compute_metrics = compute_metrics
        self.normalization = normalization
        self.clamp_range = clamp_range

        # Create or use provided update policy
        if policy is not None:
            self.policy = policy
        else:
            self.policy = create_update_policy(
                policy_type=policy_type,
                dt=dt,
                alpha=alpha
            )

        # Metrics computer
        if compute_metrics:
            self.metrics_computer = MetricsComputer()

    def evolve_operator(
        self,
        operator: nn.Module,
        initial_condition: torch.Tensor,  # [C, H, W]
        num_realizations: int = 1,
        base_seed: int = 0,
        show_progress: bool = False
    ) -> Tuple[torch.Tensor, List[List[TrajectoryMetrics]]]:
        """
        Evolve single operator from initial condition with multiple realizations.

        Args:
            operator: Neural operator (nn.Module) to evolve
            initial_condition: Initial state [C, H, W]
            num_realizations: Number of stochastic realizations
            base_seed: Base seed for reproducibility
            show_progress: Show progress bar

        Returns:
            Tuple of:
            - trajectories: [M, T, C, H, W] where M = num_realizations
            - metrics: List[List[TrajectoryMetrics]] (M realizations, T steps each)

        Example:
            ```python
            trajectories, metrics = engine.evolve_operator(
                operator=neural_operator,
                initial_condition=torch.randn(3, 64, 64),
                num_realizations=10,
                base_seed=42
            )
            print(trajectories.shape)  # [10, 100, 3, 64, 64]
            ```
        """
        operator.eval()
        trajectories = []
        all_metrics = []

        iterator = range(num_realizations)
        if show_progress:
            iterator = tqdm(iterator, desc="Realizations", leave=False)

        with torch.no_grad():
            for m in iterator:
                seed = base_seed + m
                traj, metrics = self._evolve_single_realization(
                    operator, initial_condition, seed, show_progress=show_progress
                )
                trajectories.append(traj)
                all_metrics.append(metrics)

        return torch.stack(trajectories), all_metrics

    def _evolve_single_realization(
        self,
        operator: nn.Module,
        X0: torch.Tensor,  # [C, H, W]
        seed: int,
        show_progress: bool = False
    ) -> Tuple[torch.Tensor, List[TrajectoryMetrics]]:
        """
        Evolve single stochastic realization.

        Args:
            operator: Neural operator
            X0: Initial condition [C, H, W]
            seed: Random seed
            show_progress: Show timestep progress

        Returns:
            Tuple of (trajectory [T, C, H, W], metrics [T])
        """
        trajectory = []
        metrics = []

        # Initialize state and apply post-processing to initial condition
        X_t = X0.unsqueeze(0).to(self.device)  # [1, C, H, W]
        X_t = self._postprocess(X_t)  # Apply normalization/clamping to IC
        trajectory.append(X_t.squeeze(0).clone())

        # Compute initial metrics
        if self.compute_metrics:
            m = self.metrics_computer.compute_all(X_t)
            metrics.append(m)

        # Temporal evolution
        iterator = range(1, self.num_timesteps)
        if show_progress:
            iterator = tqdm(iterator, desc="Timesteps", leave=False)

        for t in iterator:
            # Apply operator with seed for stochasticity
            torch.manual_seed(seed + t)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + t)

            O_theta_X = operator(X_t)

            # Apply update policy
            X_next = self.policy.update(X_t, O_theta_X)

            # Post-processing
            X_next = self._postprocess(X_next)

            # Store state
            trajectory.append(X_next.squeeze(0).clone())

            # Compute metrics
            if self.compute_metrics:
                m = self.metrics_computer.compute_all(X_next, X_t)
                metrics.append(m)

            # Update state
            X_t = X_next

        return torch.stack(trajectory), metrics

    def _postprocess(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply post-update normalization and clamping.

        Args:
            X: State tensor [B, C, H, W]

        Returns:
            Processed state
        """
        # Normalization
        if self.normalization == "minmax":
            X_min = X.min()
            X_max = X.max()
            if X_max - X_min > 1e-8:
                X = (X - X_min) / (X_max - X_min)

        elif self.normalization == "zscore":
            X = (X - X.mean()) / (X.std() + 1e-8)

        # Clamping
        if self.clamp_range is not None:
            X = torch.clamp(X, self.clamp_range[0], self.clamp_range[1])

        return X

    def evolve_batch(
        self,
        operators: List[nn.Module],
        initial_conditions: torch.Tensor,  # [N, C, H, W]
        num_realizations: int,
        output_path: Optional[Path] = None,
        show_progress: bool = True
    ) -> Optional[Tuple[torch.Tensor, List[List[List[TrajectoryMetrics]]]]]:
        """
        Evolve batch of operators and optionally save to HDF5.

        Args:
            operators: List of N neural operators
            initial_conditions: Initial conditions [N, C, H, W]
            num_realizations: Number of realizations per operator
            output_path: Optional HDF5 output path
            show_progress: Show progress bar

        Returns:
            If output_path is None: (trajectories [N, M, T, C, H, W], metrics)
            If output_path is provided: None (saves to disk)

        Example:
            ```python
            # Save to disk
            engine.evolve_batch(
                operators=operators,
                initial_conditions=ICs,
                num_realizations=10,
                output_path=Path("trajectories.h5")
            )

            # Return in memory
            trajectories, metrics = engine.evolve_batch(
                operators=operators,
                initial_conditions=ICs,
                num_realizations=10
            )
            ```
        """
        N = len(operators)

        if initial_conditions.shape[0] != N:
            raise ValueError(
                f"Number of initial conditions ({initial_conditions.shape[0]}) "
                f"must match number of operators ({N})"
            )

        # Setup output writer if saving to disk
        writer = None
        if output_path is not None:
            C, H, W = initial_conditions.shape[1:]
            writer = TrajectoryWriter(
                output_path=output_path,
                num_operators=N,
                num_realizations=num_realizations,
                num_timesteps=self.num_timesteps,
                grid_size=H,
                num_channels=C
            ).__enter__()

        # Accumulate results if not saving to disk
        results_traj = [] if output_path is None else None
        results_metrics = [] if output_path is None else None

        # Iterate over operators
        iterator = enumerate(operators)
        if show_progress:
            iterator = tqdm(iterator, total=N, desc="Evolving operators")

        for i, operator in iterator:
            IC = initial_conditions[i]
            traj, metrics = self.evolve_operator(
                operator, IC, num_realizations, base_seed=i * 1000,
                show_progress=False  # Don't show nested progress
            )

            if writer:
                # Save to HDF5
                for m in range(num_realizations):
                    writer.write_trajectory(i, m, traj[m], metrics[m])
            else:
                # Accumulate in memory
                results_traj.append(traj)
                results_metrics.append(metrics)

        # Write metadata and close file
        if writer:
            writer.write_metadata({
                "policy": self.policy.name(),
                "num_timesteps": self.num_timesteps,
                "num_operators": N,
                "num_realizations": num_realizations,
                "normalization": self.normalization or "none",
                "clamp_range": list(self.clamp_range) if self.clamp_range else None
            })
            writer.__exit__(None, None, None)
            return None
        else:
            return torch.stack(results_traj), results_metrics
