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
        Evolve single operator from initial condition with multiple realizations (BATCHED).

        Uses intelligent batching to process realizations in parallel for ~10x speedup.
        Automatically calibrates batch size based on GPU memory to prevent OOM.

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

        # Determine optimal batch size for realizations
        batch_size = self._calibrate_batch_size(
            operator, initial_condition, num_realizations
        )

        # Process realizations in batches
        all_trajectories = []
        all_metrics = []

        num_batches = (num_realizations + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_realizations)
                batch_num_realizations = batch_end - batch_start

                # Process this batch of realizations in parallel
                batch_trajs, batch_metrics = self._evolve_batched_realizations(
                    operator=operator,
                    initial_condition=initial_condition,
                    num_realizations=batch_num_realizations,
                    base_seed=base_seed + batch_start,
                    show_progress=show_progress and (batch_idx == 0)  # Only show for first batch
                )

                all_trajectories.append(batch_trajs)
                all_metrics.extend(batch_metrics)

        # Concatenate batches
        trajectories = torch.cat(all_trajectories, dim=0)  # [M, T, C, H, W]

        return trajectories, all_metrics

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

    def _evolve_batched_realizations(
        self,
        operator: nn.Module,
        initial_condition: torch.Tensor,  # [C, H, W]
        num_realizations: int,
        base_seed: int,
        show_progress: bool = False
    ) -> Tuple[torch.Tensor, List[List[TrajectoryMetrics]]]:
        """
        Evolve multiple realizations in parallel (BATCHED for 10x speedup).

        Args:
            operator: Neural operator
            initial_condition: Initial condition [C, H, W]
            num_realizations: Number of realizations to process in parallel
            base_seed: Base seed for reproducibility
            show_progress: Show timestep progress

        Returns:
            Tuple of (trajectories [B, T, C, H, W], metrics [B][T])
        """
        # Replicate initial condition for batch
        X_t = initial_condition.unsqueeze(0).repeat(num_realizations, 1, 1, 1)  # [B, C, H, W]
        X_t = X_t.to(self.device)
        X_t = self._postprocess(X_t)

        # Storage for trajectories (list of tensors, will stack at end)
        trajectory_storage = [X_t.clone()]  # List of [B, C, H, W]

        # Metrics per realization
        all_metrics = [[] for _ in range(num_realizations)]
        if self.compute_metrics:
            for b in range(num_realizations):
                m = self.metrics_computer.compute_all(X_t[b:b+1])
                all_metrics[b].append(m)

        # Temporal evolution
        iterator = range(1, self.num_timesteps)
        if show_progress:
            iterator = tqdm(iterator, desc="Timesteps", leave=False)

        for t in iterator:
            # Set seeds for each realization independently
            # Note: We need different random state per realization in the batch
            # This is handled by operator's internal dropout/noise using batch dimension
            seeds = [base_seed + b * self.num_timesteps + t for b in range(num_realizations)]

            # For simplicity, use base_seed + t (realizations differ via dropout)
            torch.manual_seed(base_seed + t)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed + t)

            # Forward pass for entire batch in parallel
            O_theta_X = operator(X_t)  # [B, C, H, W]

            # Apply update policy
            X_next = self.policy.update(X_t, O_theta_X)  # [B, C, H, W]

            # Post-processing
            X_next = self._postprocess(X_next)

            # Store
            trajectory_storage.append(X_next.clone())

            # Compute metrics per realization
            if self.compute_metrics:
                for b in range(num_realizations):
                    m = self.metrics_computer.compute_all(X_next[b:b+1], X_t[b:b+1])
                    all_metrics[b].append(m)

            # Update state
            X_t = X_next

        # Stack trajectories: [T, B, C, H, W] -> [B, T, C, H, W]
        trajectories = torch.stack(trajectory_storage, dim=0)  # [T, B, C, H, W]
        trajectories = trajectories.transpose(0, 1)  # [B, T, C, H, W]

        return trajectories, all_metrics

    def _calibrate_batch_size(
        self,
        operator: nn.Module,
        initial_condition: torch.Tensor,
        num_realizations: int,
    ) -> int:
        """
        Calibrate optimal batch size based on GPU memory.

        Runs a test forward pass to measure memory usage, then determines
        safe batch size based on available GPU RAM.

        Args:
            operator: Neural operator to test
            initial_condition: Sample IC [C, H, W]
            num_realizations: Target number of realizations

        Returns:
            Optimal batch size (between 1 and num_realizations)
        """
        if not torch.cuda.is_available():
            # CPU: use smaller batches to avoid excessive RAM
            return min(num_realizations, 4)

        # Start with conservative estimate
        test_batch_size = min(num_realizations, 8)

        try:
            # Clear cache aggressively to minimize fragmentation
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()

            # Measure baseline memory before test
            baseline_allocated = allocated_memory

            # Test forward pass with test batch size
            with torch.no_grad():
                X_test = initial_condition.unsqueeze(0).repeat(test_batch_size, 1, 1, 1).to(self.device)
                _ = operator(X_test)
                torch.cuda.synchronize()

            # Measure memory used by test batch
            peak_memory = torch.cuda.max_memory_allocated()
            memory_per_sample = (peak_memory - baseline_allocated) / test_batch_size

            # Estimate total memory needed for full evolution:
            # 1. Forward pass memory (measured above)
            # 2. Trajectory storage: num_timesteps × batch_size × state_size
            C, H, W = initial_condition.shape
            bytes_per_state = C * H * W * 4  # float32
            trajectory_memory_per_sample = self.num_timesteps * bytes_per_state
            total_memory_per_sample = memory_per_sample + trajectory_memory_per_sample

            # Calculate truly available memory (accounting for what's already used + fragmentation)
            # OPTIMIZATION: Use 70% of free memory (increased from 35% for better throughput)
            free_memory = total_memory - reserved_memory  # Account for reserved (includes fragmentation)
            usable_memory = free_memory * 0.70  # 70% safe, 30% margin for fragmentation
            safe_batch_size = int(usable_memory / total_memory_per_sample)

            # Clamp to reasonable range
            batch_size = max(1, min(safe_batch_size, num_realizations))

            # Clean up test allocation
            del X_test
            torch.cuda.empty_cache()

            return batch_size

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("⚠ GPU memory fragmentation detected during calibration.")
                print("  Consider setting: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                # Fallback to very conservative batch size
                return min(num_realizations, 2)
            raise

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
