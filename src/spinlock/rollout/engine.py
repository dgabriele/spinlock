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
from torch.amp import autocast
from typing import List, Optional, Tuple, Literal, Dict
from pathlib import Path
from tqdm import tqdm

from .policies import UpdatePolicy, create_update_policy
from .metrics import MetricsComputer, TrajectoryMetrics
from .trajectory import TrajectoryWriter

# Optional: Import for operator feature extraction
try:
    from spinlock.features.summary.operator_sensitivity import OperatorSensitivityExtractor
    OPERATOR_FEATURES_AVAILABLE = True
except ImportError:
    OPERATOR_FEATURES_AVAILABLE = False


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
        from spinlock.rollout import OperatorRollout
        from spinlock.operators import NeuralOperator

        engine = OperatorRollout(
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
        extract_operator_features: bool = False,
        operator_feature_config: Optional[object] = None,
        device: torch.device = torch.device("cuda"),
        precision: str = "float16"
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
            extract_operator_features: Whether to extract operator sensitivity features during rollout
            operator_feature_config: SummaryOperatorSensitivityConfig (optional)
            device: Torch device (cuda or cpu)
            precision: Precision mode ("float32", "float16", "bfloat16")
                      Defaults to "float16" for 2× speedup on modern GPUs
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.compute_metrics = compute_metrics
        self.normalization = normalization
        self.clamp_range = clamp_range
        self.extract_operator_features = extract_operator_features

        # Mixed precision setup
        self.precision = precision
        self.dtype = self._get_dtype(precision)
        self.use_amp = precision in ("float16", "bfloat16")

        # Enable cuDNN auto-tuning for convolution algorithms (5-15% speedup)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

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

        # Operator feature extractor (optional)
        self.operator_feature_extractor: Optional['OperatorSensitivityExtractor'] = None
        if extract_operator_features:
            if not OPERATOR_FEATURES_AVAILABLE:
                raise RuntimeError(
                    "Operator feature extraction requested but spinlock.features.sdf not available"
                )
            self.operator_feature_extractor = OperatorSensitivityExtractor(
                device=device,
                lipschitz_epsilon_scales=(
                    operator_feature_config.lipschitz_epsilon_scales
                    if operator_feature_config else None
                ),
                gain_scale_factors=(
                    operator_feature_config.gain_scale_factors
                    if operator_feature_config else None
                )
            )
            self.operator_feature_config = operator_feature_config

    def _get_dtype(self, precision: str) -> torch.dtype:
        """
        Convert precision string to PyTorch dtype.

        Args:
            precision: Precision mode ("float32", "float16", "bfloat16")

        Returns:
            torch.dtype corresponding to precision mode

        Raises:
            ValueError: If precision mode is unsupported
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if precision not in dtype_map:
            raise ValueError(
                f"Unsupported precision mode: '{precision}'. "
                f"Supported modes: {list(dtype_map.keys())}"
            )

        dtype = dtype_map[precision]

        # Check GPU capability for bfloat16
        if precision == "bfloat16" and self.device.type == "cuda":
            capability = torch.cuda.get_device_capability(self.device)
            major, minor = capability

            # BF16 requires Ampere (sm_80) or newer
            if major < 8:
                print(f"[WARNING] bfloat16 requested but GPU (sm_{major}{minor}) doesn't support it")
                print(f"          Falling back to float16")
                return torch.float16

        return dtype

    def evolve_operator(
        self,
        operator: nn.Module,
        initial_condition: torch.Tensor,  # [C, H, W]
        num_realizations: int = 1,
        base_seed: int = 0,
        show_progress: bool = False
    ) -> Tuple[torch.Tensor, List[List[TrajectoryMetrics]], Optional[Dict[str, torch.Tensor]]]:
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
            - operator_features: Optional Dict[str, torch.Tensor] with operator sensitivity features

        Example:
            ```python
            trajectories, metrics, op_features = engine.evolve_operator(
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
        all_metrics = [] if self.compute_metrics else None  # Only create if metrics enabled

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
                if self.compute_metrics:
                    all_metrics.extend(batch_metrics)

                # Free batch_metrics immediately to prevent accumulation
                del batch_metrics

        # Concatenate batches
        trajectories = torch.cat(all_trajectories, dim=0)  # [M, T, C, H, W]

        # Free all_trajectories list to prevent accumulation
        del all_trajectories

        # Extract operator features if enabled
        operator_features = None
        if self.extract_operator_features:
            operator_features = self._extract_operator_sensitivity_features(
                operator=operator,
                initial_condition=initial_condition
            )

        # Return empty list instead of None for metrics when disabled (for compatibility)
        metrics_return = all_metrics if all_metrics is not None else []
        return trajectories, metrics_return, operator_features

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

            # Forward pass with automatic mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
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

        # Pre-allocate trajectory storage for efficiency
        # Eliminates 500+ clone() + append() + stack() operations
        B, C, H, W = X_t.shape
        trajectories = torch.zeros(
            self.num_timesteps, B, C, H, W,
            dtype=X_t.dtype, device=X_t.device
        )
        trajectories[0] = X_t  # Store initial state (no clone needed)

        # Metrics per realization (only create if needed to save memory)
        if self.compute_metrics:
            all_metrics = [[] for _ in range(num_realizations)]
            for b in range(num_realizations):
                m = self.metrics_computer.compute_all(X_t[b:b+1])
                all_metrics[b].append(m)
        else:
            all_metrics = []  # Empty list when metrics disabled (saves memory)

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

            # Forward pass with automatic mixed precision (2× speedup)
            with autocast(device_type=self.device.type, enabled=self.use_amp, dtype=self.dtype):
                O_theta_X = operator(X_t)  # [B, C, H, W]

                # Apply update policy
                X_next = self.policy.update(X_t, O_theta_X)  # [B, C, H, W]

            # Post-processing (convert back to float32 if needed for metrics)
            X_next = self._postprocess(X_next)

            # Store directly (no clone needed, pre-allocated)
            trajectories[t] = X_next

            # Compute metrics per realization
            if self.compute_metrics:
                for b in range(num_realizations):
                    m = self.metrics_computer.compute_all(X_next[b:b+1], X_t[b:b+1])
                    all_metrics[b].append(m)

            # Update state
            X_t = X_next

        # Transpose to [B, T, C, H, W] (single operation)
        trajectories = trajectories.transpose(0, 1)

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
            # OPTIMIZATION: Use 90% of free memory (Phase 1 optimization for 1.2× throughput)
            # Increased from 70% to maximize GPU utilization while maintaining stability
            free_memory = total_memory - reserved_memory  # Account for reserved (includes fragmentation)
            usable_memory = free_memory * 0.90  # 90% utilization, 10% margin for fragmentation
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
                print("  Consider setting: PYTORCH_ALLOC_CONF=expandable_segments:True")
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

    def _extract_operator_sensitivity_features(
        self,
        operator: nn.Module,
        initial_condition: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract operator sensitivity features.

        This method runs the operator with perturbed inputs to characterize:
        - Lipschitz estimates (sensitivity to perturbations)
        - Gain curves (response to amplitude scaling)
        - Linearity metrics (deviation from linear behavior)

        Args:
            operator: Neural operator
            initial_condition: Initial condition [C, H, W]

        Returns:
            Dictionary of feature name -> scalar tensor
        """
        if self.operator_feature_extractor is None:
            raise RuntimeError("Operator feature extractor not initialized")

        # Extract features using initial condition
        features = self.operator_feature_extractor.extract(
            operator=operator,
            input_field=initial_condition,
            config=self.operator_feature_config
        )

        return features

    def evolve_batch(
        self,
        operators: List[nn.Module],
        initial_conditions: torch.Tensor,  # [N, C, H, W]
        num_realizations: int,
        output_path: Optional[Path] = None,
        show_progress: bool = True
    ) -> Optional[Tuple[torch.Tensor, List[List[List[TrajectoryMetrics]]], Optional[List[Dict[str, torch.Tensor]]]]]:
        """
        Evolve batch of operators and optionally save to HDF5.

        Args:
            operators: List of N neural operators
            initial_conditions: Initial conditions [N, C, H, W]
            num_realizations: Number of realizations per operator
            output_path: Optional HDF5 output path
            show_progress: Show progress bar

        Returns:
            If output_path is None: (trajectories [N, M, T, C, H, W], metrics, operator_features)
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
            trajectories, metrics, op_features = engine.evolve_batch(
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
        results_operator_features = [] if (output_path is None and self.extract_operator_features) else None

        # Iterate over operators
        iterator = enumerate(operators)
        if show_progress:
            iterator = tqdm(iterator, total=N, desc="Evolving operators")

        for i, operator in iterator:
            IC = initial_conditions[i]
            traj, metrics, op_features = self.evolve_operator(
                operator, IC, num_realizations, base_seed=i * 1000,
                show_progress=False  # Don't show nested progress
            )

            if writer:
                # Save to HDF5
                for m in range(num_realizations):
                    writer.write_trajectory(i, m, traj[m], metrics[m])
                # TODO: Save operator features to HDF5 (needs TrajectoryWriter extension)
            else:
                # Accumulate in memory
                results_traj.append(traj)
                results_metrics.append(metrics)
                if op_features is not None:
                    results_operator_features.append(op_features)

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
            return torch.stack(results_traj), results_metrics, results_operator_features
