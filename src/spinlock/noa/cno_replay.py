"""CNO Replay - Reconstruct and rollout CNO operators for NOA supervision.

This module enables state-level supervision for NOA training by:
1. Reconstructing CNO operators from stored Sobol parameter vectors
2. Rolling out CNO trajectories to produce target states
3. Comparing NOA trajectories to CNO trajectories with MSE loss

The 100k dataset stores only parameter vectors and features (not raw trajectories),
so we must replay CNO operators to get target trajectories during training.

Key insight: Since we know the exact parameter space used for generation,
we can reconstruct the same operators and get identical trajectories
(given the same IC and stochastic seed).

Usage:
    replayer = CNOReplayer.from_config("configs/experiments/local_100k_optimized.yaml")

    # During training:
    target_trajectory = replayer.rollout(
        params_vector=params[i],  # [12,] from dataset
        ic=ic,                     # [C, H, W] from dataset
        timesteps=64,
        num_realizations=1,
    )
"""

import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path
import yaml

from spinlock.operators.builder import OperatorBuilder
from spinlock.operators.parameters import OperatorParameters


class CNOReplayer:
    """Reconstruct and rollout CNO operators from stored parameter vectors.

    Enables state-level supervision for NOA training without storing
    full trajectory data in the dataset.

    Attributes:
        parameter_space: Full parameter space config from original dataset generation
        builder: OperatorBuilder for constructing operators
        device: Computation device
        operator_cache: LRU cache for avoiding repeated operator construction
    """

    def __init__(
        self,
        parameter_space: Dict[str, Dict[str, Any]],
        device: str = "cuda",
        cache_size: int = 8,
    ):
        """Initialize CNO replayer.

        Args:
            parameter_space: Full parameter space config (architecture, stochastic, operator, evolution)
            device: Computation device
            cache_size: Number of operators to cache (0 to disable)
        """
        self.parameter_space = parameter_space
        self.device = torch.device(device)
        self.builder = OperatorBuilder()

        # Flatten parameter space for map_parameters (same order as during generation)
        self._flat_param_spec = self._flatten_parameter_space(parameter_space)

        # Simple LRU cache (operator construction is expensive)
        self._cache_size = cache_size
        self._operator_cache: Dict[tuple, nn.Module] = {}
        self._cache_order: list = []

    def _flatten_parameter_space(
        self,
        parameter_space: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Flatten nested parameter space to match Sobol vector ordering.

        The Sobol sampler iterates through categories in order:
        architecture → stochastic → operator → evolution
        """
        flat = {}
        for category in ["architecture", "stochastic", "operator", "evolution"]:
            if category in parameter_space:
                for name, spec in parameter_space[category].items():
                    flat[f"{category}.{name}"] = spec
        return flat

    def _map_parameters(
        self,
        unit_params: NDArray[np.float64],
    ) -> Dict[str, Any]:
        """Map Sobol unit parameters to actual values.

        Args:
            unit_params: [0,1]^d parameter vector from dataset

        Returns:
            Dictionary with mapped parameter values
        """
        # Use builder's map_parameters with flattened spec
        flat_mapped = self.builder.map_parameters(unit_params, self._flat_param_spec)

        # Reconstruct nested structure for build_simple_cnn
        # and add default values it expects
        params = {
            "input_channels": 1,  # Always 1 for our datasets
            "output_channels": 1,
        }

        for flat_name, value in flat_mapped.items():
            # Extract just the parameter name (e.g., "architecture.num_layers" -> "num_layers")
            name = flat_name.split(".")[-1]
            params[name] = value

        return params

    def _get_cache_key(self, unit_params: NDArray[np.float64]) -> tuple:
        """Create hashable cache key from parameter vector."""
        return tuple(unit_params.tolist())

    def _build_operator(self, params: Dict[str, Any]) -> nn.Module:
        """Build SimpleCNN operator from mapped parameters."""
        operator = self.builder.build_simple_cnn(params)
        operator = operator.to(self.device)
        operator.eval()  # Inference mode
        return operator

    def get_operator(
        self,
        unit_params: Union[NDArray[np.float64], torch.Tensor],
    ) -> nn.Module:
        """Get or build operator from parameter vector.

        Uses caching to avoid repeated construction.

        Args:
            unit_params: [0,1]^d parameter vector from dataset

        Returns:
            SimpleCNNOperator ready for inference
        """
        if isinstance(unit_params, torch.Tensor):
            unit_params = unit_params.cpu().numpy()

        cache_key = self._get_cache_key(unit_params)

        # Check cache
        if cache_key in self._operator_cache:
            # Move to end of LRU order
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._operator_cache[cache_key]

        # Build new operator
        mapped_params = self._map_parameters(unit_params)
        operator = self._build_operator(mapped_params)

        # Add to cache
        if self._cache_size > 0:
            if len(self._operator_cache) >= self._cache_size:
                # Evict oldest
                oldest = self._cache_order.pop(0)
                del self._operator_cache[oldest]

            self._operator_cache[cache_key] = operator
            self._cache_order.append(cache_key)

        return operator

    @torch.no_grad()
    def rollout(
        self,
        params_vector: Union[NDArray[np.float64], torch.Tensor],
        ic: torch.Tensor,
        timesteps: int,
        num_realizations: int = 1,
        seed: Optional[int] = None,
        return_all_steps: bool = True,
    ) -> torch.Tensor:
        """Roll out CNO operator from IC to produce target trajectory.

        Args:
            params_vector: [d,] Sobol unit parameter vector from dataset
            ic: [B, C, H, W] or [C, H, W] initial condition
            timesteps: Number of timesteps to generate
            num_realizations: Number of stochastic realizations (M)
            seed: Random seed for reproducibility (None for random)
            return_all_steps: If True, return full trajectory including IC

        Returns:
            If return_all_steps and num_realizations == 1:
                Trajectory [B, T+1, C, H, W]
            If return_all_steps and num_realizations > 1:
                Trajectories [B, M, T+1, C, H, W]
            If not return_all_steps:
                Final state [B, C, H, W] or [B, M, C, H, W]
        """
        # Handle single sample IC
        if ic.dim() == 3:
            ic = ic.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

        ic = ic.to(self.device)

        # Get or build operator
        operator = self.get_operator(params_vector)

        if num_realizations == 1:
            return self._single_rollout(operator, ic, timesteps, seed, return_all_steps)
        else:
            return self._multi_rollout(operator, ic, timesteps, num_realizations, seed, return_all_steps)

    def _single_rollout(
        self,
        operator: nn.Module,
        ic: torch.Tensor,
        timesteps: int,
        seed: Optional[int],
        return_all_steps: bool,
    ) -> torch.Tensor:
        """Single realization rollout with NaN/Inf detection."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        trajectory = [ic] if return_all_steps else []
        x = ic

        for step in range(timesteps):
            try:
                x = operator(x)

                # Check for NaN/Inf to prevent hangs
                if torch.isnan(x).any() or torch.isinf(x).any():
                    raise ValueError(
                        f"NaN/Inf detected in operator output at step {step+1}/{timesteps}"
                    )

                if return_all_steps:
                    trajectory.append(x)

            except RuntimeError as e:
                # CUDA errors can cause hangs - fail fast
                raise RuntimeError(
                    f"CUDA error in operator rollout at step {step+1}/{timesteps}: {e}"
                ) from e

        if return_all_steps:
            return torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]
        else:
            return x

    def _multi_rollout(
        self,
        operator: nn.Module,
        ic: torch.Tensor,
        timesteps: int,
        num_realizations: int,
        seed: Optional[int],
        return_all_steps: bool,
    ) -> torch.Tensor:
        """Multiple realization rollout."""
        realizations = []

        for m in range(num_realizations):
            m_seed = seed + m if seed is not None else None
            traj = self._single_rollout(operator, ic, timesteps, m_seed, return_all_steps)
            realizations.append(traj)

        if return_all_steps:
            # Each traj is [B, T+1, C, H, W] -> stack to [B, M, T+1, C, H, W]
            return torch.stack(realizations, dim=1)
        else:
            # Each traj is [B, C, H, W] -> stack to [B, M, C, H, W]
            return torch.stack(realizations, dim=1)

    def clear_cache(self):
        """Clear operator cache (useful for memory management)."""
        self._operator_cache.clear()
        self._cache_order.clear()

    @classmethod
    def from_config(
        cls,
        config_path: str,
        device: str = "cuda",
        cache_size: int = 8,
    ) -> "CNOReplayer":
        """Create replayer from YAML config file.

        Args:
            config_path: Path to config YAML used for dataset generation
            device: Computation device
            cache_size: Operator cache size

        Returns:
            CNOReplayer configured with the same parameter space
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        if "parameter_space" not in config:
            raise ValueError(f"Config missing 'parameter_space': {config_path}")

        return cls(
            parameter_space=config["parameter_space"],
            device=device,
            cache_size=cache_size,
        )
