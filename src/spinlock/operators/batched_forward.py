"""
Batched Operator Execution with vmap.

EXPERIMENTAL: Phase 2 CUDA optimization attempt.

Uses torch.func.vmap and functional_call to batch multiple operators.
In theory, this should enable batch execution with ONE kernel launch instead of N.

**BENCHMARK RESULTS (January 2026):**
    - vmap batched execution is ~25% SLOWER than sequential
    - Overhead from functional_call + vmap exceeds kernel fusion benefits
    - Instance normalization and complex module structures don't vectorize well
    - NOT RECOMMENDED for production use

This module is retained for:
    1. Reference implementation
    2. Future PyTorch improvements to vmap
    3. Testing on simpler model architectures

For production, use Phase 1 (architecture partitioning + torch.compile) instead.

Example:
    >>> from spinlock.operators.batched_forward import run_batched_partition
    >>> # Given partition with 100 operators
    >>> outputs = run_batched_partition(
    ...     template=compiled_template,
    ...     state_dicts=list_of_100_state_dicts,
    ...     inputs=batched_inputs,  # [100, C, H, W]
    ... )
    >>> # ONE kernel launch for all 100 operators

Author: Claude (Anthropic)
Date: January 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import torch.func as func


def stack_parameters(
    state_dicts: List[Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Stack N operator state_dicts into batched parameter dict.

    Each parameter becomes [N, ...] where N is batch dimension.

    Args:
        state_dicts: List of N state_dicts with identical keys
        device: Target device (uses first state_dict's device if None)

    Returns:
        Dict mapping param names to batched tensors [N, ...]

    Raises:
        ValueError: If state_dicts have different keys or shapes
    """
    if not state_dicts:
        raise ValueError("state_dicts cannot be empty")

    # Validate all state_dicts have same keys
    reference_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], 1):
        if set(sd.keys()) != reference_keys:
            raise ValueError(
                f"State dict {i} has different keys than reference. "
                f"Missing: {reference_keys - set(sd.keys())}, "
                f"Extra: {set(sd.keys()) - reference_keys}"
            )

    # Stack each parameter
    batched = {}
    target_device = device or next(iter(state_dicts[0].values())).device

    for key in state_dicts[0].keys():
        tensors = [sd[key].to(target_device) for sd in state_dicts]
        batched[key] = torch.stack(tensors, dim=0)  # [N, ...]

    return batched


def create_batched_forward(
    template: nn.Module,
    in_dims: tuple = (0, 0),
    randomness: str = "different",
) -> callable:
    """
    Create a vmap-compatible batched forward function.

    Uses torch.func.functional_call to separate parameters from model structure,
    then vmaps over both parameters and inputs.

    Args:
        template: Template model (defines graph structure)
        in_dims: vmap input dimensions (params_dim, inputs_dim)
        randomness: vmap randomness mode ("same", "different", "error")
            - "different": Each batch element gets independent random samples
            - "same": All batch elements share the same random samples
            - "error": Raise error if randomness is encountered (default torch behavior)

    Returns:
        Batched forward function: (batched_params, batched_inputs) -> batched_outputs
    """
    # Get base model for functional call
    # Handle both regular modules and torch.compile wrapped modules
    if hasattr(template, "_orig_mod"):
        base_model = template._orig_mod
    else:
        base_model = template

    def functional_forward(
        params: Dict[str, torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Single-operator forward using functional_call."""
        return func.functional_call(base_model, params, (x,))

    # Vectorize over params (dim 0) and inputs (dim 0)
    # Use randomness="different" to allow stochastic blocks with independent noise
    batched_forward = func.vmap(
        functional_forward,
        in_dims=in_dims,
        randomness=randomness,
    )

    return batched_forward


def run_batched_partition(
    template: nn.Module,
    state_dicts: List[Dict[str, torch.Tensor]],
    inputs: torch.Tensor,  # [N, C, H, W]
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Run all operators in partition with batched execution.

    ONE kernel launch for all N operators (vs N launches for sequential).

    Args:
        template: Compiled template operator (defines graph structure)
        state_dicts: List of N state_dicts (one per operator)
        inputs: Batched inputs [N, C, H, W]
        chunk_size: Optional chunk size for memory-limited execution

    Returns:
        Batched outputs [N, C, H, W]

    Raises:
        ValueError: If batch sizes don't match
    """
    N = len(state_dicts)
    if inputs.shape[0] != N:
        raise ValueError(
            f"Batch size mismatch: {N} state_dicts but {inputs.shape[0]} inputs"
        )

    if N == 0:
        return torch.empty(0, *inputs.shape[1:], device=inputs.device)

    # Stack parameters: each param becomes [N, ...]
    batched_params = stack_parameters(state_dicts, device=inputs.device)

    # Create batched forward function
    batched_forward = create_batched_forward(template)

    if chunk_size is None or chunk_size >= N:
        # Single batched execution
        with torch.no_grad():
            outputs = batched_forward(batched_params, inputs)
    else:
        # Chunked execution for memory-limited scenarios
        outputs_list = []
        for i in range(0, N, chunk_size):
            end = min(i + chunk_size, N)
            chunk_params = {k: v[i:end] for k, v in batched_params.items()}
            chunk_inputs = inputs[i:end]
            with torch.no_grad():
                chunk_outputs = batched_forward(chunk_params, chunk_inputs)
            outputs_list.append(chunk_outputs)
        outputs = torch.cat(outputs_list, dim=0)

    return outputs


def run_batched_timestep(
    template: nn.Module,
    state_dicts: List[Dict[str, torch.Tensor]],
    x_t: torch.Tensor,  # [N, C, H, W] - current state
    policy: str = "residual",
    dt: float = 0.01,
    alpha: float = 0.5,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Run one timestep of batched temporal evolution.

    Applies evolution policy (residual, convex, or autoregressive) to batched outputs.

    Args:
        template: Compiled template operator
        state_dicts: List of N state_dicts
        x_t: Current state [N, C, H, W]
        policy: Evolution policy ("residual", "convex", "autoregressive")
        dt: Time step for residual policy
        alpha: Mixing weight for convex policy
        chunk_size: Optional chunk size for memory-limited execution

    Returns:
        Next state x_{t+1} [N, C, H, W]
    """
    # Get operator outputs for current state
    f_x = run_batched_partition(
        template=template,
        state_dicts=state_dicts,
        inputs=x_t,
        chunk_size=chunk_size,
    )

    # Apply evolution policy
    if policy == "autoregressive":
        return f_x
    elif policy == "residual":
        return x_t + dt * f_x
    elif policy == "convex":
        return alpha * f_x + (1 - alpha) * x_t
    else:
        raise ValueError(f"Unknown evolution policy: {policy}")


class BatchedOperatorExecutor:
    """
    High-level executor for batched operator processing.

    Manages template caching, parameter stacking, and batched execution.
    Designed for integration with DatasetGenerationPipeline.
    """

    def __init__(
        self,
        template: nn.Module,
        device: torch.device,
        max_batch_size: int = 64,
    ):
        """
        Initialize batched executor.

        Args:
            template: Compiled template operator
            device: Target device
            max_batch_size: Maximum batch size (for memory management)
        """
        self.template = template
        self.device = device
        self.max_batch_size = max_batch_size
        self._batched_forward = None

    def _get_batched_forward(self) -> callable:
        """Lazily create batched forward function."""
        if self._batched_forward is None:
            self._batched_forward = create_batched_forward(self.template)
        return self._batched_forward

    def execute(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute batched operators.

        Automatically chunks if batch exceeds max_batch_size.

        Args:
            state_dicts: List of N operator state_dicts
            inputs: Batched inputs [N, C, H, W]

        Returns:
            Batched outputs [N, C, H, W]
        """
        return run_batched_partition(
            template=self.template,
            state_dicts=state_dicts,
            inputs=inputs,
            chunk_size=self.max_batch_size,
        )

    def evolve_trajectory(
        self,
        state_dicts: List[Dict[str, torch.Tensor]],
        initial_conditions: torch.Tensor,  # [N, C, H, W]
        num_timesteps: int,
        policy: str = "residual",
        dt: float = 0.01,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Evolve batched trajectories over time.

        Args:
            state_dicts: List of N operator state_dicts
            initial_conditions: Initial states [N, C, H, W]
            num_timesteps: Number of timesteps to evolve
            policy: Evolution policy
            dt: Time step for residual policy
            alpha: Mixing weight for convex policy

        Returns:
            Trajectories [N, T, C, H, W]
        """
        N = len(state_dicts)
        device = initial_conditions.device
        C, H, W = initial_conditions.shape[1:]

        # Pre-allocate output
        trajectories = torch.zeros(
            N, num_timesteps, C, H, W,
            device=device,
            dtype=initial_conditions.dtype,
        )

        # Initial state
        x_t = initial_conditions
        trajectories[:, 0] = x_t

        # Stack parameters once
        batched_params = stack_parameters(state_dicts, device=device)
        batched_forward = self._get_batched_forward()

        # Temporal evolution
        for t in range(1, num_timesteps):
            with torch.no_grad():
                f_x = batched_forward(batched_params, x_t)

            # Apply evolution policy
            if policy == "autoregressive":
                x_t = f_x
            elif policy == "residual":
                x_t = x_t + dt * f_x
            elif policy == "convex":
                x_t = alpha * f_x + (1 - alpha) * x_t

            trajectories[:, t] = x_t

        return trajectories
