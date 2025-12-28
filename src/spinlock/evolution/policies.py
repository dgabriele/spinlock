"""
Temporal update policies for neural operator evolution.

Implements three update strategies from the Spinlock specification:
1. Autoregressive: X_t = O_θ(X_{t-1})
2. Residual: X_t = X_{t-1} + dt * O_θ(X_{t-1})
3. Convex: X_t = α * X_{t-1} + (1-α) * O_θ(X_{t-1})

Design principles:
- Strategy pattern: Polymorphic update policies
- DRY: Shared base class with factory method
- Type-safe: Clear signatures and documentation
"""

from abc import ABC, abstractmethod
import torch
from typing import Optional


class UpdatePolicy(ABC):
    """
    Abstract base class for temporal update strategies.

    All update policies implement the `update` method which computes
    the next state from the previous state and operator output.
    """

    @abstractmethod
    def update(
        self,
        X_prev: torch.Tensor,
        O_theta_X: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute next state from previous state and operator output.

        Args:
            X_prev: Previous state [B, C, H, W]
            O_theta_X: Operator output O_θ(X_prev) [B, C, H, W]
            **kwargs: Policy-specific parameters

        Returns:
            Next state X_next [B, C, H, W]
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return policy name for logging and identification."""
        pass


class AutoregressivePolicy(UpdatePolicy):
    """
    Autoregressive update: X_t = O_θ(X_{t-1})

    Maximally expressive policy where the operator output becomes
    the next state directly. Sensitive to operator scale and can
    exhibit chaotic behavior for long horizons.

    Use cases:
    - Short-horizon dynamics
    - Chaos discovery
    - Stress-testing operator expressivity

    Example:
        ```python
        policy = AutoregressivePolicy()
        X_next = policy.update(X_prev, O_theta_X)
        # X_next == O_theta_X
        ```
    """

    def update(
        self,
        X_prev: torch.Tensor,
        O_theta_X: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Autoregressive update: directly use operator output."""
        return O_theta_X

    def name(self) -> str:
        return "autoregressive"


class ResidualPolicy(UpdatePolicy):
    """
    Residual update: X_t = X_{t-1} + dt * O_θ(X_{t-1})

    Interprets the operator as a learned vector field and integrates
    with step size dt. Stability depends on step size and operator scale.

    Use cases:
    - Smooth continuous-time approximations
    - Small-step dynamical systems
    - PDE-like evolution

    Example:
        ```python
        policy = ResidualPolicy(dt=0.01)
        X_next = policy.update(X_prev, O_theta_X)
        # X_next == X_prev + 0.01 * O_theta_X
        ```
    """

    def __init__(self, dt: float = 0.01):
        """
        Initialize residual policy.

        Args:
            dt: Integration step size (default: 0.01)
        """
        if dt <= 0:
            raise ValueError(f"Step size dt must be positive, got {dt}")
        self.dt = dt

    def update(
        self,
        X_prev: torch.Tensor,
        O_theta_X: torch.Tensor,
        dt: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Residual update: integrate operator output.

        Args:
            X_prev: Previous state
            O_theta_X: Operator output
            dt: Override step size (optional)

        Returns:
            Next state
        """
        step_size = dt if dt is not None else self.dt
        return X_prev + step_size * O_theta_X

    def name(self) -> str:
        return f"residual(dt={self.dt})"


class ConvexPolicy(UpdatePolicy):
    """
    Convex (damped) update: X_t = α * X_{t-1} + (1-α) * O_θ(X_{t-1})

    Default policy providing a principled compromise between expressivity
    and stability. The mixing parameter α ∈ [0,1) controls temporal
    persistence vs. operator influence.

    Characteristics:
    - Discrete-time relaxation process
    - Implicit stability (bounded for bounded operators)
    - Tunable temporal persistence
    - Robust under stochasticity

    Use cases:
    - Long-horizon evolution (DEFAULT)
    - Representation learning
    - Dataset generation for downstream abstraction

    Example:
        ```python
        policy = ConvexPolicy(alpha=0.7)
        X_next = policy.update(X_prev, O_theta_X)
        # X_next == 0.7 * X_prev + 0.3 * O_theta_X
        ```
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize convex policy.

        Args:
            alpha: Mixing parameter in [0, 1) (default: 0.5)
                  Higher values = more persistence of previous state
                  Lower values = more influence from operator
        """
        if not (0 <= alpha < 1):
            raise ValueError(f"Alpha must be in [0,1), got {alpha}")
        self.alpha = alpha

    def update(
        self,
        X_prev: torch.Tensor,
        O_theta_X: torch.Tensor,
        alpha: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Convex update: weighted combination of state and operator output.

        Args:
            X_prev: Previous state
            O_theta_X: Operator output
            alpha: Override mixing parameter (optional)

        Returns:
            Next state
        """
        mix_param = alpha if alpha is not None else self.alpha
        return mix_param * X_prev + (1.0 - mix_param) * O_theta_X

    def name(self) -> str:
        return f"convex(alpha={self.alpha})"


def create_update_policy(
    policy_type: str,
    dt: Optional[float] = None,
    alpha: Optional[float] = None
) -> UpdatePolicy:
    """
    Factory function for creating update policies.

    Args:
        policy_type: One of "autoregressive", "residual", "convex"
        dt: Step size for residual policy (default: 0.01)
        alpha: Mixing parameter for convex policy (default: 0.5)

    Returns:
        UpdatePolicy instance

    Raises:
        ValueError: If policy_type is unknown

    Example:
        ```python
        # Create convex policy
        policy = create_update_policy("convex", alpha=0.7)

        # Create residual policy
        policy = create_update_policy("residual", dt=0.01)

        # Create autoregressive policy
        policy = create_update_policy("autoregressive")
        ```
    """
    policy_type = policy_type.lower()

    if policy_type == "autoregressive":
        return AutoregressivePolicy()

    elif policy_type == "residual":
        return ResidualPolicy(dt=dt or 0.01)

    elif policy_type == "convex":
        return ConvexPolicy(alpha=alpha or 0.5)

    else:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Must be one of: 'autoregressive', 'residual', 'convex'"
        )
