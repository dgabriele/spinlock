"""
Operator rollout for stochastic neural operators.

Provides core functionality for rolling out neural operators autoregressively
over time with configurable update policies, metrics computation, and
trajectory storage.

"Rollout" is the standard ML term for autoregressively applying a model
(common in neural ODEs, model-based RL, sequence modeling).

Key components:
- Update policies: Autoregressive, residual, convex
- Rollout engine: Orchestrates temporal dynamics
- Metrics: Energy, entropy, autocorrelation, variance
- Trajectory storage: HDF5-based storage
- Initial conditions: Dataset, GRF, zeros

Example:
    ```python
    from spinlock.rollout import OperatorRollout, ConvexPolicy

    policy = ConvexPolicy(alpha=0.7)
    rollout = OperatorRollout(
        policy=policy,
        num_timesteps=100,
        device=torch.device("cuda")
    )

    trajectories, metrics = rollout.evolve_operator(
        operator=neural_operator,
        initial_condition=X0,
        num_realizations=10
    )
    ```
"""

from .policies import (
    UpdatePolicy,
    AutoregressivePolicy,
    ResidualPolicy,
    ConvexPolicy,
    create_update_policy,
)
from .metrics import TrajectoryMetrics, MetricsComputer
from .engine import OperatorRollout
from .initializers import InitialConditionSampler
from .trajectory import TrajectoryWriter

__all__ = [
    # Policies
    "UpdatePolicy",
    "AutoregressivePolicy",
    "ResidualPolicy",
    "ConvexPolicy",
    "create_update_policy",
    # Metrics
    "TrajectoryMetrics",
    "MetricsComputer",
    # Rollout
    "OperatorRollout",
    # Utilities
    "InitialConditionSampler",
    "TrajectoryWriter",
]
