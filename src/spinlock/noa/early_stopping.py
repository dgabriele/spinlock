"""Early stopping criteria for episode termination.

Determines when to stop MNO rollouts during autonomous operation.

Criteria:
- Convergence: Equilibrium reached (||u_t - u_{t-1}|| < ε)
- Token Stability: Behavioral loop detected (token_t == token_{t-k})
- Max Steps: Safety fallback (t >= T_max)
- Composite: Combine multiple criteria with AND/OR logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import torch
from torch import Tensor


@dataclass
class EpisodeState:
    """Current state of episode execution for early stopping checks.

    Attributes:
        t: Current timestep
        state: Current MNO state [C, H, W] or [B, C, H, W]
        prev_state: Previous state (None if t == 0)
        tokens: Token sequence so far, list of [num_categories * num_levels] tensors
        metadata: Additional episode metadata
    """

    t: int
    state: Tensor
    prev_state: Optional[Tensor] = None
    tokens: Optional[List[Tensor]] = None
    metadata: Optional[dict] = None


class EarlyStoppingCriterion(ABC):
    """Abstract base class for early stopping criteria."""

    @abstractmethod
    def should_stop(self, episode_state: EpisodeState) -> Tuple[bool, str]:
        """Check if episode should stop.

        Args:
            episode_state: Current episode state

        Returns:
            Tuple of (should_stop, reason)
            - should_stop: True if criterion met
            - reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset internal state for new episode."""
        pass


class ConvergenceStop(EarlyStoppingCriterion):
    """Stop when state converges to equilibrium.

    Detects when ||u_t - u_{t-1}|| < threshold, indicating the system
    has reached a stable fixed point.
    """

    def __init__(self, threshold: float = 1e-4, norm: str = "l2"):
        """Initialize convergence stopping criterion.

        Args:
            threshold: Convergence threshold (state difference below this stops)
            norm: Norm type ("l2", "l1", "linf")
        """
        self.threshold = threshold
        self.norm = norm

    def should_stop(self, episode_state: EpisodeState) -> Tuple[bool, str]:
        """Check if state has converged.

        Args:
            episode_state: Current episode state

        Returns:
            (True, reason) if converged, else (False, "")
        """
        if episode_state.prev_state is None:
            return False, ""

        # Compute state difference
        diff = episode_state.state - episode_state.prev_state

        # Compute norm
        if self.norm == "l2":
            error = torch.norm(diff)
        elif self.norm == "l1":
            error = torch.abs(diff).sum()
        elif self.norm == "linf":
            error = torch.abs(diff).max()
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        if error < self.threshold:
            return True, f"Converged (||Δu||_{self.norm} = {error:.2e} < {self.threshold:.2e})"

        return False, ""

    def reset(self):
        """No internal state to reset."""
        pass


class TokenStabilityStop(EarlyStoppingCriterion):
    """Stop when token sequence enters stable loop.

    Detects when tokens repeat for k consecutive steps, indicating
    the behavioral state has entered a limit cycle.
    """

    def __init__(self, window: int = 5, consecutive: int = 3):
        """Initialize token stability stopping criterion.

        Args:
            window: Look-back window for token comparison
            consecutive: Number of consecutive stable steps required
        """
        self.window = window
        self.consecutive = consecutive
        self._stable_count = 0

    def should_stop(self, episode_state: EpisodeState) -> Tuple[bool, str]:
        """Check if tokens have stabilized.

        Args:
            episode_state: Current episode state

        Returns:
            (True, reason) if stable, else (False, "")
        """
        if episode_state.tokens is None or len(episode_state.tokens) < self.window + 1:
            return False, ""

        # Get current and window-past token
        current_token = episode_state.tokens[-1]
        past_token = episode_state.tokens[-self.window - 1]

        # Check if tokens match
        if torch.equal(current_token, past_token):
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self.consecutive:
            return (
                True,
                f"Token stability (repeated {self.consecutive} times with period {self.window})",
            )

        return False, ""

    def reset(self):
        """Reset stable count for new episode."""
        self._stable_count = 0


class MaxStepsStop(EarlyStoppingCriterion):
    """Stop when maximum timestep reached.

    Safety fallback to prevent infinite loops.
    """

    def __init__(self, max_steps: int = 256):
        """Initialize max steps stopping criterion.

        Args:
            max_steps: Maximum number of timesteps
        """
        self.max_steps = max_steps

    def should_stop(self, episode_state: EpisodeState) -> Tuple[bool, str]:
        """Check if max steps reached.

        Args:
            episode_state: Current episode state

        Returns:
            (True, reason) if max steps reached, else (False, "")
        """
        if episode_state.t >= self.max_steps:
            return True, f"Max steps reached (t = {episode_state.t} >= {self.max_steps})"

        return False, ""

    def reset(self):
        """No internal state to reset."""
        pass


class CompositeEarlyStopping(EarlyStoppingCriterion):
    """Combine multiple stopping criteria with AND/OR logic.

    Enables flexible early stopping policies:
    - OR: Stop if ANY criterion met (default, most permissive)
    - AND: Stop only if ALL criteria met (most conservative)

    Example:
        >>> # Stop if converged OR max steps
        >>> criteria = CompositeEarlyStopping(
        ...     [ConvergenceStop(), MaxStepsStop()],
        ...     logic="OR"
        ... )
        >>>
        >>> # Stop only if both converged AND tokens stable
        >>> criteria = CompositeEarlyStopping(
        ...     [ConvergenceStop(), TokenStabilityStop()],
        ...     logic="AND"
        ... )
    """

    def __init__(
        self,
        criteria: List[EarlyStoppingCriterion],
        logic: Literal["OR", "AND"] = "OR",
    ):
        """Initialize composite stopping criterion.

        Args:
            criteria: List of stopping criteria to combine
            logic: Combination logic ("OR" or "AND")
        """
        if not criteria:
            raise ValueError("Must provide at least one criterion")

        self.criteria = criteria
        self.logic = logic

    def should_stop(self, episode_state: EpisodeState) -> Tuple[bool, str]:
        """Check if any/all criteria met.

        Args:
            episode_state: Current episode state

        Returns:
            (True, reason) if criteria met, else (False, "")
        """
        results = [criterion.should_stop(episode_state) for criterion in self.criteria]
        stops = [r[0] for r in results]
        reasons = [r[1] for r in results if r[0]]

        if self.logic == "OR":
            if any(stops):
                combined_reason = " OR ".join(reasons)
                return True, combined_reason
        elif self.logic == "AND":
            if all(stops):
                combined_reason = " AND ".join(reasons)
                return True, combined_reason
        else:
            raise ValueError(f"Unknown logic: {self.logic}")

        return False, ""

    def reset(self):
        """Reset all component criteria."""
        for criterion in self.criteria:
            criterion.reset()


# Predefined stopping policies
class StandardStoppingPolicy(CompositeEarlyStopping):
    """Standard early stopping policy for most episodes.

    Stops if:
    - State converges (||Δu|| < 1e-4), OR
    - Tokens stabilize (5-step period, 3 consecutive), OR
    - Max 256 steps reached
    """

    def __init__(self):
        super().__init__(
            criteria=[
                ConvergenceStop(threshold=1e-4),
                TokenStabilityStop(window=5, consecutive=3),
                MaxStepsStop(max_steps=256),
            ],
            logic="OR",
        )


class HighFidelityStoppingPolicy(CompositeEarlyStopping):
    """High-fidelity policy for validation runs.

    Stricter convergence, longer max steps.

    Stops if:
    - State converges (||Δu|| < 1e-5), OR
    - Tokens stabilize (10-step period, 5 consecutive), OR
    - Max 512 steps reached
    """

    def __init__(self):
        super().__init__(
            criteria=[
                ConvergenceStop(threshold=1e-5),
                TokenStabilityStop(window=10, consecutive=5),
                MaxStepsStop(max_steps=512),
            ],
            logic="OR",
        )


class FastStoppingPolicy(CompositeEarlyStopping):
    """Fast policy for large-scale exploration.

    Looser convergence, shorter max steps.

    Stops if:
    - State converges (||Δu|| < 1e-3), OR
    - Tokens stabilize (3-step period, 2 consecutive), OR
    - Max 128 steps reached
    """

    def __init__(self):
        super().__init__(
            criteria=[
                ConvergenceStop(threshold=1e-3),
                TokenStabilityStop(window=3, consecutive=2),
                MaxStepsStop(max_steps=128),
            ],
            logic="OR",
        )
