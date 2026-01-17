"""Base interface for perturbations applied to MNO rollouts.

Design Principles:
- Domain-agnostic: Works with any state representation (reaction-diffusion, fluids, etc.)
- Composable: Multiple perturbations can be combined (Phase 3+)
- Metadata-complete: All parameters logged for episodic memory retrieval
- DRY expansion: New perturbation types inherit from BasePerturbation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
from torch import Tensor


class BasePerturbation(ABC):
    """Abstract base class for all perturbation types.

    A perturbation modifies the state of a dynamical system during MNO rollout.
    Perturbations can be:
    - Impulse: Single-timestep delta-function (eigenmode excitation)
    - Sustained: Multi-timestep forcing (driven dynamics)
    - Structured: Pattern-based learned perturbations (Phase 4+)

    All perturbations must implement:
    1. apply() - Modify state at given timestep
    2. is_active() - Check if perturbation active at timestep
    3. get_metadata() - Return parameters for logging/memory
    """

    def __init__(self, device: str = "cpu"):
        """Initialize perturbation.

        Args:
            device: Device to place tensors on ("cpu" or "cuda")
        """
        self.device = device

    @abstractmethod
    def apply(self, state: Tensor, t: int) -> Tensor:
        """Apply perturbation to state at timestep t.

        Args:
            state: Current state [B, C, H, W] or [C, H, W]
            t: Current timestep (0-indexed)

        Returns:
            Perturbed state with same shape as input

        Note:
            This method should be pure (no side effects on state).
            Implementation should handle both batched [B, C, H, W] and
            unbatched [C, H, W] inputs.
        """
        pass

    @abstractmethod
    def is_active(self, t: int) -> bool:
        """Check if perturbation is active at timestep t.

        Args:
            t: Current timestep (0-indexed)

        Returns:
            True if perturbation should be applied at timestep t
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for logging and episodic memory storage.

        Returns:
            Dictionary containing perturbation parameters:
            - type: Perturbation class name
            - device: Device placement
            - Additional subclass-specific parameters

        Note:
            Subclasses should call super().get_metadata() and extend
            the returned dictionary with their own parameters.
        """
        return {
            "type": self.__class__.__name__,
            "device": self.device,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        metadata = self.get_metadata()
        params = ", ".join(f"{k}={v}" for k, v in metadata.items() if k != "type")
        return f"{metadata['type']}({params})"


class NullPerturbation(BasePerturbation):
    """No-op perturbation for baseline comparisons.

    Useful for:
    - Control experiments (unperturbed baseline)
    - Testing episode infrastructure
    - Continuous mode initialization (no initial perturbation)
    """

    def apply(self, state: Tensor, t: int) -> Tensor:
        """Return state unchanged."""
        return state

    def is_active(self, t: int) -> bool:
        """Never active (no perturbation)."""
        return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return minimal metadata."""
        metadata = super().get_metadata()
        metadata["description"] = "No perturbation (baseline)"
        return metadata


class CompositePerturbation(BasePerturbation):
    """Combine multiple perturbations (Phase 3+).

    Applies multiple perturbations in sequence at each timestep.
    Useful for:
    - Multi-site forcing
    - Combined impulse + sustained perturbations
    - Complex experimental designs

    Example:
        >>> impulse = ImpulsePerturbation(t_start=0, ...)
        >>> sustained = SustainedPerturbation(t_start=10, t_end=50, ...)
        >>> combined = CompositePerturbation([impulse, sustained])
    """

    def __init__(self, perturbations: list[BasePerturbation], device: str = "cpu"):
        """Initialize composite perturbation.

        Args:
            perturbations: List of perturbations to apply in sequence
            device: Device placement
        """
        super().__init__(device)
        self.perturbations = perturbations

    def apply(self, state: Tensor, t: int) -> Tensor:
        """Apply all active perturbations in sequence.

        Args:
            state: Current state [B, C, H, W] or [C, H, W]
            t: Current timestep

        Returns:
            State after applying all active perturbations
        """
        current_state = state
        for perturbation in self.perturbations:
            if perturbation.is_active(t):
                current_state = perturbation.apply(current_state, t)
        return current_state

    def is_active(self, t: int) -> bool:
        """Active if any component perturbation is active.

        Args:
            t: Current timestep

        Returns:
            True if at least one perturbation is active at timestep t
        """
        return any(p.is_active(t) for p in self.perturbations)

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata including all component perturbations.

        Returns:
            Dictionary with list of component perturbation metadata
        """
        metadata = super().get_metadata()
        metadata["num_perturbations"] = len(self.perturbations)
        metadata["perturbations"] = [p.get_metadata() for p in self.perturbations]
        return metadata
