"""Perturbation framework for autonomous MNO operation.

This module provides a flexible, modular interface for applying perturbations
to MNO rollouts. Perturbations excite learned dynamics and enable exploration
of the MNO's behavioral manifold without parameter conditioning.

Design Principles:
- Domain-agnostic: Works with any state representation
- Composable: Multiple perturbations can be combined
- Metadata-complete: All parameters logged for episodic memory
- DRY expansion: Easy to add new perturbation types

Usage:
    >>> from spinlock.perturbations import ImpulsePerturbation, ImpulsePerturbationFactory
    >>>
    >>> # Create center blob perturbation
    >>> perturbation = ImpulsePerturbationFactory.center_blob(
    ...     t_impulse=0,
    ...     amplitude=1.0,
    ...     spatial_size=(64, 64),
    ...     sigma=0.1
    ... )
    >>>
    >>> # Apply to state
    >>> perturbed_state = perturbation.apply(state, t=0)
    >>>
    >>> # Get metadata for logging
    >>> metadata = perturbation.get_metadata()
"""

from .base import BasePerturbation, NullPerturbation, CompositePerturbation
from .impulse import ImpulsePerturbation, ImpulsePerturbationFactory

__all__ = [
    "BasePerturbation",
    "NullPerturbation",
    "CompositePerturbation",
    "ImpulsePerturbation",
    "ImpulsePerturbationFactory",
]
