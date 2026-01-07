"""Abstract base class for NOA training losses.

Provides standardized interface for computing NOA losses with:
- Unified LossOutput format for consistent logging
- Abstract compute() method for loss computation
- Metadata about leading vs auxiliary loss terms

Design Pattern:
    All NOA loss functions inherit from BaseNOALoss to enable:
    - Swappable training objectives (MSE-led vs VQ-led)
    - Unified training loops that work with any loss
    - Consistent logging and metrics tracking

Two Training Paradigms:
    MSE-led: Physics fidelity first
        Loss = λ_traj * L_traj + λ_commit * L_commit + λ_latent * L_latent
               ═══════════════
                  PRIMARY

    VQ-led: Symbolic coherence first
        Loss = λ_recon * L_recon + λ_commit * L_commit + λ_traj * L_traj
               ════════════════
                  PRIMARY

Example:
    >>> class MyLoss(BaseNOALoss):
    ...     def compute(self, pred, target, ic, noa):
    ...         loss = F.mse_loss(pred, target)
    ...         return LossOutput(total=loss, components={'mse': loss}, metrics={'mse': loss.item()})
    ...     @property
    ...     def leading_loss_name(self): return 'mse'
    ...     @property
    ...     def auxiliary_loss_names(self): return []
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch
import torch.nn as nn


@dataclass
class LossOutput:
    """Standardized loss output format for NOA training.

    Provides a consistent structure for loss computation results,
    enabling unified training loops and logging across different
    loss functions.

    Attributes:
        total: The total weighted loss (scalar tensor for backprop)
        components: Individual loss components as tensors (for analysis)
        metrics: Detached float values for logging (won't affect gradients)

    Example:
        >>> output = LossOutput(
        ...     total=combined_loss,
        ...     components={'traj': traj_loss, 'commit': commit_loss},
        ...     metrics={'traj': 0.5, 'commit': 0.001, 'total': 0.501}
        ... )
        >>> output.total.backward()  # Only total is used for backprop
        >>> print(output.metrics)    # For logging
    """

    total: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure metrics dict includes total if not already present."""
        if 'total' not in self.metrics and self.total is not None:
            self.metrics['total'] = self.total.item() if self.total.numel() == 1 else 0.0


class BaseNOALoss(nn.Module, ABC):
    """Abstract base for NOA loss functions.

    Defines the interface for computing training losses for NOA.
    Subclasses implement different training paradigms:
    - MSE-led: Prioritize physics fidelity (L_traj primary)
    - VQ-led: Prioritize symbolic coherence (L_recon primary)

    Subclasses must implement:
    - compute(): Calculate all loss components and return LossOutput
    - leading_loss_name: Property identifying the primary loss term
    - auxiliary_loss_names: Property listing auxiliary loss terms

    The leading vs auxiliary distinction is informational—for logging
    and understanding the training objective. The actual weighting is
    determined by lambda parameters in the loss function.
    """

    @abstractmethod
    def compute(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        ic: Optional[torch.Tensor] = None,
        noa: Optional[nn.Module] = None,
    ) -> LossOutput:
        """Compute loss components.

        This is the main computation method called by the training loop.
        Implementations should:
        1. Compute all loss terms (leading and auxiliary)
        2. Apply weighting factors
        3. Return a LossOutput with total, components, and metrics

        Args:
            pred_trajectory: NOA predicted trajectory [B, T, C, H, W]
            target_trajectory: CNO ground truth trajectory [B, T, C, H, W]
            ic: Initial condition [B, C, H, W] (optional, for feature extraction)
            noa: NOA backbone reference (optional, for L_latent computation)

        Returns:
            LossOutput containing:
            - total: Weighted sum of all losses (for backprop)
            - components: Dict of individual loss tensors
            - metrics: Dict of float values for logging
        """
        pass

    @property
    @abstractmethod
    def leading_loss_name(self) -> str:
        """Name of the primary (leading) loss term.

        The leading loss defines the training paradigm:
        - "traj" for MSE-led (physics fidelity)
        - "recon" for VQ-led (symbolic coherence)

        Returns:
            String name of the primary loss component
        """
        pass

    @property
    @abstractmethod
    def auxiliary_loss_names(self) -> List[str]:
        """Names of auxiliary loss terms.

        Auxiliary losses provide regularization but are not the
        primary training objective.

        Returns:
            List of auxiliary loss component names
        """
        pass

    @property
    def all_loss_names(self) -> List[str]:
        """All loss component names (leading + auxiliary)."""
        return [self.leading_loss_name] + self.auxiliary_loss_names

    def format_log_string(self, metrics: Dict[str, float]) -> str:
        """Format metrics dictionary into a log string.

        Creates a consistent logging format:
        - Leading loss shown first with emphasis
        - Auxiliary losses follow
        - Total shown last

        Args:
            metrics: Dictionary of metric name → value

        Returns:
            Formatted string like "traj=0.5000 commit=0.0010 total=0.5010"
        """
        parts = []

        # Leading loss first
        if self.leading_loss_name in metrics:
            parts.append(f"{self.leading_loss_name}={metrics[self.leading_loss_name]:.4f}")

        # Auxiliary losses
        for name in self.auxiliary_loss_names:
            if name in metrics:
                parts.append(f"{name}={metrics[name]:.6f}")

        # Total last
        if 'total' in metrics:
            parts.append(f"total={metrics['total']:.4f}")

        return " ".join(parts)
