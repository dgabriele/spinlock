"""Difficulty-proportional candidate budget allocation.

Samples above the acceptance threshold get zero extra candidates.
Below-threshold samples get a budget that scales with distance to the threshold:
easy rejects (just below) get few extra; hard rejects (far below) get many.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spinlock.experimental.diffusion.config import AdaptiveBudgetConfig


class CandidateBudgetAllocator:
    """Allocate D3PM re-sampling and perturbation budgets per sample."""

    def __init__(self, config: AdaptiveBudgetConfig, threshold: float) -> None:
        self.config = config
        self.threshold = threshold

    def allocate(
        self,
        agreements: Tensor,  # [B] initial agreement scores
    ) -> tuple[Tensor, Tensor]:
        """Compute per-sample budgets for D3PM re-sampling and perturbation.

        Args:
            agreements: Best agreement scores from the initial D3PM round.

        Returns:
            (d3pm_budget [B], perturbation_budget [B]) — integer tensors.
            Samples at or above threshold get (0, 0).
        """
        cfg = self.config
        device = agreements.device

        # Distance to threshold: 0 = at threshold, 1 = zero agreement
        distance = (1.0 - agreements / self.threshold).clamp(0.0, 1.0)

        # Scale budget by distance
        if cfg.budget_scaling == "sqrt":
            scaled = distance.sqrt()
        else:  # linear
            scaled = distance

        total = cfg.min_extra_candidates + torch.ceil(
            scaled * (cfg.max_extra_candidates - cfg.min_extra_candidates)
        )

        # Zero out for already-accepted samples
        accepted = agreements >= self.threshold
        total[accepted] = 0

        total = total.int()
        d3pm_budget = torch.ceil(total.float() * cfg.d3pm_fraction).int()
        perturbation_budget = total - d3pm_budget

        return d3pm_budget, perturbation_budget
