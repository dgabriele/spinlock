"""Generic parameter schedules for training hyperparameter annealing.

Maps training progress in [0, 1] to a float value, supporting constant,
linear, cosine, and step schedules.  Used for dropout annealing (start high
for codebook protection, decay to let aux heads see full information) and
declarative weight scheduling (e.g. gate_sparsity_weight ramping).
"""

import math
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ScheduleConfig(BaseModel):
    """Declarative schedule configuration: type + endpoints."""

    type: Literal["constant", "linear", "cosine", "step"] = "linear"
    start: float = Field(..., description="Value at progress=0")
    end: float = Field(default=0.0, description="Value at progress=1 (unused for step/constant)")

    # Step schedule only:
    milestones: Optional[List[float]] = Field(
        default=None,
        description="Progress fractions where value changes, e.g. [0.3, 0.7]",
    )
    values: Optional[List[float]] = Field(
        default=None,
        description="Values per segment (len = len(milestones) + 1)",
    )

    @model_validator(mode="after")
    def _validate_step(self) -> "ScheduleConfig":
        if self.type == "step":
            if self.milestones is None or self.values is None:
                raise ValueError(
                    "Step schedule requires both 'milestones' and 'values'"
                )
            if len(self.values) != len(self.milestones) + 1:
                raise ValueError(
                    f"len(values) must be len(milestones)+1, "
                    f"got {len(self.values)} values and {len(self.milestones)} milestones"
                )
            if sorted(self.milestones) != self.milestones:
                raise ValueError("milestones must be in ascending order")
        return self


class ParameterSchedule:
    """Maps training progress [0, 1] to a parameter value.

    Usage::

        sched = ParameterSchedule(ScheduleConfig(type="cosine", start=0.6, end=0.1))
        for epoch in range(num_epochs):
            progress = epoch / max(num_epochs - 1, 1)
            dropout_p = sched(progress)
    """

    def __init__(self, config: ScheduleConfig) -> None:
        self.config = config

    def __call__(self, progress: float) -> float:
        """Evaluate schedule at the given training progress.

        Args:
            progress: Training progress in [0, 1].

        Returns:
            Scheduled parameter value.
        """
        progress = max(0.0, min(1.0, progress))
        cfg = self.config

        if cfg.type == "constant":
            return cfg.start

        if cfg.type == "linear":
            return cfg.start + (cfg.end - cfg.start) * progress

        if cfg.type == "cosine":
            return cfg.end + (cfg.start - cfg.end) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if cfg.type == "step":
            assert cfg.milestones is not None and cfg.values is not None
            for i, milestone in enumerate(cfg.milestones):
                if progress < milestone:
                    return cfg.values[i]
            return cfg.values[-1]

        raise ValueError(f"Unknown schedule type: {cfg.type}")

    def __repr__(self) -> str:
        cfg = self.config
        if cfg.type == "step":
            return f"ParameterSchedule(step, milestones={cfg.milestones}, values={cfg.values})"
        return f"ParameterSchedule({cfg.type}, {cfg.start:.4f} -> {cfg.end:.4f})"
