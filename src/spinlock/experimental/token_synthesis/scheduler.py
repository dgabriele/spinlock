"""Mode scheduler for explore/refine switching.

Controls the alternation between exploration (generating novel tokens and
computing surprisal) and refinement (fine-tuning the diffusion model on
high-surprisal samples).

Supports two strategies:
- Fixed: Alternate explore_steps batches with refine_epochs epochs
- Adaptive: Additionally monitor queue fill level to switch early
"""

import logging
from enum import Enum

from spinlock.experimental.token_synthesis.config import SchedulerConfig
from spinlock.experimental.token_synthesis.priority_queue import SurprisalPriorityQueue

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    EXPLORE = "explore"
    REFINE = "refine"


class ModeScheduler:
    """Schedule explore/refine mode switching.

    Fixed mode:
        - Run explore_steps exploration batches
        - Then refine_epochs refinement epochs
        - Repeat for max_cycles

    Adaptive mode (additionally):
        - queue > threshold_high → switch to REFINE early
        - queue < threshold_low → switch to EXPLORE early

    Args:
        config: SchedulerConfig
        queue: SurprisalPriorityQueue for fill-level monitoring
    """

    def __init__(self, config: SchedulerConfig, queue: SurprisalPriorityQueue):
        self.config = config
        self._queue = queue
        self._current_mode = Mode.EXPLORE
        self._current_cycle = 0
        self._steps_in_mode = 0

    @property
    def current_mode(self) -> Mode:
        return self._current_mode

    @property
    def current_cycle(self) -> int:
        return self._current_cycle

    @property
    def is_complete(self) -> bool:
        return self._current_cycle >= self.config.max_cycles

    def get_explore_steps_remaining(self) -> int:
        """Steps remaining in current explore phase."""
        return max(0, self.config.explore_steps - self._steps_in_mode)

    def get_refine_epochs_remaining(self) -> int:
        """Epochs remaining in current refine phase."""
        return max(0, self.config.refine_epochs - self._steps_in_mode)

    def should_switch(self) -> bool:
        """Check if mode should switch.

        Fixed: compare step count against configured limit.
        Adaptive: also check queue fill level thresholds.
        """
        # Fixed threshold
        match self._current_mode:
            case Mode.EXPLORE:
                if self._steps_in_mode >= self.config.explore_steps:
                    return True
            case Mode.REFINE:
                if self._steps_in_mode >= self.config.refine_epochs:
                    return True

        # Adaptive thresholds
        if self.config.adaptive:
            fill = self._queue.fill_fraction
            if self._current_mode == Mode.EXPLORE and fill >= self.config.queue_threshold_high:
                logger.info(
                    f"Adaptive switch: queue fill {fill:.1%} >= "
                    f"{self.config.queue_threshold_high:.1%} → REFINE"
                )
                return True
            elif self._current_mode == Mode.REFINE and fill <= self.config.queue_threshold_low:
                logger.info(
                    f"Adaptive switch: queue fill {fill:.1%} <= "
                    f"{self.config.queue_threshold_low:.1%} → EXPLORE"
                )
                return True

        return False

    def step(self) -> None:
        """Record one step/epoch in the current mode."""
        self._steps_in_mode += 1

    def advance(self) -> Mode:
        """Advance state machine, potentially switching modes.

        Call this after should_switch() returns True or at the end of
        a mode's allotted steps.

        Returns:
            The new mode after advancing.
        """
        match self._current_mode:
            case Mode.EXPLORE:
                self._current_mode = Mode.REFINE
                self._steps_in_mode = 0
                logger.info(f"Cycle {self._current_cycle}: EXPLORE → REFINE")
            case Mode.REFINE:
                self._current_mode = Mode.EXPLORE
                self._steps_in_mode = 0
                self._current_cycle += 1
                logger.info(
                    f"Cycle {self._current_cycle - 1} → {self._current_cycle}: "
                    f"REFINE → EXPLORE"
                )

        return self._current_mode
