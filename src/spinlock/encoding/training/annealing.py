"""Temperature annealing schedules for Gumbel-Softmax."""

import math


class TemperatureScheduler:
    """Temperature annealing for Gumbel-Softmax."""

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.1,
        total_epochs: int = 1000,
        schedule: str = "linear"
    ):
        self.start = start
        self.end = end
        self.total_epochs = total_epochs
        self.schedule = schedule

    def get_temperature(self, epoch: int) -> float:
        """Get temperature for current epoch."""
        if epoch >= self.total_epochs:
            return self.end

        progress = epoch / self.total_epochs

        if self.schedule == "linear":
            return self.start + (self.end - self.start) * progress
        elif self.schedule == "exponential":
            return self.start * (self.end / self.start) ** progress
        elif self.schedule == "cosine":
            return self.end + (self.start - self.end) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def __call__(self, epoch: int) -> float:
        """Alias for get_temperature."""
        return self.get_temperature(epoch)
