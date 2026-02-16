"""
Salad.com launcher for distributed jobs.

This module provides backwards compatibility by re-exporting the training launcher.
For new code, import from the specific launcher modules:
- training_launcher: Distributed DDP training
- generation_launcher: Distributed dataset generation
"""

from typing import List

from .training_launcher import SaladTrainingLauncher, launch_salad_training


# Backwards compatibility: SaladLauncher is now SaladTrainingLauncher
SaladLauncher = SaladTrainingLauncher


__all__ = [
    "SaladLauncher",  # Backwards compatibility
    "SaladTrainingLauncher",
    "launch_salad_training",
]
