"""
ARCHITECTURE (Neural Operator Parameter) feature family.

Extracts parameter-derived features from [0,1]^P unit hypercube
and mapped operator configurations.
"""

from .config import (
    ArchitectureConfig,
    ArchitectureParamsConfig,
    ArchitectureStochasticConfig,
    ArchitectureOperatorConfig,
    ArchitectureEvolutionConfig,
    ArchitectureStratificationConfig,
)
from .extractors import ArchitectureExtractor

__all__ = [
    'ArchitectureConfig',
    'ArchitectureParamsConfig',
    'ArchitectureStochasticConfig',
    'ArchitectureOperatorConfig',
    'ArchitectureEvolutionConfig',
    'ArchitectureStratificationConfig',
    'ArchitectureExtractor',
]
