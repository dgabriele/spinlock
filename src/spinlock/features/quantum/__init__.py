"""Quantum-specific temporal feature extraction.

This module provides feature extractors for quantum systems evolving under
the Caldeira-Leggett model (Quantum Brownian Motion). Features capture:
- Decoherence dynamics
- Quantum coherence
- Entanglement & mixedness
- Phase space structure
- Uncertainty evolution
"""

from .config import QuantumConfig
from .quantum import QuantumFeatureExtractor

__all__ = ["QuantumConfig", "QuantumFeatureExtractor"]
