"""
Feature extraction system for Spinlock datasets.

Provides modular feature extraction for downstream VQ-VAE training and analysis.
Two sibling feature families:
- TEMPORAL: Per-timestep time series [N, T, D] (spatial, spectral, cross_channel)
- SUMMARY: Aggregated scalars [N, D] (temporal dynamics, causality, invariant_drift)

Main Components:
    - FeatureRegistry: Maps feature names to integer indices
    - FeatureExtractor: Main orchestrator for feature extraction
    - SummaryExtractor: SUMMARY feature implementation
    - HDF5FeatureWriter: Stores features in HDF5 format

Example:
    >>> from spinlock.features import FeatureExtractor
    >>> from spinlock.features.config import FeatureExtractionConfig, TemporalConfig
    >>>
    >>> config = FeatureExtractionConfig(
    ...     input_dataset="datasets/benchmark_10k.h5",
    ...     temporal=TemporalConfig(enabled=False),  # Disable TEMPORAL
    ...     summary=SummaryConfig()
    ... )
    >>> extractor = FeatureExtractor(config)
    >>> extractor.extract()
"""

from spinlock.features.registry import FeatureRegistry

__all__ = [
    "FeatureRegistry",
]

__version__ = "1.0.0"
