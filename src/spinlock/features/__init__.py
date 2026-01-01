"""
Feature extraction system for Spinlock datasets.

Provides modular feature extraction for downstream VQ-VAE training and analysis.
Currently supports Summary Descriptor Features (SDF) with extensible architecture
for future feature families.

Main Components:
    - FeatureRegistry: Maps feature names to integer indices
    - FeatureExtractor: Main orchestrator for feature extraction
    - SummaryExtractor: Summary Descriptor Features implementation
    - HDF5FeatureWriter: Stores features in HDF5 format

Example:
    >>> from spinlock.features import FeatureExtractor
    >>> from spinlock.features.config import FeatureExtractionConfig
    >>>
    >>> config = FeatureExtractionConfig(
    ...     input_dataset="datasets/benchmark_10k.h5",
    ...     sdf=SummaryConfig()
    ... )
    >>> extractor = FeatureExtractor(config)
    >>> extractor.extract()
"""

from spinlock.features.registry import FeatureRegistry

__all__ = [
    "FeatureRegistry",
]

__version__ = "1.0.0"
