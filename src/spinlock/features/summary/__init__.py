"""
Summary Descriptor Features (SDF) v2.1.

Comprehensive feature extraction covering 12 scientific domains:

v1.0 categories (base):
- Spatial statistics (moments, gradients, curvature)
- Spectral features (FFT, power spectrum, frequencies)
- Temporal dynamics (growth rates, oscillations, stability)

v2.0 categories (operator-aware):
- Operator sensitivity (Lipschitz estimates, gain curves, linearity)
- Cross-channel interactions (correlation spectra, coherence, mutual information)
- Causality/directionality (lagged correlations, transfer entropy, Granger causality)
- Invariant drift (multi-scale norm-based drift tracking)

v2.1 categories (Phase 2 - complete implementation):
- Distributional (entropy, complexity, compression, multiscale entropy)
- Structural (topology, connected components, edges, GLCM texture)
- Statistical physics (correlations, structure factor, density fluctuations)
- Morphological (shape descriptors, image moments, granulometry)
- Multiscale (wavelet decomposition, Laplacian pyramid)

v2.1 Improvements:
- ✅ Numerical stability fixes (FFT normalization, kurtosis overflow, PACF)
- ✅ Intelligent T-normalization (works across different trajectory lengths)
- ✅ NaN elimination (rolling windows, zero-variance handling)
- ✅ Complete feature set (all 12 categories implemented)

Total: ~420-520 aggregated features (config-dependent, all valid).
"""

__version__ = "2.1.0"

from spinlock.features.summary.extractors import SummaryExtractor
from spinlock.features.summary.config import (
    SummaryConfig,
    SummarySpatialConfig,
    SummarySpectralConfig,
    SummaryTemporalConfig,
    SummaryOperatorSensitivityConfig,
    SummaryCrossChannelConfig,
    SummaryCausalityConfig,
    SummaryInvariantDriftConfig,
    SummaryDistributionalConfig,
    SummaryStructuralConfig,
    SummaryPhysicsConfig,
    SummaryMorphologicalConfig,
    SummaryMultiscaleConfig,
)
from spinlock.features.summary.spatial import SpatialFeatureExtractor
from spinlock.features.summary.spectral import SpectralFeatureExtractor
from spinlock.features.summary.temporal import TemporalFeatureExtractor
from spinlock.features.summary.operator_sensitivity import OperatorSensitivityExtractor
from spinlock.features.summary.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.summary.causality import CausalityFeatureExtractor
from spinlock.features.summary.invariant_drift import InvariantDriftExtractor
# Phase 2 extractors (v2.1)
from spinlock.features.summary.distributional import DistributionalFeatureExtractor
from spinlock.features.summary.structural import StructuralFeatureExtractor
from spinlock.features.summary.physics import PhysicsFeatureExtractor
from spinlock.features.summary.morphological import MorphologicalFeatureExtractor
from spinlock.features.summary.multiscale import MultiscaleFeatureExtractor

__all__ = [
    "SummaryExtractor",
    "SummaryConfig",
    "SummarySpatialConfig",
    "SummarySpectralConfig",
    "SummaryTemporalConfig",
    "SummaryOperatorSensitivityConfig",
    "SummaryCrossChannelConfig",
    "SummaryCausalityConfig",
    "SummaryInvariantDriftConfig",
    "SummaryDistributionalConfig",
    "SummaryStructuralConfig",
    "SummaryPhysicsConfig",
    "SummaryMorphologicalConfig",
    "SummaryMultiscaleConfig",
    "SpatialFeatureExtractor",
    "SpectralFeatureExtractor",
    "TemporalFeatureExtractor",
    "OperatorSensitivityExtractor",
    "CrossChannelFeatureExtractor",
    "CausalityFeatureExtractor",
    "InvariantDriftExtractor",
    "DistributionalFeatureExtractor",
    "StructuralFeatureExtractor",
    "PhysicsFeatureExtractor",
    "MorphologicalFeatureExtractor",
    "MultiscaleFeatureExtractor",
]
