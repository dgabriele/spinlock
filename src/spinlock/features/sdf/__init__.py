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

from spinlock.features.sdf.extractors import SDFExtractor
from spinlock.features.sdf.config import (
    SDFConfig,
    SDFSpatialConfig,
    SDFSpectralConfig,
    SDFTemporalConfig,
    SDFOperatorSensitivityConfig,
    SDFCrossChannelConfig,
    SDFCausalityConfig,
    SDFInvariantDriftConfig,
    SDFDistributionalConfig,
    SDFStructuralConfig,
    SDFPhysicsConfig,
    SDFMorphologicalConfig,
    SDFMultiscaleConfig,
)
from spinlock.features.sdf.spatial import SpatialFeatureExtractor
from spinlock.features.sdf.spectral import SpectralFeatureExtractor
from spinlock.features.sdf.temporal import TemporalFeatureExtractor
from spinlock.features.sdf.operator_sensitivity import OperatorSensitivityExtractor
from spinlock.features.sdf.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.sdf.causality import CausalityFeatureExtractor
from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor
# Phase 2 extractors (v2.1)
from spinlock.features.sdf.distributional import DistributionalFeatureExtractor
from spinlock.features.sdf.structural import StructuralFeatureExtractor
from spinlock.features.sdf.physics import PhysicsFeatureExtractor
from spinlock.features.sdf.morphological import MorphologicalFeatureExtractor
from spinlock.features.sdf.multiscale import MultiscaleFeatureExtractor

__all__ = [
    "SDFExtractor",
    "SDFConfig",
    "SDFSpatialConfig",
    "SDFSpectralConfig",
    "SDFTemporalConfig",
    "SDFOperatorSensitivityConfig",
    "SDFCrossChannelConfig",
    "SDFCausalityConfig",
    "SDFInvariantDriftConfig",
    "SDFDistributionalConfig",
    "SDFStructuralConfig",
    "SDFPhysicsConfig",
    "SDFMorphologicalConfig",
    "SDFMultiscaleConfig",
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
