"""
Summary Descriptor Features (SDF) v2.0.

Comprehensive feature extraction covering 12 scientific domains:

v1.0 categories:
- Spatial statistics (moments, gradients, curvature)
- Spectral features (FFT, power spectrum, frequencies)
- Temporal dynamics (growth rates, oscillations, stability)
- Information theory (entropy, complexity, compression)
- Structural features (topology, edges, texture)
- Statistical physics (correlations, structure factor)
- Morphological features (shape, moments)
- Multiscale analysis (wavelets, pyramids)

v2.0 categories (operator-aware features):
- Operator sensitivity (Lipschitz estimates, gain curves, linearity)
- Cross-channel interactions (correlation spectra, coherence, mutual information)
- Causality/directionality (lagged correlations, transfer entropy, Granger causality)
- Invariant drift (multi-scale norm-based drift tracking)

Total: ~260-350 features with v2.0 features enabled (default multiscale configurations).
"""

__version__ = "2.0.0"

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
)
from spinlock.features.sdf.spatial import SpatialFeatureExtractor
from spinlock.features.sdf.spectral import SpectralFeatureExtractor
from spinlock.features.sdf.temporal import TemporalFeatureExtractor
from spinlock.features.sdf.operator_sensitivity import OperatorSensitivityExtractor
from spinlock.features.sdf.cross_channel import CrossChannelFeatureExtractor
from spinlock.features.sdf.causality import CausalityFeatureExtractor
from spinlock.features.sdf.invariant_drift import InvariantDriftExtractor

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
    "SpatialFeatureExtractor",
    "SpectralFeatureExtractor",
    "TemporalFeatureExtractor",
    "OperatorSensitivityExtractor",
    "CrossChannelFeatureExtractor",
    "CausalityFeatureExtractor",
    "InvariantDriftExtractor",
]
