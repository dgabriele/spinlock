#!/usr/bin/env python3
"""
Register all SDF (Summary Descriptor Features) in the feature registry.

This script creates a comprehensive registry of all SDF features including:
- Spatial statistics (19 base + 8 new = 27 features)
- Spectral statistics (27 base + 4 new = 31 features)
- Temporal statistics (13 features, T≥3 required)
- Cross-channel interactions (10 base + 2 new = 12 features)
- Causality (15 features, T≥3 required)
- Invariant drift (60 base + 4 new = 64 features, T>1 required)
- Operator sensitivity (12 features)

Total: ~164 features (all valid for T=500)

Usage:
    python scripts/dev/register_sdf_features.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spinlock.features.registry import FeatureRegistry


def register_all_sdf_features() -> FeatureRegistry:
    """
    Register all SDF features in the registry.

    Returns:
        FeatureRegistry with all SDF features registered
    """
    registry = FeatureRegistry(family_name="summary")

    # =========================================================================
    # SPATIAL FEATURES (27 features)
    # =========================================================================

    # Basic moments (8 features)
    registry.register("spatial_mean", "spatial", "Spatial mean")
    registry.register("spatial_variance", "spatial", "Spatial variance")
    registry.register("spatial_std", "spatial", "Spatial standard deviation")
    registry.register("spatial_skewness", "spatial", "Spatial skewness (3rd moment)")
    registry.register("spatial_kurtosis", "spatial", "Spatial kurtosis (4th moment)")
    registry.register("spatial_min", "spatial", "Spatial minimum")
    registry.register("spatial_max", "spatial", "Spatial maximum")
    registry.register("spatial_range", "spatial", "Spatial range (max - min)")

    # Robust statistics (2 features)
    registry.register("spatial_iqr", "spatial", "Interquartile range (Q3 - Q1)")
    registry.register("spatial_mad", "spatial", "Median absolute deviation")

    # Gradients (6 features)
    registry.register("gradient_magnitude_mean", "spatial", "Mean gradient magnitude")
    registry.register("gradient_magnitude_std", "spatial", "Std gradient magnitude")
    registry.register("gradient_magnitude_max", "spatial", "Max gradient magnitude")
    registry.register("gradient_x_mean", "spatial", "Mean horizontal gradient")
    registry.register("gradient_y_mean", "spatial", "Mean vertical gradient")
    registry.register("gradient_anisotropy", "spatial", "Gradient anisotropy ratio")

    # Curvature (3 features)
    registry.register("laplacian_mean", "spatial", "Mean Laplacian (curvature)")
    registry.register("laplacian_std", "spatial", "Std Laplacian")
    registry.register("laplacian_energy", "laplacian_energy", "Laplacian energy per pixel")

    # NEW: Effective dimensionality (3 features)
    registry.register("effective_rank", "spatial", "SVD-based effective rank")
    registry.register("participation_ratio", "spatial", "Inverse Simpson index of SVs")
    registry.register("explained_variance_90", "spatial", "# SVs for 90% variance")

    # NEW: Gradient saturation (2 features)
    registry.register("gradient_saturation_ratio", "spatial", "Fraction with low gradients")
    registry.register("gradient_flatness", "spatial", "Kurtosis of gradient distribution")

    # NEW: Coherence structure (3 features)
    registry.register("coherence_length", "spatial", "Autocorrelation decay length (1/e)")
    registry.register("correlation_anisotropy", "spatial", "Directional correlation bias")
    registry.register("structure_factor_peak", "spatial", "Characteristic length from power spectrum")

    # =========================================================================
    # SPECTRAL FEATURES (31 features)
    # =========================================================================

    # Power spectrum statistics (8 features)
    registry.register("spectral_power_mean", "spectral", "Mean spectral power")
    registry.register("spectral_power_std", "spectral", "Std spectral power")
    registry.register("spectral_power_max", "spectral", "Max spectral power")
    registry.register("spectral_power_kurtosis", "spectral", "Spectral power kurtosis")
    registry.register("spectral_centroid", "spectral", "Spectral centroid (center of mass)")
    registry.register("spectral_spread", "spectral", "Spectral spread (variance around centroid)")
    registry.register("spectral_rolloff", "spectral", "Spectral rolloff (95% energy threshold)")
    registry.register("spectral_flatness", "spectral", "Spectral flatness (geometric/arithmetic mean)")

    # Frequency band energies (3 features)
    registry.register("low_freq_energy", "spectral", "Energy in low frequencies")
    registry.register("mid_freq_energy", "spectral", "Energy in mid frequencies")
    registry.register("high_freq_energy", "spectral", "Energy in high frequencies")

    # Dominant frequency (4 features)
    registry.register("dominant_frequency_magnitude", "spectral", "Magnitude of dominant frequency")
    registry.register("dominant_frequency_x", "spectral", "X-component of dominant frequency")
    registry.register("dominant_frequency_y", "spectral", "Y-component of dominant frequency")
    registry.register("dominant_frequency_radial", "spectral", "Radial dominant frequency")

    # Entropy (2 features)
    registry.register("spectral_entropy", "spectral", "Spectral entropy")
    registry.register("spectral_entropy_normalized", "spectral", "Normalized spectral entropy")

    # Anisotropy (3 features)
    registry.register("spectral_anisotropy_ratio", "spectral", "Power anisotropy ratio")
    registry.register("spectral_horizontal_energy", "spectral", "Horizontal energy")
    registry.register("spectral_vertical_energy", "spectral", "Vertical energy")

    # Multiscale (3 features)
    registry.register("spectral_slope", "spectral", "Power law slope (1/f^α)")
    registry.register("spectral_knee_freq", "spectral", "Knee frequency (if present)")
    registry.register("spectral_whiteness", "spectral", "Whiteness measure")

    # Phase statistics (4 features)
    registry.register("phase_coherence_mean", "spectral", "Mean phase coherence")
    registry.register("phase_coherence_std", "spectral", "Std phase coherence")
    registry.register("phase_uniformity", "spectral", "Phase distribution uniformity")
    registry.register("phase_concentration", "spectral", "Phase concentration (von Mises)")

    # NEW: Harmonic content (4 features)
    registry.register("harmonic_ratio_2f", "spectral", "2nd harmonic / fundamental ratio")
    registry.register("harmonic_ratio_3f", "spectral", "3rd harmonic / fundamental ratio")
    registry.register("total_harmonic_distortion", "spectral", "THD (nonlinearity measure)")
    registry.register("fundamental_purity", "spectral", "Fundamental / total power")

    # =========================================================================
    # TEMPORAL FEATURES (13 features, T≥3 required)
    # =========================================================================

    # Trajectory statistics (6 features)
    registry.register("temporal_mean_drift", "temporal", "Mean temporal drift")
    registry.register("temporal_variance_growth", "temporal", "Variance growth rate")
    registry.register("temporal_autocorr_lag1", "temporal", "Autocorrelation at lag 1")
    registry.register("temporal_autocorr_decay", "temporal", "Autocorrelation decay rate")
    registry.register("temporal_stationarity", "temporal", "Stationarity measure")
    registry.register("temporal_trend_strength", "temporal", "Trend strength")

    # Stability (4 features)
    registry.register("temporal_stability_score", "temporal", "Overall stability score")
    registry.register("temporal_regime_change_count", "temporal", "# regime changes detected")
    registry.register("temporal_max_deviation", "temporal", "Max deviation from initial")
    registry.register("temporal_settling_time", "temporal", "Time to settle (if converges)")

    # Periodicity (3 features)
    registry.register("temporal_dominant_period", "temporal", "Dominant period (if periodic)")
    registry.register("temporal_periodicity_strength", "temporal", "Periodicity strength")
    registry.register("temporal_quasi_periodicity", "temporal", "Quasi-periodicity measure")

    # =========================================================================
    # CROSS-CHANNEL FEATURES (12 features, C>1 required)
    # =========================================================================

    # Correlation spectrum (5 features)
    registry.register("cross_channel_corr_mean", "cross_channel", "Mean pairwise correlation")
    registry.register("cross_channel_corr_std", "cross_channel", "Std pairwise correlation")
    registry.register("cross_channel_corr_max", "cross_channel", "Max pairwise correlation")
    registry.register("cross_channel_corr_min", "cross_channel", "Min pairwise correlation")
    registry.register("cross_channel_effective_rank", "cross_channel", "Effective rank of corr matrix")

    # Mutual information (2 features, opt-in)
    registry.register("cross_channel_mi_mean", "cross_channel", "Mean mutual information")
    registry.register("cross_channel_mi_max", "cross_channel", "Max mutual information")

    # NEW: Conditional mutual information (2 features, opt-in)
    registry.register("cross_channel_cmi_mean", "cross_channel", "Mean conditional MI I(X;Y|Z)")
    registry.register("cross_channel_cmi_max", "cross_channel", "Max conditional MI")

    # Coherence (3 features, opt-in)
    registry.register("cross_spectral_coherence_mean", "cross_channel", "Mean cross-spectral coherence")
    registry.register("cross_spectral_coherence_max", "cross_channel", "Max cross-spectral coherence")
    registry.register("cross_spectral_phase_locking", "cross_channel", "Phase-locking index")

    # =========================================================================
    # CAUSALITY FEATURES (15 features, T≥3 required)
    # =========================================================================

    # Transfer entropy (5 features)
    registry.register("transfer_entropy_forward_mean", "causality", "Mean TE (i→j)")
    registry.register("transfer_entropy_backward_mean", "causality", "Mean TE (j→i)")
    registry.register("transfer_entropy_asymmetry", "causality", "TE asymmetry measure")
    registry.register("transfer_entropy_net_flow", "causality", "Net information flow")
    registry.register("transfer_entropy_total", "causality", "Total TE (bidirectional)")

    # Granger causality (5 features)
    registry.register("granger_causality_forward_mean", "causality", "Mean Granger (i→j)")
    registry.register("granger_causality_backward_mean", "causality", "Mean Granger (j→i)")
    registry.register("granger_causality_asymmetry", "causality", "Granger asymmetry")
    registry.register("granger_causality_net_flow", "causality", "Net Granger flow")
    registry.register("granger_causality_total", "causality", "Total Granger")

    # Time-delayed correlations (5 features)
    registry.register("delayed_correlation_peak", "causality", "Peak delayed correlation")
    registry.register("delayed_correlation_lag", "causality", "Lag of peak correlation")
    registry.register("delayed_correlation_asymmetry", "causality", "Delayed corr asymmetry")
    registry.register("delayed_correlation_forward_strength", "causality", "Forward corr strength")
    registry.register("delayed_correlation_backward_strength", "causality", "Backward corr strength")

    # =========================================================================
    # INVARIANT DRIFT FEATURES (64 features, T>1 required)
    # =========================================================================

    # Generic norm drift: 5 norms × 4 metrics × 3 scales = 60 features
    norms = ["L1", "L2", "Linf", "entropy", "tv"]
    metrics = ["mean_drift", "drift_variance", "final_initial_ratio", "monotonicity"]
    scales = ["raw", "lowpass", "highpass"]

    for norm in norms:
        for metric in metrics:
            for scale in scales:
                name = f"{norm}_{metric}_{scale}"
                desc = f"{norm} norm {metric} ({scale} scale)"
                registry.register(name, "invariant_drift", desc)

    # NEW: Scale-specific dissipation (4 features)
    registry.register("dissipation_rate_lowfreq", "invariant_drift", "Low-freq energy decay rate")
    registry.register("dissipation_rate_highfreq", "invariant_drift", "High-freq energy decay rate")
    registry.register("dissipation_selectivity", "invariant_drift", "High/low dissipation ratio")
    registry.register("energy_cascade_direction", "invariant_drift", "Upscale vs downscale transfer")

    # =========================================================================
    # OPERATOR SENSITIVITY FEATURES (12 features)
    # =========================================================================

    # Lipschitz estimates (3 features)
    registry.register("lipschitz_estimate_small", "operator_sensitivity", "Lipschitz at small perturbation")
    registry.register("lipschitz_estimate_medium", "operator_sensitivity", "Lipschitz at medium perturbation")
    registry.register("lipschitz_estimate_large", "operator_sensitivity", "Lipschitz at large perturbation")

    # Gain curves (4 features)
    registry.register("gain_small", "operator_sensitivity", "Output gain at small input")
    registry.register("gain_medium", "operator_sensitivity", "Output gain at medium input")
    registry.register("gain_large", "operator_sensitivity", "Output gain at large input")
    registry.register("gain_saturation", "operator_sensitivity", "Gain saturation degree")

    # Linearity (3 features)
    registry.register("linearity_r2", "operator_sensitivity", "R² correlation (linearity)")
    registry.register("linearity_compression_ratio", "operator_sensitivity", "Output/input compression")
    registry.register("linearity_deviation_max", "operator_sensitivity", "Max deviation from linear")

    # Extras (2 features)
    registry.register("response_asymmetry", "operator_sensitivity", "Asymmetry of response")
    registry.register("multi_channel_consistency", "operator_sensitivity", "Cross-channel consistency")

    return registry


def main():
    """Main entry point."""
    print("=" * 70)
    print("SDF Feature Registry")
    print("=" * 70)

    # Register all features
    registry = register_all_sdf_features()

    # Print summary
    print(f"\n{registry}\n")

    # Print details by category
    print("=" * 70)
    print("Features by Category")
    print("=" * 70)

    for category in sorted(registry.categories):
        features = registry.get_features_by_category(category)
        print(f"\n{category.upper()} ({len(features)} features):")
        for feat in features:
            print(f"  [{feat.index:3d}] {feat.name:50s} - {feat.description}")

    # Export to JSON
    json_path = Path(__file__).parent.parent.parent / "src" / "spinlock" / "features" / "sdf" / "feature_registry.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        f.write(registry.to_json())

    print(f"\n✓ Exported registry to: {json_path}")
    print(f"  Total features: {registry.num_features}")
    print(f"  Categories: {len(registry.categories)}")


if __name__ == "__main__":
    main()
