"""Unit tests for behavioral filter metrics, dedup, and dynamics classification.

Tests cover:
    1. Constant trajectory → all complexity metrics ≈ 0, classified fixed_point
    2. Linear ramp → positive metrics, NOT fixed_point
    3. Spatially uniform oscillation → low spatial_var, low gradient_energy
    4. Default config → no filtering (backward compat)
    5. Active config → trivial samples rejected
    6. Dedup buffer: identical fingerprints → marked as duplicates
    7. Dedup buffer: distinct fingerprints → both accepted
    8. Spectral flatness: periodic signal → low flatness
    9. Config schema: new fields have correct defaults
"""

import pytest
import torch
import numpy as np

from spinlock.lenia.replayer import (
    BehavioralDedupBuffer,
    DynamicsClass,
    LeniaReplayer,
    TemporalActivityMetrics,
    _WelfordNormalizer,
)


# ── Helpers ──

def _make_constant_traj(B=4, T=64, C=3, H=16, W=16, value=0.5):
    """Trajectory where every frame is identical — trivial fixed point."""
    return torch.full((B, T, C, H, W), value)


def _make_ramp_traj(B=4, T=64, C=3, H=16, W=16):
    """Linearly increasing trajectory — guaranteed non-trivial temporal activity."""
    t = torch.linspace(0, 1, T).view(1, T, 1, 1, 1)
    spatial = torch.rand(B, 1, C, H, W)
    return spatial * t.expand(B, T, C, H, W)


def _make_uniform_oscillation(B=4, T=64, C=3, H=16, W=16, period=4):
    """Spatially uniform oscillation — whole grid pulsing identically."""
    t = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1)
    osc = 0.5 + 0.3 * torch.sin(2 * np.pi * t / period)
    return osc.expand(B, T, C, H, W).clone()


def _make_complex_traj(B=4, T=64, C=3, H=16, W=16):
    """Spatially structured, temporally varying trajectory — non-trivial."""
    t = torch.linspace(0, 1, T).view(1, T, 1, 1, 1)
    # Spatial structure: gradient across H
    h = torch.linspace(0, 1, H).view(1, 1, 1, H, 1)
    w = torch.linspace(0, 1, W).view(1, 1, 1, 1, W)
    base = (h * 0.5 + w * 0.3).expand(B, T, C, H, W)
    # Add temporal variation (not perfectly periodic)
    temporal = t * 0.2 + 0.1 * torch.sin(2 * np.pi * t * 3.7)
    noise = torch.rand(B, T, C, H, W) * 0.05
    return (base + temporal + noise).clamp(0, 1)


# ── Test: Temporal Activity Metrics ──

class TestTemporalActivityMetrics:
    """Tests for _compute_temporal_activity."""

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_constant_traj_zero_activity(self):
        """Constant trajectory should have near-zero temporal activity."""
        replayer = self._make_replayer()
        traj = _make_constant_traj()
        metrics = replayer._compute_temporal_activity(traj)

        assert metrics.early_late_mse.shape == (4,)
        assert (metrics.early_late_mse < 1e-10).all()
        assert (metrics.late_evolution_rate < 1e-10).all()
        assert (metrics.late_half_mean_var < 1e-10).all()

    def test_ramp_traj_positive_activity(self):
        """Linearly increasing trajectory should have positive temporal activity."""
        replayer = self._make_replayer()
        traj = _make_ramp_traj()
        metrics = replayer._compute_temporal_activity(traj)

        assert (metrics.early_late_mse > 1e-5).all()
        assert (metrics.late_evolution_rate > 1e-6).all()

    def test_shapes(self):
        """Output tensors have correct shapes."""
        replayer = self._make_replayer()
        B, T = 8, 32
        traj = torch.rand(B, T, 3, 16, 16)
        metrics = replayer._compute_temporal_activity(traj)

        assert metrics.early_late_mse.shape == (B,)
        assert metrics.quarter_late_mse.shape == (B,)
        assert metrics.late_half_mean_var.shape == (B,)
        assert metrics.late_evolution_rate.shape == (B,)


# ── Test: Spatial Complexity Metrics ──

class TestSpatialComplexityMetrics:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_constant_traj_zero_spatial_var(self):
        """Constant trajectory should have zero spatial variance."""
        replayer = self._make_replayer()
        traj = _make_constant_traj()
        sv = replayer._compute_late_spatial_variance(traj)
        assert (sv < 1e-10).all()

    def test_structured_traj_positive_spatial_var(self):
        """Structured trajectory should have positive spatial variance."""
        replayer = self._make_replayer()
        traj = _make_complex_traj()
        sv = replayer._compute_late_spatial_variance(traj)
        assert (sv > 1e-4).all()

    def test_constant_traj_zero_gradient_energy(self):
        """Constant trajectory should have zero gradient energy."""
        replayer = self._make_replayer()
        traj = _make_constant_traj()
        ge = replayer._compute_gradient_energy(traj)
        assert (ge < 1e-10).all()

    def test_structured_traj_positive_gradient_energy(self):
        """Spatially structured trajectory should have positive gradient energy."""
        replayer = self._make_replayer()
        traj = _make_complex_traj()
        ge = replayer._compute_gradient_energy(traj)
        assert (ge > 1e-6).all()

    def test_uniform_oscillation_low_spatial_metrics(self):
        """Spatially uniform oscillation should have low spatial_var and gradient_energy."""
        replayer = self._make_replayer()
        traj = _make_uniform_oscillation()
        sv = replayer._compute_late_spatial_variance(traj)
        ge = replayer._compute_gradient_energy(traj)
        assert (sv < 1e-8).all(), f"spatial_var too high: {sv}"
        assert (ge < 1e-8).all(), f"gradient_energy too high: {ge}"


# ── Test: Spectral Flatness ──

class TestSpectralFlatness:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_periodic_signal_low_flatness(self):
        """Pure periodic oscillation should have low spectral flatness."""
        replayer = self._make_replayer()
        traj = _make_uniform_oscillation(period=4)
        sf = replayer._compute_spectral_flatness(traj)
        assert (sf < 0.3).all(), f"Expected low flatness for periodic signal, got {sf}"

    def test_noise_signal_high_flatness(self):
        """White noise trajectory should have high spectral flatness."""
        replayer = self._make_replayer()
        torch.manual_seed(42)
        traj = torch.rand(4, 64, 3, 16, 16) * 0.5 + 0.25
        sf = replayer._compute_spectral_flatness(traj)
        assert (sf > 0.4).all(), f"Expected high flatness for noise, got {sf}"

    def test_constant_traj_returns_finite(self):
        """Constant trajectory should return finite spectral flatness."""
        replayer = self._make_replayer()
        traj = _make_constant_traj()
        sf = replayer._compute_spectral_flatness(traj)
        assert torch.isfinite(sf).all()


# ── Test: Extended _find_bad_samples ──

class TestFindBadSamples:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_default_config_no_extra_filtering(self):
        """With default thresholds (all 0), only dead/saturated/oscillating are rejected."""
        replayer = self._make_replayer()
        # Alive, non-trivial trajectory
        traj = _make_complex_traj(T=64)
        bad = replayer._find_bad_samples(traj)
        assert len(bad) == 0, f"Expected no bad samples with default config, got {bad}"

    def test_constant_traj_rejected_by_temporal_filter(self):
        """Constant trajectory should be rejected when temporal filter is active."""
        replayer = self._make_replayer(min_temporal_activity=1e-5)
        traj = _make_constant_traj()
        bad = replayer._find_bad_samples(traj)
        assert len(bad) == 4, f"Expected all 4 constant samples rejected, got {bad}"

    def test_constant_traj_rejected_by_early_late_mse(self):
        """Constant trajectory should be rejected by early-late MSE filter."""
        replayer = self._make_replayer(min_early_late_mse=1e-5)
        traj = _make_constant_traj()
        bad = replayer._find_bad_samples(traj)
        assert len(bad) == 4

    def test_complex_traj_passes_all_filters(self):
        """Complex trajectory should pass all filters."""
        replayer = self._make_replayer(
            min_temporal_activity=1e-8,
            min_early_late_mse=1e-8,
            spatial_var_threshold=1e-6,
            gradient_energy_threshold=1e-8,
        )
        traj = _make_complex_traj(T=64)
        bad = replayer._find_bad_samples(traj)
        assert len(bad) == 0, f"Expected complex traj to pass all filters, got {bad}"

    def test_uniform_oscillation_rejected_by_spatial_var(self):
        """Spatially uniform oscillation should be rejected by spatial_var_threshold."""
        replayer = self._make_replayer(spatial_var_threshold=1e-4)
        traj = _make_uniform_oscillation()
        bad = replayer._find_bad_samples(traj)
        assert len(bad) == 4, f"Expected all uniform oscillation rejected, got {bad}"


# ── Test: Behavioral Dedup Buffer ──

class TestBehavioralDedupBuffer:

    def test_first_batch_all_accepted(self):
        """First batch should always be accepted (empty buffer)."""
        buf = BehavioralDedupBuffer(threshold=0.5, device="cpu")
        fp = torch.randn(10, 8)
        dup = buf.check_and_add(fp)
        assert not dup.any(), "First batch should have no duplicates"

    def test_identical_fingerprints_detected(self):
        """Identical fingerprints added twice should be detected as duplicates."""
        buf = BehavioralDedupBuffer(threshold=0.5, device="cpu")
        fp = torch.ones(5, 8)
        # First batch: accepted
        dup1 = buf.check_and_add(fp)
        assert not dup1.any()
        # Second batch: identical → duplicates
        dup2 = buf.check_and_add(fp.clone())
        assert dup2.all(), f"Expected all duplicates, got {dup2}"

    def test_distinct_fingerprints_accepted(self):
        """Very different fingerprints should both be accepted."""
        buf = BehavioralDedupBuffer(threshold=0.5, device="cpu")
        fp1 = torch.zeros(5, 8)
        fp2 = torch.ones(5, 8) * 100  # very far away
        dup1 = buf.check_and_add(fp1)
        dup2 = buf.check_and_add(fp2)
        assert not dup1.any()
        assert not dup2.any(), f"Expected distinct fingerprints accepted, got {dup2}"

    def test_buffer_grows_correctly(self):
        """Buffer should grow only with non-duplicate fingerprints."""
        buf = BehavioralDedupBuffer(threshold=0.5, device="cpu")
        fp1 = torch.randn(5, 8)
        buf.check_and_add(fp1)
        assert buf._buffer.shape[0] == 5

        fp2 = fp1.clone()  # duplicates
        buf.check_and_add(fp2)
        assert buf._buffer.shape[0] == 5, "Buffer should not grow with duplicates"

        fp3 = torch.randn(3, 8) * 100  # very different
        buf.check_and_add(fp3)
        assert buf._buffer.shape[0] == 8


# ── Test: Welford Normalizer ──

class TestWelfordNormalizer:

    def test_unit_variance_normalization(self):
        """After many samples, normalized output should have ~unit variance."""
        norm = _WelfordNormalizer(dim=4)
        torch.manual_seed(0)
        for _ in range(100):
            batch = torch.randn(10, 4) * 3 + 5  # mean=5, std=3
            norm.update(batch)

        std = norm.std()
        # Should be close to 3
        assert (std - 3.0).abs().max() < 0.5, f"Expected std ≈ 3, got {std}"

        # Normalized should be ~unit variance
        test = torch.randn(100, 4) * 3 + 5
        normalized = norm.normalize(test)
        assert normalized.std(dim=0).mean() < 1.5  # rough check


# ── Test: Dynamics Classification ──

class TestDynamicsClassification:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", classify_dynamics=True, **kwargs)

    def test_constant_classified_fixed_point(self):
        """Constant trajectory should be classified as fixed_point."""
        replayer = self._make_replayer()
        traj = _make_constant_traj()
        classes = replayer._classify_dynamics(traj)
        assert all(c == "fixed_point" for c in classes), f"Expected fixed_point, got {classes}"

    def test_ramp_not_fixed_point(self):
        """Linearly increasing trajectory should NOT be classified as fixed_point."""
        replayer = self._make_replayer()
        traj = _make_ramp_traj(T=64)
        classes = replayer._classify_dynamics(traj)
        assert all(c != "fixed_point" for c in classes), f"Expected non-fixed_point, got {classes}"

    def test_dynamics_class_enum_values(self):
        """DynamicsClass enum should have expected values."""
        assert DynamicsClass.FIXED_POINT.value == "fixed_point"
        assert DynamicsClass.PERIODIC.value == "periodic"
        assert DynamicsClass.APERIODIC.value == "aperiodic"
        assert DynamicsClass.TRANSIENT.value == "transient"

    def test_classification_stored_in_side_channel(self):
        """_classify_dynamics should store result in _last_dynamics_classes."""
        replayer = self._make_replayer()
        traj = _make_constant_traj(B=3)
        replayer._classify_dynamics(traj)
        assert replayer._last_dynamics_classes is not None
        assert len(replayer._last_dynamics_classes) == 3


# ── Test: filter_duplicates integration ──

class TestFilterDuplicates:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_disabled_returns_no_dups(self):
        """When dedup_enabled=False, filter_duplicates should return all False."""
        replayer = self._make_replayer(dedup_enabled=False)
        outputs = torch.rand(4, 3, 64, 3, 16, 16)  # [B, M, T, C, H, W]
        dup = replayer.filter_duplicates(outputs)
        assert not dup.any()

    def test_enabled_detects_identical(self):
        """When dedup enabled, identical trajectories should be detected."""
        replayer = self._make_replayer(dedup_enabled=True, dedup_threshold=0.5)
        # Single constant frame repeated — all fingerprints identical
        traj = torch.full((4, 3, 64, 3, 16, 16), 0.5)
        dup1 = replayer.filter_duplicates(traj)
        assert not dup1.any(), "First batch should not be duplicates"

        dup2 = replayer.filter_duplicates(traj)
        assert dup2.all(), "Second identical batch should all be duplicates"


# ── Test: Config Schema ──

class TestConfigSchema:

    def test_defaults_reproduce_old_behavior(self):
        """Default config values should disable all new filters."""
        from spinlock.config.schema import LeniaSimulationConfig
        cfg = LeniaSimulationConfig()
        assert cfg.min_temporal_activity == 0.0
        assert cfg.min_early_late_mse == 0.0
        assert cfg.spatial_var_threshold == 0.0
        assert cfg.gradient_energy_threshold == 0.0
        assert cfg.spectral_flatness_threshold == 0.0
        assert cfg.dedup_enabled is False
        assert cfg.dedup_threshold == 0.5
        assert cfg.classify_dynamics is False

    def test_config_accepts_filter_values(self):
        """Config should accept non-zero filter values."""
        from spinlock.config.schema import LeniaSimulationConfig
        cfg = LeniaSimulationConfig(
            min_temporal_activity=1e-5,
            min_early_late_mse=1e-5,
            spatial_var_threshold=0.001,
            gradient_energy_threshold=1e-5,
            spectral_flatness_threshold=0.1,
            dedup_enabled=True,
            dedup_threshold=0.3,
            classify_dynamics=True,
        )
        assert cfg.min_temporal_activity == 1e-5
        assert cfg.dedup_enabled is True
        assert cfg.classify_dynamics is True


# ── Test: Fingerprint computation ──

class TestFingerprint:

    def _make_replayer(self, **kwargs):
        return LeniaReplayer(device="cpu", **kwargs)

    def test_fingerprint_shape(self):
        """Fingerprint should be [B, 8] for 3-channel trajectory."""
        replayer = self._make_replayer()
        traj = torch.rand(4, 64, 3, 16, 16)
        fp = replayer._compute_fingerprint(traj)
        assert fp.shape == (4, 8), f"Expected (4, 8), got {fp.shape}"

    def test_fingerprint_deterministic(self):
        """Same trajectory should produce same fingerprint."""
        replayer = self._make_replayer()
        traj = torch.rand(4, 64, 3, 16, 16)
        fp1 = replayer._compute_fingerprint(traj)
        fp2 = replayer._compute_fingerprint(traj)
        assert torch.allclose(fp1, fp2)

    def test_different_trajs_different_fingerprints(self):
        """Different trajectories should produce different fingerprints."""
        replayer = self._make_replayer()
        traj1 = _make_constant_traj()
        traj2 = _make_complex_traj()
        fp1 = replayer._compute_fingerprint(traj1)
        fp2 = replayer._compute_fingerprint(traj2)
        assert not torch.allclose(fp1, fp2), "Different trajectories should have different fingerprints"

    def test_fingerprint_finite(self):
        """Fingerprint should contain no NaN or inf."""
        replayer = self._make_replayer()
        traj = _make_complex_traj()
        fp = replayer._compute_fingerprint(traj)
        assert torch.isfinite(fp).all(), f"Fingerprint contains non-finite values: {fp}"
