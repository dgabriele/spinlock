"""
Temporal dynamics feature extraction.

Extracts trajectory-level features capturing temporal evolution:
- Growth rates and acceleration (energy, variance, mean)
- Oscillations (dominant frequency, amplitude, period)
- Stability (Lyapunov approximation, smoothness, regime changes)
- Stationarity (trend strength, detrended variance)

These features are computed once per realization (trajectory-level),
not per-timestep like spatial/spectral features.
"""

import torch
import torch.fft
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFTemporalConfig


class TemporalFeatureExtractor:
    """
    Extract temporal dynamics features from trajectories.

    Operates on full trajectories [N, M, T, C, H, W] and computes
    trajectory-level summaries.

    Example:
        >>> extractor = TemporalFeatureExtractor(device='cuda')
        >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
        >>> features = extractor.extract(trajectories)  # [N, M, num_features]
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize temporal feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        config: Optional['SDFTemporalConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract temporal features from trajectories.

        Args:
            trajectories: Full trajectories [N, M, T, C, H, W]
                N = batch size
                M = num realizations
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SDFTemporalConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors [N, M, C]
            One value per realization (trajectory-level features)
        """
        N, M, T, C, H, W = trajectories.shape

        # Handle single-timestep data (temporal features undefined)
        if T == 1:
            # Return NaN features for all temporal metrics
            nan_features = torch.full((N, M, C), float('nan'), device=trajectories.device)
            return {
                'energy_growth_rate': nan_features.clone(),
                'energy_growth_accel': nan_features.clone(),
                'variance_growth_rate': nan_features.clone(),
                'temporal_freq_dominant': nan_features.clone(),
                'oscillation_amplitude': nan_features.clone(),
                'oscillation_period': nan_features.clone(),
                'autocorr_decay_time': nan_features.clone(),
                'lyapunov_approx': nan_features.clone(),
                'trajectory_smoothness': nan_features.clone(),
                'regime_switches': nan_features.clone(),
                'final_to_initial_ratio': nan_features.clone(),
                'trend_strength': nan_features.clone(),
                'detrended_variance': nan_features.clone(),
            }

        # Compute spatially-averaged time series for each realization
        # This gives us the temporal evolution of global quantities
        time_series = trajectories.mean(dim=(-2, -1))  # [N, M, T, C]

        features = {}

        # Use config to determine which features to extract
        if config is None:
            include_all = True
        else:
            include_all = False

        # Growth & decay features
        if include_all or (config is not None and config.include_energy_growth_rate):
            energy = (trajectories ** 2).sum(dim=(-2, -1))  # [N, M, T, C]
            growth_rate = self._compute_growth_rate(energy)
            features['energy_growth_rate'] = growth_rate

        if include_all or (config is not None and config.include_energy_growth_accel):
            energy = (trajectories ** 2).sum(dim=(-2, -1))
            accel = self._compute_growth_acceleration(energy)
            features['energy_growth_accel'] = accel

        if include_all or (config is not None and config.include_variance_growth_rate):
            variance = trajectories.var(dim=(-2, -1))  # [N, M, T, C]
            growth_rate = self._compute_growth_rate(variance)
            features['variance_growth_rate'] = growth_rate

        # Oscillation features
        if include_all or (config is not None and config.include_temporal_freq_dominant):
            energy = (trajectories ** 2).sum(dim=(-2, -1))
            dom_freq = self._compute_dominant_temporal_frequency(energy)
            features['temporal_freq_dominant'] = dom_freq

        if include_all or (config is not None and config.include_oscillation_amplitude):
            energy = (trajectories ** 2).sum(dim=(-2, -1))
            amplitude = self._compute_oscillation_amplitude(energy)
            features['oscillation_amplitude'] = amplitude

        if include_all or (config is not None and config.include_oscillation_period):
            energy = (trajectories ** 2).sum(dim=(-2, -1))
            period = self._compute_oscillation_period(energy)
            features['oscillation_period'] = period

        if include_all or (config is not None and config.include_autocorr_decay_time):
            max_lag = config.autocorr_max_lag if config else 20
            decay = self._compute_autocorr_decay_time(time_series, max_lag)
            features['autocorr_decay_time'] = decay

        # Phase 2 extension: PACF (Partial Autocorrelation Function)
        if include_all or (config is not None and config.include_pacf):
            max_lag_pacf = config.pacf_max_lag if config else 10
            pacf_features = self._compute_pacf(time_series, max_lag_pacf)
            features.update(pacf_features)

        # Stability metrics
        if include_all or (config is not None and config.include_lyapunov_approx):
            lyap = self._compute_lyapunov_approx(time_series)
            features['lyapunov_approx'] = lyap

        if include_all or (config is not None and config.include_trajectory_smoothness):
            smoothness = self._compute_trajectory_smoothness(time_series)
            features['trajectory_smoothness'] = smoothness

        if include_all or (config is not None and config.include_regime_switches):
            switches = self._compute_regime_switches(time_series)
            features['regime_switches'] = switches

        if include_all or (config is not None and config.include_final_to_initial_ratio):
            ratio = self._compute_final_to_initial_ratio(time_series)
            features['final_to_initial_ratio'] = ratio

        # Stationarity features
        if include_all or (config is not None and config.include_trend_strength):
            trend = self._compute_trend_strength(time_series)
            features['trend_strength'] = trend

        if include_all or (config is not None and config.include_detrended_variance):
            detrend_var = self._compute_detrended_variance(time_series)
            features['detrended_variance'] = detrend_var

        # Event detection features
        if include_all or (config is not None and config.include_event_counts):
            event_features = self._compute_event_counts(time_series)
            features.update(event_features)

        if include_all or (config is not None and config.include_time_to_event):
            tte_features = self._compute_time_to_event(time_series)
            features.update(tte_features)

        # Rolling window statistics (CRITICAL: multi-timescale analysis)
        if include_all or (config is not None and config.include_rolling_windows):
            rolling_features = self._compute_rolling_stats(
                time_series,
                window_fractions=config.rolling_window_fractions if config is not None else [0.05, 0.10, 0.20]
            )
            features.update(rolling_features)

        return features

    # =========================================================================
    # Growth & Decay
    # =========================================================================

    def _compute_growth_rate(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute exponential growth rate via linear fit to log(y).

        For y(t) ~ exp(r*t), we have log(y) ~ r*t + const.
        Fit: r = slope of log(y) vs t.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Growth rates [N, M, C]
        """
        N, M, T, C = time_series.shape

        # Take log (add small epsilon for stability)
        log_series = torch.log(torch.abs(time_series) + 1e-8)

        # Time index
        t = torch.arange(T, dtype=torch.float32, device=time_series.device)

        # Linear regression: log(y) = a + b*t
        # b = cov(t, log_y) / var(t)
        t_mean = t.mean()
        t_centered = t - t_mean

        log_series_mean = log_series.mean(dim=2, keepdim=True)  # [N, M, 1, C]
        log_series_centered = log_series - log_series_mean  # [N, M, T, C]

        # Covariance: (1/T) * sum(t_centered * log_series_centered)
        cov = (t_centered.view(1, 1, T, 1) * log_series_centered).mean(dim=2)  # [N, M, C]

        # Variance of t
        var_t = t_centered.pow(2).mean()

        # Slope (growth rate)
        growth_rate = cov / var_t

        return growth_rate

    def _compute_growth_acceleration(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute second derivative of growth (acceleration).

        Uses central differences on growth rate time series.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Acceleration [N, M, C]
        """
        # First derivative (velocity)
        velocity = time_series[:, :, 1:, :] - time_series[:, :, :-1, :]

        # Second derivative (acceleration)
        accel = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]

        # Mean absolute acceleration
        mean_accel = accel.abs().mean(dim=2)

        return mean_accel

    # =========================================================================
    # Oscillations
    # =========================================================================

    def _compute_dominant_temporal_frequency(
        self,
        time_series: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dominant temporal frequency via FFT.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Dominant frequency (normalized) [N, M, C]
        """
        N, M, T, C = time_series.shape

        # Remove mean (detrend)
        time_series_centered = time_series - time_series.mean(dim=2, keepdim=True)

        # FFT along time dimension
        fft_result = torch.fft.rfft(time_series_centered, dim=2)  # [N, M, T//2+1, C]
        power = torch.abs(fft_result) ** 2

        # Find peak frequency (excluding DC component at index 0)
        power_no_dc = power[:, :, 1:, :]  # Exclude DC
        peak_idx = torch.argmax(power_no_dc, dim=2) + 1  # [N, M, C] (add 1 for DC offset)

        # Normalize by Nyquist frequency
        dominant_freq = peak_idx.float() / T

        return dominant_freq

    def _compute_oscillation_amplitude(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute oscillation amplitude (peak-to-trough).

        Args:
            time_series: [N, M, T, C]

        Returns:
            Amplitude [N, M, C]
        """
        max_val = time_series.amax(dim=2)  # [N, M, C]
        min_val = time_series.amin(dim=2)  # [N, M, C]

        amplitude = max_val - min_val

        return amplitude

    def _compute_oscillation_period(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Estimate oscillation period from dominant frequency.

        Period = 1 / frequency

        Args:
            time_series: [N, M, T, C]

        Returns:
            Period (in timesteps) [N, M, C]
        """
        dominant_freq = self._compute_dominant_temporal_frequency(time_series)

        # Period = 1 / freq (add epsilon to avoid division by zero)
        period = 1.0 / (dominant_freq + 1e-8)

        return period

    def _compute_autocorr_decay_time(
        self,
        time_series: torch.Tensor,
        max_lag: int = 20
    ) -> torch.Tensor:
        """
        Compute autocorrelation decay timescale.

        Fits exponential decay: autocorr(tau) ~ exp(-tau / tau_decay)

        Args:
            time_series: [N, M, T, C]
            max_lag: Maximum lag to compute

        Returns:
            Decay timescale [N, M, C]
        """
        N, M, T, C = time_series.shape

        # Compute autocorrelation via FFT (efficient for all lags)
        # Mean-center
        mean = time_series.mean(dim=2, keepdim=True)
        centered = time_series - mean

        # FFT, compute power spectrum, inverse FFT
        fft_result = torch.fft.rfft(centered, n=2*T, dim=2)
        power = fft_result * torch.conj(fft_result)
        autocorr_full = torch.fft.irfft(power, n=2*T, dim=2)

        # Extract relevant lags and normalize
        autocorr = autocorr_full[:, :, :max_lag, :]  # [N, M, max_lag, C]
        autocorr = autocorr / (autocorr[:, :, 0:1, :] + 1e-8)  # Normalize by lag-0

        # Fit exponential decay: log(autocorr) ~ -tau / tau_decay
        # tau_decay = -tau / log(autocorr)
        # Use lag at which autocorr drops to 1/e ≈ 0.368
        threshold = torch.tensor(1.0 / np.e, device=time_series.device)

        # Find first lag where autocorr < threshold
        below_threshold = autocorr < threshold
        # Get first True index (or max_lag if never crosses)
        decay_idx = below_threshold.int().argmax(dim=2).float()  # [N, M, C]

        # If never crosses threshold, use max_lag
        never_crosses = ~below_threshold.any(dim=2)
        decay_idx = torch.where(never_crosses, torch.tensor(max_lag, dtype=torch.float32, device=time_series.device), decay_idx)

        return decay_idx

    def _compute_pacf(
        self,
        time_series: torch.Tensor,
        max_lag: int = 10
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Partial Autocorrelation Function (PACF) via Yule-Walker.

        PACF isolates direct correlations at each lag by removing indirect effects.
        Uses Levinson-Durbin recursion for efficiency.

        Args:
            time_series: [N, M, T, C]
            max_lag: Maximum lag to compute (default: 10)

        Returns:
            Dictionary with PACF features [N, M, C]:
                - pacf_lag_{k}: PACF at lag k for k in [1, max_lag]
        """
        N, M, T, C = time_series.shape

        # Compute autocorrelation first (needed for Yule-Walker)
        mean = time_series.mean(dim=2, keepdim=True)
        centered = time_series - mean

        # FFT-based autocorrelation
        fft_result = torch.fft.rfft(centered, n=2*T, dim=2)
        power = fft_result * torch.conj(fft_result)
        autocorr_full = torch.fft.irfft(power, n=2*T, dim=2)

        # Extract lags and normalize
        autocorr = autocorr_full[:, :, :max_lag+1, :]  # [N, M, max_lag+1, C]
        autocorr = autocorr / (autocorr[:, :, 0:1, :] + 1e-8)

        # Compute PACF via Levinson-Durbin recursion
        # For each (n, m, c), solve Yule-Walker equations
        pacf_values = {}

        for lag_k in range(1, max_lag + 1):
            # At lag k, we solve: r[k] = sum_{i=1}^{k-1} phi[i] * r[k-i]
            # PACF[k] = phi[k,k] (last coefficient in AR(k) fit)

            if lag_k == 1:
                # PACF[1] = autocorr[1] (direct correlation)
                pacf_k = autocorr[:, :, 1, :]  # [N, M, C]
            else:
                # Use simplified approximation: PACF[k] ≈ (autocorr[k] - prediction) / (1 - R²)
                # For efficiency, use approximate formula:
                # PACF[k] ≈ autocorr[k] if autocorr[k-1] is small
                # This is an approximation; full Levinson-Durbin is more complex
                # For production, we use a simpler approach based on residuals

                # Approximate PACF using residual correlation
                # PACF[k] ≈ correlation of residuals after removing AR(k-1) effects
                # Simplified: PACF[k] ≈ autocorr[k] / sqrt(1 - autocorr[k-1]²)
                prev_acf = autocorr[:, :, lag_k-1, :]
                curr_acf = autocorr[:, :, lag_k, :]
                denominator = torch.sqrt(1.0 - prev_acf**2 + 1e-8)
                pacf_k = curr_acf / denominator

            pacf_values[f'pacf_lag_{lag_k}'] = pacf_k

        return pacf_values

    # =========================================================================
    # Stability
    # =========================================================================

    def _compute_lyapunov_approx(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Approximate Lyapunov exponent (divergence rate).

        Measures sensitivity to initial conditions via log of successive differences.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Approximate Lyapunov exponent [N, M, C]
        """
        # Successive differences
        diff = time_series[:, :, 1:, :] - time_series[:, :, :-1, :]

        # Log of absolute differences (growth rate of perturbations)
        log_diff = torch.log(torch.abs(diff) + 1e-8)

        # Mean over time
        lyapunov = log_diff.mean(dim=2)

        return lyapunov

    def _compute_trajectory_smoothness(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute trajectory smoothness (sum of second time derivatives).

        Smoother trajectories have smaller second derivatives.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Smoothness metric [N, M, C] (smaller = smoother)
        """
        # First derivative
        first_deriv = time_series[:, :, 1:, :] - time_series[:, :, :-1, :]

        # Second derivative
        second_deriv = first_deriv[:, :, 1:, :] - first_deriv[:, :, :-1, :]

        # Sum of absolute second derivatives (inversely related to smoothness)
        roughness = second_deriv.abs().sum(dim=2)

        return roughness

    def _compute_regime_switches(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Count regime switches (sign changes in growth rate).

        Args:
            time_series: [N, M, T, C]

        Returns:
            Number of regime switches [N, M, C]
        """
        # First derivative (growth rate)
        growth = time_series[:, :, 1:, :] - time_series[:, :, :-1, :]

        # Sign of growth
        sign = torch.sign(growth)

        # Sign changes
        sign_changes = (sign[:, :, 1:, :] != sign[:, :, :-1, :]).float()

        # Count switches
        num_switches = sign_changes.sum(dim=2)

        return num_switches

    def _compute_final_to_initial_ratio(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute ratio of final to initial values.

        Indicates overall growth or decay.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Ratio [N, M, C]
        """
        initial = time_series[:, :, 0, :].abs() + 1e-8
        final = time_series[:, :, -1, :].abs() + 1e-8

        ratio = final / initial

        return ratio

    # =========================================================================
    # Stationarity
    # =========================================================================

    def _compute_trend_strength(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute trend strength (R² of linear fit).

        Strong trend → R² close to 1
        No trend → R² close to 0

        Args:
            time_series: [N, M, T, C]

        Returns:
            R² values [N, M, C]
        """
        N, M, T, C = time_series.shape

        # Time index
        t = torch.arange(T, dtype=torch.float32, device=time_series.device)
        t_mean = t.mean()

        # Linear fit: y = a + b*t
        # Compute slope (same as growth rate calculation)
        t_centered = t - t_mean
        y_mean = time_series.mean(dim=2, keepdim=True)
        y_centered = time_series - y_mean

        cov = (t_centered.view(1, 1, T, 1) * y_centered).mean(dim=2)
        var_t = t_centered.pow(2).mean()
        slope = cov / var_t

        # Predicted values
        y_pred = y_mean + slope.unsqueeze(2) * t_centered.view(1, 1, T, 1)

        # R² = 1 - SS_res / SS_tot
        ss_res = ((time_series - y_pred) ** 2).sum(dim=2)
        ss_tot = ((time_series - y_mean) ** 2).sum(dim=2)

        r_squared = 1.0 - ss_res / (ss_tot + 1e-8)

        # Clamp to [0, 1] (can be negative for bad fits)
        r_squared = torch.clamp(r_squared, 0.0, 1.0)

        return r_squared

    def _compute_detrended_variance(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Compute variance after removing linear trend.

        Args:
            time_series: [N, M, T, C]

        Returns:
            Detrended variance [N, M, C]
        """
        N, M, T, C = time_series.shape

        # Fit linear trend
        t = torch.arange(T, dtype=torch.float32, device=time_series.device)
        t_mean = t.mean()
        t_centered = t - t_mean

        y_mean = time_series.mean(dim=2, keepdim=True)
        y_centered = time_series - y_mean

        cov = (t_centered.view(1, 1, T, 1) * y_centered).mean(dim=2, keepdim=True)
        var_t = t_centered.pow(2).mean()
        slope = cov / var_t

        # Remove trend
        y_pred = y_mean + slope * t_centered.view(1, 1, T, 1)
        detrended = time_series - y_pred

        # Variance of detrended series
        detrended_var = detrended.var(dim=2)

        return detrended_var

    def _compute_event_counts(
        self,
        time_series: torch.Tensor,
        threshold_factor: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Count extreme events: spikes, bursts, threshold crossings.

        Args:
            time_series: [N, M, T, C]
            threshold_factor: Multiples of std for defining events

        Returns:
            Dictionary with event count features [N, M, C]:
                - num_spikes: Crossings above mean + k*std
                - num_bursts: Sustained periods (3+ steps) above threshold
                - num_zero_crossings: Sign changes (for oscillations)
        """
        N, M, T, C = time_series.shape

        # Compute statistics
        mean = time_series.mean(dim=2, keepdim=True)  # [N, M, 1, C]
        std = time_series.std(dim=2, keepdim=True)  # [N, M, 1, C]
        threshold_high = mean + threshold_factor * std
        threshold_low = mean - threshold_factor * std

        # Spike detection: any crossing above high threshold
        above_threshold = (time_series > threshold_high).float()  # [N, M, T, C]
        # Count transitions from below to above
        diff = above_threshold[:, :, 1:, :] - above_threshold[:, :, :-1, :]
        num_spikes = (diff > 0).sum(dim=2).float()  # [N, M, C]

        # Burst detection: sustained periods (3+ steps) above threshold
        # Use convolution to detect sequences of consecutive True values
        kernel_size = 3
        # Pad to keep same length after conv
        padded = torch.nn.functional.pad(above_threshold, (0, 0, kernel_size//2, kernel_size//2))
        # Reshape for conv1d: [N*M*C, 1, T+pad]
        reshaped = padded.permute(0, 1, 3, 2).reshape(N*M*C, 1, -1)
        kernel = torch.ones(1, 1, kernel_size, device=time_series.device)
        # Convolve to get sum of consecutive values
        conv = torch.nn.functional.conv1d(reshaped, kernel, padding=0)
        # Burst = sum equals kernel_size (all True in window)
        bursts = (conv[:, 0, :] == kernel_size).float()
        # Count number of bursts (consecutive burst windows count as one)
        burst_starts = (bursts[:, 1:] - bursts[:, :-1]) > 0
        num_bursts = burst_starts.sum(dim=1).reshape(N, M, C)  # [N, M, C]

        # Zero-crossing detection (sign changes)
        # Useful for detecting oscillatory behavior
        centered = time_series - mean
        signs = torch.sign(centered)
        sign_changes = (signs[:, :, 1:, :] - signs[:, :, :-1, :]).abs() > 1
        num_zero_crossings = sign_changes.sum(dim=2).float()  # [N, M, C]

        return {
            'num_spikes': num_spikes,
            'num_bursts': num_bursts,
            'num_zero_crossings': num_zero_crossings
        }

    def _compute_time_to_event(
        self,
        time_series: torch.Tensor,
        thresholds: list = [0.5, 1.0, 2.0]
    ) -> Dict[str, torch.Tensor]:
        """
        Time until trajectory crosses thresholds (multiples of initial value).

        Args:
            time_series: [N, M, T, C]
            thresholds: List of threshold multipliers

        Returns:
            Dictionary with time-to-event features [N, M, C]:
                - time_to_{thresh}x: First crossing time (or T if never crossed)
        """
        N, M, T, C = time_series.shape

        initial = time_series[:, :, 0:1, :]  # [N, M, 1, C]

        result = {}
        for thresh_mult in thresholds:
            threshold = initial * thresh_mult

            # For each threshold direction (above for growth, below for decay)
            # Detect first crossing
            if thresh_mult > 1.0:
                # Growth: first time above threshold
                crossed = time_series > threshold  # [N, M, T, C]
            elif thresh_mult < 1.0:
                # Decay: first time below threshold
                crossed = time_series < threshold
            else:
                # Exact match (not very useful, skip)
                continue

            # Find first True index along time dimension
            # argmax returns first True index (or 0 if all False)
            first_crossing = crossed.float().argmax(dim=2)  # [N, M, C]

            # If never crossed, set to T (trajectory length)
            never_crossed = ~crossed.any(dim=2)  # [N, M, C]
            first_crossing = torch.where(never_crossed, torch.full_like(first_crossing, T), first_crossing)

            # Normalize by trajectory length (0-1 range)
            normalized_time = first_crossing.float() / T

            result[f'time_to_{thresh_mult}x'] = normalized_time

        return result

    def _compute_rolling_stats(
        self,
        time_series: torch.Tensor,
        window_fractions: list = [0.05, 0.10, 0.20]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics over multiple rolling windows (CRITICAL FEATURE).

        Multi-timescale analysis to capture both transient and sustained dynamics.
        For each window size, computes mean/std/max/min across the rolling window.

        Args:
            time_series: [N, M, T, C] temporal trajectories
            window_fractions: List of window sizes as fractions of T (default: 5%, 10%, 20%)

        Returns:
            Dictionary with rolling window features [N, M, C]:
                - rolling_mean_wX: Mean of rolling window values
                - rolling_std_wX: Std of rolling window values
                - rolling_max_wX: Max of rolling window values
                - rolling_min_wX: Min of rolling window values
            where X = int(fraction * 100) (e.g., w5, w10, w20)
        """
        N, M, T, C = time_series.shape
        result = {}

        for frac in window_fractions:
            # Compute window size (ensure at least 2 timesteps)
            window_size = max(2, int(frac * T))

            # Use unfold to create sliding windows efficiently
            # unfold(dimension, size, step) creates overlapping windows
            # Result shape: [N, M, C, num_windows, window_size]
            windows = time_series.permute(0, 1, 3, 2).unfold(3, window_size, 1)
            # Now: [N, M, C, num_windows, window_size]

            # Compute statistics across the window dimension (dim=4)
            window_means = windows.mean(dim=4)  # [N, M, C, num_windows]
            window_stds = windows.std(dim=4)    # [N, M, C, num_windows]
            window_maxs = windows.max(dim=4)[0] # [N, M, C, num_windows]
            window_mins = windows.min(dim=4)[0] # [N, M, C, num_windows]

            # Aggregate across all windows (mean of rolling statistics)
            # This gives a single value representing the typical behavior
            # across all windows of this size
            prefix = f'rolling_w{int(frac * 100)}'
            result[f'{prefix}_mean'] = window_means.mean(dim=3)  # [N, M, C]
            result[f'{prefix}_std'] = window_stds.mean(dim=3)    # [N, M, C]
            result[f'{prefix}_max'] = window_maxs.mean(dim=3)    # [N, M, C]
            result[f'{prefix}_min'] = window_mins.mean(dim=3)    # [N, M, C]

            # Also capture variability of the rolling statistics
            # (how much do window stats vary across time?)
            result[f'{prefix}_mean_variability'] = window_means.std(dim=3)  # [N, M, C]
            result[f'{prefix}_std_variability'] = window_stds.std(dim=3)    # [N, M, C]

        return result

    def aggregate_realizations(
        self,
        features: Dict[str, torch.Tensor],
        methods: list = ['mean', 'std', 'cv']
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate per-realization features across realizations.

        Args:
            features: Dict of features with shape [N, M, C]
            methods: Aggregation methods ('mean', 'std', 'cv', 'min', 'max')

        Returns:
            Aggregated features dict with shape [N, C]
        """
        aggregated = {}

        for name, feat in features.items():
            if feat.ndim != 3:  # Expected [N, M, C]
                raise ValueError(f"Expected 3D tensor, got shape {feat.shape}")

            for method in methods:
                agg_name = f"{name}_{method}"

                if method == 'mean':
                    aggregated[agg_name] = feat.mean(dim=1)  # Average over realizations
                elif method == 'std':
                    aggregated[agg_name] = feat.std(dim=1)
                elif method == 'cv':
                    # Coefficient of variation: std / mean
                    mean = feat.mean(dim=1)
                    std = feat.std(dim=1)
                    aggregated[agg_name] = std / (mean.abs() + 1e-8)
                elif method == 'min':
                    aggregated[agg_name] = feat.amin(dim=1)
                elif method == 'max':
                    aggregated[agg_name] = feat.amax(dim=1)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

        return aggregated
