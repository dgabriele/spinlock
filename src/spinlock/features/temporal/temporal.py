"""Enhanced Temporal Feature Extractor (v3.1).

Extracts per-timestep temporal features using windowed history.
All features are online-computable from current state + fixed-size buffer.

Version 3.1 additions:
- Temporal skewness/kurtosis (captures asymmetry and tail behavior)
- Multi-lag autocorrelation (temporal memory at lags 2,5,10,20)
- Sample entropy proxy (regularity/predictability measure)
- Enstrophy (vorticity/turbulence indicator)
- Divergence proxy (field expansion/contraction)

Use get_feature_dimensions() to query actual dimension counts dynamically.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Tuple
import numpy as np

class TemporalFeatureExtractor:
    """Extract enhanced temporal dynamics features per timestep.

    Five sub-components:
    1. Instantaneous Dynamics - Derivatives, rates, enstrophy, divergence
    2. Local Temporal - Statistics, skewness, kurtosis, multi-lag autocorr
    3. Local Stability - Stability metrics, sample entropy
    4. Phase Space Geometry - Trajectory geometry
    5. Multi-scale Temporal - Multi-window aggregations

    All features use windowed history for online computation.

    Args:
        device: Torch device
        window_size: Primary window size (default: 5)
        short_window: Short-term window (default: 5)
        medium_window: Medium-term window (default: 20)
        long_window: Long-term window (default: 50)
    """

    def __init__(
        self,
        device: torch.device,
        window_size: int = 5,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
    ):
        self.device = device
        self.window_size = window_size
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

        # History buffers
        self.history_buffer = deque(maxlen=long_window)
        self.derivative_buffer = deque(maxlen=medium_window)

    def reset(self):
        """Reset history buffers."""
        self.history_buffer.clear()
        self.derivative_buffer.clear()

    def get_feature_dimensions(self) -> dict:
        """Get actual feature dimensions for each component.

        Returns:
            Dictionary mapping component names to their dimensions
        """
        # Reset to clear any existing history with different batch size
        self.reset()

        # Extract features from a dummy input to get actual dimensions
        dummy = torch.zeros(1, 3, 64, 64, device=self.device)

        # Feed enough history
        for _ in range(max(self.long_window, 10)):
            _ = self.extract(dummy)

        # Now extract and measure
        inst = self._extract_instantaneous(dummy)
        local = self._extract_local_temporal(dummy)
        stab = self._extract_local_stability(dummy)
        phase = self._extract_phase_space(dummy)
        multi = self._extract_multiscale(dummy)

        # Reset again to clear dummy data
        self.reset()

        return {
            'instantaneous': inst.shape[1],
            'local_temporal': local.shape[1],
            'local_stability': stab.shape[1],
            'phase_space': phase.shape[1],
            'multiscale': multi.shape[1],
            'total': sum([inst.shape[1], local.shape[1], stab.shape[1],
                         phase.shape[1], multi.shape[1]])
        }

    def extract(self, u: torch.Tensor) -> torch.Tensor:
        """Extract enhanced temporal features.

        Args:
            u: State tensor [B, C, H, W]

        Returns:
            features: [B, D] temporal features where D depends on configuration
        """
        # Update history
        self.history_buffer.append(u.detach().cpu())

        # Extract each component
        inst = self._extract_instantaneous(u)
        local = self._extract_local_temporal(u)
        stab = self._extract_local_stability(u)
        phase = self._extract_phase_space(u)
        multi = self._extract_multiscale(u)

        # Defensive dimension checks
        features_list = [inst, local, stab, phase, multi]
        features_list = [
            f.flatten(start_dim=1) if f.dim() > 2 else f
            for f in features_list
        ]

        return torch.cat(features_list, dim=-1)

    def _extract_instantaneous(self, u: torch.Tensor) -> torch.Tensor:
        """Extract instantaneous dynamics features.

        Features:
        - Time derivatives (first/second order)
        - Rate of change metrics
        - Energy flux and momentum
        - Enstrophy (vorticity proxy)
        - Divergence proxy

        Args:
            u: [B, C, H, W]

        Returns:
            [B, D] instantaneous features
        """
        B, C, H, W = u.shape
        features = []

        # Compute spatial gradients
        u_x = torch.diff(u, dim=3)  # [B, C, H, W-1]
        u_y = torch.diff(u, dim=2)  # [B, C, H-1, W]

        # 1. Gradient magnitude per channel (3D)
        grad_mag_x = torch.norm(u_x, p=2, dim=(2, 3))  # [B, C]
        grad_mag_y = torch.norm(u_y, p=2, dim=(2, 3))  # [B, C]
        features.append(grad_mag_x)
        features.append(grad_mag_y)

        # 2. Mean gradient magnitude (1D)
        mean_grad = (grad_mag_x.mean(dim=1, keepdim=True) +
                    grad_mag_y.mean(dim=1, keepdim=True)) / 2
        features.append(mean_grad)

        # 3. Temporal derivative (if history available) (C*3 = 9D)
        if len(self.history_buffer) >= 2:
            u_prev = self.history_buffer[-2].to(u.device)
            dt_u = u - u_prev

            # L2 norm per channel
            dt_norm = torch.norm(dt_u, p=2, dim=(2, 3))  # [B, C]
            features.append(dt_norm)

            # Mean and std across channels
            features.append(dt_norm.mean(dim=1, keepdim=True))
            features.append(dt_norm.std(dim=1, keepdim=True))

            # Store for second derivative
            self.derivative_buffer.append(dt_u.detach().cpu())

            # 4. Second derivative (acceleration) (C = 3D)
            if len(self.derivative_buffer) >= 2:
                dt_u_prev = self.derivative_buffer[-2].to(u.device)
                dtt_u = dt_u - dt_u_prev
                dtt_norm = torch.norm(dtt_u, p=2, dim=(2, 3))  # [B, C]
                features.append(dtt_norm)
            else:
                features.append(torch.zeros(B, C, device=u.device))
        else:
            # No history - use zeros
            features.append(torch.zeros(B, C, device=u.device))  # dt_norm
            features.append(torch.zeros(B, 1, device=u.device))  # dt_mean
            features.append(torch.zeros(B, 1, device=u.device))  # dt_std
            features.append(torch.zeros(B, C, device=u.device))  # dtt_norm

        # 5. Energy (L2 norm) (1D)
        energy = torch.norm(u, p=2, dim=(1, 2, 3), keepdim=True)
        features.append(energy)

        # 6. Energy per channel (C = 3D)
        energy_per_ch = torch.norm(u, p=2, dim=(2, 3))  # [B, C]
        features.append(energy_per_ch)

        # 7. Enstrophy (vorticity proxy) (NEW: 1D)
        # For scalar fields, use gradient magnitude squared as proxy
        # Enstrophy measures the intensity of small-scale vorticity/turbulence
        vorticity_proxy = (u_x ** 2).mean(dim=(2, 3)) + (u_y ** 2).mean(dim=(2, 3))  # [B, C]
        enstrophy = vorticity_proxy.mean(dim=1, keepdim=True)  # [B, 1]
        features.append(enstrophy)

        # 8. Divergence proxy (NEW: 1D)
        # For 2D fields, approximate divergence as sum of spatial gradients
        # Measures expansion/contraction of the field
        divergence = (u_x.abs().mean(dim=(2, 3)) + u_y.abs().mean(dim=(2, 3))).mean(dim=1, keepdim=True)  # [B, 1]
        features.append(divergence)

        # Flatten and concatenate
        features = [f.reshape(B, -1) for f in features]
        result = torch.cat(features, dim=-1)  # Should be ~24D

        return result

    def _extract_local_temporal(self, u: torch.Tensor) -> torch.Tensor:
        """Extract local temporal statistics.

        Features:
        - Short-window mean/std/variance
        - Min/max/range values
        - Temporal skewness/kurtosis
        - Autocorrelation (single and multi-lag)
        - Trend estimation

        Args:
            u: [B, C, H, W]

        Returns:
            [B, D] local temporal features
        """
        B, C, H, W = u.shape
        features = []

        # Get window history
        window = min(self.short_window, len(self.history_buffer))

        if window >= 2:
            # Stack history: [window, B, C, H, W]
            history = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, window + 1)
            ], dim=0)

            # 1. Temporal mean per channel (C = 3D)
            temp_mean = history.mean(dim=0)  # [B, C, H, W]
            temp_mean_norm = torch.norm(temp_mean, p=2, dim=(2, 3))  # [B, C]
            features.append(temp_mean_norm)

            # 2. Temporal std per channel (C = 3D)
            temp_std = history.std(dim=0)  # [B, C, H, W]
            temp_std_norm = torch.norm(temp_std, p=2, dim=(2, 3))  # [B, C]
            features.append(temp_std_norm)

            # 3. Temporal variance (spatial L2) (C = 3D)
            temp_var = history.var(dim=0)  # [B, C, H, W]
            temp_var_norm = torch.norm(temp_var, p=2, dim=(2, 3))  # [B, C]
            features.append(temp_var_norm)

            # 3b. Temporal skewness per channel (NEW: C = 3D)
            # Skewness = E[(X - μ)³] / σ³
            eps = 1e-8
            centered = history - temp_mean.unsqueeze(0)  # [T, B, C, H, W]
            temp_std_full = history.std(dim=0)  # [B, C, H, W]
            temp_skew = ((centered ** 3).mean(dim=0)) / (temp_std_full ** 3 + eps)  # [B, C, H, W]
            temp_skew_norm = torch.norm(temp_skew, p=2, dim=(2, 3))  # [B, C]
            features.append(temp_skew_norm)

            # 3c. Temporal kurtosis per channel (NEW: C = 3D)
            # Kurtosis = E[(X - μ)⁴] / σ⁴
            temp_kurt = ((centered ** 4).mean(dim=0)) / (temp_std_full ** 4 + eps)  # [B, C, H, W]
            temp_kurt_norm = torch.norm(temp_kurt, p=2, dim=(2, 3))  # [B, C]
            features.append(temp_kurt_norm)

            # 4. Min/max across time (C*2 = 6D)
            temp_min, _ = torch.min(torch.norm(history, p=2, dim=(3, 4)), dim=0)  # [B, C]
            temp_max, _ = torch.max(torch.norm(history, p=2, dim=(3, 4)), dim=0)  # [B, C]
            features.append(temp_min)
            features.append(temp_max)

            # 5. Range (max - min) (C = 3D)
            temp_range = temp_max - temp_min
            features.append(temp_range)

            # 6. Mean absolute deviation (C = 3D)
            mad = torch.mean(torch.abs(history - temp_mean.unsqueeze(0)), dim=0)
            mad_norm = torch.norm(mad, p=2, dim=(2, 3))  # [B, C]
            features.append(mad_norm)

            # 7. Autocorrelation lag-1 (C = 3D)
            if window >= 3:
                autocorr = self._compute_autocorr(history)  # [B, C]
                features.append(autocorr)
            else:
                features.append(torch.zeros(B, C, device=u.device))

            # 7b. Multi-lag autocorrelation (NEW: 5D)
            # Compute autocorrelations at lags 2, 5, 10, 20, and their mean
            lags = [2, 5, 10, 20]
            for lag in lags:
                if window > lag:
                    autocorr_lag = self._compute_autocorr(history, lag=lag)  # [B, C]
                    # Take mean across channels for global measure
                    features.append(autocorr_lag.mean(dim=1, keepdim=True))  # [B, 1]
                else:
                    features.append(torch.zeros(B, 1, device=u.device))

            # Add mean of all multi-lag autocorrelations
            if window > 2:
                all_lags = [self._compute_autocorr(history, lag=lag).mean(dim=1, keepdim=True)
                           for lag in [1, 2, 5, 10, 20] if window > lag]
                if all_lags:
                    mean_autocorr = torch.stack(all_lags, dim=0).mean(dim=0)  # [B, 1]
                    features.append(mean_autocorr)
                else:
                    features.append(torch.zeros(B, 1, device=u.device))
            else:
                features.append(torch.zeros(B, 1, device=u.device))

            # 8. Trend (linear fit slope) (C = 3D)
            trend = self._compute_trend(history)  # [B, C]
            features.append(trend)

        else:
            # Not enough history - use zeros (39D total)
            features.append(torch.zeros(B, C, device=u.device))  # mean
            features.append(torch.zeros(B, C, device=u.device))  # std
            features.append(torch.zeros(B, C, device=u.device))  # var
            features.append(torch.zeros(B, C, device=u.device))  # skew (NEW)
            features.append(torch.zeros(B, C, device=u.device))  # kurt (NEW)
            features.append(torch.zeros(B, C, device=u.device))  # min
            features.append(torch.zeros(B, C, device=u.device))  # max
            features.append(torch.zeros(B, C, device=u.device))  # range
            features.append(torch.zeros(B, C, device=u.device))  # mad
            features.append(torch.zeros(B, C, device=u.device))  # autocorr
            # Multi-lag autocorrelations (5D: lags 2, 5, 10, 20, mean)
            for _ in range(5):
                features.append(torch.zeros(B, 1, device=u.device))
            features.append(torch.zeros(B, C, device=u.device))  # trend

        # Flatten and concatenate
        features = [f.reshape(B, -1) for f in features]
        result = torch.cat(features, dim=-1)  # Should be ~28D

        return result

    def _extract_local_stability(self, u: torch.Tensor) -> torch.Tensor:
        """Extract local stability metrics.

        Features:
        - Lyapunov-like exponents
        - Divergence and contraction rates
        - Perturbation sensitivity
        - Sample entropy proxy
        - Recurrence and entropy proxies

        Args:
            u: [B, C, H, W]

        Returns:
            [B, D] stability features
        """
        B, C, H, W = u.shape
        features = []

        window = min(self.medium_window, len(self.history_buffer))

        if window >= 3:
            history = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, window + 1)
            ], dim=0)

            # 1. Local Lyapunov exponent approximation (C = 3D)
            # Compute log(|u(t) - u(t-1)| / |u(t-1) - u(t-2)|)
            diff_curr = torch.norm(history[0] - history[1], p=2, dim=(2, 3))  # [B, C]
            diff_prev = torch.norm(history[1] - history[2], p=2, dim=(2, 3))  # [B, C]
            lyap = torch.log((diff_curr + 1e-8) / (diff_prev + 1e-8))
            features.append(lyap)

            # 2. Divergence rate (C = 3D)
            # Rate of change of distances
            divergence = (diff_curr - diff_prev) / (diff_prev + 1e-8)
            features.append(divergence)

            # 3. Stability indicator (variance of differences) (C = 3D)
            diffs = torch.stack([
                torch.norm(history[i] - history[i+1], p=2, dim=(2, 3))
                for i in range(window - 1)
            ], dim=0)  # [window-1, B, C]
            stability = diffs.var(dim=0)  # [B, C]
            features.append(stability)

            # 4. Recurrence (how often state returns to neighborhood) (C = 3D)
            recurrence = self._compute_recurrence(history)  # [B, C]
            features.append(recurrence)

            # 5. Entropy proxy (distribution of values) (C = 3D)
            entropy = self._compute_entropy(history)  # [B, C]
            features.append(entropy)

            # 6. Contraction/expansion (C = 3D)
            # Compare distances in first vs last half of window
            half = window // 2
            dist_early = torch.stack([
                torch.norm(history[i] - history[i+1], p=2, dim=(2, 3))
                for i in range(half)
            ], dim=0).mean(dim=0)  # [B, C]
            dist_late = torch.stack([
                torch.norm(history[i] - history[i+1], p=2, dim=(2, 3))
                for i in range(half, window - 1)
            ], dim=0).mean(dim=0)  # [B, C]
            expansion = (dist_late - dist_early) / (dist_early + 1e-8)
            features.append(expansion)

            # 7. Maximum Lyapunov (C = 3D)
            max_lyap = lyap.max(dim=1, keepdim=True)[0].expand(-1, C)
            features.append(max_lyap)

            # 8. Stability margin (C = 3D)
            margin = torch.abs(lyap)
            features.append(margin)

            # 9. Sample entropy proxy (NEW: 1D)
            # Simplified: standard deviation of consecutive differences
            # Measures regularity/predictability of the time series
            consecutive_diffs = torch.stack([
                torch.norm(history[i] - history[i+1], p=2, dim=(2, 3))
                for i in range(window - 1)
            ], dim=0)  # [window-1, B, C]
            sample_entropy = consecutive_diffs.std(dim=0).mean(dim=1, keepdim=True)  # [B, 1]
            features.append(sample_entropy)

        else:
            # Not enough history - 25D zeros
            for _ in range(8):
                features.append(torch.zeros(B, C, device=u.device))
            features.append(torch.zeros(B, 1, device=u.device))  # sample entropy

        # Flatten and concatenate
        features = [f.reshape(B, -1) for f in features]
        result = torch.cat(features, dim=-1)  # Should be ~24D

        return result

    def _extract_phase_space(self, u: torch.Tensor) -> torch.Tensor:
        """Extract phase space geometry features (26D).

        Features:
        - Trajectory curvature
        - Phase space volume
        - Attractor dimensions

        Args:
            u: [B, C, H, W]

        Returns:
            [B, 26] phase space features
        """
        B, C, H, W = u.shape
        features = []

        window = min(self.medium_window, len(self.history_buffer))

        if window >= 5:
            history = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, window + 1)
            ], dim=0)

            # 1. Trajectory curvature (C = 3D)
            curvature = self._compute_curvature(history)  # [B, C]
            features.append(curvature)

            # 2. Phase space volume (correlation dimension proxy) (C = 3D)
            volume = self._compute_phase_volume(history)  # [B, C]
            features.append(volume)

            # 3. Trajectory length (C = 3D)
            length = torch.stack([
                torch.norm(history[i] - history[i+1], p=2, dim=(2, 3))
                for i in range(window - 1)
            ], dim=0).sum(dim=0)  # [B, C]
            features.append(length)

            # 4. Straightness (end-to-end / path length) (C = 3D)
            end_to_end = torch.norm(history[0] - history[-1], p=2, dim=(2, 3))  # [B, C]
            straightness = end_to_end / (length + 1e-8)
            features.append(straightness)

            # 5. Tortuosity (path length / end-to-end) (C = 3D)
            tortuosity = length / (end_to_end + 1e-8)
            features.append(tortuosity)

            # 6. Direction changes (C = 3D)
            direction_changes = self._compute_direction_changes(history)  # [B, C]
            features.append(direction_changes)

            # 7. Mean curvature (1D)
            mean_curv = curvature.mean(dim=1, keepdim=True)
            features.append(mean_curv)

            # 8. Max curvature (1D)
            max_curv = curvature.max(dim=1, keepdim=True)[0]
            features.append(max_curv)

            # 9. Phase space diameter (max pairwise distance) (C = 3D)
            diameter = self._compute_diameter(history)  # [B, C]
            features.append(diameter)

        else:
            # Not enough history - 26D zeros
            features.append(torch.zeros(B, C, device=u.device))  # curvature
            features.append(torch.zeros(B, C, device=u.device))  # volume
            features.append(torch.zeros(B, C, device=u.device))  # length
            features.append(torch.zeros(B, C, device=u.device))  # straightness
            features.append(torch.zeros(B, C, device=u.device))  # tortuosity
            features.append(torch.zeros(B, C, device=u.device))  # direction changes
            features.append(torch.zeros(B, 1, device=u.device))  # mean curvature
            features.append(torch.zeros(B, 1, device=u.device))  # max curvature
            features.append(torch.zeros(B, C, device=u.device))  # diameter

        # Flatten and concatenate
        features = [f.reshape(B, -1) for f in features]
        result = torch.cat(features, dim=-1)  # Should be ~26D

        return result

    def _extract_multiscale(self, u: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale temporal features (30D).

        Features across multiple time windows:
        - Short-term (5 steps)
        - Medium-term (20 steps)
        - Long-term (50 steps)

        Args:
            u: [B, C, H, W]

        Returns:
            [B, 30] multi-scale features
        """
        B, C, H, W = u.shape
        features = []

        # Short-term window
        short_window = min(self.short_window, len(self.history_buffer))
        if short_window >= 2:
            short_hist = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, short_window + 1)
            ], dim=0)
            short_mean = torch.norm(short_hist.mean(dim=0), p=2, dim=(2, 3))  # [B, C]
            short_std = torch.norm(short_hist.std(dim=0), p=2, dim=(2, 3))  # [B, C]
            features.append(short_mean)
            features.append(short_std)
        else:
            features.append(torch.zeros(B, C, device=u.device))
            features.append(torch.zeros(B, C, device=u.device))

        # Medium-term window
        medium_window = min(self.medium_window, len(self.history_buffer))
        if medium_window >= 2:
            medium_hist = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, medium_window + 1)
            ], dim=0)
            medium_mean = torch.norm(medium_hist.mean(dim=0), p=2, dim=(2, 3))  # [B, C]
            medium_std = torch.norm(medium_hist.std(dim=0), p=2, dim=(2, 3))  # [B, C]
            features.append(medium_mean)
            features.append(medium_std)
        else:
            features.append(torch.zeros(B, C, device=u.device))
            features.append(torch.zeros(B, C, device=u.device))

        # Long-term window
        long_window = min(self.long_window, len(self.history_buffer))
        if long_window >= 2:
            long_hist = torch.stack([
                self.history_buffer[-i].to(u.device)
                for i in range(1, long_window + 1)
            ], dim=0)
            long_mean = torch.norm(long_hist.mean(dim=0), p=2, dim=(2, 3))  # [B, C]
            long_std = torch.norm(long_hist.std(dim=0), p=2, dim=(2, 3))  # [B, C]
            features.append(long_mean)
            features.append(long_std)
        else:
            features.append(torch.zeros(B, C, device=u.device))
            features.append(torch.zeros(B, C, device=u.device))

        # Cross-scale ratios (C*3 = 9D)
        if short_window >= 2 and medium_window >= 2:
            short_to_medium = short_mean / (medium_mean + 1e-8)
            features.append(short_to_medium)
        else:
            features.append(torch.zeros(B, C, device=u.device))

        if medium_window >= 2 and long_window >= 2:
            medium_to_long = medium_mean / (long_mean + 1e-8)
            features.append(medium_to_long)
        else:
            features.append(torch.zeros(B, C, device=u.device))

        if short_window >= 2 and long_window >= 2:
            short_to_long = short_mean / (long_mean + 1e-8)
            features.append(short_to_long)
        else:
            features.append(torch.zeros(B, C, device=u.device))

        # Flatten and concatenate
        features = [f.reshape(B, -1) for f in features]
        result = torch.cat(features, dim=-1)  # Should be ~30D

        return result

    # Helper methods

    def _compute_autocorr(self, history: torch.Tensor, lag: int = 1) -> torch.Tensor:
        """Compute autocorrelation at specified lag.

        Args:
            history: [T, B, C, H, W]
            lag: Time lag for autocorrelation (default: 1)

        Returns:
            [B, C] autocorrelation coefficients
        """
        T, B, C, H, W = history.shape
        if T < lag + 2:
            return torch.zeros(B, C, device=history.device)

        # Flatten spatial dimensions
        hist_flat = history.reshape(T, B, C, -1)  # [T, B, C, H*W]

        # Compute mean
        mean = hist_flat.mean(dim=0, keepdim=True)  # [1, B, C, H*W]

        # Center
        centered = hist_flat - mean  # [T, B, C, H*W]

        # Compute lag-k correlation
        corr = (centered[:-lag] * centered[lag:]).sum(dim=(0, 3))  # [B, C]
        var = (centered ** 2).sum(dim=(0, 3))  # [B, C]

        autocorr = corr / (var + 1e-8)

        return autocorr

    def _compute_trend(self, history: torch.Tensor) -> torch.Tensor:
        """Compute linear trend (slope).

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] trend coefficients
        """
        T, B, C, H, W = history.shape

        # Compute L2 norm over time
        norms = torch.norm(history, p=2, dim=(3, 4))  # [T, B, C]

        # Linear regression: y = ax + b
        t = torch.arange(T, dtype=torch.float32, device=history.device)
        t_mean = t.mean()

        # Slope: cov(t, y) / var(t)
        y_mean = norms.mean(dim=0, keepdim=True)  # [1, B, C]
        cov = ((t.unsqueeze(1).unsqueeze(2) - t_mean) * (norms - y_mean)).sum(dim=0)  # [B, C]
        var_t = ((t - t_mean) ** 2).sum()

        slope = cov / (var_t + 1e-8)

        return slope

    def _compute_recurrence(self, history: torch.Tensor) -> torch.Tensor:
        """Compute recurrence metric.

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] recurrence scores
        """
        T, B, C, H, W = history.shape

        # Compute pairwise distances (sample subset for efficiency)
        subset_size = min(10, T)
        indices = torch.linspace(0, T-1, subset_size, dtype=torch.long)
        subset = history[indices]  # [subset_size, B, C, H, W]

        # Pairwise distances
        dists = torch.zeros(subset_size, subset_size, B, C, device=history.device)
        for i in range(subset_size):
            for j in range(i+1, subset_size):
                dist = torch.norm(subset[i] - subset[j], p=2, dim=(2, 3))  # [B, C]
                dists[i, j] = dist
                dists[j, i] = dist

        # Recurrence: fraction of distances below threshold (median)
        threshold = dists.median()
        recurrence = (dists < threshold).float().mean(dim=(0, 1))  # [B, C]

        return recurrence

    def _compute_entropy(self, history: torch.Tensor) -> torch.Tensor:
        """Compute entropy proxy (variance of histogram).

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] entropy estimates
        """
        T, B, C, H, W = history.shape

        # Flatten and compute histogram variance per channel
        hist_flat = history.reshape(T, B, C, -1)  # [T, B, C, H*W]

        # Compute variance across all elements (proxy for entropy)
        entropy = hist_flat.var(dim=(0, 3))  # [B, C]

        return entropy

    def _compute_curvature(self, history: torch.Tensor) -> torch.Tensor:
        """Compute trajectory curvature.

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] mean curvature
        """
        T, B, C, H, W = history.shape

        if T < 3:
            return torch.zeros(B, C, device=history.device)

        # Compute second derivative
        first_diff = history[:-1] - history[1:]  # [T-1, B, C, H, W]
        second_diff = first_diff[:-1] - first_diff[1:]  # [T-2, B, C, H, W]

        # Curvature = |d²x/dt²|
        curvature = torch.norm(second_diff, p=2, dim=(3, 4))  # [T-2, B, C]

        # Mean curvature
        mean_curvature = curvature.mean(dim=0)  # [B, C]

        return mean_curvature

    def _compute_phase_volume(self, history: torch.Tensor) -> torch.Tensor:
        """Compute phase space volume (correlation dimension proxy).

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] volume estimates
        """
        T, B, C, H, W = history.shape

        # Use variance as proxy for volume
        volume = history.var(dim=0)  # [B, C, H, W]
        volume_norm = torch.norm(volume, p=2, dim=(2, 3))  # [B, C]

        return volume_norm

    def _compute_direction_changes(self, history: torch.Tensor) -> torch.Tensor:
        """Compute number of direction changes in trajectory.

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] direction change counts
        """
        T, B, C, H, W = history.shape

        if T < 3:
            return torch.zeros(B, C, device=history.device)

        # Compute velocity vectors
        velocities = history[:-1] - history[1:]  # [T-1, B, C, H, W]

        # Dot product of consecutive velocities
        dots = (velocities[:-1] * velocities[1:]).sum(dim=(3, 4))  # [T-2, B, C]

        # Direction changes when dot product is negative
        changes = (dots < 0).float().sum(dim=0)  # [B, C]

        return changes

    def _compute_diameter(self, history: torch.Tensor) -> torch.Tensor:
        """Compute phase space diameter (max pairwise distance).

        Args:
            history: [T, B, C, H, W]

        Returns:
            [B, C] diameter
        """
        T, B, C, H, W = history.shape

        # Sample subset for efficiency
        subset_size = min(10, T)
        indices = torch.linspace(0, T-1, subset_size, dtype=torch.long)
        subset = history[indices]  # [subset_size, B, C, H, W]

        # Compute max pairwise distance
        max_dist = torch.zeros(B, C, device=history.device)
        for i in range(subset_size):
            for j in range(i+1, subset_size):
                dist = torch.norm(subset[i] - subset[j], p=2, dim=(2, 3))  # [B, C]
                max_dist = torch.max(max_dist, dist)

        return max_dist
