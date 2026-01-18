"""
Temporal feature extraction (per-timestep).

Extracts rich temporal dynamics features at each timestep for online operation.
Replaces old trajectory-level temporal features with per-timestep equivalents:
- Instantaneous dynamics (22D): energy, dissipation, spectral, structure, stats
- Local temporal (28D): autocorr, trends, windowed stats, oscillations, growth
- Local stability (24D): Lipschitz, stability, divergence, regularity
- Phase space geometry (26D): flow, vorticity, strain, topology, manifold
- Multi-scale temporal (30D): hierarchical averaging, cross-scale, persistence

Total: 130D per-timestep features (vs old trajectory-level features).

Example:
    >>> from spinlock.features.temporal.temporal import TemporalFeatureExtractor
    >>> extractor = TemporalFeatureExtractor(device='cuda')
    >>> state = torch.randn(4, 2, 64, 64, device='cuda')  # [B, C, H, W]
    >>> features = extractor.extract(state)  # [B, 130]
"""

import torch
from torch import Tensor
from collections import deque
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.temporal.config import TemporalConfig


class TemporalFeatureExtractor:
    """Extract per-timestep temporal dynamics features.

    Computes 130D features from current state and short history:
    - Instantaneous: Energy, dissipation, spectral dynamics
    - Local temporal: Autocorr, trends, windowed stats
    - Local stability: Lipschitz, Jacobian-free metrics
    - Phase space: Flow, vorticity, strain, topology
    - Multi-scale: Hierarchical temporal averaging

    Maintains circular buffers for windowed features. Call reset() at episode boundaries.

    Args:
        device: Computation device
        window_size: History window for local temporal features (default: 5)
        short_window: Short-term averaging window (default: 5)
        medium_window: Medium-term averaging window (default: 20)
        long_window: Long-term averaging window (default: 50)

    Example:
        >>> extractor = TemporalFeatureExtractor(device='cuda', window_size=5)
        >>> for t in range(num_timesteps):
        ...     state = mno_step(...)  # [B, C, H, W]
        ...     features = extractor.extract(state)  # [B, 130]
        >>> extractor.reset()  # Clear buffers for next episode
    """

    def __init__(
        self,
        device: torch.device = torch.device('cuda'),
        window_size: int = 5,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50
    ):
        self.device = device
        self.window_size = window_size
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window

        # Circular buffers for windowed features
        self.history_buffer = deque(maxlen=window_size)
        self.short_buffer = deque(maxlen=short_window)
        self.medium_buffer = deque(maxlen=medium_window)
        self.long_buffer = deque(maxlen=long_window)

    def extract(self, u: Tensor) -> Tensor:
        """Extract all 130D temporal features from current state.

        Args:
            u: Current state [B, C, H, W]

        Returns:
            features: [B, 130] concatenated temporal features
        """
        # Extract each category
        instantaneous = self._extract_instantaneous(u)  # [B, 22]
        local_temporal = self._extract_local_temporal(u)  # [B, 28]
        local_stability = self._extract_local_stability(u)  # [B, 24]
        phase_space = self._extract_phase_space(u)  # [B, 26]
        multiscale = self._extract_multiscale(u)  # [B, 30]

        # Concatenate all features
        features = torch.cat([
            instantaneous,
            local_temporal,
            local_stability,
            phase_space,
            multiscale
        ], dim=-1)

        return features  # [B, 130]

    def _extract_instantaneous(self, u: Tensor) -> Tensor:
        """Extract instantaneous dynamics features (22D).

        Features computed from single snapshot without history:
        - Energy metrics (4D): kinetic, enstrophy, potential, total
        - Dissipation rates (3D): viscous, reaction, total
        - Spectral dynamics (5D): centroid, bandwidth, rolloff, high-freq, entropy
        - Structure metrics (5D): gradient norms, Laplacian stats, divergence
        - Statistical moments (5D): skewness, kurtosis, entropy, Gini, IQR

        Args:
            u: State [B, C, H, W]

        Returns:
            features: [B, 22]
        """
        B, C, H, W = u.shape
        features_list = []

        # Energy metrics (4D)
        kinetic_energy = (u ** 2).mean(dim=(-2, -1))  # [B, C]
        enstrophy = self._compute_gradient_energy(u)  # [B, C]
        potential_energy = (u ** 4).mean(dim=(-2, -1))  # [B, C] (reaction-diffusion)
        total_energy = kinetic_energy + potential_energy  # [B, C]
        features_list.extend([
            kinetic_energy.mean(dim=-1, keepdim=True),  # Average over channels
            enstrophy.mean(dim=-1, keepdim=True),
            potential_energy.mean(dim=-1, keepdim=True),
            total_energy.mean(dim=-1, keepdim=True),
        ])  # 4 features

        # Dissipation rates (3D)
        laplacian = self._compute_laplacian(u)  # [B, C, H, W]
        viscous_dissipation = (laplacian ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        reaction_dissipation = (u ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        total_dissipation = viscous_dissipation + reaction_dissipation  # [B, 1]
        features_list.extend([viscous_dissipation, reaction_dissipation, total_dissipation])  # 3 features

        # Spectral dynamics (5D)
        fft = torch.fft.rfft2(u)  # [B, C, H, W//2+1]
        power_spectrum = (fft.abs() ** 2).mean(dim=1)  # [B, H, W//2+1], average over channels
        power_flat = power_spectrum.flatten(start_dim=1)  # [B, H*(W//2+1)]

        # Spectral centroid (frequency-weighted center of mass)
        freqs = torch.arange(power_flat.shape[1], device=u.device).float()
        spectral_centroid = (power_flat * freqs).sum(dim=1, keepdim=True) / (power_flat.sum(dim=1, keepdim=True) + 1e-8)  # [B, 1]

        # Spectral bandwidth (spread)
        centroid_expanded = spectral_centroid.expand_as(power_flat)
        spectral_bandwidth = torch.sqrt((power_flat * (freqs - centroid_expanded) ** 2).sum(dim=1, keepdim=True) / (power_flat.sum(dim=1, keepdim=True) + 1e-8))  # [B, 1]

        # Spectral rolloff (85% energy frequency)
        cumsum = torch.cumsum(power_flat, dim=1)
        total = cumsum[:, -1:]
        rolloff_idx = (cumsum > 0.85 * total).to(torch.float).argmax(dim=1, keepdim=True).float()  # [B, 1]
        spectral_rolloff = rolloff_idx / power_flat.shape[1]  # Normalize to [0, 1]

        # High-frequency ratio
        mid_point = power_flat.shape[1] // 2
        high_freq_energy = power_flat[:, mid_point:].sum(dim=1, keepdim=True)
        total_energy_spec = power_flat.sum(dim=1, keepdim=True) + 1e-8
        high_freq_ratio = high_freq_energy / total_energy_spec  # [B, 1]

        # Spectral entropy
        power_normalized = power_flat / (total_energy_spec + 1e-8)
        spectral_entropy = -(power_normalized * torch.log(power_normalized + 1e-8)).sum(dim=1, keepdim=True)  # [B, 1]

        features_list.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff, high_freq_ratio, spectral_entropy])  # 5 features

        # Structure metrics (5D)
        grad_x, grad_y = self._compute_gradients(u)  # [B, C, H, W] each
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # [B, C, H, W]
        gradient_norm_mean = grad_norm.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        gradient_norm_std = grad_norm.std(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        laplacian_mean = laplacian.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        laplacian_std = laplacian.std(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Divergence (if multi-channel, treat as vector field)
        if C >= 2:
            divergence = grad_x[:, 0] + grad_y[:, min(1, C-1)]  # [B, H, W]
            divergence_mean = divergence.mean(dim=(-2, -1)).unsqueeze(1)  # [B, 1]
        else:
            divergence_mean = torch.zeros(B, 1, device=u.device)

        features_list.extend([gradient_norm_mean, gradient_norm_std, laplacian_mean, laplacian_std, divergence_mean])  # 5 features

        # Statistical moments (5D)
        u_flat = u.flatten(start_dim=1)  # [B, C*H*W]
        mean = u_flat.mean(dim=1, keepdim=True)
        std = u_flat.std(dim=1, keepdim=True) + 1e-8

        # Skewness (third moment)
        skewness = ((u_flat - mean) ** 3).mean(dim=1, keepdim=True) / (std ** 3)  # [B, 1]

        # Kurtosis (fourth moment)
        kurtosis = ((u_flat - mean) ** 4).mean(dim=1, keepdim=True) / (std ** 4)  # [B, 1]

        # Entropy (histogram-based)
        u_sorted = torch.sort(u_flat, dim=1)[0]
        hist_bins = 20
        hist_indices = (torch.arange(B, device=u.device).unsqueeze(1),
                       (torch.arange(hist_bins, device=u.device).float() / hist_bins * u_flat.shape[1]).long())
        hist = torch.zeros(B, hist_bins, device=u.device)
        for b in range(B):
            hist[b] = torch.histc(u_flat[b], bins=hist_bins)
        hist_normalized = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(hist_normalized * torch.log(hist_normalized + 1e-8)).sum(dim=1, keepdim=True)  # [B, 1]

        # Gini coefficient (inequality measure)
        u_sorted_flat = u_flat.sort(dim=1)[0]
        n = u_flat.shape[1]
        index = torch.arange(1, n + 1, device=u.device).float()
        gini = (2 * (u_sorted_flat * index).sum(dim=1, keepdim=True)) / (n * u_sorted_flat.sum(dim=1, keepdim=True) + 1e-8) - (n + 1) / n  # [B, 1]

        # IQR (interquartile range)
        q1 = torch.quantile(u_flat, 0.25, dim=1, keepdim=True)
        q3 = torch.quantile(u_flat, 0.75, dim=1, keepdim=True)
        iqr = q3 - q1  # [B, 1]

        features_list.extend([skewness, kurtosis, entropy, gini, iqr])  # 5 features

        # Concatenate all instantaneous features
        features = torch.cat(features_list, dim=-1)  # [B, 22]

        # Handle NaNs/Infs
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def _extract_local_temporal(self, u: Tensor) -> Tensor:
        """Extract local temporal dynamics using history buffer (28D).

        Features computed from recent K timesteps:
        - Autocorrelation (3D): lag-1, lag-2, lag-5
        - Trend detection (4D): linear slope, R², acceleration, jerk
        - Windowed statistics (6D): mean, std, min, max, range, CV
        - Change detection (5D): delta mean/std, L1/L2/Linf norms
        - Oscillation detection (4D): zero crossings, peaks, troughs, amplitude
        - Growth rates (6D): exponential, power-law, relative, doubling time, energy/enstrophy growth

        Args:
            u: State [B, C, H, W]

        Returns:
            features: [B, 28]
        """
        B = u.shape[0]

        # Update history buffer
        self.history_buffer.append(u.detach().cpu())

        # If not enough history, return zeros
        if len(self.history_buffer) < 2:
            return torch.zeros(B, 28, device=u.device)

        # Convert buffer to tensor [K, B, C, H, W]
        history = torch.stack(list(self.history_buffer), dim=0).to(u.device)
        K = history.shape[0]

        features_list = []

        # Flatten spatial dimensions for easier computation
        history_flat = history.flatten(start_dim=2)  # [K, B, C*H*W]
        current_flat = u.flatten(start_dim=1)  # [B, C*H*W]

        # Autocorrelation (3D)
        def compute_autocorr(lag):
            if lag >= K:
                return torch.zeros(B, 1, device=u.device)
            older = history_flat[max(0, K-1-lag)]  # [B, C*H*W]
            newer = history_flat[-1]  # [B, C*H*W]
            corr = (older * newer).mean(dim=-1, keepdim=True) / (torch.sqrt((older ** 2).mean(dim=-1, keepdim=True) * (newer ** 2).mean(dim=-1, keepdim=True)) + 1e-8)
            return corr

        autocorr_lag1 = compute_autocorr(1)
        autocorr_lag2 = compute_autocorr(2)
        autocorr_lag5 = compute_autocorr(min(5, K-1))
        features_list.extend([autocorr_lag1, autocorr_lag2, autocorr_lag5])  # 3 features

        # Trend detection (4D) - fit linear model to temporal mean
        if K >= 3:
            temporal_means = history_flat.mean(dim=-1)  # [K, B]
            t = torch.arange(K, device=u.device, dtype=torch.float32).unsqueeze(1)  # [K, 1]
            t_mean = t.mean()

            # Linear regression: y = slope * t + intercept
            numerator = ((t - t_mean) * (temporal_means - temporal_means.mean(dim=0, keepdim=True))).sum(dim=0, keepdim=True)  # [1, B]
            denominator = ((t - t_mean) ** 2).sum() + 1e-8
            slope = (numerator / denominator).t()  # [B, 1]

            # R² (goodness of fit)
            predictions = slope * t + (temporal_means.mean(dim=0, keepdim=True).t())  # [K, B]
            ss_res = ((temporal_means - predictions.t()) ** 2).sum(dim=0, keepdim=True).t()
            ss_tot = ((temporal_means - temporal_means.mean(dim=0, keepdim=True)) ** 2).sum(dim=0, keepdim=True).t() + 1e-8
            r2 = 1 - ss_res / ss_tot  # [B, 1]

            # Acceleration (second derivative)
            if K >= 4:
                accel = temporal_means[-1] - 2 * temporal_means[-2] + temporal_means[-3]  # [B]
                accel = accel.unsqueeze(1)  # [B, 1]
            else:
                accel = torch.zeros(B, 1, device=u.device)

            # Jerk (third derivative)
            if K >= 5:
                jerk = temporal_means[-1] - 3 * temporal_means[-2] + 3 * temporal_means[-3] - temporal_means[-4]  # [B]
                jerk = jerk.unsqueeze(1)  # [B, 1]
            else:
                jerk = torch.zeros(B, 1, device=u.device)

            features_list.extend([slope, r2, accel, jerk])  # 4 features
        else:
            features_list.extend([torch.zeros(B, 1, device=u.device) for _ in range(4)])

        # Windowed statistics (6D)
        window_mean = history_flat.mean(dim=0).mean(dim=-1, keepdim=True)  # [B, 1]
        window_std = history_flat.std(dim=0).mean(dim=-1, keepdim=True)  # [B, 1]
        window_min = history_flat.min(dim=0)[0].min(dim=-1, keepdim=True)[0]  # [B, 1]
        window_max = history_flat.max(dim=0)[0].max(dim=-1, keepdim=True)[0]  # [B, 1]
        window_range = window_max - window_min  # [B, 1]
        window_cv = window_std / (window_mean.abs() + 1e-8)  # [B, 1]
        features_list.extend([window_mean, window_std, window_min, window_max, window_range, window_cv])  # 6 features

        # Change detection (5D) - compare current to oldest in buffer
        oldest_flat = history_flat[0]  # [B, C*H*W]
        delta_mean = (current_flat - oldest_flat).mean(dim=-1, keepdim=True)  # [B, 1]
        delta_std = (current_flat - oldest_flat).std(dim=-1, keepdim=True)  # [B, 1]
        delta_l1 = (current_flat - oldest_flat).abs().mean(dim=-1, keepdim=True)  # [B, 1]
        delta_l2 = torch.sqrt(((current_flat - oldest_flat) ** 2).mean(dim=-1, keepdim=True))  # [B, 1]
        delta_linf = (current_flat - oldest_flat).abs().max(dim=-1, keepdim=True)[0]  # [B, 1]
        features_list.extend([delta_mean, delta_std, delta_l1, delta_l2, delta_linf])  # 5 features

        # Oscillation detection (4D)
        if K >= 3:
            temporal_means = history_flat.mean(dim=-1)  # [K, B]

            # Zero crossing rate
            signs = torch.sign(temporal_means - temporal_means.mean(dim=0, keepdim=True))
            sign_changes = (signs[1:] != signs[:-1]).float().sum(dim=0, keepdim=True).t() / (K - 1)  # [B, 1]

            # Peak/trough count (simple local maxima/minima)
            is_peak = (temporal_means[1:-1] > temporal_means[:-2]) & (temporal_means[1:-1] > temporal_means[2:])
            is_trough = (temporal_means[1:-1] < temporal_means[:-2]) & (temporal_means[1:-1] < temporal_means[2:])
            peak_count = is_peak.float().sum(dim=0, keepdim=True).t() / (K - 2)  # [B, 1]
            trough_count = is_trough.float().sum(dim=0, keepdim=True).t() / (K - 2)  # [B, 1]

            # Oscillation amplitude (max - min in window)
            osc_amplitude = (temporal_means.max(dim=0)[0] - temporal_means.min(dim=0)[0]).unsqueeze(1)  # [B, 1]

            features_list.extend([sign_changes, peak_count, trough_count, osc_amplitude])  # 4 features
        else:
            features_list.extend([torch.zeros(B, 1, device=u.device) for _ in range(4)])

        # Growth rates (6D)
        if K >= 2:
            current_energy = (current_flat ** 2).mean(dim=-1, keepdim=True)  # [B, 1]
            oldest_energy = (oldest_flat ** 2).mean(dim=-1, keepdim=True)  # [B, 1]

            # Exponential growth rate: λ such that E(t) = E(0) * exp(λ * t)
            exp_growth_rate = torch.log((current_energy + 1e-8) / (oldest_energy + 1e-8)) / K  # [B, 1]

            # Power-law exponent: α such that E(t) = E(0) * t^α
            power_law_exp = torch.log((current_energy + 1e-8) / (oldest_energy + 1e-8)) / (torch.log(torch.tensor(K, dtype=torch.float32, device=u.device)))  # [B, 1]

            # Relative growth rate
            rel_growth_rate = (current_energy - oldest_energy) / (oldest_energy + 1e-8)  # [B, 1]

            # Doubling time (time to 2x current value)
            doubling_time = torch.log(torch.tensor(2.0, device=u.device)) / (exp_growth_rate + 1e-8)  # [B, 1]

            # Energy growth rate (derivative)
            energy_growth = (current_energy - oldest_energy) / K  # [B, 1]

            # Enstrophy growth rate
            current_enstrophy = self._compute_gradient_energy(u).mean(dim=-1, keepdim=True)  # [B, 1]
            oldest_enstrophy = self._compute_gradient_energy(history[0].to(u.device)).mean(dim=-1, keepdim=True)  # [B, 1]
            enstrophy_growth = (current_enstrophy - oldest_enstrophy) / K  # [B, 1]

            features_list.extend([exp_growth_rate, power_law_exp, rel_growth_rate, doubling_time, energy_growth, enstrophy_growth])  # 6 features
        else:
            features_list.extend([torch.zeros(B, 1, device=u.device) for _ in range(6)])

        # Concatenate all local temporal features
        features = torch.cat(features_list, dim=-1)  # [B, 28]

        # Handle NaNs/Infs
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def _extract_local_stability(self, u: Tensor) -> Tensor:
        """Extract local stability metrics (24D).

        Features measuring instantaneous stability/sensitivity:
        - Lipschitz estimates (4D): spatial, x/y directional, anisotropy
        - Jacobian-free stability (5D): gradient alignment, flow consistency, contraction/expansion, stability index
        - Local linearization (5D): local gain, sensitivity norm, condition number, spectral radius, damping
        - Divergence metrics (5D): trajectory divergence, prediction error, curvature change, phase space distance, Lyapunov proxy
        - Regularity measures (5D): smoothness, Hölder exponent, TV, roughness, irregularity index

        Args:
            u: State [B, C, H, W]

        Returns:
            features: [B, 24]
        """
        B = u.shape[0]

        # If not enough history, return zeros
        if len(self.history_buffer) < 2:
            return torch.zeros(B, 24, device=u.device)

        features_list = []

        # Get previous state
        u_prev = list(self.history_buffer)[-2].to(u.device)  # [B, C, H, W]

        # Lipschitz estimates (4D)
        grad_x, grad_y = self._compute_gradients(u)
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        u_norm = torch.sqrt((u ** 2).mean())

        spatial_lipschitz = grad_norm.max() / (u_norm + 1e-8)  # Scalar, expand to [B, 1]
        spatial_lipschitz = spatial_lipschitz.unsqueeze(0).unsqueeze(1).expand(B, 1)

        local_lipschitz_x = grad_x.abs().mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        local_lipschitz_y = grad_y.abs().mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        lipschitz_anisotropy = local_lipschitz_x / (local_lipschitz_y + 1e-8)  # [B, 1]

        features_list.extend([spatial_lipschitz, local_lipschitz_x, local_lipschitz_y, lipschitz_anisotropy])  # 4 features

        # Jacobian-free stability (5D)
        grad_x_prev, grad_y_prev = self._compute_gradients(u_prev)
        grad_norm_prev = torch.sqrt(grad_x_prev ** 2 + grad_y_prev ** 2)

        # Gradient alignment (cosine similarity)
        grad_alignment = ((grad_x * grad_x_prev + grad_y * grad_y_prev) / (grad_norm * grad_norm_prev + 1e-8)).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Flow consistency (correlation of velocity fields)
        u_flat = u.flatten(start_dim=1)
        u_prev_flat = u_prev.flatten(start_dim=1)
        flow_consistency = (u_flat * u_prev_flat).mean(dim=-1, keepdim=True) / (torch.sqrt((u_flat ** 2).mean(dim=-1, keepdim=True) * (u_prev_flat ** 2).mean(dim=-1, keepdim=True)) + 1e-8)  # [B, 1]

        # Contraction rate
        delta_norm = torch.sqrt(((u - u_prev) ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))
        if len(self.history_buffer) >= 3:
            u_prev2 = list(self.history_buffer)[-3].to(u.device)
            delta_norm_prev = torch.sqrt(((u_prev - u_prev2) ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))
            contraction_rate = delta_norm / (delta_norm_prev + 1e-8)  # [B, 1]
        else:
            contraction_rate = torch.ones(B, 1, device=u.device)

        # Expansion rate (max local stretching)
        expansion_rate = grad_norm.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(1)  # [B, 1]

        # Stability index (empirical)
        stability_index = torch.exp(-delta_norm)  # [B, 1]

        features_list.extend([grad_alignment, flow_consistency, contraction_rate, expansion_rate, stability_index])  # 5 features

        # Local linearization (5D)
        # Local gain: ||δu_t|| / ||δu_{t-1}||
        local_gain = delta_norm  # [B, 1] (already computed above)

        # Sensitivity norm (finite difference Jacobian estimate)
        sensitivity_norm = grad_norm.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Condition number (ratio of max/min singular values, approximated)
        grad_max = grad_norm.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(1)
        grad_min = grad_norm.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(1) + 1e-8
        condition_number = grad_max / grad_min  # [B, 1]

        # Spectral radius (largest eigenvalue magnitude estimate)
        spectral_radius = expansion_rate  # Use max gradient as proxy

        # Damping ratio (energy decay rate)
        energy_current = (u ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)
        energy_prev = (u_prev ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)
        damping_ratio = (energy_prev - energy_current) / (energy_prev + 1e-8)  # [B, 1]

        features_list.extend([local_gain, sensitivity_norm, condition_number, spectral_radius, damping_ratio])  # 5 features

        # Divergence metrics (5D)
        # Trajectory divergence (not applicable without reference, use delta_norm)
        trajectory_divergence = delta_norm  # [B, 1]

        # Prediction error (if we had a predictor, for now use change magnitude)
        prediction_error = delta_norm  # [B, 1]

        # Curvature change
        laplacian = self._compute_laplacian(u)
        laplacian_prev = self._compute_laplacian(u_prev)
        curvature_change = torch.sqrt(((laplacian - laplacian_prev) ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))  # [B, 1]

        # Phase space distance (L2 distance in flattened space)
        phase_space_distance = torch.sqrt(((u_flat - u_prev_flat) ** 2).mean(dim=-1, keepdim=True))  # [B, 1]

        # Lyapunov proxy (log divergence rate over window)
        if len(self.history_buffer) >= 2:
            lyapunov_proxy = torch.log(delta_norm + 1e-8)  # [B, 1]
        else:
            lyapunov_proxy = torch.zeros(B, 1, device=u.device)

        features_list.extend([trajectory_divergence, prediction_error, curvature_change, phase_space_distance, lyapunov_proxy])  # 5 features

        # Regularity measures (5D)
        # Smoothness (Sobolev norm ratio)
        h1_norm = torch.sqrt((u ** 2 + grad_norm ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))
        l2_norm = torch.sqrt((u ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))
        smoothness = l2_norm / (h1_norm + 1e-8)  # [B, 1]

        # Hölder exponent (estimated from gradient decay)
        holder_exponent = -torch.log(grad_norm.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True) + 1e-8)  # [B, 1]

        # Total variation
        tv = (grad_norm).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Roughness (high-frequency energy fraction)
        fft = torch.fft.rfft2(u)
        power = (fft.abs() ** 2).mean(dim=1)  # [B, H, W//2+1]
        high_freq_start = power.shape[1] // 2
        high_freq_energy = power[:, high_freq_start:].sum(dim=(-2, -1), keepdim=True).flatten(start_dim=1)
        total_energy = power.sum(dim=(-2, -1), keepdim=True).flatten(start_dim=1) + 1e-8
        roughness = high_freq_energy / total_energy  # [B, 1]

        # Irregularity index (deviation from smooth approximation)
        u_smooth = torch.nn.functional.avg_pool2d(u, kernel_size=3, stride=1, padding=1)  # [B, C, H, W]
        irregularity = torch.sqrt(((u - u_smooth) ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))  # [B, 1]

        features_list.extend([smoothness, holder_exponent, tv, roughness, irregularity])  # 5 features

        # Concatenate all local stability features
        features = torch.cat(features_list, dim=-1)  # [B, 24]

        # Handle NaNs/Infs
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def _extract_phase_space(self, u: Tensor) -> Tensor:
        """Extract phase space geometry features (26D).

        Features characterizing local flow geometry:
        - Flow alignment (6D): velocity magnitude/direction, coherence, curl, divergence
        - Vorticity measures (5D): mean, std, enstrophy density, vortex count, helicity
        - Strain rate (4D): shear, normal (xx, yy), magnitude
        - Topological features (5D): critical point density, saddles, attractors, repellers, topological entropy
        - Manifold geometry (6D): local dimension, tangent space angle, curvature, geodesic deviation, metric tensor det, volume element

        Args:
            u: State [B, C, H, W]

        Returns:
            features: [B, 26]
        """
        B, C, H, W = u.shape

        # If not enough history, return zeros
        if len(self.history_buffer) < 2:
            return torch.zeros(B, 26, device=u.device)

        features_list = []

        u_prev = list(self.history_buffer)[-2].to(u.device)

        # Compute gradients and velocity
        grad_x, grad_y = self._compute_gradients(u)
        velocity_x = (u - u_prev)[:, 0] if C >= 1 else torch.zeros(B, H, W, device=u.device)  # [B, H, W]
        velocity_y = (u - u_prev)[:, min(1, C-1)] if C >= 2 else torch.zeros(B, H, W, device=u.device)

        # Flow alignment (6D)
        velocity_magnitude = torch.sqrt(velocity_x ** 2 + velocity_y ** 2).mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]

        # Velocity direction (mean components)
        velocity_direction_x = velocity_x.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
        velocity_direction_y = velocity_y.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]

        # Flow coherence (alignment of local velocity vectors)
        velocity_norm = torch.sqrt(velocity_x ** 2 + velocity_y ** 2) + 1e-8
        flow_coherence = ((velocity_x / velocity_norm) ** 2 + (velocity_y / velocity_norm) ** 2).mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]

        # Flow curl (rotational component)
        if C >= 2:
            curl = grad_x[:, 1] - grad_y[:, 0]  # [B, H, W]
            flow_curl = curl.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
        else:
            flow_curl = torch.zeros(B, 1, device=u.device)

        # Flow divergence (compressibility)
        if C >= 2:
            div = grad_x[:, 0] + grad_y[:, 1]  # [B, H, W]
            flow_div = div.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
        else:
            flow_div = torch.zeros(B, 1, device=u.device)

        features_list.extend([velocity_magnitude, velocity_direction_x, velocity_direction_y, flow_coherence, flow_curl, flow_div])  # 6 features

        # Vorticity measures (5D)
        if C >= 2:
            vorticity = curl  # Already computed
            vorticity_mean = vorticity.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
            vorticity_std = vorticity.std(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]

            # Enstrophy density
            enstrophy_density = (vorticity ** 2).mean(dim=(-2, -1), keepdim=True).unsqueeze(1) / (H * W)  # [B, 1]

            # Vortex count (number of local extrema in vorticity)
            vorticity_padded = torch.nn.functional.pad(vorticity.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
            is_vortex = ((vorticity.unsqueeze(1) > vorticity_padded[:, :, :-2, 1:-1]) &
                         (vorticity.unsqueeze(1) > vorticity_padded[:, :, 2:, 1:-1]) &
                         (vorticity.unsqueeze(1) > vorticity_padded[:, :, 1:-1, :-2]) &
                         (vorticity.unsqueeze(1) > vorticity_padded[:, :, 1:-1, 2:]))
            vortex_count = is_vortex.float().sum(dim=(-2, -1)).unsqueeze(1).float() / (H * W)  # [B, 1]

            # Helicity (u · curl(u) for 3D, approximation for 2D)
            helicity = (u[:, 0] * curl).mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
        else:
            vorticity_mean = torch.zeros(B, 1, device=u.device)
            vorticity_std = torch.zeros(B, 1, device=u.device)
            enstrophy_density = torch.zeros(B, 1, device=u.device)
            vortex_count = torch.zeros(B, 1, device=u.device)
            helicity = torch.zeros(B, 1, device=u.device)

        features_list.extend([vorticity_mean, vorticity_std, enstrophy_density, vortex_count, helicity])  # 5 features

        # Strain rate (4D)
        if C >= 2:
            # Strain tensor components
            S_xx = grad_x[:, 0]  # ∂u_x/∂x
            S_yy = grad_y[:, 1]  # ∂u_y/∂y
            S_xy = 0.5 * (grad_y[:, 0] + grad_x[:, 1])  # ½(∂u_x/∂y + ∂u_y/∂x)

            shear_strain = S_xy.abs().mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
            normal_strain_xx = S_xx.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
            normal_strain_yy = S_yy.mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
            strain_rate_magnitude = torch.sqrt(S_xx ** 2 + S_yy ** 2 + 2 * S_xy ** 2).mean(dim=(-2, -1), keepdim=True).unsqueeze(1)  # [B, 1]
        else:
            shear_strain = torch.zeros(B, 1, device=u.device)
            normal_strain_xx = torch.zeros(B, 1, device=u.device)
            normal_strain_yy = torch.zeros(B, 1, device=u.device)
            strain_rate_magnitude = torch.zeros(B, 1, device=u.device)

        features_list.extend([shear_strain, normal_strain_xx, normal_strain_yy, strain_rate_magnitude])  # 4 features

        # Topological features (5D) - simplified approximations
        # Critical point density (gradient near zero)
        grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        is_critical = (grad_norm < grad_norm.mean() * 0.1).float()
        critical_point_density = is_critical.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Saddle count (heuristic: laplacian sign changes)
        laplacian = self._compute_laplacian(u)
        laplacian_signs = torch.sign(laplacian)
        saddle_count = (laplacian_signs.diff(dim=-1).abs() > 0).float().mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Attractor count (local minima)
        is_attractor = is_critical * (laplacian > 0).float()
        attractor_count = is_attractor.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Repeller count (local maxima)
        is_repeller = is_critical * (laplacian < 0).float()
        repeller_count = is_repeller.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Topological entropy (simplified: spatial entropy of gradient field)
        grad_hist = torch.histc(grad_norm.flatten(start_dim=1), bins=20)
        grad_hist_normalized = grad_hist / (grad_hist.sum() + 1e-8)
        topological_entropy = -(grad_hist_normalized * torch.log(grad_hist_normalized + 1e-8)).sum().unsqueeze(0).unsqueeze(1).expand(B, 1)  # [B, 1]

        features_list.extend([critical_point_density, saddle_count, attractor_count, repeller_count, topological_entropy])  # 5 features

        # Manifold geometry (6D) - approximations
        # Local dimension (correlation dimension estimate from gradient distribution)
        local_dimension = torch.log(torch.tensor(H * W, dtype=torch.float32, device=u.device)) / (torch.log(grad_norm.max() / (grad_norm.min() + 1e-8) + 1.0))
        local_dimension = local_dimension.unsqueeze(0).unsqueeze(1).expand(B, 1)  # [B, 1]

        # Tangent space angle (angle between successive gradient directions)
        grad_x_prev, grad_y_prev = self._compute_gradients(u_prev)
        tangent_angle = ((grad_x * grad_x_prev + grad_y * grad_y_prev) /
                        (torch.sqrt(grad_x ** 2 + grad_y ** 2) * torch.sqrt(grad_x_prev ** 2 + grad_y_prev ** 2) + 1e-8))
        tangent_angle = tangent_angle.mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Curvature tensor norm (second derivatives)
        laplacian_norm = (laplacian ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]

        # Geodesic deviation (how much trajectories deviate from straight lines)
        geodesic_deviation = torch.sqrt(((u - u_prev) ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True))  # [B, 1]

        # Metric tensor determinant (volume scaling)
        metric_tensor_det = (grad_x ** 2 * grad_y ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # Simplified

        # Volume element
        volume_element = torch.sqrt(metric_tensor_det + 1e-8)  # [B, 1]

        features_list.extend([local_dimension, tangent_angle, laplacian_norm, geodesic_deviation, metric_tensor_det, volume_element])  # 6 features

        # Concatenate all phase space features
        features = torch.cat(features_list, dim=-1)  # [B, 26]

        # Handle NaNs/Infs
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def _extract_multiscale(self, u: Tensor) -> Tensor:
        """Extract multi-scale temporal features (30D).

        Features from hierarchical temporal averaging:
        - Multi-scale energy (5D): instant, short, medium, long, trend
        - Multi-scale gradients (5D): same scales
        - Multi-scale spectral (5D): centroid at each scale + drift
        - Multi-scale entropy (5D): instant through long + production rate
        - Cross-scale correlations (5D): instant-short, short-medium, medium-long, coherence, separation
        - Temporal persistence (5D): Hurst exponent, DFA, memory depth, persistence score, anti-persistence

        Args:
            u: State [B, C, H, W]

        Returns:
            features: [B, 30]
        """
        B = u.shape[0]

        # Update all buffers
        self.short_buffer.append(u.detach().cpu())
        self.medium_buffer.append(u.detach().cpu())
        self.long_buffer.append(u.detach().cpu())

        # If not enough history, return zeros
        if len(self.short_buffer) < 2:
            return torch.zeros(B, 30, device=u.device)

        features_list = []

        # Helper to compute energy from buffer
        def buffer_energy(buffer):
            if len(buffer) == 0:
                return torch.zeros(B, 1, device=u.device)
            states = torch.stack(list(buffer), dim=0).to(u.device)  # [K, B, C, H, W]
            energy = (states ** 2).mean(dim=(-2, -1)).mean(dim=-1).mean(dim=0, keepdim=True).t()  # [B, 1]
            return energy

        # Multi-scale energy (5D)
        energy_instant = (u ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        energy_short = buffer_energy(self.short_buffer)  # [B, 1]
        energy_medium = buffer_energy(self.medium_buffer)  # [B, 1]
        energy_long = buffer_energy(self.long_buffer)  # [B, 1]

        # Energy trend (slope of log-energy vs log-time)
        if len(self.long_buffer) >= 3:
            energies = torch.cat([energy_instant, energy_short, energy_medium, energy_long], dim=1)  # [B, 4]
            log_energies = torch.log(energies + 1e-8)
            log_times = torch.log(torch.tensor([1.0, float(self.short_window), float(self.medium_window), float(self.long_window)], device=u.device))
            # Simple linear regression
            energy_trend = (log_energies[:, -1:] - log_energies[:, :1]) / (log_times[-1] - log_times[0])  # [B, 1]
        else:
            energy_trend = torch.zeros(B, 1, device=u.device)

        features_list.extend([energy_instant, energy_short, energy_medium, energy_long, energy_trend])  # 5 features

        # Multi-scale gradients (5D)
        def buffer_gradients(buffer):
            if len(buffer) == 0:
                return torch.zeros(B, 1, device=u.device)
            states = torch.stack(list(buffer), dim=0).to(u.device)
            grad_x, grad_y = self._compute_gradients(states[-1])  # Use most recent
            grad_norm = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)
            return grad_norm

        grad_x, grad_y = self._compute_gradients(u)
        grad_instant = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=(-2, -1)).mean(dim=-1, keepdim=True)  # [B, 1]
        grad_short = buffer_gradients(self.short_buffer)
        grad_medium = buffer_gradients(self.medium_buffer)
        grad_long = buffer_gradients(self.long_buffer)
        grad_trend = (grad_instant - grad_long) / (len(self.long_buffer) + 1) if len(self.long_buffer) > 0 else torch.zeros(B, 1, device=u.device)

        features_list.extend([grad_instant, grad_short, grad_medium, grad_long, grad_trend])  # 5 features

        # Multi-scale spectral (5D)
        def buffer_spectral_centroid(buffer):
            if len(buffer) == 0:
                return torch.zeros(B, 1, device=u.device)
            states = torch.stack(list(buffer), dim=0).to(u.device)
            fft = torch.fft.rfft2(states[-1])
            power = (fft.abs() ** 2).mean(dim=1).flatten(start_dim=1)
            freqs = torch.arange(power.shape[1], device=u.device).float()
            centroid = (power * freqs).sum(dim=1, keepdim=True) / (power.sum(dim=1, keepdim=True) + 1e-8)
            return centroid

        fft = torch.fft.rfft2(u)
        power = (fft.abs() ** 2).mean(dim=1).flatten(start_dim=1)
        freqs = torch.arange(power.shape[1], device=u.device).float()
        spectral_instant = (power * freqs).sum(dim=1, keepdim=True) / (power.sum(dim=1, keepdim=True) + 1e-8)
        spectral_short = buffer_spectral_centroid(self.short_buffer)
        spectral_medium = buffer_spectral_centroid(self.medium_buffer)
        spectral_long = buffer_spectral_centroid(self.long_buffer)
        spectral_drift = (spectral_instant - spectral_long) / (len(self.long_buffer) + 1) if len(self.long_buffer) > 0 else torch.zeros(B, 1, device=u.device)

        features_list.extend([spectral_instant, spectral_short, spectral_medium, spectral_long, spectral_drift])  # 5 features

        # Multi-scale entropy (5D)
        def buffer_entropy(buffer):
            if len(buffer) == 0:
                return torch.zeros(B, 1, device=u.device)
            states = torch.stack(list(buffer), dim=0).to(u.device)
            state_flat = states[-1].flatten(start_dim=1)
            hist = torch.histc(state_flat[0], bins=20)  # Use first sample
            hist_norm = hist / (hist.sum() + 1e-8)
            entropy = -(hist_norm * torch.log(hist_norm + 1e-8)).sum().unsqueeze(0).unsqueeze(1).expand(B, 1)
            return entropy

        u_flat = u.flatten(start_dim=1)
        hist = torch.histc(u_flat[0], bins=20)
        hist_norm = hist / (hist.sum() + 1e-8)
        entropy_instant = -(hist_norm * torch.log(hist_norm + 1e-8)).sum().unsqueeze(0).unsqueeze(1).expand(B, 1)
        entropy_short = buffer_entropy(self.short_buffer)
        entropy_medium = buffer_entropy(self.medium_buffer)
        entropy_long = buffer_entropy(self.long_buffer)
        entropy_production = (entropy_instant - entropy_long) / (len(self.long_buffer) + 1) if len(self.long_buffer) > 0 else torch.zeros(B, 1, device=u.device)

        features_list.extend([entropy_instant, entropy_short, entropy_medium, entropy_long, entropy_production])  # 5 features

        # Cross-scale correlations (5D)
        instant_short_corr = (energy_instant * energy_short).sum() / (torch.sqrt((energy_instant ** 2).sum() * (energy_short ** 2).sum()) + 1e-8)
        instant_short_corr = instant_short_corr.unsqueeze(0).unsqueeze(1).expand(B, 1)

        short_medium_corr = (energy_short * energy_medium).sum() / (torch.sqrt((energy_short ** 2).sum() * (energy_medium ** 2).sum()) + 1e-8)
        short_medium_corr = short_medium_corr.unsqueeze(0).unsqueeze(1).expand(B, 1)

        medium_long_corr = (energy_medium * energy_long).sum() / (torch.sqrt((energy_medium ** 2).sum() * (energy_long ** 2).sum()) + 1e-8)
        medium_long_corr = medium_long_corr.unsqueeze(0).unsqueeze(1).expand(B, 1)

        # Overall scale coherence
        all_energies = torch.cat([energy_instant, energy_short, energy_medium, energy_long], dim=1)
        scale_coherence = all_energies.std(dim=1, keepdim=True) / (all_energies.mean(dim=1, keepdim=True) + 1e-8)

        # Scale separation (independence measure)
        scale_separation = 1.0 - (instant_short_corr + short_medium_corr + medium_long_corr) / 3.0

        features_list.extend([instant_short_corr, short_medium_corr, medium_long_corr, scale_coherence, scale_separation])  # 5 features

        # Temporal persistence (5D) - simplified estimates
        # Hurst exponent (long-range dependence)
        if len(self.long_buffer) >= 5:
            long_states = torch.stack(list(self.long_buffer), dim=0).to(u.device)
            energy_series = (long_states ** 2).mean(dim=(-2, -1)).mean(dim=-1)[:, 0]  # [K]
            K = energy_series.shape[0]

            # R/S statistic for Hurst estimation
            mean_energy = energy_series.mean()
            cumsum = torch.cumsum(energy_series - mean_energy, dim=0)
            R = cumsum.max() - cumsum.min()
            S = energy_series.std() + 1e-8
            hurst = torch.log(R / S) / torch.log(torch.tensor(K, dtype=torch.float32, device=u.device))
            hurst = hurst.unsqueeze(0).unsqueeze(1).expand(B, 1)
        else:
            hurst = torch.full((B, 1), 0.5, device=u.device)  # Default to 0.5 (uncorrelated)

        # DFA (detrended fluctuation analysis) - simplified
        dfa = hurst  # Use same estimate for simplicity

        # Memory depth (autocorrelation decay length)
        if len(self.history_buffer) >= 3:
            history = torch.stack(list(self.history_buffer), dim=0).to(u.device)
            autocorrs = []
            for lag in range(1, min(len(self.history_buffer), 5)):
                older = history[-1-lag].flatten(start_dim=1)
                newer = history[-1].flatten(start_dim=1)
                corr = (older * newer).mean() / (torch.sqrt((older ** 2).mean() * (newer ** 2).mean()) + 1e-8)
                autocorrs.append(corr)
            # Find where autocorr drops below threshold
            autocorrs_tensor = torch.stack(autocorrs)
            memory_depth = (autocorrs_tensor > 0.5).float().sum().unsqueeze(0).unsqueeze(1).expand(B, 1)
        else:
            memory_depth = torch.ones(B, 1, device=u.device)

        # Persistence score (tendency to continue current trend)
        if len(self.history_buffer) >= 3:
            history = torch.stack(list(self.history_buffer), dim=0).to(u.device)
            deltas = history[1:] - history[:-1]
            same_sign = ((deltas[:-1] * deltas[1:]) > 0).float().mean()
            persistence_score = same_sign.unsqueeze(0).unsqueeze(1).expand(B, 1)
        else:
            persistence_score = torch.full((B, 1), 0.5, device=u.device)

        # Anti-persistence (tendency to reverse)
        anti_persistence = 1.0 - persistence_score

        features_list.extend([hurst, dfa, memory_depth, persistence_score, anti_persistence])  # 5 features

        # Concatenate all multiscale features
        features = torch.cat(features_list, dim=-1)  # [B, 30]

        # Handle NaNs/Infs
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    def reset(self):
        """Reset all buffers (call at episode boundaries)."""
        self.history_buffer.clear()
        self.short_buffer.clear()
        self.medium_buffer.clear()
        self.long_buffer.clear()

    # Helper methods

    def _compute_gradients(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute spatial gradients using finite differences.

        Args:
            u: State [B, C, H, W] or [B, C, H, W]

        Returns:
            grad_x, grad_y: Gradients [B, C, H, W] each
        """
        # Central differences with periodic boundaries
        grad_x = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / 2.0
        grad_y = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / 2.0
        return grad_x, grad_y

    def _compute_laplacian(self, u: Tensor) -> Tensor:
        """Compute Laplacian using finite differences.

        Args:
            u: State [B, C, H, W]

        Returns:
            laplacian: [B, C, H, W]
        """
        # 5-point stencil
        laplacian = (
            torch.roll(u, -1, dims=-1) + torch.roll(u, 1, dims=-1) +
            torch.roll(u, -1, dims=-2) + torch.roll(u, 1, dims=-2) - 4 * u
        )
        return laplacian

    def _compute_gradient_energy(self, u: Tensor) -> Tensor:
        """Compute gradient energy (enstrophy).

        Args:
            u: State [B, C, H, W]

        Returns:
            enstrophy: [B, C]
        """
        grad_x, grad_y = self._compute_gradients(u)
        enstrophy = (grad_x ** 2 + grad_y ** 2).mean(dim=(-2, -1))  # [B, C]
        return enstrophy
