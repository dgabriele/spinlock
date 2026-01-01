"""
Invariant drift feature extraction.

Extracts trajectory-level features measuring norm-based invariant drift:
- Generic norms (L1, L2, L∞, entropy, total variation)
- Drift metrics (mean drift rate, variance, final/initial ratio, monotonicity)
- Multi-scale filtering (raw, low-pass, high-pass) for scale-specific behavior

These features characterize operator stability, dissipation, and scale-dependent dynamics
without assuming physical conservation laws. Drift is measured as a latent operator property.

Example:
    >>> extractor = InvariantDriftExtractor(device='cuda')
    >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
    >>> features = extractor.extract(trajectories)  # [N, M, num_features]
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryInvariantDriftConfig


class InvariantDriftExtractor:
    """
    Extract invariant drift features from trajectories.

    Measures generic norm-based drift across raw, low-pass, and high-pass filtered fields
    to characterize operator stability and scale-specific dissipation.

    Operates on full trajectories [N, M, T, C, H, W] and computes
    trajectory-level drift summaries.

    Example:
        >>> extractor = InvariantDriftExtractor(device='cuda')
        >>> trajectories = torch.randn(8, 10, 50, 3, 128, 128, device='cuda')
        >>> features = extractor.extract(trajectories)
        >>> # Returns dict with ~60 features shaped [N, M, C]
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize invariant drift feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

        # Pre-compute Gaussian kernel for low-pass filtering (cache for efficiency)
        self._gaussian_kernel = None
        self._kernel_sigma = 2.0
        self._kernel_size = 9

    def _adaptive_outlier_clip(
        self,
        values: torch.Tensor,
        iqr_multiplier: float = 10.0
    ) -> torch.Tensor:
        """Adaptive outlier clipping based on IQR."""
        # Convert to float if needed (quantile requires float/double)
        if not values.is_floating_point():
            values = values.float()

        values_flat = values.flatten()
        valid_mask = ~torch.isnan(values_flat)
        valid_values = values_flat[valid_mask]

        if valid_values.numel() < 4:
            return values

        q1 = torch.quantile(valid_values, 0.25)
        q3 = torch.quantile(valid_values, 0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        values_clipped = torch.clamp(values, min=lower_bound, max=upper_bound)

        return values_clipped

    def extract(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        config: Optional['SummaryInvariantDriftConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract invariant drift features from trajectories.

        Args:
            trajectories: Full trajectories [N, M, T, C, H, W]
                N = batch size
                M = num realizations
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryInvariantDriftConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors [N, M, C]
            One value per realization (trajectory-level features)
        """
        N, M, T, C, H, W = trajectories.shape

        # Handle single-timestep data (drift features undefined)
        if T == 1:
            # Return NaN features for all drift metrics
            return self._return_nan_features(N, M, C, trajectories.device, config)

        features = {}

        # Use config to determine which features to extract
        if config is None:
            include_all = True
        else:
            include_all = False

        # Determine which scales to compute
        num_scales = config.num_scales if config else 3
        scales = ['raw']
        if num_scales >= 2:
            scales.append('lowpass')
        if num_scales >= 3:
            scales.append('highpass')

        # Compute multi-scale fields
        multiscale_fields = self._compute_multiscale_fields(trajectories, config)

        # Extract drift features for each norm and scale
        norms_to_compute = []

        # Generic norms (always computed by default)
        if include_all or (config and config.include_L1_drift):
            norms_to_compute.append('L1')
        if include_all or (config and config.include_L2_drift):
            norms_to_compute.append('L2')
        if include_all or (config and config.include_Linf_drift):
            norms_to_compute.append('Linf')
        if include_all or (config and config.include_entropy_drift):
            norms_to_compute.append('entropy')
        if include_all or (config and config.include_tv_drift):
            norms_to_compute.append('tv')

        # Compute drift for each norm and scale combination
        for norm_type in norms_to_compute:
            for scale in scales:
                fields = multiscale_fields[scale]
                drift_features = self._compute_norm_drift(fields, norm_type, scale, config)
                features.update(drift_features)

        # Optional physical invariants (config-gated)
        if config is not None:
            if config.include_mass_drift:
                mass_features = self._compute_mass_drift(multiscale_fields, config)
                features.update(mass_features)

            if config.include_energy_drift:
                energy_features = self._compute_energy_drift(multiscale_fields, config)
                features.update(energy_features)

        # Scale-specific dissipation features (config-gated or default)
        if include_all or (config is not None and getattr(config, 'include_scale_specific_dissipation', False)):
            dissipation_features = self._compute_scale_specific_dissipation(multiscale_fields, config)
            features.update(dissipation_features)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    def _return_nan_features(
        self,
        N: int,
        M: int,
        C: int,
        device: torch.device,
        config: Optional['SummaryInvariantDriftConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return NaN features for T=1 edge case.

        Args:
            N: Batch size
            M: Number of realizations
            C: Number of channels
            device: Device for tensors
            config: Optional config

        Returns:
            Dictionary of NaN features
        """
        nan_features = torch.full((N, M, C), float('nan'), device=device)
        features = {}

        # Generic norms × 4 metrics × 3 scales = 60 features
        norms = ['L1', 'L2', 'Linf', 'entropy', 'tv']
        metrics = ['mean_drift', 'drift_variance', 'final_initial_ratio', 'monotonicity']
        scales = ['raw', 'lowpass', 'highpass']

        for norm in norms:
            for metric in metrics:
                for scale in scales:
                    feat_name = f"{norm}_{metric}_{scale}"
                    features[feat_name] = nan_features.clone()

        # Optional features (if enabled)
        if config is not None:
            if config.include_mass_drift:
                for metric in metrics:
                    for scale in scales:
                        feat_name = f"mass_{metric}_{scale}"
                        features[feat_name] = nan_features.clone()

            if config.include_energy_drift:
                for energy_type in ['L2', 'gradient']:
                    for metric in metrics:
                        for scale in scales:
                            feat_name = f"energy_{energy_type}_{metric}_{scale}"
                            features[feat_name] = nan_features.clone()

        return features

    # =========================================================================
    # Multi-Scale Filtering
    # =========================================================================

    def _compute_multiscale_fields(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W]
        config: Optional['SummaryInvariantDriftConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute raw, low-pass, and high-pass filtered fields.

        Uses Gaussian blur for low-pass filtering to separate smooth-mode
        behavior from small-scale instabilities.

        Args:
            fields: Trajectory fields [N, M, T, C, H, W]
            config: Optional config

        Returns:
            Dictionary with keys 'raw', 'lowpass', 'highpass'
            Each has shape [N, M, T, C, H, W]
        """
        # Ensure Gaussian kernel is initialized
        if self._gaussian_kernel is None:
            sigma = config.gaussian_sigma if config else self._kernel_sigma
            self._gaussian_kernel = self._create_gaussian_kernel(sigma, self._kernel_size)
            self._gaussian_kernel = self._gaussian_kernel.to(self.device)

        # Reshape for batch processing: [N*M*T*C, 1, H, W]
        N, M, T, C, H, W = fields.shape
        fields_flat = fields.reshape(N * M * T * C, 1, H, W)

        # Low-pass filter: Gaussian blur (σ=2 pixels, kernel_size=9)
        low_pass_flat = F.conv2d(
            fields_flat,
            self._gaussian_kernel,
            padding=self._kernel_size // 2  # 'same' padding
        )

        # High-pass: residual (raw - low_pass)
        high_pass_flat = fields_flat - low_pass_flat

        # Reshape back to [N, M, T, C, H, W]
        low_pass = low_pass_flat.reshape(N, M, T, C, H, W)
        high_pass = high_pass_flat.reshape(N, M, T, C, H, W)

        return {
            'raw': fields,
            'lowpass': low_pass,
            'highpass': high_pass
        }

    def _create_gaussian_kernel(self, sigma: float, kernel_size: int) -> torch.Tensor:
        """
        Create 2D Gaussian kernel for low-pass filtering.

        Args:
            sigma: Standard deviation of Gaussian
            kernel_size: Kernel size (must be odd)

        Returns:
            Gaussian kernel [1, 1, kernel_size, kernel_size]
        """
        # Create 1D Gaussian
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Outer product for 2D kernel
        gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d / gauss_2d.sum()

        # Reshape to [1, 1, kernel_size, kernel_size]
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

        return kernel

    # =========================================================================
    # Norm Computations
    # =========================================================================

    def _compute_norm_drift(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W]
        norm_type: str,
        scale: str,
        config: Optional['SummaryInvariantDriftConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute drift metrics for a specific norm and scale.

        Args:
            fields: Trajectory fields [N, M, T, C, H, W]
            norm_type: 'L1', 'L2', 'Linf', 'entropy', 'tv'
            scale: 'raw', 'lowpass', 'highpass'
            config: Optional config

        Returns:
            Dictionary with 4 features (mean_drift, variance, ratio, monotonicity)
            Each shaped [N, M, C]
        """
        # Compute norm time series [N, M, T, C]
        if norm_type == 'L1':
            norm_series = self._compute_L1_norm(fields)
        elif norm_type == 'L2':
            norm_series = self._compute_L2_norm(fields)
        elif norm_type == 'Linf':
            norm_series = self._compute_Linf_norm(fields)
        elif norm_type == 'entropy':
            entropy_bins = config.entropy_num_bins if config else 32
            norm_series = self._compute_entropy(fields, num_bins=entropy_bins)
        elif norm_type == 'tv':
            norm_series = self._compute_total_variation(fields)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        # Compute drift metrics from norm time series
        drift_metrics = self._compute_drift_metrics(norm_series)

        # Rename features with norm and scale
        features = {}
        for metric_name, metric_value in drift_metrics.items():
            feat_name = f"{norm_type}_{metric_name}_{scale}"
            features[feat_name] = metric_value

        return features

    def _compute_L1_norm(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute L1 norm: ∫|u| dx."""
        # Sum absolute values over spatial dimensions
        return fields.abs().sum(dim=(-2, -1))  # [N, M, T, C]

    def _compute_L2_norm(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm: √(∫u² dx)."""
        # Sum of squares over spatial dimensions, then sqrt
        return torch.sqrt((fields ** 2).sum(dim=(-2, -1)) + 1e-8)  # [N, M, T, C]

    def _compute_Linf_norm(self, fields: torch.Tensor) -> torch.Tensor:
        """Compute L∞ norm: max|u|."""
        # Maximum absolute value over spatial dimensions
        return fields.abs().amax(dim=(-2, -1))  # [N, M, T, C]

    def _compute_entropy(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W]
        num_bins: int = 32
    ) -> torch.Tensor:
        """
        Compute histogram-based entropy: -∫p log p dx (GPU-optimized batched version).

        Uses binning for efficiency (KDE too expensive for large grids).
        All operations stay on GPU - no .item() calls.

        Args:
            fields: Trajectory fields [N, M, T, C, H, W]
            num_bins: Number of histogram bins

        Returns:
            Entropy time series [N, M, T, C]
        """
        N, M, T, C, H, W = fields.shape

        # Flatten spatial dimensions for histogram computation
        fields_flat = fields.reshape(N, M, T, C, H * W)  # [N, M, T, C, H*W]

        # Flatten batch dimensions for batched processing
        NMTC = N * M * T * C
        fields_flat_batched = fields_flat.reshape(NMTC, H * W)  # [NMTC, H*W]

        # Normalize to [0, 1] per sample (no .item() calls - stays on GPU)
        min_vals = fields_flat_batched.min(dim=1, keepdim=True).values  # [NMTC, 1]
        max_vals = fields_flat_batched.max(dim=1, keepdim=True).values  # [NMTC, 1]
        normalized = (fields_flat_batched - min_vals) / (max_vals - min_vals + 1e-8)  # [NMTC, H*W]

        # Discretize to bins
        discrete = torch.floor(normalized * (num_bins - 1)).long()
        discrete = torch.clamp(discrete, 0, num_bins - 1)  # [NMTC, H*W]

        # Batched histogram computation using bincount
        entropies = torch.zeros(NMTC, device=fields.device)

        for i in range(NMTC):
            # Compute histogram for this sample
            hist = torch.bincount(discrete[i], minlength=num_bins).float()  # [num_bins]
            hist = hist / (H * W)  # Normalize to probabilities

            # Entropy: -Σ p log p (vectorized)
            entropies[i] = -(hist * torch.log(hist + 1e-10)).sum()

        # Reshape to [N, M, T, C]
        return entropies.reshape(N, M, T, C)

    def _compute_total_variation(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation: ∫|∇u| dx.

        Uses discrete approximation with central differences.

        Args:
            fields: Trajectory fields [N, M, T, C, H, W]

        Returns:
            TV time series [N, M, T, C]
        """
        N, M, T, C, H, W = fields.shape

        # Reshape for batch gradient computation
        fields_flat = fields.reshape(N * M * T, C, H, W)

        # Compute gradients using central differences
        # Gradient in x direction (circular boundary)
        grad_x = torch.roll(fields_flat, shifts=-1, dims=3) - torch.roll(fields_flat, shifts=1, dims=3)
        grad_x = grad_x / 2.0  # Central difference

        # Gradient in y direction (circular boundary)
        grad_y = torch.roll(fields_flat, shifts=-1, dims=2) - torch.roll(fields_flat, shifts=1, dims=2)
        grad_y = grad_y / 2.0

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # Total variation: sum over spatial dims
        tv = grad_mag.sum(dim=(-2, -1))  # [NMT, C]

        # Reshape to [N, M, T, C]
        return tv.reshape(N, M, T, C)

    # =========================================================================
    # Drift Metrics
    # =========================================================================

    def _compute_drift_metrics(
        self,
        invariant_series: torch.Tensor  # [N, M, T, C]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute drift metrics from time series of invariant.

        Args:
            invariant_series: Time series of invariant values [N, M, T, C]

        Returns:
            Dict with keys:
            - 'mean_drift': Mean drift rate [N, M, C]
            - 'drift_variance': Variance of drift [N, M, C]
            - 'final_initial_ratio': Final/initial ratio [N, M, C]
            - 'monotonicity': Sign consistency of drift [N, M, C]
        """
        # Compute temporal differences
        diffs = invariant_series[:, :, 1:, :] - invariant_series[:, :, :-1, :]  # [N, M, T-1, C]

        # Mean drift rate: average temporal change
        mean_drift = diffs.mean(dim=2)  # [N, M, C]

        # Drift variance: variability of temporal changes
        drift_var = diffs.var(dim=2)  # [N, M, C]

        # Final/initial ratio: overall growth or decay
        final_val = invariant_series[:, :, -1, :]
        initial_val = invariant_series[:, :, 0, :] + 1e-8
        ratio = final_val / initial_val

        # Monotonicity: consistency of drift direction
        # Range: -1 (always decreasing) to +1 (always increasing)
        signs = torch.sign(diffs)
        monotonicity = signs.mean(dim=2)  # [N, M, C]

        return {
            'mean_drift': mean_drift,
            'drift_variance': drift_var,
            'final_initial_ratio': ratio,
            'monotonicity': monotonicity
        }

    # =========================================================================
    # Optional Physical Invariants
    # =========================================================================

    def _compute_mass_drift(
        self,
        multiscale_fields: Dict[str, torch.Tensor],
        config: 'SummaryInvariantDriftConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mass drift: ∫u dx (scalar fields only).

        Args:
            multiscale_fields: Dict with 'raw', 'lowpass', 'highpass' fields
            config: Config

        Returns:
            Dictionary with mass drift features for each scale
        """
        features = {}

        num_scales = config.num_scales if config else 3
        scales = ['raw']
        if num_scales >= 2:
            scales.append('lowpass')
        if num_scales >= 3:
            scales.append('highpass')

        for scale in scales:
            fields = multiscale_fields[scale]  # [N, M, T, C, H, W]

            # Mass = ∫u dx (sum over spatial dimensions)
            mass_series = fields.sum(dim=(-2, -1))  # [N, M, T, C]

            # Compute drift metrics
            drift_metrics = self._compute_drift_metrics(mass_series)

            # Rename features
            for metric_name, metric_value in drift_metrics.items():
                feat_name = f"mass_{metric_name}_{scale}"
                features[feat_name] = metric_value

        return features

    def _compute_energy_drift(
        self,
        multiscale_fields: Dict[str, torch.Tensor],
        config: 'SummaryInvariantDriftConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy drift.

        Includes:
        - L2 energy: ∫u² dx
        - Gradient energy: ∫|∇u|² dx

        Args:
            multiscale_fields: Dict with 'raw', 'lowpass', 'highpass' fields
            config: Config

        Returns:
            Dictionary with energy drift features
        """
        features = {}

        num_scales = config.num_scales if config else 3
        scales = ['raw']
        if num_scales >= 2:
            scales.append('lowpass')
        if num_scales >= 3:
            scales.append('highpass')

        for scale in scales:
            fields = multiscale_fields[scale]  # [N, M, T, C, H, W]

            # L2 energy: ∫u² dx
            l2_energy_series = (fields ** 2).sum(dim=(-2, -1))  # [N, M, T, C]

            # Gradient energy: ∫|∇u|² dx
            N, M, T, C, H, W = fields.shape
            fields_flat = fields.reshape(N * M * T, C, H, W)

            # Compute gradients
            grad_x = torch.roll(fields_flat, shifts=-1, dims=3) - torch.roll(fields_flat, shifts=1, dims=3)
            grad_x = grad_x / 2.0
            grad_y = torch.roll(fields_flat, shifts=-1, dims=2) - torch.roll(fields_flat, shifts=1, dims=2)
            grad_y = grad_y / 2.0

            # Gradient energy
            grad_energy_flat = (grad_x ** 2 + grad_y ** 2).sum(dim=(-2, -1))  # [NMT, C]
            grad_energy_series = grad_energy_flat.reshape(N, M, T, C)

            # Compute drift metrics for L2 energy
            l2_drift = self._compute_drift_metrics(l2_energy_series)
            for metric_name, metric_value in l2_drift.items():
                feat_name = f"energy_L2_{metric_name}_{scale}"
                features[feat_name] = metric_value

            # Compute drift metrics for gradient energy
            grad_drift = self._compute_drift_metrics(grad_energy_series)
            for metric_name, metric_value in grad_drift.items():
                feat_name = f"energy_gradient_{metric_name}_{scale}"
                features[feat_name] = metric_value

        return features

    def _compute_scale_specific_dissipation(
        self,
        multiscale_fields: Dict[str, torch.Tensor],
        config: Optional['SummaryInvariantDriftConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute scale-specific dissipation features.

        Characterizes energy dissipation across frequency bands:
        - dissipation_rate_lowfreq: Energy decay in low frequencies
        - dissipation_rate_highfreq: Energy decay in high frequencies
        - dissipation_selectivity: Ratio of high/low dissipation rates
        - energy_cascade_direction: Upscale vs. downscale energy transfer

        These features detect frequency-dependent dissipation and energy cascades
        (e.g., turbulent cascades, diffusion selectivity).

        Args:
            multiscale_fields: Dict with 'raw', 'lowpass', 'highpass' fields [N, M, T, C, H, W]
            config: Optional config

        Returns:
            Dictionary with 4 dissipation features [N, M, C]
        """
        features = {}

        # Extract fields
        fields_raw = multiscale_fields['raw']  # [N, M, T, C, H, W]
        N, M, T, C, H, W = fields_raw.shape

        # Compute spectral energy over time
        # Use FFT to decompose into frequency bands
        fields_flat = fields_raw.reshape(N * M * T, C, H, W)

        # Compute 2D FFT
        fft = torch.fft.rfft2(fields_flat, dim=(-2, -1))  # [NMT, C, H, W//2+1]
        power = torch.abs(fft) ** 2  # Spectral power

        # Reshape back to time series
        power_series = power.reshape(N, M, T, C, H, W // 2 + 1)  # [N, M, T, C, H, W//2+1]

        # Define frequency bands
        # Low frequency: radial freq < 0.25 * Nyquist
        # High frequency: radial freq > 0.75 * Nyquist
        freq_y = torch.fft.fftfreq(H, d=1.0, device=fields_raw.device)[:, None]  # [H, 1]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=fields_raw.device)[None, :]  # [1, W//2+1]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)  # [H, W//2+1]

        # Broadcast to match power_series shape [N, M, T, C, H, W//2+1]
        freq_radial = freq_radial.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 1, H, W//2+1]

        # Frequency thresholds
        nyquist_freq = 0.5
        low_freq_threshold = 0.25 * nyquist_freq
        high_freq_threshold = 0.75 * nyquist_freq

        # Masks for frequency bands
        mask_lowfreq = (freq_radial < low_freq_threshold).float()  # [1, 1, 1, 1, H, W//2+1]
        mask_highfreq = (freq_radial > high_freq_threshold).float()  # [1, 1, 1, 1, H, W//2+1]

        # Compute energy in each band over time
        energy_lowfreq = (power_series * mask_lowfreq).sum(dim=(-2, -1))  # [N, M, T, C]
        energy_highfreq = (power_series * mask_highfreq).sum(dim=(-2, -1))  # [N, M, T, C]

        # Compute dissipation rate: -d(log E)/dt
        # For numerical stability, use finite differences
        # dissipation_rate ≈ -(log E[t+1] - log E[t]) / dt

        # Low frequency dissipation rate
        log_energy_low = torch.log(energy_lowfreq + 1e-8)  # [N, M, T, C]
        dlog_dt_low = log_energy_low[:, :, 1:, :] - log_energy_low[:, :, :-1, :]  # [N, M, T-1, C]
        dissipation_rate_lowfreq = -dlog_dt_low.mean(dim=2)  # [N, M, C] (average over time)

        # High frequency dissipation rate
        log_energy_high = torch.log(energy_highfreq + 1e-8)  # [N, M, T, C]
        dlog_dt_high = log_energy_high[:, :, 1:, :] - log_energy_high[:, :, :-1, :]  # [N, M, T-1, C]
        dissipation_rate_highfreq = -dlog_dt_high.mean(dim=2)  # [N, M, C]

        # Dissipation selectivity: ratio of high/low rates
        dissipation_selectivity = dissipation_rate_highfreq / (dissipation_rate_lowfreq + 1e-8)  # [N, M, C]

        # Energy cascade direction: sign of low-freq energy drift
        # Positive = energy accumulating at low freq (downscale cascade)
        # Negative = energy dissipating from low freq (upscale cascade)
        # Use mean drift direction over trajectory
        energy_cascade_direction = torch.sign(dlog_dt_low.mean(dim=2))  # [N, M, C]

        # Store features
        features['dissipation_rate_lowfreq'] = dissipation_rate_lowfreq
        features['dissipation_rate_highfreq'] = dissipation_rate_highfreq
        features['dissipation_selectivity'] = dissipation_selectivity
        features['energy_cascade_direction'] = energy_cascade_direction

        return features
