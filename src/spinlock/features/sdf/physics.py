"""
Statistical physics feature extraction.

Extracts physics-inspired features from 2D fields:
- Spatial correlations (correlation length, peak correlation)
- Structure factor (S(k) peak, width, integral)
- Density fluctuations (variance/mean², compressibility)
- Clustering coefficient (local density correlation)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFPhysicsConfig


class PhysicsFeatureExtractor:
    """
    Extract statistical physics features from 2D fields.

    Features measure correlations, structure, and fluctuations inspired
    by condensed matter physics and statistical mechanics.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = PhysicsFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize physics feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

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
        fields: torch.Tensor,  # [N, T, C, H, W]
        config: Optional['SDFPhysicsConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract physics features from fields.

        Args:
            fields: Input fields [N, T, C, H, W]
                N = batch size
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SDFPhysicsConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors
            Each tensor has shape [N, T, C]
        """
        N, T, C, H, W = fields.shape

        # Reshape to [N*T, C, H, W] for batched computation
        NT = N * T
        fields_flat = fields.reshape(NT, C, H, W)

        features = {}

        # Use config to determine which features to extract
        if config is None:
            # Default: extract all features
            include_all = True
        else:
            include_all = False

        # Spatial correlation features
        if include_all or (config is not None and (
            config.include_correlation_length or
            config.include_correlation_peak
        )):
            corr_features = self._compute_correlation_features(fields_flat)
            features.update(corr_features)

        # Structure factor features
        if include_all or (config is not None and (
            config.include_structure_factor_peak or
            config.include_structure_factor_width or
            config.include_structure_factor_integral
        )):
            sf_features = self._compute_structure_factor_features(fields_flat)
            features.update(sf_features)

        # Density fluctuation features
        if include_all or (config is not None and (
            config.include_density_fluctuation or
            config.include_compressibility_proxy
        )):
            density_features = self._compute_density_fluctuations(fields_flat)
            features.update(density_features)

        # Clustering features
        if include_all or (config is not None and config.include_clustering_coefficient):
            clustering_features = self._compute_clustering_coefficient(fields_flat)
            features.update(clustering_features)

        # Reshape all features from [NT, C] -> [N, T, C]
        for key in features:
            if features[key].shape[0] == NT:
                features[key] = features[key].reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Spatial Correlations
    # =========================================================================

    def _compute_correlation_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute radial correlation function and extract correlation length.

        Correlation function: C(r) = <f(x)f(x+r)> - <f>^2
        Correlation length: Fit C(r) ~ exp(-r/ξ) to extract ξ

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - correlation_length: Characteristic decay length
                - correlation_peak: Maximum correlation at non-zero lag
        """
        NT, C, H, W = x.shape

        corr_length = torch.zeros(NT, C, device=self.device)
        corr_peak = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            # Extract channel
            x_c = x[:, c, :, :]  # [NT, H, W]

            # Compute autocorrelation via FFT
            # C(r) = IFFT(|FFT(f)|^2) / N
            fft = torch.fft.rfft2(x_c, dim=(-2, -1))  # [NT, H, W//2+1]
            power = torch.abs(fft) ** 2
            autocorr = torch.fft.irfft2(power, s=(H, W), dim=(-2, -1))  # [NT, H, W]

            # Normalize
            autocorr = autocorr / (H * W)

            # Shift zero-frequency to center
            autocorr = torch.fft.fftshift(autocorr, dim=(-2, -1))

            # Compute radial average
            center_y, center_x = H // 2, W // 2

            # Create radial distance map
            y_coords = torch.arange(H, device=self.device) - center_y
            x_coords = torch.arange(W, device=self.device) - center_x
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            r_map = torch.sqrt(xx**2 + yy**2)  # [H, W]

            # Radially average (simple binning)
            max_r = int(min(H, W) // 2)
            radial_profile = torch.zeros(NT, max_r, device=self.device)

            for r_idx in range(max_r):
                mask = (r_map >= r_idx) & (r_map < r_idx + 1)
                if mask.sum() > 0:
                    radial_profile[:, r_idx] = autocorr[:, mask].mean(dim=1)

            # Find correlation length via exponential decay fit
            # C(r) ~ exp(-r/ξ) => log(C(r)) ~ -r/ξ
            # Use first 1/4 of profile for fitting
            fit_range = max_r // 4
            if fit_range > 2:
                r_vals = torch.arange(1, fit_range, device=self.device).float()

                for i in range(NT):
                    c_vals = radial_profile[i, 1:fit_range]

                    # Skip if correlation is negative or zero
                    if (c_vals > 1e-6).any():
                        log_c = torch.log(torch.clamp(c_vals, min=1e-6))

                        # Linear fit: log(C) = a - r/ξ
                        # Correlation length ξ = -1 / slope
                        # Simple least squares
                        mean_r = r_vals.mean()
                        mean_logc = log_c.mean()
                        slope = ((r_vals - mean_r) * (log_c - mean_logc)).sum() / ((r_vals - mean_r) ** 2).sum()

                        if slope < -1e-6:
                            corr_length[i, c] = -1.0 / slope
                            # Normalize by grid size
                            corr_length[i, c] = corr_length[i, c] / max_r
                        else:
                            corr_length[i, c] = 1.0  # No decay
                    else:
                        corr_length[i, c] = 0.0

                # Correlation peak (max at r > 0)
                if max_r > 1:
                    corr_peak[:, c] = radial_profile[:, 1:].max(dim=1)[0]

        # Clamp correlation length to reasonable range
        corr_length = torch.clamp(corr_length, min=0.0, max=2.0)

        return {
            'correlation_length': corr_length,
            'correlation_peak': corr_peak
        }

    # =========================================================================
    # Structure Factor
    # =========================================================================

    def _compute_structure_factor_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute structure factor S(k) and extract features.

        Structure factor: S(k) = |FFT(f)|^2 / N
        Features: peak location, width, integral

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - structure_factor_peak_k: k-value at S(k) maximum
                - structure_factor_peak_value: Maximum S(k) value
                - structure_factor_width: FWHM of peak
                - structure_factor_integral: Total integrated S(k)
        """
        NT, C, H, W = x.shape

        peak_k = torch.zeros(NT, C, device=self.device)
        peak_value = torch.zeros(NT, C, device=self.device)
        sf_width = torch.zeros(NT, C, device=self.device)
        sf_integral = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            # Extract channel
            x_c = x[:, c, :, :]  # [NT, H, W]

            # Compute structure factor
            fft = torch.fft.rfft2(x_c, dim=(-2, -1))  # [NT, H, W//2+1]
            sf = torch.abs(fft) ** 2 / (H * W)

            # Radially average S(k)
            # Create k-space radial distance map
            ky = torch.fft.fftfreq(H, d=1.0, device=self.device) * 2 * np.pi
            kx = torch.fft.rfftfreq(W, d=1.0, device=self.device) * 2 * np.pi
            kyy, kxx = torch.meshgrid(ky, kx, indexing='ij')
            k_map = torch.sqrt(kxx**2 + kyy**2)  # [H, W//2+1]

            # Radially average
            max_k = k_map.max()
            num_k_bins = min(50, int(max_k * min(H, W) / (2 * np.pi)))

            if num_k_bins > 2:
                k_bins = torch.linspace(0, max_k, num_k_bins, device=self.device)
                radial_sf = torch.zeros(NT, num_k_bins, device=self.device)

                for k_idx in range(num_k_bins - 1):
                    k_min = k_bins[k_idx]
                    k_max = k_bins[k_idx + 1]
                    mask = (k_map >= k_min) & (k_map < k_max)

                    if mask.sum() > 0:
                        radial_sf[:, k_idx] = sf[:, mask].mean(dim=1)

                # Find peak (skip k=0)
                if num_k_bins > 2:
                    peak_idx = radial_sf[:, 1:].argmax(dim=1) + 1
                    for i in range(NT):
                        peak_k[i, c] = k_bins[peak_idx[i]] / max_k  # Normalize
                        peak_value[i, c] = radial_sf[i, peak_idx[i]]

                    # Estimate FWHM (half-max width)
                    for i in range(NT):
                        half_max = peak_value[i, c] / 2.0
                        profile = radial_sf[i, :]

                        # Find indices where profile crosses half-max
                        above_half = profile > half_max
                        if above_half.sum() > 1:
                            left = torch.where(above_half)[0].min()
                            right = torch.where(above_half)[0].max()
                            sf_width[i, c] = (k_bins[right] - k_bins[left]) / max_k
                        else:
                            sf_width[i, c] = 0.1  # Narrow peak

                    # Integral of S(k) (sum of radial profile)
                    sf_integral[:, c] = radial_sf.mean(dim=1)

        return {
            'structure_factor_peak_k': peak_k,
            'structure_factor_peak_value': peak_value,
            'structure_factor_width': sf_width,
            'structure_factor_integral': sf_integral
        }

    # =========================================================================
    # Density Fluctuations
    # =========================================================================

    def _compute_density_fluctuations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute density fluctuation measures.

        Density fluctuation: σ²/<ρ>²
        Compressibility: χ ~ <δρ²> / <ρ>

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - density_fluctuation: Normalized variance
                - compressibility_proxy: Susceptibility-like measure
        """
        NT, C, H, W = x.shape

        # Spatial mean (density)
        mean = x.mean(dim=(-2, -1))  # [NT, C]

        # Spatial variance (fluctuations)
        variance = x.var(dim=(-2, -1))  # [NT, C]

        # Density fluctuation: σ²/<ρ>²
        density_fluctuation = variance / (mean**2 + 1e-8)

        # Compressibility proxy: <δρ²> / <ρ>
        compressibility = variance / (mean.abs() + 1e-8)

        # Clamp to reasonable range
        density_fluctuation = torch.clamp(density_fluctuation, min=0.0, max=100.0)
        compressibility = torch.clamp(compressibility, min=0.0, max=100.0)

        return {
            'density_fluctuation': density_fluctuation,
            'compressibility_proxy': compressibility
        }

    # =========================================================================
    # Clustering
    # =========================================================================

    def _compute_clustering_coefficient(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute clustering coefficient (local density correlation).

        Measures how correlated local densities are with their neighbors.
        Uses spatial covariance of local averages.

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - clustering_coefficient: Local correlation measure
        """
        NT, C, H, W = x.shape

        clustering = torch.zeros(NT, C, device=self.device)

        # Compute local averages via 3×3 convolution
        kernel_size = 3
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size**2)

        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [NT, 1, H, W]

            # Local average
            local_avg = F.conv2d(x_c, kernel, padding=kernel_size//2)  # [NT, 1, H, W]

            # Compute spatial autocorrelation of local averages
            # Correlation between local_avg and its shifted version
            # Use horizontal shift by 1 pixel
            left = local_avg[:, :, :, :-1]
            right = local_avg[:, :, :, 1:]

            # Pearson correlation
            mean_left = left.mean(dim=(-2, -1), keepdim=True)
            mean_right = right.mean(dim=(-2, -1), keepdim=True)

            cov = ((left - mean_left) * (right - mean_right)).mean(dim=(-2, -1))
            std_left = left.std(dim=(-2, -1))
            std_right = right.std(dim=(-2, -1))

            corr = cov / (std_left * std_right + 1e-8)
            clustering[:, c] = corr.squeeze()

        # Clamp to [-1, 1]
        clustering = torch.clamp(clustering, min=-1.0, max=1.0)

        return {
            'clustering_coefficient': clustering
        }
