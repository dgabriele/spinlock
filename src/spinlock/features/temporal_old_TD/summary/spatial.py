"""
Spatial statistics feature extraction.

Extracts spatial features from 2D fields:
- Basic moments (mean, std, variance, skewness, kurtosis, min, max, range)
- Robust statistics (IQR, MAD)
- Gradients (magnitude, directional, anisotropy)
- Curvature (Laplacian, Hessian)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummarySpatialConfig


class SpatialFeatureExtractor:
    """
    Extract spatial statistics features from 2D fields.

    Features are computed per-timestep and can be aggregated temporally.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = SpatialFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize spatial feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W] or [N, T, C, H, W]
        config: Optional['SummarySpatialConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract spatial features from fields.

        Args:
            fields: Input fields [N, M, T, C, H, W] or [N, T, C, H, W]
                N = batch size
                M = num realizations (optional)
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummarySpatialConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors
            Each tensor has shape [N, T, C] (averaged over spatial dimensions)
        """
        # Handle both [N,M,T,C,H,W] and [N,T,C,H,W] inputs
        if fields.ndim == 6:
            # [N, M, T, C, H, W] -> merge M into batch dimension
            N, M, T, C, H, W = fields.shape
            fields = fields.reshape(N * M, T, C, H, W)
            has_realizations = True
        else:
            N, T, C, H, W = fields.shape
            M = 1
            has_realizations = False

        # Reshape to [N*T, C, H, W] for batched computation
        NT = fields.shape[0] * T
        fields_flat = fields.reshape(NT, C, H, W)

        features = {}

        # Use config to determine which features to extract
        if config is None:
            # Default: extract all features
            include_all = True
        else:
            include_all = False

        # Basic moments
        if include_all or (config is not None and config.include_mean):
            features['spatial_mean'] = self._compute_mean(fields_flat)

        if include_all or (config is not None and config.include_variance):
            features['spatial_variance'] = self._compute_variance(fields_flat)

        if include_all or (config is not None and config.include_std):
            features['spatial_std'] = self._compute_std(fields_flat)

        if include_all or (config is not None and config.include_skewness):
            features['spatial_skewness'] = self._compute_skewness(fields_flat)

        if include_all or (config is not None and config.include_kurtosis):
            features['spatial_kurtosis'] = self._compute_kurtosis(fields_flat)

        if include_all or (config is not None and config.include_min):
            features['spatial_min'] = self._compute_min(fields_flat)

        if include_all or (config is not None and config.include_max):
            features['spatial_max'] = self._compute_max(fields_flat)

        if include_all or (config is not None and config.include_range):
            features['spatial_range'] = self._compute_range(fields_flat)

        # Robust statistics
        if include_all or (config is not None and config.include_iqr):
            features['spatial_iqr'] = self._compute_iqr(fields_flat)

        if include_all or (config is not None and config.include_mad):
            features['spatial_mad'] = self._compute_mad(fields_flat)

        # Percentiles (distribution shape characterization)
        if include_all or (config is not None and config.include_percentiles):
            percentile_features = self._compute_percentiles(fields_flat)
            features.update(percentile_features)  # Add all percentile_X features

        # Phase 2 extension: Histogram/occupancy (state space coverage)
        if include_all or (config is not None and config.include_histogram):
            num_bins = config.histogram_num_bins if config is not None else 16
            histogram_features = self._compute_histogram_features(fields_flat, num_bins=num_bins)
            features.update(histogram_features)

        # Gradients
        if include_all or (config is not None and config.include_gradient_magnitude):
            grad_mag = self._compute_gradient_magnitude(fields_flat)
            features['gradient_magnitude_mean'] = grad_mag.mean(dim=(-2, -1))
            features['gradient_magnitude_std'] = grad_mag.std(dim=(-2, -1))
            features['gradient_magnitude_max'] = grad_mag.amax(dim=(-2, -1))

        if include_all or (config is not None and config.include_gradient_x_mean):
            grad_x = self._compute_gradient_x(fields_flat)
            features['gradient_x_mean'] = grad_x.mean(dim=(-2, -1))

        if include_all or (config is not None and config.include_gradient_y_mean):
            grad_y = self._compute_gradient_y(fields_flat)
            features['gradient_y_mean'] = grad_y.mean(dim=(-2, -1))

        if include_all or (config is not None and config.include_gradient_anisotropy):
            features['gradient_anisotropy'] = self._compute_gradient_anisotropy(fields_flat)

        # Curvature
        if include_all or (config is not None and config.include_laplacian):
            laplacian = self._compute_laplacian(fields_flat)
            features['laplacian_mean'] = laplacian.mean(dim=(-2, -1))
            features['laplacian_std'] = laplacian.std(dim=(-2, -1))
            # Normalize energy by grid size to get energy per pixel
            features['laplacian_energy'] = (laplacian ** 2).sum(dim=(-2, -1)) / (H * W)

        # Effective dimensionality (SVD-based intrinsic dimensionality)
        if include_all or (config is not None and getattr(config, 'include_effective_dimensionality', False)):
            eff_dim = self._compute_effective_dimensionality(fields_flat)
            features['effective_rank'] = eff_dim['effective_rank']
            features['participation_ratio'] = eff_dim['participation_ratio']
            features['explained_variance_90'] = eff_dim['explained_variance_90']

        # Gradient saturation (amplitude limiting/thresholding detection)
        if include_all or (config is not None and getattr(config, 'include_gradient_saturation', False)):
            grad_sat = self._compute_gradient_saturation(fields_flat)
            features['gradient_saturation_ratio'] = grad_sat['saturation_ratio']
            features['gradient_flatness'] = grad_sat['flatness']

        # Coherence structure (spatial correlation metrics)
        if include_all or (config is not None and getattr(config, 'include_coherence_structure', False)):
            coherence = self._compute_coherence_structure(fields_flat, H, W)
            features['coherence_length'] = coherence['coherence_length']
            features['correlation_anisotropy'] = coherence['correlation_anisotropy']
            features['structure_factor_peak'] = coherence['structure_factor_peak']

        # Reshape all features back to [N*M, T, C]
        for name, feat in features.items():
            if feat.ndim == 2:  # [NT, C]
                if has_realizations:
                    features[name] = feat.reshape(N, M, T, C)
                else:
                    features[name] = feat.reshape(N, T, C)

        return features

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _adaptive_outlier_clip(
        self,
        values: torch.Tensor,
        iqr_multiplier: float = 10.0
    ) -> torch.Tensor:
        """
        Adaptive outlier clipping based on Interquartile Range (IQR).

        Uses IQR-based bounds that adapt to the actual distribution of values,
        preserving valid extreme values while clipping numerical errors.

        Args:
            values: Tensor of feature values [NT, C] or any shape
            iqr_multiplier: Multiplier for IQR fence (default 10.0)
                - 1.5: Standard outlier detection (aggressive)
                - 3.0: Far outlier detection (moderate)
                - 10.0: Extreme outlier detection (conservative, preserves heavy tails)
                - 15.0: Very conservative (for features like kurtosis with valid extremes)

        Returns:
            Clipped values with same shape as input

        Note:
            - Computes bounds from non-NaN values only
            - If < 4 non-NaN values, returns values unchanged (can't compute quartiles)
            - Adapts to actual data distribution (not hardcoded limits)
        """
        # Flatten to [N] for percentile computation
        original_shape = values.shape
        # Convert to float if needed (quantile requires float/double)
        if not values.is_floating_point():
            values = values.float()

        values_flat = values.flatten()

        # Filter out NaN values
        valid_mask = ~torch.isnan(values_flat)
        valid_values = values_flat[valid_mask]

        # Need at least 4 values to compute quartiles
        if valid_values.numel() < 4:
            return values  # Return unchanged

        # Compute quartiles (robust to outliers)
        q1 = torch.quantile(valid_values, 0.25)
        q3 = torch.quantile(valid_values, 0.75)
        iqr = q3 - q1

        # IQR fence: [Q1 - k*IQR, Q3 + k*IQR]
        # This adapts to the actual spread of the data
        # k=10 means we only clip values 10× IQR away from quartiles
        # (preserves 99.9%+ of valid values, only catches numerical errors)
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Clip values
        values_clipped = torch.clamp(values, min=lower_bound, max=upper_bound)

        return values_clipped

    # =========================================================================
    # Basic Statistics
    # =========================================================================

    def _compute_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial mean."""
        return x.mean(dim=(-2, -1))  # [NT, C]

    def _compute_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial variance."""
        return x.var(dim=(-2, -1))  # [NT, C]

    def _compute_std(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial standard deviation."""
        return x.std(dim=(-2, -1))  # [NT, C]

    def _compute_min(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial minimum."""
        return x.amin(dim=(-2, -1))  # [NT, C]

    def _compute_max(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial maximum."""
        return x.amax(dim=(-2, -1))  # [NT, C]

    def _compute_range(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial range (max - min)."""
        return self._compute_max(x) - self._compute_min(x)

    def _compute_skewness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial skewness (third standardized moment) with robust variance handling.

        Skewness measures asymmetry of the distribution.
        Returns 0 for zero-variance fields (symmetric by definition when all values equal).
        Returns 0 for symmetric distributions (e.g., structured ICs like sine waves).
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)

        # Variance threshold for numerical stability (prevents overflow from near-zero division)
        variance_threshold = 1e-4
        zero_variance_mask = (std < variance_threshold)

        # Standardize with clamped denominator to prevent overflow
        z = (x - mean) / torch.clamp(std, min=variance_threshold)
        skew = (z ** 3).mean(dim=(-2, -1))  # [NT, C]

        # Zero for zero-variance fields (uniform distribution = no asymmetry)
        skew = torch.where(
            zero_variance_mask.squeeze(-1).squeeze(-1),
            torch.zeros_like(skew),
            skew
        )

        # Adaptive outlier protection: clip extreme numerical errors while preserving valid extremes
        # Use IQR-based bounds computed from non-NaN values in current batch
        skew = self._adaptive_outlier_clip(skew, iqr_multiplier=10.0)

        # Final safety: replace any remaining NaN/Inf with 0
        # (can occur for symmetric distributions like structured ICs)
        skew = torch.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0)

        return skew

    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial kurtosis (fourth standardized moment - 3) with robust variance handling.

        Kurtosis measures tail heaviness. Excess kurtosis is 0 for Gaussian.
        Returns 0 for zero-variance fields (degenerate distribution).
        Returns 0 for symmetric distributions (e.g., structured ICs like sine waves).
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)

        # Variance threshold for numerical stability (prevents overflow from near-zero division)
        variance_threshold = 1e-4
        zero_variance_mask = (std < variance_threshold)

        # Standardize with clamped denominator to prevent overflow
        z = (x - mean) / torch.clamp(std, min=variance_threshold)
        kurt = (z ** 4).mean(dim=(-2, -1)) - 3.0  # Excess kurtosis [NT, C]

        # Zero for zero-variance fields (degenerate distribution)
        kurt = torch.where(
            zero_variance_mask.squeeze(-1).squeeze(-1),
            torch.zeros_like(kurt),
            kurt
        )

        # Adaptive outlier protection: clip extreme numerical errors while preserving valid extremes
        # Kurtosis can be very large for heavy-tailed distributions, so use wider multiplier
        kurt = self._adaptive_outlier_clip(kurt, iqr_multiplier=15.0)

        # Final safety: replace any remaining NaN/Inf with 0
        # (can occur for symmetric distributions like structured ICs)
        kurt = torch.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0)

        return kurt

    # =========================================================================
    # Robust Statistics
    # =========================================================================

    def _compute_iqr(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute interquartile range (Q3 - Q1).

        Robust measure of spread.
        """
        # Flatten spatial dimensions
        x_flat = x.flatten(start_dim=2)  # [NT, C, H*W]

        # Compute Q1 (25th percentile) and Q3 (75th percentile)
        q1 = torch.quantile(x_flat, 0.25, dim=2)  # [NT, C]
        q3 = torch.quantile(x_flat, 0.75, dim=2)  # [NT, C]

        iqr = q3 - q1
        return iqr

    def _compute_mad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute median absolute deviation.

        Robust measure of variability: MAD = median(|x - median(x)|)
        """
        # Flatten spatial dimensions
        x_flat = x.flatten(start_dim=2)  # [NT, C, H*W]

        # Compute median
        median = torch.median(x_flat, dim=2, keepdim=True).values  # [NT, C, 1]

        # Compute MAD
        mad = torch.median(torch.abs(x_flat - median), dim=2).values  # [NT, C]

        return mad

    def _compute_percentiles(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute distribution percentiles (5%, 25%, 50%, 75%, 95%).

        Returns percentiles as a dictionary for better distribution characterization
        beyond mean/variance. Useful for detecting asymmetry and tail behavior.

        Args:
            x: Input tensor [NT, C, H, W]

        Returns:
            Dictionary mapping percentile names to tensors [NT, C]
        """
        # Flatten spatial dimensions
        x_flat = x.flatten(start_dim=2)  # [NT, C, H*W]

        # Define percentile levels
        percentiles = [5, 25, 50, 75, 95]

        # Compute all percentiles in one call for efficiency
        # torch.quantile expects quantiles in [0, 1]
        quantiles = torch.tensor([p / 100.0 for p in percentiles], device=x.device)
        # Result shape: [num_quantiles, NT, C]
        percentile_values = torch.quantile(x_flat, quantiles, dim=2)

        # Build dictionary
        result = {}
        for i, p in enumerate(percentiles):
            result[f'percentile_{p}'] = percentile_values[i]  # [NT, C]

        return result

    def _compute_histogram_features(
        self,
        x: torch.Tensor,
        num_bins: int = 16
    ) -> Dict[str, torch.Tensor]:
        """
        Compute histogram/occupancy features (state space coverage).

        Measures how values are distributed across bins, capturing:
        - Histogram entropy: Uniformity of state space coverage
        - Peak bin fraction: Dominance of most common value range
        - Effective bins: Number of bins with significant mass

        Args:
            x: Input tensor [NT, C, H, W]
            num_bins: Number of histogram bins (default: 16)

        Returns:
            Dictionary mapping feature names to tensors [NT, C]
        """
        NT, C, H, W = x.shape

        # Flatten spatial dimensions
        x_flat = x.flatten(start_dim=2)  # [NT, C, H*W]

        result = {}

        # Compute histogram for each (n, c) pair
        for nt in range(NT):
            for c in range(C):
                values = x_flat[nt, c]  # [H*W]

                # Compute histogram (bins between min and max)
                hist = torch.histc(values, bins=num_bins, min=values.min().item(), max=values.max().item())
                hist = hist / hist.sum()  # Normalize to probabilities

                # 1. Histogram entropy: -sum(p * log(p))
                # High entropy → uniform distribution, low entropy → peaked distribution
                nonzero_bins = hist[hist > 1e-10]
                if len(nonzero_bins) > 0:
                    entropy = -(nonzero_bins * torch.log(nonzero_bins)).sum()
                else:
                    entropy = torch.tensor(0.0, device=x.device)

                # 2. Peak bin fraction: Mass in most populated bin
                peak_fraction = hist.max()

                # 3. Effective bins: Number of bins with > 1% of mass
                effective_bins = (hist > 0.01).sum().float()

                # Store features (accumulate across NT, C for efficiency)
                if nt == 0 and c == 0:
                    result['histogram_entropy'] = torch.zeros(NT, C, device=x.device)
                    result['histogram_peak_fraction'] = torch.zeros(NT, C, device=x.device)
                    result['histogram_effective_bins'] = torch.zeros(NT, C, device=x.device)

                result['histogram_entropy'][nt, c] = entropy
                result['histogram_peak_fraction'][nt, c] = peak_fraction
                result['histogram_effective_bins'][nt, c] = effective_bins

        return result

    # =========================================================================
    # Gradients
    # =========================================================================

    def _compute_gradient_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute horizontal gradient using central differences.

        Uses circular padding (periodic boundary conditions).
        """
        # Central differences: (x[i+1] - x[i-1]) / 2
        # Pad with circular boundary
        x_padded = F.pad(x, (1, 1, 0, 0), mode='circular')
        grad_x = (x_padded[:, :, :, 2:] - x_padded[:, :, :, :-2]) * 0.5

        return grad_x

    def _compute_gradient_y(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute vertical gradient using central differences.

        Uses circular padding (periodic boundary conditions).
        """
        # Central differences: (x[j+1] - x[j-1]) / 2
        # Pad with circular boundary
        x_padded = F.pad(x, (0, 0, 1, 1), mode='circular')
        grad_y = (x_padded[:, :, 2:, :] - x_padded[:, :, :-2, :]) * 0.5

        return grad_y

    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude: sqrt(grad_x^2 + grad_y^2).
        """
        grad_x = self._compute_gradient_x(x)
        grad_y = self._compute_gradient_y(x)

        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return grad_mag

    def _compute_gradient_anisotropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient anisotropy: ratio of directional gradient strengths.

        Measures preferential directionality in gradients.
        Returns |grad_x| / |grad_y| (averaged over space).
        """
        grad_x = self._compute_gradient_x(x)
        grad_y = self._compute_gradient_y(x)

        # Mean absolute gradients
        mean_grad_x = grad_x.abs().mean(dim=(-2, -1))
        mean_grad_y = grad_y.abs().mean(dim=(-2, -1))

        # Anisotropy ratio
        anisotropy = mean_grad_x / (mean_grad_y + 1e-8)

        return anisotropy

    # =========================================================================
    # Curvature
    # =========================================================================

    def _compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian (second spatial derivative): ∇²x = ∂²x/∂x² + ∂²x/∂y².

        Uses 5-point stencil with circular boundary conditions.
        """
        # 5-point stencil: Laplacian = (N + S + E + W - 4*C)
        # Pad with circular boundary
        x_padded = F.pad(x, (1, 1, 1, 1), mode='circular')

        # Extract neighbors
        center = x_padded[:, :, 1:-1, 1:-1]  # Center
        north = x_padded[:, :, :-2, 1:-1]    # i-1, j
        south = x_padded[:, :, 2:, 1:-1]     # i+1, j
        west = x_padded[:, :, 1:-1, :-2]     # i, j-1
        east = x_padded[:, :, 1:-1, 2:]      # i, j+1

        laplacian = north + south + west + east - 4 * center

        return laplacian

    # =========================================================================
    # Effective Dimensionality (SVD-based intrinsic dimensionality)
    # =========================================================================

    def _compute_effective_dimensionality(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute effective dimensionality via SVD-based measures (GPU-optimized batched version).

        Measures intrinsic dimensionality of spatial structure:
        - Effective rank: Stable rank (sum of singular values squared / largest SV squared)
        - Participation ratio: Inverse Simpson index of SV distribution
        - Explained variance 90: Number of SVs needed to explain 90% variance

        Args:
            x: Input fields [NT, C, H, W]

        Returns:
            Dict with keys: effective_rank, participation_ratio, explained_variance_90
            Each has shape [NT, C]
        """
        NT, C, H, W = x.shape

        # Reshape to [NT*C, H, W] for batched processing
        x_flat = x.reshape(NT * C, H, W)

        # Batched SVD: Process all samples at once
        # torch.linalg.svd can handle batched input [N, H, W]
        U, S, Vh = torch.linalg.svd(x_flat, full_matrices=False)
        # S has shape [NT*C, K] where K = min(H, W)

        # Vectorized metric computation (no loops!)
        S_squared = S ** 2  # [NT*C, K]
        total_variance = S_squared.sum(dim=1, keepdim=True)  # [NT*C, 1]

        # Normalize to get variance explained per sample
        variance_explained = S_squared / (total_variance + 1e-8)  # [NT*C, K]

        # 1. Effective rank (stable rank): sum(S²) / max(S²)
        # For each sample: sum over K dimension, divide by first element
        effective_rank = S_squared.sum(dim=1) / (S_squared[:, 0] + 1e-8)  # [NT*C]

        # 2. Participation ratio: 1 / sum(p_i²) where p_i are normalized SVs
        participation_ratio = 1.0 / ((variance_explained ** 2).sum(dim=1) + 1e-8)  # [NT*C]

        # 3. Explained variance 90: Number of SVs needed to explain 90% variance
        cumulative_variance = torch.cumsum(variance_explained, dim=1)  # [NT*C, K]
        # Find first index where cumulative variance >= 0.9 for each sample
        # Count how many values are < 0.9, then add 1
        explained_variance_90 = (cumulative_variance < 0.9).sum(dim=1).float() + 1.0  # [NT*C]

        # Reshape back to [NT, C]
        effective_rank = effective_rank.reshape(NT, C)
        participation_ratio = participation_ratio.reshape(NT, C)
        explained_variance_90 = explained_variance_90.reshape(NT, C)

        return {
            'effective_rank': effective_rank,
            'participation_ratio': participation_ratio,
            'explained_variance_90': explained_variance_90,
        }

    # =========================================================================
    # Gradient Saturation (amplitude limiting/thresholding detection)
    # =========================================================================

    def _compute_gradient_saturation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gradient saturation features.

        Detects amplitude limiting/thresholding behavior:
        - Saturation ratio: Fraction of pixels with low gradients (flat regions)
        - Flatness: Kurtosis of gradient distribution (heavy tails = intermittent)

        Args:
            x: Input fields [NT, C, H, W]

        Returns:
            Dict with keys: saturation_ratio, flatness
            Each has shape [NT, C]
        """
        # Compute gradient magnitude
        grad_mag = self._compute_gradient_magnitude(x)  # [NT, C, H, W]

        # 1. Saturation ratio: fraction of pixels with gradient < threshold
        # Use adaptive threshold: 5% of the maximum gradient per sample
        grad_max = grad_mag.amax(dim=(-2, -1), keepdim=True)  # [NT, C, 1, 1]
        threshold = 0.05 * grad_max
        saturation_mask = grad_mag < threshold  # [NT, C, H, W]
        saturation_ratio = saturation_mask.float().mean(dim=(-2, -1))  # [NT, C]

        # 2. Gradient flatness: kurtosis of gradient magnitude distribution
        # High kurtosis = heavy tails (intermittent sharp edges + flat regions)
        # Low kurtosis = uniform gradients
        grad_mag_flat = grad_mag.flatten(start_dim=2)  # [NT, C, H*W]
        mean = grad_mag_flat.mean(dim=2, keepdim=True)  # [NT, C, 1]
        std = grad_mag_flat.std(dim=2, keepdim=True)  # [NT, C, 1]
        z = (grad_mag_flat - mean) / (std + 1e-8)  # Standardize
        flatness = (z ** 4).mean(dim=2) - 3.0  # Excess kurtosis [NT, C]

        return {
            'saturation_ratio': saturation_ratio,
            'flatness': flatness,
        }

    # =========================================================================
    # Coherence Structure (spatial correlation metrics)
    # =========================================================================

    def _compute_coherence_structure(
        self,
        x: torch.Tensor,  # [NT, C, H, W]
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial coherence structure features.

        Measures spatial correlation and characteristic length scales:
        - Coherence length: Autocorrelation decay length
        - Correlation anisotropy: Directional bias in correlations
        - Structure factor peak: Characteristic length scale from power spectrum

        Args:
            x: Input fields [NT, C, H, W]
            H, W: Spatial dimensions

        Returns:
            Dict with keys: coherence_length, correlation_anisotropy, structure_factor_peak
            Each has shape [NT, C]
        """
        NT, C, H_in, W_in = x.shape

        # 1. Coherence length: Autocorrelation decay length
        # Compute 2D autocorrelation via FFT
        # R(r) = F^{-1}[|F[x]|²] (Wiener-Khinchin theorem)
        fft = torch.fft.rfft2(x, dim=(-2, -1))  # [NT, C, H, W//2+1]
        power = torch.abs(fft) ** 2  # Power spectrum
        autocorr = torch.fft.irfft2(power, s=(H, W), dim=(-2, -1))  # [NT, C, H, W]

        # Normalize autocorrelation by zero-lag value
        autocorr_normalized = autocorr / (autocorr[:, :, 0:1, 0:1] + 1e-8)  # [NT, C, H, W]

        # Compute radial autocorrelation profile
        # Create radial distance grid
        y = torch.arange(H, device=x.device, dtype=torch.float32) - H // 2
        x_coord = torch.arange(W, device=x.device, dtype=torch.float32) - W // 2
        Y, X = torch.meshgrid(y, x_coord, indexing='ij')
        r = torch.sqrt(Y ** 2 + X ** 2)  # Radial distance from center

        # Shift autocorrelation to center zero-lag
        autocorr_centered = torch.fft.fftshift(autocorr_normalized, dim=(-2, -1))  # [NT, C, H, W]

        # Find coherence length: where autocorrelation drops to 1/e ≈ 0.368 (batched version)
        threshold = 1.0 / torch.e

        # Batched computation: Process all samples at once
        autocorr_flat = autocorr_centered.reshape(NT * C, H, W)  # [NT*C, H, W]

        # Find where autocorr drops below threshold for all samples
        below_threshold = (autocorr_flat < threshold).float()  # [NT*C, H, W]

        # Mask radial distances where autocorr < threshold
        # Broadcast r to match batch dimension: [NT*C, H, W]
        r_expanded = r.unsqueeze(0).expand(NT * C, -1, -1)  # [NT*C, H, W]
        r_masked = r_expanded * below_threshold + 1e6 * (1 - below_threshold)  # [NT*C, H, W]

        # Coherence length = minimum radius where below threshold (per sample)
        coherence_length = r_masked.amin(dim=(-2, -1))  # [NT*C]

        # If never drops below threshold, use max radius (vectorized)
        max_radius = torch.tensor(max(H, W) / 2.0, device=x.device)
        coherence_length = torch.where(
            coherence_length > 1e5,
            max_radius,
            coherence_length
        )  # [NT*C]

        coherence_length = coherence_length.reshape(NT, C)  # [NT, C]

        # 2. Correlation anisotropy: Directional bias
        # Compute autocorrelation in x and y directions
        autocorr_x = autocorr_centered[:, :, H//2, :].abs().mean(dim=-1)  # [NT, C] (average along x-axis)
        autocorr_y = autocorr_centered[:, :, :, W//2].abs().mean(dim=-1)  # [NT, C] (average along y-axis)

        # Anisotropy ratio
        correlation_anisotropy = autocorr_x / (autocorr_y + 1e-8)  # [NT, C]

        # 3. Structure factor peak: Characteristic length scale from power spectrum (batched version)
        # Find dominant spatial frequency (peak in radial power spectrum)
        # Create radial frequency grid
        freq_y = torch.fft.fftfreq(H, d=1.0, device=x.device)[:, None]  # [H, 1]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=x.device)[None, :]  # [1, W//2+1]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)  # [H, W//2+1]

        # Batched computation: Process all samples at once
        power_flat = power.reshape(NT * C, H, W // 2 + 1)  # [NT*C, H, W//2+1]

        # Find peak (ignore DC component at freq=0)
        mask_nonzero = (freq_radial > 0.01).float()  # [H, W//2+1]
        # Broadcast mask to all samples
        power_masked = power_flat * mask_nonzero.unsqueeze(0)  # [NT*C, H, W//2+1]

        # Find frequency of peak power for all samples
        power_masked_flat = power_masked.reshape(NT * C, -1)  # [NT*C, H*(W//2+1)]
        max_indices = power_masked_flat.argmax(dim=1)  # [NT*C]

        # Convert flat indices to 2D coordinates
        max_idx_y = max_indices // (W // 2 + 1)  # [NT*C]
        max_idx_x = max_indices % (W // 2 + 1)   # [NT*C]

        # Get peak frequencies using advanced indexing
        # Need to flatten freq_radial and index into it
        freq_radial_flat = freq_radial.flatten()  # [H*(W//2+1)]
        peak_freq = freq_radial_flat[max_indices]  # [NT*C]

        # Characteristic length = 1 / peak_freq
        structure_factor_peak = 1.0 / (peak_freq + 1e-8)  # [NT*C]

        structure_factor_peak = structure_factor_peak.reshape(NT, C)  # [NT, C]

        return {
            'coherence_length': coherence_length,
            'correlation_anisotropy': correlation_anisotropy,
            'structure_factor_peak': structure_factor_peak,
        }

    def aggregate_temporal(
        self,
        features: Dict[str, torch.Tensor],
        methods: list = ['mean', 'std']
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate per-timestep features over time.

        Args:
            features: Dict of features with shape [N, T, C] or [N, M, T, C]
            methods: Aggregation methods ('mean', 'std', 'min', 'max', 'final')

        Returns:
            Aggregated features dict with shape [N, C] or [N, M, C]
        """
        aggregated = {}

        for name, feat in features.items():
            # Determine time dimension
            if feat.ndim == 3:  # [N, T, C]
                time_dim = 1
            elif feat.ndim == 4:  # [N, M, T, C]
                time_dim = 2
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")

            for method in methods:
                agg_name = f"{name}_{method}"

                if method == 'mean':
                    aggregated[agg_name] = feat.mean(dim=time_dim)
                elif method == 'std':
                    aggregated[agg_name] = feat.std(dim=time_dim)
                elif method == 'min':
                    aggregated[agg_name] = feat.amin(dim=time_dim)
                elif method == 'max':
                    aggregated[agg_name] = feat.amax(dim=time_dim)
                elif method == 'final':
                    # Extract final timestep
                    if feat.ndim == 3:
                        aggregated[agg_name] = feat[:, -1, :]
                    else:
                        aggregated[agg_name] = feat[:, :, -1, :]
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

        return aggregated
