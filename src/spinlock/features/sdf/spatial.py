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
    from spinlock.features.sdf.config import SDFSpatialConfig


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
        config: Optional['SDFSpatialConfig'] = None
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
            config: Optional SDFSpatialConfig for feature selection

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
        Compute spatial skewness (third standardized moment).

        Skewness measures asymmetry of the distribution.
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        z = (x - mean) / (std + 1e-8)  # Standardize
        skew = (z ** 3).mean(dim=(-2, -1))  # [NT, C]
        return skew

    def _compute_kurtosis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial kurtosis (fourth standardized moment - 3).

        Kurtosis measures tail heaviness. Excess kurtosis is 0 for Gaussian.
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        z = (x - mean) / (std + 1e-8)  # Standardize
        kurt = (z ** 4).mean(dim=(-2, -1)) - 3.0  # Excess kurtosis
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
        Compute effective dimensionality via SVD-based measures.

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

        # Reshape to [NT*C, H, W]
        x_flat = x.reshape(NT * C, H, W)

        # Flatten spatial dimensions: [NT*C, H*W]
        x_matrix = x_flat.reshape(NT * C, H * W)

        # Process each sample separately (SVD per sample)
        # Need to loop because torch.linalg.svd doesn't batch well for this use case
        effective_rank_list = []
        participation_ratio_list = []
        explained_variance_90_list = []

        for i in range(NT * C):
            sample = x_matrix[i:i+1, :]  # [1, H*W]

            try:
                # Compute SVD for single sample
                U, S, Vh = torch.linalg.svd(sample, full_matrices=False)
                # S has shape [min(1, H*W)] = [1] for single sample
                # This is wrong - we need the SVD across spatial dimensions

                # Actually, for spatial structure analysis, we want SVD of the spatial field
                # Reshape back to [H, W] and compute SVD
                sample_2d = x_flat[i].unsqueeze(0)  # [1, H, W]
                sample_2d = sample_2d.reshape(H, W)  # [H, W]

                # Compute SVD of 2D field
                U, S, Vh = torch.linalg.svd(sample_2d, full_matrices=False)
                # S has shape [min(H, W)]

            except RuntimeError:
                # Fallback: use smaller truncated SVD if OOM
                print("⚠️  SVD OOM, using randomized truncated SVD")
                from torch.svd_lowrank import svd_lowrank
                k = min(50, min(H, W))
                U, S, Vh = svd_lowrank(sample_2d, q=k)

            # S is 1D tensor of singular values [K] where K = min(H, W)
            S_squared = S ** 2
            total_variance = S_squared.sum()  # scalar

            # Normalize to get variance explained
            variance_explained = S_squared / (total_variance + 1e-8)  # [K]

            # 1. Effective rank (stable rank): sum(S²) / max(S²)
            eff_rank = S_squared.sum() / (S_squared[0] + 1e-8)  # scalar

            # 2. Participation ratio: 1 / sum(p_i²) where p_i are normalized SVs
            part_ratio = 1.0 / (variance_explained ** 2).sum()  # scalar

            # 3. Explained variance 90: Number of SVs needed to explain 90% variance
            cumulative_variance = torch.cumsum(variance_explained, dim=0)  # [K]
            # Find first index where cumulative variance > 0.9
            exp_var_90 = (cumulative_variance < 0.9).sum().float() + 1.0  # scalar

            effective_rank_list.append(eff_rank)
            participation_ratio_list.append(part_ratio)
            explained_variance_90_list.append(exp_var_90)

        # Stack results
        effective_rank = torch.stack(effective_rank_list)  # [NT*C]
        participation_ratio = torch.stack(participation_ratio_list)  # [NT*C]
        explained_variance_90 = torch.stack(explained_variance_90_list)  # [NT*C]

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

        # Find coherence length: where autocorrelation drops to 1/e ≈ 0.368
        threshold = 1.0 / torch.e

        # For each sample/channel, find minimum radius where autocorr < threshold
        coherence_length_list = []

        for i in range(NT * C):
            sample = autocorr_centered.reshape(NT * C, H, W)[i]  # [H, W]

            # Find where autocorr drops below threshold
            below_threshold = (sample < threshold).float()

            # Mask radial distances where autocorr < threshold
            r_masked = r * below_threshold + 1e6 * (1 - below_threshold)  # Large value for above threshold

            # Coherence length = minimum radius where below threshold
            coh_len = r_masked.min()

            # If never drops below threshold, use max radius
            if coh_len > 1e5:
                coh_len = torch.tensor(max(H, W) / 2.0, device=x.device)

            coherence_length_list.append(coh_len)

        coherence_length = torch.stack(coherence_length_list).reshape(NT, C)  # [NT, C]

        # 2. Correlation anisotropy: Directional bias
        # Compute autocorrelation in x and y directions
        autocorr_x = autocorr_centered[:, :, H//2, :].abs().mean(dim=-1)  # [NT, C] (average along x-axis)
        autocorr_y = autocorr_centered[:, :, :, W//2].abs().mean(dim=-1)  # [NT, C] (average along y-axis)

        # Anisotropy ratio
        correlation_anisotropy = autocorr_x / (autocorr_y + 1e-8)  # [NT, C]

        # 3. Structure factor peak: Characteristic length scale from power spectrum
        # Find dominant spatial frequency (peak in radial power spectrum)
        # Create radial frequency grid
        freq_y = torch.fft.fftfreq(H, d=1.0, device=x.device)[:, None]  # [H, 1]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=x.device)[None, :]  # [1, W//2+1]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)  # [H, W//2+1]

        # Compute radial power spectrum
        # For each sample/channel, find peak frequency
        structure_factor_peak_list = []

        for i in range(NT * C):
            power_sample = power.reshape(NT * C, H, W // 2 + 1)[i]  # [H, W//2+1]

            # Find peak (ignore DC component at freq=0)
            mask_nonzero = (freq_radial > 0.01).float()  # Exclude near-zero frequencies
            power_masked = power_sample * mask_nonzero

            # Find frequency of peak power
            max_idx = power_masked.flatten().argmax()
            max_idx_y = max_idx // (W // 2 + 1)
            max_idx_x = max_idx % (W // 2 + 1)

            peak_freq = freq_radial[max_idx_y, max_idx_x]

            # Characteristic length = 1 / peak_freq
            char_length = 1.0 / (peak_freq + 1e-8)

            structure_factor_peak_list.append(char_length)

        structure_factor_peak = torch.stack(structure_factor_peak_list).reshape(NT, C)  # [NT, C]

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
