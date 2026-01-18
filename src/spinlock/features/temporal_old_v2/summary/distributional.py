"""
Distributional feature extraction.

Extracts complexity and information-theoretic features from 2D fields:
- Sample entropy (regularity measure)
- Approximate entropy (pattern predictability)
- SVD entropy (entropy of singular value spectrum)
- Participation ratio (effective dimensionality)
- PCA compression ratio (variance capture efficiency)
- Multiscale entropy (complexity across scales)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryDistributionalConfig


class DistributionalFeatureExtractor:
    """
    Extract distributional and information-theoretic features from 2D fields.

    Features measure complexity, regularity, and effective dimensionality.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = DistributionalFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize distributional feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def _adaptive_outlier_clip(
        self,
        values: torch.Tensor,
        iqr_multiplier: float = 10.0
    ) -> torch.Tensor:
        """
        Adaptive outlier clipping based on Interquartile Range (IQR).

        Args:
            values: Input tensor (any shape)
            iqr_multiplier: Multiplier for IQR fence (default: 10.0 for conservative clipping)

        Returns:
            Clipped tensor with same shape as input
        """
        # Flatten and filter NaN
        original_shape = values.shape
        # Convert to float if needed (quantile requires float/double)
        if not values.is_floating_point():
            values = values.float()

        values_flat = values.flatten()
        valid_mask = ~torch.isnan(values_flat)
        valid_values = values_flat[valid_mask]

        if valid_values.numel() < 4:
            return values  # Can't compute quartiles

        # Compute quartiles (robust to outliers)
        q1 = torch.quantile(valid_values, 0.25)
        q3 = torch.quantile(valid_values, 0.75)
        iqr = q3 - q1

        # IQR fence: [Q1 - k*IQR, Q3 + k*IQR]
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # Clip values
        values_clipped = torch.clamp(values, min=lower_bound, max=upper_bound)

        return values_clipped

    def extract(
        self,
        fields: torch.Tensor,  # [N, T, C, H, W]
        config: Optional['SummaryDistributionalConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract distributional features from fields.

        Args:
            fields: Input fields [N, T, C, H, W]
                N = batch size
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryDistributionalConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors
            Each tensor has shape [N, T, C] (averaged over spatial dimensions)
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

        # Entropy measures
        if include_all or (config is not None and config.include_sample_entropy):
            features['sample_entropy'] = self._compute_sample_entropy(fields_flat)

        if include_all or (config is not None and config.include_approximate_entropy):
            features['approximate_entropy'] = self._compute_approximate_entropy(fields_flat)

        # OPTIMIZATION: Compute SVD once for all SVD-based features
        # Three features use SVD: svd_entropy, participation_ratio, pca_compression_ratio
        need_svd = (
            include_all or
            (config is not None and (
                config.include_svd_entropy or
                config.include_participation_ratio or
                config.include_compression_ratio_pca
            ))
        )

        if need_svd:
            # Compute SVD once: reshape to [NT*C, H, W]
            x_batched = fields_flat.reshape(NT * C, H, W)
            try:
                U, S, Vh = torch.linalg.svd(x_batched, full_matrices=False)
                svd_available = True
            except Exception:
                # SVD failed - features will use fallback
                U, S, Vh = None, None, None
                svd_available = False
        else:
            U, S, Vh = None, None, None
            svd_available = False

        # SVD-based features (reuse cached SVD)
        if include_all or (config is not None and config.include_svd_entropy):
            features['svd_entropy'] = self._compute_svd_entropy(
                fields_flat, cached_svd=(U, S, Vh) if svd_available else None
            )

        # Dimensionality measures
        if include_all or (config is not None and config.include_participation_ratio):
            features['participation_ratio'] = self._compute_participation_ratio(
                fields_flat, cached_svd=(U, S, Vh) if svd_available else None
            )

        if include_all or (config is not None and config.include_compression_ratio_pca):
            features['pca_compression_ratio'] = self._compute_pca_compression_ratio(
                fields_flat, cached_svd=(U, S, Vh) if svd_available else None
            )

        # Multiscale entropy (controlled by num_entropy_scales)
        if include_all or (config is not None and config.include_entropy and config.num_entropy_scales > 1):
            # Use first num_entropy_scales from [2, 4, 8]
            num_scales = config.num_entropy_scales if config else 3
            scales = [2, 4, 8][:num_scales]
            multiscale_features = self._compute_multiscale_entropy(fields_flat, scales=scales)
            features.update(multiscale_features)

        # Reshape all features from [NT, C] -> [N, T, C]
        for key in features:
            if features[key].shape[0] == NT:
                features[key] = features[key].reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Entropy Measures
    # =========================================================================

    def _compute_sample_entropy(
        self,
        x: torch.Tensor,
        m: int = 2,
        r: float = 0.2
    ) -> torch.Tensor:
        """
        Compute sample entropy (regularity measure).

        Sample entropy measures the regularity of a time series by counting
        similar patterns of length m vs m+1. Lower values = more regular.

        Args:
            x: [NT, C, H, W] input fields
            m: Pattern length (default 2)
            r: Tolerance threshold as fraction of std (default 0.2)

        Returns:
            Sample entropy [NT, C]
        """
        NT, C, H, W = x.shape

        # Flatten spatial dimensions to create 1D signal per channel
        x_flat = x.reshape(NT, C, H * W)  # [NT, C, N_points]

        # Compute sample entropy per channel
        entropy = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            for i in range(NT):
                signal = x_flat[i, c, :]  # [N_points]

                # Use sliding window to get sufficient samples
                # For H×W = 64×64 = 4096 points, subsample to ~256 for speed
                if len(signal) > 256:
                    indices = torch.linspace(0, len(signal) - 1, 256, device=self.device).long()
                    signal = signal[indices]

                # Compute threshold
                r_threshold = r * signal.std()

                # Count template matches for m and m+1
                # Simplified: use autocorrelation-based approximation
                # True sample entropy requires template matching (too slow)
                # Approximation: -log(autocorr at lag m+1 / autocorr at lag m)

                if len(signal) > m + 1:
                    autocorr = self._compute_autocorr_for_entropy(signal, max_lag=m+1)

                    # Sample entropy ≈ -log(autocorr[m+1] / autocorr[m])
                    if autocorr[m] > 1e-6 and autocorr[m+1] > 1e-6:
                        sample_ent = -torch.log(autocorr[m+1] / (autocorr[m] + 1e-8))
                        entropy[i, c] = torch.clamp(sample_ent, min=0.0, max=10.0)
                    else:
                        entropy[i, c] = 0.0
                else:
                    entropy[i, c] = 0.0

        return entropy

    def _compute_autocorr_for_entropy(
        self,
        signal: torch.Tensor,
        max_lag: int
    ) -> torch.Tensor:
        """Helper: compute autocorrelation for entropy estimation."""
        N = len(signal)
        mean = signal.mean()
        var = signal.var()

        autocorr = torch.zeros(max_lag + 1, device=signal.device)

        for lag in range(max_lag + 1):
            if lag < N:
                c = ((signal[:N-lag] - mean) * (signal[lag:] - mean)).mean()
                autocorr[lag] = c / (var + 1e-8)

        return autocorr.abs()  # Use absolute value for stability

    def _compute_approximate_entropy(
        self,
        x: torch.Tensor,
        m: int = 2,
        r: float = 0.2
    ) -> torch.Tensor:
        """
        Compute approximate entropy (pattern predictability).

        Approximate entropy (ApEn) measures irregularity. Similar to sample
        entropy but includes self-matches.

        Args:
            x: [NT, C, H, W] input fields
            m: Pattern length (default 2)
            r: Tolerance threshold as fraction of std (default 0.2)

        Returns:
            Approximate entropy [NT, C]
        """
        NT, C, H, W = x.shape

        # Simplified: use spectral entropy as proxy for ApEn
        # True ApEn requires template matching (too slow for large fields)

        # Flatten spatial dimensions
        x_flat = x.reshape(NT, C, H * W)

        # Compute spectral entropy (frequency domain irregularity)
        entropy = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            # FFT of each channel
            fft = torch.fft.rfft(x_flat[:, c, :], dim=1)  # [NT, freq_bins]
            power = torch.abs(fft) ** 2

            # Normalize to probability distribution
            power_sum = power.sum(dim=1, keepdim=True) + 1e-8
            prob = power / power_sum

            # Shannon entropy
            entropy[:, c] = -(prob * torch.log(prob + 1e-8)).sum(dim=1)

        # Normalize to [0, 10] range
        entropy = torch.clamp(entropy, min=0.0, max=10.0)

        return entropy

    def _compute_svd_entropy(
        self,
        x: torch.Tensor,
        cached_svd: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Compute SVD entropy (entropy of singular value spectrum).

        SVD entropy measures the complexity of spatial patterns by computing
        Shannon entropy of normalized singular values.

        Args:
            x: [NT, C, H, W] input fields
            cached_svd: Optional tuple (U, S, Vh) from pre-computed SVD

        Returns:
            SVD entropy [NT, C]
        """
        NT, C, H, W = x.shape

        try:
            if cached_svd is not None:
                # Use cached SVD results
                U, S, Vh = cached_svd
            else:
                # Compute SVD: reshape to [NT*C, H, W]
                x_batched = x.reshape(NT * C, H, W)
                U, S, Vh = torch.linalg.svd(x_batched, full_matrices=False)

            # S: [NT*C, min(H,W)]
            # Normalize singular values to probability distribution
            S_norm = S / (S.sum(dim=1, keepdim=True) + 1e-8)

            # Shannon entropy (vectorized)
            ent = -(S_norm * torch.log(S_norm + 1e-8)).sum(dim=1)
            entropy = torch.clamp(ent, min=0.0, max=10.0)

            # Reshape back to [NT, C]
            entropy = entropy.reshape(NT, C)

        except Exception:
            # SVD failed - fallback to zeros
            entropy = torch.zeros(NT, C, device=self.device)

        return entropy

    # =========================================================================
    # Dimensionality Measures
    # =========================================================================

    def _compute_participation_ratio(
        self,
        x: torch.Tensor,
        cached_svd: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Compute participation ratio (effective dimensionality).

        Participation ratio = (sum of eigenvalues)^2 / sum of squared eigenvalues
        Measures how many modes actively participate in the dynamics.

        Args:
            x: [NT, C, H, W] input fields
            cached_svd: Optional tuple (U, S, Vh) from pre-computed SVD

        Returns:
            Participation ratio [NT, C], normalized by min(H, W)
        """
        NT, C, H, W = x.shape

        try:
            if cached_svd is not None:
                # Use cached SVD results
                U, S, Vh = cached_svd
            else:
                # Compute SVD: reshape to [NT*C, H, W]
                x_batched = x.reshape(NT * C, H, W)
                U, S, Vh = torch.linalg.svd(x_batched, full_matrices=False)

            # S: [NT*C, min(H,W)]
            # Eigenvalues = squared singular values
            eigenvalues = S ** 2  # [NT*C, min(H,W)]

            # Participation ratio (vectorized)
            sum_eig = eigenvalues.sum(dim=1)  # [NT*C]
            sum_eig_sq = (eigenvalues ** 2).sum(dim=1)  # [NT*C]

            # Avoid division by zero
            pr_val = torch.where(
                sum_eig_sq > 1e-8,
                (sum_eig ** 2) / sum_eig_sq,
                torch.zeros_like(sum_eig)
            )

            # Normalize by max possible (min(H, W))
            pr = pr_val / min(H, W)

            # Reshape back to [NT, C]
            pr = pr.reshape(NT, C)

        except Exception:
            pr = torch.zeros(NT, C, device=self.device)

        return pr

    def _compute_pca_compression_ratio(
        self,
        x: torch.Tensor,
        cached_svd: Optional[tuple] = None,
        variance_threshold: float = 0.9
    ) -> torch.Tensor:
        """
        Compute PCA compression ratio (variance capture efficiency).

        Measures how many principal components are needed to capture
        90% of variance. Lower ratio = more compressible.

        Args:
            x: [NT, C, H, W] input fields
            cached_svd: Optional tuple (U, S, Vh) from pre-computed SVD
            variance_threshold: Variance to capture (default 0.9)

        Returns:
            Compression ratio [NT, C] (num components / total components)
        """
        NT, C, H, W = x.shape

        try:
            if cached_svd is not None:
                # Use cached SVD results
                U, S, Vh = cached_svd
            else:
                # Compute SVD: reshape to [NT*C, H, W]
                x_batched = x.reshape(NT * C, H, W)
                U, S, Vh = torch.linalg.svd(x_batched, full_matrices=False)

            # S: [NT*C, min(H,W)]
            # Eigenvalues = squared singular values
            eigenvalues = S ** 2  # [NT*C, min(H,W)]

            # Cumulative variance explained (vectorized)
            total_var = eigenvalues.sum(dim=1, keepdim=True)  # [NT*C, 1]
            cumsum_var = torch.cumsum(eigenvalues, dim=1)  # [NT*C, min(H,W)]
            frac_var = cumsum_var / (total_var + 1e-8)  # [NT*C, min(H,W)]

            # Find number of components needed for threshold (vectorized)
            num_components = (frac_var < variance_threshold).sum(dim=1) + 1  # [NT*C]
            total_components = eigenvalues.shape[1]

            # Compression ratio (lower = more compressible)
            compression = num_components.float() / total_components

            # Reshape back to [NT, C]
            compression = compression.reshape(NT, C)

        except Exception:
            compression = torch.ones(NT, C, device=self.device)  # No compression

        return compression

    # =========================================================================
    # Multiscale Analysis
    # =========================================================================

    def _compute_multiscale_entropy(
        self,
        x: torch.Tensor,
        scales: list = [2, 4, 8]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multiscale entropy at different coarse-graining scales.

        Coarse-grains the field by averaging non-overlapping blocks, then
        computes spectral entropy at each scale.

        Args:
            x: [NT, C, H, W] input fields
            scales: Coarse-graining factors (default [2, 4, 8])

        Returns:
            Dictionary with keys 'multiscale_entropy_scale_X'
        """
        NT, C, H, W = x.shape
        features = {}

        for scale in scales:
            # Coarse-grain via average pooling
            if H >= scale and W >= scale:
                x_coarse = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                # Compute spectral entropy on coarse-grained field
                entropy = self._compute_spectral_entropy(x_coarse)
                features[f'multiscale_entropy_scale_{scale}'] = entropy
            else:
                # Scale too large, return zeros
                features[f'multiscale_entropy_scale_{scale}'] = torch.zeros(NT, C, device=self.device)

        return features

    def _compute_spectral_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper: compute spectral entropy via 2D FFT.

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Spectral entropy [NT, C]
        """
        NT, C, H, W = x.shape

        # Compute 2D FFT
        fft = torch.fft.rfft2(x, dim=(-2, -1))  # [NT, C, H, W//2+1]
        power = torch.abs(fft) ** 2

        # Flatten spatial frequencies
        power_flat = power.reshape(NT, C, -1)  # [NT, C, H*(W//2+1)]

        # Normalize to probability distribution
        power_sum = power_flat.sum(dim=2, keepdim=True) + 1e-8
        prob = power_flat / power_sum

        # Shannon entropy
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=2)  # [NT, C]

        # Normalize to [0, 10] range
        entropy = torch.clamp(entropy, min=0.0, max=10.0)

        return entropy
