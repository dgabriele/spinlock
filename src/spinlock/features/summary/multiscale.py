"""
Multiscale feature extraction.

Extracts multi-resolution features from 2D fields:
- Wavelet decomposition (Haar wavelets, 4 levels)
- Laplacian pyramid (Gaussian pyramid differences, 4 levels)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING, List, Tuple
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryMultiscaleConfig


class MultiscaleFeatureExtractor:
    """
    Extract multiscale features from 2D fields.

    Features measure structure at different spatial scales using
    wavelet and pyramid decompositions.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = MultiscaleFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize multiscale feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

        # Pre-compute Haar wavelet filters
        self._init_wavelet_filters()

        # Pre-compute Gaussian kernel for pyramid
        self._init_gaussian_kernel()

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

    def _init_wavelet_filters(self):
        """Initialize 2D Haar wavelet filters."""
        # 1D Haar wavelet filters (scaling and wavelet)
        # Scaling (low-pass): [1, 1] / sqrt(2)
        # Wavelet (high-pass): [1, -1] / sqrt(2)

        h = torch.tensor([1.0, 1.0], device=self.device) / np.sqrt(2)
        g = torch.tensor([1.0, -1.0], device=self.device) / np.sqrt(2)

        # 2D separable filters (LL, LH, HL, HH)
        # LL: low-pass in both directions (approximation)
        # LH: low-pass in rows, high-pass in cols (horizontal detail)
        # HL: high-pass in rows, low-pass in cols (vertical detail)
        # HH: high-pass in both directions (diagonal detail)

        self.filter_LL = torch.outer(h, h).reshape(1, 1, 2, 2)  # [1, 1, 2, 2]
        self.filter_LH = torch.outer(h, g).reshape(1, 1, 2, 2)
        self.filter_HL = torch.outer(g, h).reshape(1, 1, 2, 2)
        self.filter_HH = torch.outer(g, g).reshape(1, 1, 2, 2)

    def _init_gaussian_kernel(self, sigma: float = 1.0, kernel_size: int = 5):
        """Initialize Gaussian kernel for pyramid construction."""
        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32) - kernel_size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()

        # Create 2D Gaussian kernel via outer product
        self.gaussian_kernel = torch.outer(gaussian_1d, gaussian_1d).reshape(1, 1, kernel_size, kernel_size)

    def extract(
        self,
        fields: torch.Tensor,  # [N, T, C, H, W]
        config: Optional['SummaryMultiscaleConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multiscale features from fields.

        Args:
            fields: Input fields [N, T, C, H, W]
                N = batch size
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryMultiscaleConfig for feature selection

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
            num_wavelet_levels = 4
            num_pyramid_levels = 4
        else:
            include_all = False
            num_wavelet_levels = config.num_wavelet_levels
            num_pyramid_levels = config.num_pyramid_levels

        # Wavelet features
        if include_all or (config is not None and (
            config.include_wavelet_energy or
            config.include_wavelet_mean or
            config.include_wavelet_std
        )):
            wavelet_features = self._compute_wavelet_features(fields_flat, num_levels=num_wavelet_levels)
            features.update(wavelet_features)

        # Pyramid features
        if include_all or (config is not None and (
            config.include_pyramid_energy or
            config.include_pyramid_contrast
        )):
            pyramid_features = self._compute_pyramid_features(fields_flat, num_levels=num_pyramid_levels)
            features.update(pyramid_features)

        # Reshape all features from [NT, C] -> [N, T, C]
        for key in features:
            if features[key].shape[0] == NT:
                features[key] = features[key].reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Wavelet Decomposition
    # =========================================================================

    def _compute_wavelet_features(
        self,
        x: torch.Tensor,
        num_levels: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute wavelet decomposition features.

        Uses 2D Haar wavelet transform at multiple levels.
        Extracts energy, mean, and std from each level's detail coefficients.

        Args:
            x: [NT, C, H, W] input fields
            num_levels: Number of decomposition levels (default 4)

        Returns:
            Dictionary with wavelet features per level
        """
        NT, C, H, W = x.shape
        features = {}

        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [NT, 1, H, W]

            # Iterative decomposition
            approximation = x_c

            for level in range(num_levels):
                # Check if we can decompose further
                current_h, current_w = approximation.shape[-2:]
                if current_h < 2 or current_w < 2:
                    # Too small, pad with zeros for this level
                    energy = torch.zeros(NT, device=self.device)
                    mean = torch.zeros(NT, device=self.device)
                    std = torch.zeros(NT, device=self.device)
                else:
                    # Apply 2D Haar wavelet transform
                    LL, LH, HL, HH = self._dwt2d(approximation)

                    # Detail coefficients (LH, HL, HH)
                    details = torch.cat([LH, HL, HH], dim=1)  # [NT, 3, H/2, W/2]

                    # Energy: sum of squared coefficients
                    energy = (details ** 2).sum(dim=(1, 2, 3))

                    # Mean and std of detail coefficients
                    mean = details.mean(dim=(1, 2, 3))
                    std = details.std(dim=(1, 2, 3))

                    # Normalize by spatial size
                    energy = energy / (details.shape[2] * details.shape[3] + 1e-8)

                    # Continue with approximation (LL) for next level
                    approximation = LL

                # Store features for this level
                if c == 0:
                    # Initialize tensors for all channels
                    features[f'wavelet_energy_level_{level}'] = torch.zeros(NT, C, device=self.device)
                    features[f'wavelet_mean_level_{level}'] = torch.zeros(NT, C, device=self.device)
                    features[f'wavelet_std_level_{level}'] = torch.zeros(NT, C, device=self.device)

                features[f'wavelet_energy_level_{level}'][:, c] = energy
                features[f'wavelet_mean_level_{level}'][:, c] = mean
                features[f'wavelet_std_level_{level}'][:, c] = std

        return features

    def _dwt2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D Discrete Wavelet Transform (Haar).

        Args:
            x: [NT, 1, H, W] input

        Returns:
            LL, LH, HL, HH: Approximation and detail coefficients [NT, 1, H/2, W/2]
        """
        # Apply filters with stride 2 (downsampling)
        LL = F.conv2d(x, self.filter_LL, stride=2)
        LH = F.conv2d(x, self.filter_LH, stride=2)
        HL = F.conv2d(x, self.filter_HL, stride=2)
        HH = F.conv2d(x, self.filter_HH, stride=2)

        return LL, LH, HL, HH

    # =========================================================================
    # Laplacian Pyramid
    # =========================================================================

    def _compute_pyramid_features(
        self,
        x: torch.Tensor,
        num_levels: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Laplacian pyramid features.

        Builds Gaussian pyramid (iterative blur + downsample), then computes
        Laplacian pyramid (differences between levels).
        Extracts energy and contrast from each level.

        Args:
            x: [NT, C, H, W] input fields
            num_levels: Number of pyramid levels (default 4)

        Returns:
            Dictionary with pyramid features per level
        """
        NT, C, H, W = x.shape
        features = {}

        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [NT, 1, H, W]

            # Build Gaussian pyramid
            gaussian_pyramid = [x_c]

            for level in range(num_levels):
                # Blur with Gaussian
                current = gaussian_pyramid[-1]
                blurred = F.conv2d(current, self.gaussian_kernel, padding=self.gaussian_kernel.shape[-1]//2)

                # Downsample by factor of 2
                if blurred.shape[-2] >= 2 and blurred.shape[-1] >= 2:
                    downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
                    gaussian_pyramid.append(downsampled)
                else:
                    # Can't downsample further
                    break

            # Build Laplacian pyramid (differences)
            num_pyramid_levels = len(gaussian_pyramid) - 1

            for level in range(num_levels):
                if level < num_pyramid_levels:
                    # Get current and next level
                    current = gaussian_pyramid[level]
                    next_level = gaussian_pyramid[level + 1]

                    # Upsample next level to match current size
                    upsampled = F.interpolate(
                        next_level,
                        size=(current.shape[-2], current.shape[-1]),
                        mode='bilinear',
                        align_corners=False
                    )

                    # Laplacian = current - upsampled(next)
                    laplacian = current - upsampled

                    # Energy: sum of squared coefficients
                    energy = (laplacian ** 2).sum(dim=(1, 2, 3))
                    energy = energy / (laplacian.shape[2] * laplacian.shape[3] + 1e-8)

                    # Contrast: std of Laplacian coefficients
                    contrast = laplacian.std(dim=(1, 2, 3))

                else:
                    # Level beyond pyramid depth, use zeros
                    energy = torch.zeros(NT, device=self.device)
                    contrast = torch.zeros(NT, device=self.device)

                # Store features for this level
                if c == 0:
                    features[f'pyramid_energy_level_{level}'] = torch.zeros(NT, C, device=self.device)
                    features[f'pyramid_contrast_level_{level}'] = torch.zeros(NT, C, device=self.device)

                features[f'pyramid_energy_level_{level}'][:, c] = energy
                features[f'pyramid_contrast_level_{level}'][:, c] = contrast

        return features
