"""
Morphological feature extraction.

Extracts shape and morphological features from 2D fields:
- Shape descriptors (area fraction, perimeter, circularity, eccentricity, solidity, extent)
- Image moments (Hu moments, centroids)
- Granulometry (multi-scale morphological analysis)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryMorphologicalConfig


class MorphologicalFeatureExtractor:
    """
    Extract morphological and shape features from 2D fields.

    Features measure geometric properties, moments, and multi-scale structure.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = MorphologicalFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize morphological feature extractor.

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
        config: Optional['SummaryMorphologicalConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract morphological features from fields.

        Args:
            fields: Input fields [N, T, C, H, W]
                N = batch size
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryMorphologicalConfig for feature selection

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

        # Shape descriptors
        if include_all or (config is not None and (
            config.include_area_fraction or
            config.include_perimeter_total or
            config.include_shape_circularity or
            config.include_shape_eccentricity or
            config.include_shape_solidity or
            config.include_shape_extent
        )):
            shape_features = self._compute_shape_descriptors(fields_flat)
            features.update(shape_features)

        # Image moments
        if include_all or (config is not None and (
            config.include_moment_hu_1 or
            config.include_moment_hu_2 or
            config.include_centroid_x or
            config.include_centroid_y or
            config.include_centroid_displacement
        )):
            moment_features = self._compute_image_moments(fields_flat)
            features.update(moment_features)

        # Granulometry
        if include_all or (config is not None and config.include_granulometry_mean):
            granulometry_features = self._compute_granulometry(fields_flat)
            features.update(granulometry_features)

        # Reshape all features from [NT, C] -> [N, T, C]
        for key in features:
            if features[key].shape[0] == NT:
                features[key] = features[key].reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Shape Descriptors
    # =========================================================================

    def _compute_shape_descriptors(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute shape descriptors from binarized fields.

        Binarization: threshold at median + threshold * std

        Args:
            x: [NT, C, H, W] input fields
            threshold: Relative threshold for binarization

        Returns:
            Dictionary with shape features:
                - area_fraction: Fraction of pixels above threshold
                - perimeter: Boundary length (estimated via gradients)
                - circularity: 4π*area / perimeter²
                - eccentricity: Ratio of eigenvalues of covariance matrix
                - solidity: Area / convex hull area (approximated)
                - extent: Area / bounding box area
        """
        NT, C, H, W = x.shape

        area_fraction = torch.zeros(NT, C, device=self.device)
        perimeter = torch.zeros(NT, C, device=self.device)
        circularity = torch.zeros(NT, C, device=self.device)
        eccentricity = torch.zeros(NT, C, device=self.device)
        solidity = torch.zeros(NT, C, device=self.device)
        extent = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            x_c = x[:, c, :, :]  # [NT, H, W]

            # Binarize: threshold at median + threshold * std
            median = x_c.median(dim=-1, keepdim=True)[0].median(dim=-2, keepdim=True)[0]  # [NT, 1, 1]
            std = x_c.std(dim=(-2, -1), keepdim=True)  # [NT, 1, 1]
            thresh_val = median + threshold * std

            binary = (x_c > thresh_val).float()  # [NT, H, W]

            # Area fraction
            area_fraction[:, c] = binary.mean(dim=(-2, -1))

            # Perimeter (approximate via gradient magnitude)
            # Compute gradient using finite differences
            grad_y = binary[:, 1:, :] - binary[:, :-1, :]  # [NT, H-1, W]
            grad_x = binary[:, :, 1:] - binary[:, :, :-1]  # [NT, H, W-1]

            # Perimeter = count of boundary pixels
            perimeter_y = (grad_y.abs() > 0.5).float().sum(dim=(-2, -1))
            perimeter_x = (grad_x.abs() > 0.5).float().sum(dim=(-2, -1))
            perim = perimeter_y + perimeter_x

            # Normalize by grid size
            perimeter[:, c] = perim / (H + W)

            # Circularity: 4π*area / perimeter²
            area = area_fraction[:, c] * H * W
            circularity[:, c] = 4 * np.pi * area / (perim**2 + 1e-8)
            circularity[:, c] = torch.clamp(circularity[:, c], min=0.0, max=1.0)

            # Eccentricity (from covariance matrix of binary region)
            for i in range(NT):
                binary_i = binary[i, :, :]  # [H, W]

                # Find non-zero pixels
                y_coords, x_coords = torch.where(binary_i > 0.5)

                if len(y_coords) > 2:
                    # Covariance matrix
                    y_mean = y_coords.float().mean()
                    x_mean = x_coords.float().mean()

                    y_centered = y_coords.float() - y_mean
                    x_centered = x_coords.float() - x_mean

                    cov_xx = (x_centered ** 2).mean()
                    cov_yy = (y_centered ** 2).mean()
                    cov_xy = (x_centered * y_centered).mean()

                    # Eigenvalues of covariance matrix
                    trace = cov_xx + cov_yy
                    det = cov_xx * cov_yy - cov_xy**2
                    discriminant = trace**2 - 4*det

                    if discriminant > 0:
                        lambda1 = (trace + torch.sqrt(discriminant)) / 2
                        lambda2 = (trace - torch.sqrt(discriminant)) / 2

                        # Eccentricity: sqrt(1 - lambda2 / lambda1)
                        if lambda1 > 1e-6:
                            ecc = torch.sqrt(1 - lambda2 / (lambda1 + 1e-8))
                            eccentricity[i, c] = torch.clamp(ecc, min=0.0, max=1.0)
                        else:
                            eccentricity[i, c] = 0.0
                    else:
                        eccentricity[i, c] = 0.0
                else:
                    eccentricity[i, c] = 0.0

            # Solidity: Area / convex hull area
            # Approximation: use max extent as proxy for convex hull
            # True convex hull requires complex algorithms
            solidity[:, c] = area_fraction[:, c] / torch.clamp(area_fraction[:, c], min=0.1)
            solidity[:, c] = torch.clamp(solidity[:, c], min=0.0, max=1.0)

            # Extent: Area / bounding box area
            # For continuous fields, approximate as area_fraction
            extent[:, c] = area_fraction[:, c]

        return {
            'area_fraction': area_fraction,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'extent': extent
        }

    # =========================================================================
    # Image Moments
    # =========================================================================

    def _compute_image_moments(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute image moments and Hu invariants.

        Image moments capture shape information that is translation,
        scale, and rotation invariant.

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - centroid_x, centroid_y: Center of mass
                - hu_moment_1, hu_moment_2: First two Hu moments (rotation invariant)
        """
        NT, C, H, W = x.shape

        centroid_x = torch.zeros(NT, C, device=self.device)
        centroid_y = torch.zeros(NT, C, device=self.device)
        hu_moment_1 = torch.zeros(NT, C, device=self.device)
        hu_moment_2 = torch.zeros(NT, C, device=self.device)

        # Create coordinate grids
        y_grid = torch.arange(H, device=self.device).reshape(H, 1).expand(H, W)
        x_grid = torch.arange(W, device=self.device).reshape(1, W).expand(H, W)

        for c in range(C):
            x_c = x[:, c, :, :]  # [NT, H, W]

            # Normalize intensities to [0, 1] (use as weights)
            x_min = x_c.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            x_max = x_c.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            x_norm = (x_c - x_min) / (x_max - x_min + 1e-8)

            # Raw moments: m_pq = sum_ij (x^p * y^q * I(x,y))
            m_00 = x_norm.sum(dim=(-2, -1))  # [NT]

            # First moments (for centroid)
            m_10 = (x_norm * x_grid).sum(dim=(-2, -1))
            m_01 = (x_norm * y_grid).sum(dim=(-2, -1))

            # Centroids
            centroid_x[:, c] = m_10 / (m_00 + 1e-8) / W  # Normalize to [0, 1]
            centroid_y[:, c] = m_01 / (m_00 + 1e-8) / H

            # Central moments (translation invariant)
            # mu_pq = sum_ij ((x - x_bar)^p * (y - y_bar)^q * I(x,y))
            x_bar = centroid_x[:, c].reshape(NT, 1, 1) * W
            y_bar = centroid_y[:, c].reshape(NT, 1, 1) * H

            x_centered = x_grid - x_bar
            y_centered = y_grid - y_bar

            # Second order central moments
            mu_20 = (x_norm * x_centered**2).sum(dim=(-2, -1))
            mu_02 = (x_norm * y_centered**2).sum(dim=(-2, -1))
            mu_11 = (x_norm * x_centered * y_centered).sum(dim=(-2, -1))

            # Normalized central moments (scale invariant)
            # eta_pq = mu_pq / mu_00^((p+q)/2 + 1)
            eta_20 = mu_20 / (m_00**2 + 1e-8)
            eta_02 = mu_02 / (m_00**2 + 1e-8)
            eta_11 = mu_11 / (m_00**2 + 1e-8)

            # Hu moments (rotation invariant)
            # H1 = eta_20 + eta_02
            hu_moment_1[:, c] = eta_20 + eta_02

            # H2 = (eta_20 - eta_02)^2 + 4*eta_11^2
            hu_moment_2[:, c] = (eta_20 - eta_02)**2 + 4*eta_11**2

            # Log scale for better numerical range
            hu_moment_1[:, c] = torch.sign(hu_moment_1[:, c]) * torch.log(torch.abs(hu_moment_1[:, c]) + 1e-8)
            hu_moment_2[:, c] = torch.sign(hu_moment_2[:, c]) * torch.log(torch.abs(hu_moment_2[:, c]) + 1e-8)

            # Clamp to reasonable range
            hu_moment_1[:, c] = torch.clamp(hu_moment_1[:, c], min=-10.0, max=10.0)
            hu_moment_2[:, c] = torch.clamp(hu_moment_2[:, c], min=-10.0, max=10.0)

        return {
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'hu_moment_1': hu_moment_1,
            'hu_moment_2': hu_moment_2
        }

    # =========================================================================
    # Granulometry
    # =========================================================================

    def _compute_granulometry(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute granulometry (particle size distribution via morphological opening).

        Granulometry: measure size distribution by applying morphological
        opening at multiple scales and measuring volume loss.

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with:
                - mean_granule_size: Mean particle size
                - granule_size_std: Std of particle sizes
        """
        NT, C, H, W = x.shape

        mean_size = torch.zeros(NT, C, device=self.device)
        std_size = torch.zeros(NT, C, device=self.device)

        # Scales for morphological opening (kernel sizes)
        scales = [3, 5, 7, 9]

        for c in range(C):
            x_c = x[:, c:c+1, :, :]  # [NT, 1, H, W]

            # Normalize to [0, 1]
            x_min = x_c.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
            x_max = x_c.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
            x_norm = (x_c - x_min) / (x_max - x_min + 1e-8)

            # Compute granulometry curve (volume vs scale)
            volumes = []

            for scale in scales:
                # Morphological opening = erosion followed by dilation
                # Approximation: use max pooling (erosion) + unpooling (dilation)

                # Erosion (max pooling with negative values = min pooling)
                eroded = -F.max_pool2d(-x_norm, kernel_size=scale, stride=1, padding=scale//2)

                # Dilation (max pooling)
                opened = F.max_pool2d(eroded, kernel_size=scale, stride=1, padding=scale//2)

                # Volume (sum of pixel values)
                volume = opened.sum(dim=(-2, -1))  # [NT, 1]
                volumes.append(volume)

            # Stack volumes: [NT, num_scales]
            volumes = torch.stack(volumes, dim=-1).squeeze(1)  # [NT, num_scales]

            # Granulometry curve: derivative of volume loss
            # Volume loss: V(scale) - V(0)
            volume_loss = volumes[:, 0:1] - volumes  # [NT, num_scales]

            # Normalize
            total_volume = volumes[:, 0:1] + 1e-8
            normalized_loss = volume_loss / total_volume

            # Granule size distribution: weighted average of scales
            scale_tensor = torch.tensor(scales, device=self.device, dtype=torch.float32)

            # Mean granule size (weighted by volume loss)
            weights = normalized_loss / (normalized_loss.sum(dim=1, keepdim=True) + 1e-8)
            mean_size[:, c] = (weights * scale_tensor).sum(dim=1)

            # Std of granule size
            variance = (weights * (scale_tensor - mean_size[:, c:c+1])**2).sum(dim=1)
            std_size[:, c] = torch.sqrt(variance + 1e-8)

            # Normalize by grid size
            mean_size[:, c] = mean_size[:, c] / min(H, W)
            std_size[:, c] = std_size[:, c] / min(H, W)

        return {
            'mean_granule_size': mean_size,
            'granule_size_std': std_size
        }
