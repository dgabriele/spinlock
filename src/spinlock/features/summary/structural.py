"""
Structural feature extraction.

Extracts topological and structural features from 2D fields:
- Connected components (count, size statistics)
- Edge detection (density, length, Sobel gradients)
- GLCM texture features (contrast, homogeneity, energy, correlation)

All operations are GPU-accelerated using PyTorch.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryStructuralConfig


class StructuralFeatureExtractor:
    """
    Extract structural and topological features from 2D fields.

    Features measure connectivity, edges, and texture patterns.
    All operations are batched and GPU-accelerated.

    Example:
        >>> extractor = StructuralFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 64, 64, device='cuda')  # [N,M,T,C,H,W]
        >>> features = extractor.extract(fields)  # Dict of features
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize structural feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

        # Pre-compute Sobel kernels for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32, device=device).reshape(1, 1, 3, 3)

    def _adaptive_outlier_clip(
        self,
        values: torch.Tensor,
        iqr_multiplier: float = 10.0
    ) -> torch.Tensor:
        """Adaptive outlier clipping based on IQR (same as spatial extractor)."""
        original_shape = values.shape
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
        config: Optional['SummaryStructuralConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract structural features from fields.

        Args:
            fields: Input fields [N, T, C, H, W]
                N = batch size
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryStructuralConfig for feature selection

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

        # Connected components (CPU-only, slow)
        if include_all or (config is not None and (
            config.include_num_connected_components or
            config.include_largest_component_size or
            config.include_component_size_mean or
            config.include_component_size_std
        )):
            cc_features = self._compute_connected_components(fields_flat)
            features.update(cc_features)

        # Edge features
        if include_all or (config is not None and (
            config.include_edge_density or
            config.include_edge_length_total
        )):
            edge_features = self._compute_edge_features(fields_flat)
            features.update(edge_features)

        # GLCM texture features
        if include_all or (config is not None and (
            config.include_glcm_contrast or
            config.include_glcm_homogeneity or
            config.include_glcm_energy
        )):
            glcm_features = self._compute_glcm_features(fields_flat)
            features.update(glcm_features)

        # Reshape all features from [NT, C] -> [N, T, C]
        for key in features:
            if features[key].shape[0] == NT:
                features[key] = features[key].reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for key in features:
            features[key] = self._adaptive_outlier_clip(features[key], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Connected Components
    # =========================================================================

    def _compute_connected_components(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute connected component statistics.

        Uses thresholding to create binary image, then counts connected regions.
        Note: This is a simplified CPU-based implementation. For production,
        consider using scikit-image or OpenCV on CPU.

        Args:
            x: [NT, C, H, W] input fields
            threshold: Threshold for binarization (median-relative)

        Returns:
            Dictionary with keys:
                - num_connected_components: Count of components
                - largest_component_size: Size of largest component (fraction of H*W)
                - component_size_mean: Mean component size
                - component_size_std: Std of component sizes
        """
        NT, C, H, W = x.shape

        num_components = torch.zeros(NT, C, device=self.device)
        largest_size = torch.zeros(NT, C, device=self.device)
        mean_size = torch.zeros(NT, C, device=self.device)
        std_size = torch.zeros(NT, C, device=self.device)

        # Move to CPU for connected component labeling (no GPU implementation)
        x_cpu = x.cpu()

        for i in range(NT):
            for c in range(C):
                field = x_cpu[i, c, :, :].numpy()  # [H, W]

                # Binarize: threshold at median + threshold * std
                thresh_val = np.median(field) + threshold * np.std(field)
                binary = (field > thresh_val).astype(np.uint8)

                # Connected components (simplified: use scipy-free approach)
                # Count via 4-connectivity flood fill
                labeled, num_cc, sizes = self._label_connected_components(binary)

                num_components[i, c] = num_cc

                if num_cc > 0:
                    largest_size[i, c] = max(sizes) / (H * W)
                    mean_size[i, c] = np.mean(sizes)
                    std_size[i, c] = np.std(sizes) if len(sizes) > 1 else 0.0
                else:
                    largest_size[i, c] = 0.0
                    mean_size[i, c] = 0.0
                    std_size[i, c] = 0.0

        return {
            'num_connected_components': num_components,
            'largest_component_size': largest_size,
            'component_size_mean': mean_size,
            'component_size_std': std_size
        }

    def _label_connected_components(self, binary: np.ndarray):
        """
        Simple connected component labeling via flood fill.

        Args:
            binary: [H, W] binary image

        Returns:
            labeled: [H, W] labeled image
            num_components: Number of components
            sizes: List of component sizes
        """
        H, W = binary.shape
        labeled = np.zeros((H, W), dtype=np.int32)
        label = 0
        sizes = []

        def flood_fill(i, j, label):
            """4-connectivity flood fill."""
            stack = [(i, j)]
            size = 0
            while stack:
                y, x = stack.pop()
                if y < 0 or y >= H or x < 0 or x >= W:
                    continue
                if labeled[y, x] != 0 or binary[y, x] == 0:
                    continue

                labeled[y, x] = label
                size += 1

                # 4-connected neighbors
                stack.append((y-1, x))
                stack.append((y+1, x))
                stack.append((y, x-1))
                stack.append((y, x+1))

            return size

        # Find all unlabeled foreground pixels
        for i in range(H):
            for j in range(W):
                if binary[i, j] > 0 and labeled[i, j] == 0:
                    label += 1
                    size = flood_fill(i, j, label)
                    sizes.append(size)

        return labeled, label, sizes

    # =========================================================================
    # Edge Features
    # =========================================================================

    def _compute_edge_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute edge detection features via Sobel gradients.

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with keys:
                - edge_density: Fraction of pixels classified as edges
                - edge_magnitude_mean: Mean gradient magnitude
                - edge_magnitude_std: Std of gradient magnitude
        """
        NT, C, H, W = x.shape

        # Compute Sobel gradients per channel
        edge_density = torch.zeros(NT, C, device=self.device)
        edge_mag_mean = torch.zeros(NT, C, device=self.device)
        edge_mag_std = torch.zeros(NT, C, device=self.device)

        for c in range(C):
            # Extract channel
            x_c = x[:, c:c+1, :, :]  # [NT, 1, H, W]

            # Apply Sobel filters
            grad_x = F.conv2d(x_c, self.sobel_x, padding=1)  # [NT, 1, H, W]
            grad_y = F.conv2d(x_c, self.sobel_y, padding=1)  # [NT, 1, H, W]

            # Gradient magnitude
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)  # [NT, 1, H, W]

            # Edge density: fraction above threshold (mean + std)
            threshold = grad_mag.mean(dim=(-2, -1), keepdim=True) + grad_mag.std(dim=(-2, -1), keepdim=True)
            edges = (grad_mag > threshold).float()

            edge_density[:, c] = edges.mean(dim=(-2, -1)).squeeze()

            # Edge magnitude statistics
            edge_mag_mean[:, c] = grad_mag.mean(dim=(-2, -1)).squeeze()
            edge_mag_std[:, c] = grad_mag.std(dim=(-2, -1)).squeeze()

        return {
            'edge_density': edge_density,
            'edge_magnitude_mean': edge_mag_mean,
            'edge_magnitude_std': edge_mag_std
        }

    # =========================================================================
    # GLCM Texture Features
    # =========================================================================

    def _compute_glcm_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Gray-Level Co-occurrence Matrix (GLCM) texture features.

        Simplified GPU-friendly approximation:
        - Quantize intensities to 8 levels
        - Compute co-occurrence statistics for horizontal pairs
        - Extract contrast, homogeneity, energy, correlation

        Args:
            x: [NT, C, H, W] input fields

        Returns:
            Dictionary with GLCM features
        """
        NT, C, H, W = x.shape

        contrast = torch.zeros(NT, C, device=self.device)
        homogeneity = torch.zeros(NT, C, device=self.device)
        energy = torch.zeros(NT, C, device=self.device)
        correlation = torch.zeros(NT, C, device=self.device)

        # Quantize to 8 levels for manageable GLCM size
        num_levels = 8

        for c in range(C):
            for i in range(NT):
                field = x[i, c, :, :]  # [H, W]

                # Normalize to [0, 1]
                field_min = field.min()
                field_max = field.max()
                if field_max > field_min:
                    field_norm = (field - field_min) / (field_max - field_min + 1e-8)
                else:
                    # Constant field
                    contrast[i, c] = 0.0
                    homogeneity[i, c] = 1.0
                    energy[i, c] = 1.0
                    correlation[i, c] = 1.0
                    continue

                # Quantize to discrete levels
                field_quant = (field_norm * (num_levels - 1)).long()
                field_quant = torch.clamp(field_quant, 0, num_levels - 1)

                # Build GLCM for horizontal neighbors (offset = 1, angle = 0)
                glcm = torch.zeros(num_levels, num_levels, device=self.device)

                # Horizontal pairs: (i, j) and (i, j+1)
                left = field_quant[:, :-1]  # [H, W-1]
                right = field_quant[:, 1:]  # [H, W-1]

                # Accumulate co-occurrences
                for level_i in range(num_levels):
                    for level_j in range(num_levels):
                        matches = ((left == level_i) & (right == level_j)).float()
                        glcm[level_i, level_j] = matches.sum()

                # Normalize GLCM to probability
                glcm_sum = glcm.sum()
                if glcm_sum > 0:
                    glcm = glcm / glcm_sum
                else:
                    # Empty GLCM (shouldn't happen)
                    continue

                # Compute texture features
                # Create index grids
                ii, jj = torch.meshgrid(
                    torch.arange(num_levels, device=self.device),
                    torch.arange(num_levels, device=self.device),
                    indexing='ij'
                )

                # Contrast: sum of (i-j)^2 * P(i,j)
                contrast[i, c] = ((ii - jj) ** 2 * glcm).sum()

                # Homogeneity: sum of P(i,j) / (1 + |i-j|)
                homogeneity[i, c] = (glcm / (1.0 + torch.abs(ii - jj).float())).sum()

                # Energy: sum of P(i,j)^2
                energy[i, c] = (glcm ** 2).sum()

                # Correlation: complex formula involving means and stds
                # Simplified: use covariance proxy
                mu_i = (ii.float() * glcm).sum()
                mu_j = (jj.float() * glcm).sum()
                sigma_i = torch.sqrt(((ii.float() - mu_i) ** 2 * glcm).sum() + 1e-8)
                sigma_j = torch.sqrt(((jj.float() - mu_j) ** 2 * glcm).sum() + 1e-8)

                if sigma_i > 1e-6 and sigma_j > 1e-6:
                    correlation[i, c] = ((ii.float() - mu_i) * (jj.float() - mu_j) * glcm).sum() / (sigma_i * sigma_j)
                else:
                    correlation[i, c] = 0.0

        return {
            'glcm_contrast': contrast,
            'glcm_homogeneity': homogeneity,
            'glcm_energy': energy,
            'glcm_correlation': correlation
        }
