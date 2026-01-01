"""
Manual IC feature extractors.

Extracts 14 hand-crafted features across 4 categories:
- Spatial structure (4): clustering, localization, autocorrelation
- Spectral (3): frequency content, power laws
- Information (4): entropy, complexity, predictability
- Morphological (3): density, gradients, symmetry

All extractors are PyTorch-native for GPU acceleration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from scipy import ndimage


class ICManualExtractor:
    """Extract manual IC features in PyTorch."""

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device

    def extract_all(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Extract all 14 manual features.

        Args:
            ic: [B, M, C, H, W] initial conditions

        Returns:
            [B, M, 14] manual features
        """
        # Extract by category
        spatial = self.extract_spatial_features(ic)      # 4 features
        spectral = self.extract_spectral_features(ic)    # 3 features
        info = self.extract_information_features(ic)      # 4 features
        morph = self.extract_morphological_features(ic)  # 3 features

        # Stack in order
        feature_list = [
            # Spatial (4)
            spatial['spatial_cluster_count'],
            spatial['spatial_largest_cluster_frac'],
            spatial['spatial_autocorr'],
            spatial['spatial_centroid_dist'],
            # Spectral (3)
            spectral['spectral_dominant_freq'],
            spectral['spectral_centroid'],
            spectral['spectral_power_law_exp'],
            # Information (4)
            info['info_entropy'],
            info['info_local_entropy_var'],
            info['info_lz_complexity'],
            info['info_predictability'],
            # Morphological (3)
            morph['morph_density'],
            morph['morph_radial_gradient'],
            morph['morph_symmetry']
        ]

        # Stack to [B, M, 14]
        features = torch.stack(feature_list, dim=-1)
        return features

    def extract_spatial_features(self, ic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spatial structure features.

        Args:
            ic: [B, M, C, H, W] initial conditions

        Returns:
            Dictionary of features, each [B, M]
        """
        B, M, C, H, W = ic.shape

        # Squeeze channel dimension for processing
        ic_2d = ic.squeeze(2)  # [B, M, H, W]

        # Binarize at median threshold for clustering analysis
        ic_flat = ic_2d.view(B, M, -1)
        median = ic_flat.median(dim=-1, keepdim=True).values.unsqueeze(-1)  # [B, M, 1, 1]
        binary = (ic_2d > median).float()

        # Feature 1: Cluster count (approximate via connected components)
        cluster_count = self._count_clusters_approx(binary)  # [B, M]

        # Feature 2: Largest cluster fraction
        largest_cluster_frac = self._largest_cluster_fraction(binary)  # [B, M]

        # Feature 3: Spatial autocorrelation (Moran's I approximation)
        spatial_autocorr = self._moran_i_approx(ic_2d)  # [B, M]

        # Feature 4: Centroid distance from center (localization measure)
        centroid_dist = self._centroid_distance(ic_2d)  # [B, M]

        return {
            'spatial_cluster_count': cluster_count,
            'spatial_largest_cluster_frac': largest_cluster_frac,
            'spatial_autocorr': spatial_autocorr,
            'spatial_centroid_dist': centroid_dist
        }

    def extract_spectral_features(self, ic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract spectral (frequency domain) features.

        Args:
            ic: [B, M, C, H, W] initial conditions

        Returns:
            Dictionary of features, each [B, M]
        """
        B, M, C, H, W = ic.shape
        ic_2d = ic.squeeze(2)  # [B, M, H, W]

        # 2D FFT - rfft2 for real inputs
        fft = torch.fft.rfft2(ic_2d)  # [B, M, H, W//2+1]
        power = torch.abs(fft) ** 2

        # Feature 5: Dominant frequency (peak in power spectrum)
        dominant_freq = self._dominant_frequency(power)  # [B, M]

        # Feature 6: Spectral centroid (weighted mean frequency)
        spectral_centroid = self._spectral_centroid(power)  # [B, M]

        # Feature 7: Power law exponent (1/f^β behavior)
        power_law_exp = self._estimate_power_law(power)  # [B, M]

        return {
            'spectral_dominant_freq': dominant_freq,
            'spectral_centroid': spectral_centroid,
            'spectral_power_law_exp': power_law_exp
        }

    def extract_information_features(self, ic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract information-theoretic features.

        Args:
            ic: [B, M, C, H, W] initial conditions

        Returns:
            Dictionary of features, each [B, M]
        """
        B, M, C, H, W = ic.shape
        ic_2d = ic.squeeze(2)  # [B, M, H, W]

        # Feature 8: Shannon entropy (histogram-based)
        entropy = self._shannon_entropy(ic_2d)  # [B, M]

        # Feature 9: Local entropy variance (spatial non-uniformity)
        local_entropy_var = self._local_entropy_variance(ic_2d)  # [B, M]

        # Feature 10: Approximate LZ complexity (via compression proxy)
        lz_complexity = self._lz_complexity_approx(ic_2d)  # [B, M]

        # Feature 11: Predictability score (autocorrelation decay)
        predictability = self._predictability_score(ic_2d)  # [B, M]

        return {
            'info_entropy': entropy,
            'info_local_entropy_var': local_entropy_var,
            'info_lz_complexity': lz_complexity,
            'info_predictability': predictability
        }

    def extract_morphological_features(self, ic: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract morphological (shape/density) features.

        Args:
            ic: [B, M, C, H, W] initial conditions

        Returns:
            Dictionary of features, each [B, M]
        """
        B, M, C, H, W = ic.shape
        ic_2d = ic.squeeze(2)  # [B, M, H, W]

        # Feature 12: Density (fraction of high-valued pixels)
        ic_flat = ic_2d.view(B, M, -1)
        median = ic_flat.median(dim=-1, keepdim=True).values
        density = (ic_flat > median).float().mean(dim=-1)  # [B, M]

        # Feature 13: Radial gradient strength (edge detection)
        radial_gradient = self._radial_gradient_strength(ic_2d)  # [B, M]

        # Feature 14: Symmetry score (4-fold rotational symmetry)
        symmetry = self._symmetry_score(ic_2d)  # [B, M]

        return {
            'morph_density': density,
            'morph_radial_gradient': radial_gradient,
            'morph_symmetry': symmetry
        }

    # ============================================================================
    # Spatial Feature Helpers
    # ============================================================================

    def _count_clusters_approx(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Approximate cluster count via local maxima in erosion-dilation.

        Args:
            binary: [B, M, H, W] binary mask

        Returns:
            [B, M] cluster counts
        """
        B, M, H, W = binary.shape

        # Use max pooling as erosion approximation
        eroded = -F.max_pool2d(-binary, kernel_size=3, stride=1, padding=1)

        # Count connected regions via number of local peaks
        # Use max pooling to find local maxima
        local_max = F.max_pool2d(eroded, kernel_size=5, stride=1, padding=2)
        peaks = (eroded == local_max) & (eroded > 0.5)

        cluster_count = peaks.view(B, M, -1).sum(dim=-1).float()

        # Normalize by grid size (log scale)
        cluster_count = torch.log1p(cluster_count)

        return cluster_count

    def _largest_cluster_fraction(self, binary: torch.Tensor) -> torch.Tensor:
        """
        Approximate largest cluster as fraction of total active pixels.

        Args:
            binary: [B, M, H, W] binary mask

        Returns:
            [B, M] largest cluster fractions
        """
        B, M, H, W = binary.shape

        # Use max pooling with large kernel to find largest connected region
        # Larger kernel → more connectivity
        dilated = F.max_pool2d(binary, kernel_size=11, stride=1, padding=5)

        # Fraction of dilated area to total active area
        total_active = binary.view(B, M, -1).sum(dim=-1)
        largest_region = dilated.view(B, M, -1).sum(dim=-1)

        frac = largest_region / (total_active + 1e-6)
        return frac.clamp(0, 1)

    def _moran_i_approx(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Approximate Moran's I (spatial autocorrelation).

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] autocorrelation scores
        """
        B, M, H, W = ic.shape

        # Compute mean
        mean = ic.view(B, M, -1).mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, M, 1, 1]

        # Centered values
        centered = ic - mean

        # Spatial lag: average of neighbors (using convolution)
        # Define neighbor weights (von Neumann neighborhood)
        kernel = torch.tensor([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype=ic.dtype, device=ic.device)
        kernel = kernel.view(1, 1, 3, 3) / 4.0  # Normalize

        # Compute spatial lag
        centered_flat = centered.view(B * M, 1, H, W)
        spatial_lag = F.conv2d(centered_flat, kernel, padding=1)  # [B*M, 1, H, W]
        spatial_lag = spatial_lag.view(B, M, H, W)

        # Moran's I: correlation between value and spatial lag
        numerator = (centered * spatial_lag).view(B, M, -1).sum(dim=-1)
        denominator = (centered ** 2).view(B, M, -1).sum(dim=-1)

        moran_i = numerator / (denominator + 1e-6)
        return moran_i.clamp(-1, 1)

    def _centroid_distance(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Distance of centroid from grid center.

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] centroid distances (normalized)
        """
        B, M, H, W = ic.shape

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=ic.device).view(1, 1, H, 1)
        x_coords = torch.linspace(0, 1, W, device=ic.device).view(1, 1, 1, W)

        # Weight coordinates by IC values (use absolute values)
        weights = torch.abs(ic)
        total_weight = weights.view(B, M, -1).sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B, M, 1, 1]

        # Compute weighted centroid
        cy = (weights * y_coords).view(B, M, -1).sum(dim=-1) / (total_weight.squeeze() + 1e-6)
        cx = (weights * x_coords).view(B, M, -1).sum(dim=-1) / (total_weight.squeeze() + 1e-6)

        # Distance from center (0.5, 0.5)
        dist = torch.sqrt((cy - 0.5)**2 + (cx - 0.5)**2)

        # Normalize to [0, 1] (max distance is sqrt(2)/2)
        dist = dist / (np.sqrt(2) / 2)

        return dist.clamp(0, 1)

    # ============================================================================
    # Spectral Feature Helpers
    # ============================================================================

    def _dominant_frequency(self, power: torch.Tensor) -> torch.Tensor:
        """
        Find dominant frequency (peak in power spectrum).

        Args:
            power: [B, M, H, W//2+1] power spectrum

        Returns:
            [B, M] dominant frequency (normalized)
        """
        B, M, H, W_half = power.shape

        # Find peak location
        power_flat = power.view(B, M, -1)
        peak_idx = power_flat.argmax(dim=-1)  # [B, M]

        # Convert to normalized frequency [0, 1]
        total_bins = H * W_half
        dominant_freq = peak_idx.float() / total_bins

        return dominant_freq

    def _spectral_centroid(self, power: torch.Tensor) -> torch.Tensor:
        """
        Spectral centroid (weighted mean frequency).

        Args:
            power: [B, M, H, W//2+1] power spectrum

        Returns:
            [B, M] spectral centroids
        """
        B, M, H, W_half = power.shape

        # Create frequency axis
        freqs = torch.arange(H * W_half, dtype=power.dtype, device=power.device)
        freqs = freqs.view(1, 1, -1)  # [1, 1, H*W_half]

        # Flatten power
        power_flat = power.view(B, M, -1)  # [B, M, H*W_half]

        # Weighted average
        total_power = power_flat.sum(dim=-1, keepdim=True)
        centroid = (power_flat * freqs).sum(dim=-1) / (total_power.squeeze() + 1e-6)

        # Normalize
        centroid = centroid / (H * W_half)

        return centroid

    def _estimate_power_law(self, power: torch.Tensor) -> torch.Tensor:
        """
        Estimate power law exponent β in 1/f^β.

        Args:
            power: [B, M, H, W//2+1] power spectrum

        Returns:
            [B, M] power law exponents
        """
        B, M, H, W_half = power.shape

        # Radial profile: bin by distance from origin
        # Create radial distance map
        ky = torch.fft.fftfreq(H, device=power.device).view(H, 1)
        kx = torch.fft.rfftfreq(H, device=power.device).view(1, W_half)  # Assuming square grid

        k_radial = torch.sqrt(ky**2 + kx**2)  # [H, W_half]

        # Average power in radial bins
        # Simple approach: log-log fit of power vs frequency
        power_flat = power.view(B, M, -1)  # [B, M, H*W_half]
        k_flat = k_radial.view(-1)  # [H*W_half]

        # Remove DC component (k=0)
        valid_mask = k_flat > 0
        k_valid = k_flat[valid_mask]
        power_valid = power_flat[:, :, valid_mask]  # [B, M, N_valid]

        # Log-log coordinates
        log_k = torch.log(k_valid + 1e-8)  # [N_valid]
        log_power = torch.log(power_valid + 1e-8)  # [B, M, N_valid]

        # Linear regression slope (simple mean-based estimator)
        # β = -d(log P)/d(log k)
        # Approximate as correlation
        mean_log_k = log_k.mean()
        mean_log_power = log_power.mean(dim=-1, keepdim=True)  # [B, M, 1]

        numerator = ((log_k - mean_log_k) * (log_power - mean_log_power)).sum(dim=-1)
        denominator = ((log_k - mean_log_k) ** 2).sum()

        beta = -numerator / (denominator + 1e-6)

        return beta.clamp(-5, 5)  # Typical range

    # ============================================================================
    # Information Feature Helpers
    # ============================================================================

    def _shannon_entropy(self, ic: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
        """
        Shannon entropy via histogram.

        Args:
            ic: [B, M, H, W]
            num_bins: Number of histogram bins

        Returns:
            [B, M] entropy values
        """
        B, M, H, W = ic.shape

        # Normalize to [0, 1]
        ic_flat = ic.view(B, M, -1)  # [B, M, H*W]
        ic_min = ic_flat.min(dim=-1, keepdim=True).values
        ic_max = ic_flat.max(dim=-1, keepdim=True).values
        ic_norm = (ic_flat - ic_min) / (ic_max - ic_min + 1e-8)

        # Compute histogram (approximate via binning)
        # Use torch.histc for each sample
        entropy_list = []
        for b in range(B):
            for m in range(M):
                hist = torch.histc(ic_norm[b, m], bins=num_bins, min=0, max=1)
                prob = hist / (H * W)
                prob = prob[prob > 0]  # Remove zeros
                ent = -(prob * torch.log2(prob + 1e-8)).sum()
                entropy_list.append(ent)

        entropy = torch.tensor(entropy_list, device=ic.device).view(B, M)

        # Normalize by max entropy (log2(num_bins))
        entropy = entropy / np.log2(num_bins)

        return entropy

    def _local_entropy_variance(self, ic: torch.Tensor, patch_size: int = 8) -> torch.Tensor:
        """
        Variance of local patch entropies.

        Args:
            ic: [B, M, H, W]
            patch_size: Size of local patches

        Returns:
            [B, M] local entropy variances
        """
        B, M, H, W = ic.shape

        # Extract patches
        patches = F.unfold(ic.view(B * M, 1, H, W),
                          kernel_size=patch_size,
                          stride=patch_size // 2)  # [B*M, patch_size^2, num_patches]

        num_patches = patches.shape[-1]
        patches = patches.view(B, M, patch_size * patch_size, num_patches)

        # Compute entropy for each patch (simplified: just variance as proxy)
        patch_entropy = patches.var(dim=2)  # [B, M, num_patches]

        # Variance of patch entropies
        local_ent_var = patch_entropy.var(dim=-1)  # [B, M]

        return local_ent_var

    def _lz_complexity_approx(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Approximate LZ complexity via discrete derivative count.

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] complexity scores
        """
        B, M, H, W = ic.shape

        # Binarize
        ic_flat = ic.view(B, M, -1)
        median = ic_flat.median(dim=-1, keepdim=True).values
        binary = (ic_flat > median).float()

        # Count transitions (approximate complexity)
        transitions = (binary[:, :, 1:] != binary[:, :, :-1]).float().sum(dim=-1)

        # Normalize by length
        complexity = transitions / (H * W)

        return complexity

    def _predictability_score(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Predictability via autocorrelation decay.

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] predictability scores
        """
        B, M, H, W = ic.shape

        # Compute autocorrelation at lag 1 (horizontal)
        ic_shifted = torch.roll(ic, shifts=1, dims=-1)

        # Correlation
        ic_centered = ic - ic.mean(dim=(-2, -1), keepdim=True)
        ic_shifted_centered = ic_shifted - ic_shifted.mean(dim=(-2, -1), keepdim=True)

        numerator = (ic_centered * ic_shifted_centered).view(B, M, -1).sum(dim=-1)
        denominator = torch.sqrt((ic_centered ** 2).view(B, M, -1).sum(dim=-1) *
                                (ic_shifted_centered ** 2).view(B, M, -1).sum(dim=-1))

        autocorr = numerator / (denominator + 1e-6)

        # Predictability = |autocorrelation|
        predictability = torch.abs(autocorr)

        return predictability.clamp(0, 1)

    # ============================================================================
    # Morphological Feature Helpers
    # ============================================================================

    def _radial_gradient_strength(self, ic: torch.Tensor) -> torch.Tensor:
        """
        Radial gradient strength (edge detection).

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] gradient strengths
        """
        B, M, H, W = ic.shape

        # Sobel filters for gradients
        sobel_x = torch.tensor([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=ic.dtype, device=ic.device)
        sobel_y = torch.tensor([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]], dtype=ic.dtype, device=ic.device)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        # Compute gradients
        ic_flat = ic.view(B * M, 1, H, W)
        grad_x = F.conv2d(ic_flat, sobel_x, padding=1)
        grad_y = F.conv2d(ic_flat, sobel_y, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = grad_mag.view(B, M, H, W)

        # Average gradient strength
        gradient_strength = grad_mag.view(B, M, -1).mean(dim=-1)

        return gradient_strength

    def _symmetry_score(self, ic: torch.Tensor) -> torch.Tensor:
        """
        4-fold rotational symmetry score.

        Args:
            ic: [B, M, H, W]

        Returns:
            [B, M] symmetry scores
        """
        B, M, H, W = ic.shape

        # Rotate 90, 180, 270 degrees
        ic_90 = torch.rot90(ic, k=1, dims=(-2, -1))
        ic_180 = torch.rot90(ic, k=2, dims=(-2, -1))
        ic_270 = torch.rot90(ic, k=3, dims=(-2, -1))

        # Compute correlation with rotated versions
        ic_centered = ic - ic.mean(dim=(-2, -1), keepdim=True)

        def corr(ic1, ic2):
            ic1_c = ic1 - ic1.mean(dim=(-2, -1), keepdim=True)
            ic2_c = ic2 - ic2.mean(dim=(-2, -1), keepdim=True)
            num = (ic1_c * ic2_c).reshape(B, M, -1).sum(dim=-1)
            denom = torch.sqrt((ic1_c ** 2).reshape(B, M, -1).sum(dim=-1) *
                             (ic2_c ** 2).reshape(B, M, -1).sum(dim=-1))
            return num / (denom + 1e-6)

        corr_90 = corr(ic, ic_90)
        corr_180 = corr(ic, ic_180)
        corr_270 = corr(ic, ic_270)

        # Average absolute correlation (symmetry = high correlation with rotations)
        symmetry = (torch.abs(corr_90) + torch.abs(corr_180) + torch.abs(corr_270)) / 3.0

        return symmetry.clamp(0, 1)
