"""
Spectral feature extraction using 2D FFT.

Extracts frequency-domain features from 2D fields:
- FFT power spectrum (multiscale frequency bands)
- Dominant frequencies and magnitudes
- Spectral centroids (power-weighted frequency centers)
- Spectral ratios (energy distribution across bands)
- Spectral flatness and rolloff
- Anisotropy and orientation

All operations use PyTorch's optimized cuFFT backend for GPU acceleration.
"""

import torch
import torch.fft
from typing import Dict, Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from spinlock.features.temporal.config import SummarySpectralConfig
    from spinlock.features.ops import FeatureOps


class SpectralFeatureExtractor:
    """
    Extract spectral features from 2D fields using FFT.

    Uses PyTorch's optimized cuFFT for GPU-accelerated transforms.
    Features adapt to grid size (relative frequency bands, not hardcoded).

    Example:
        >>> extractor = SpectralFeatureExtractor(device='cuda')
        >>> fields = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
        >>> features = extractor.extract(fields, num_scales=5)
    """

    def __init__(self, device: torch.device = torch.device('cuda'), ops: Optional['FeatureOps'] = None):
        """
        Initialize spectral feature extractor.

        Args:
            device: Computation device (cuda or cpu)
            ops: FeatureOps provider for gradient-safe operations.
                 If None, creates StandardOps (backward-compatible).
        """
        self.device = device
        if ops is None:
            from spinlock.features.ops import StandardOps
            ops = StandardOps()
        self.ops = ops

    def extract(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W] or [N, T, C, H, W]
        config: Optional['SummarySpectralConfig'] = None,
        num_scales: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Extract spectral features from fields.

        Args:
            fields: Input fields [N, M, T, C, H, W] or [N, T, C, H, W]
            config: Optional SummarySpectralConfig for feature selection
            num_scales: Number of frequency band scales (default: 5)

        Returns:
            Dictionary mapping feature names to tensors
            Each tensor has shape [N, T, C] or [N, M, T, C]
        """
        # Handle both [N,M,T,C,H,W] and [N,T,C,H,W] inputs
        if fields.ndim == 6:
            N, M, T, C, H, W = fields.shape
            fields = fields.reshape(N * M, T, C, H, W)
            has_realizations = True
        else:
            N, T, C, H, W = fields.shape
            M = 1
            has_realizations = False

        # Reshape to [N*T, C, H, W] for batched FFT
        NT = fields.shape[0] * T
        fields_flat = fields.reshape(NT, C, H, W)

        # Compute 2D FFT power spectrum with orthonormal normalization
        # norm='ortho' ensures DC component doesn't scale with grid size
        fft_result = torch.fft.rfft2(fields_flat, dim=(-2, -1), norm='ortho')
        power = torch.abs(fft_result) ** 2  # [NT, C, H, W//2+1]

        features = {}

        # Use config to determine which features to extract
        if config is None:
            include_all = True
        else:
            include_all = False
            if hasattr(config, 'num_fft_scales'):
                num_scales = config.num_fft_scales

        # FFT power spectrum (multiscale)
        if include_all or (config is not None and config.include_fft_power):
            power_features = self._compute_power_spectrum_features(
                power, H, W, num_scales
            )
            features.update(power_features)

        # Dominant frequencies
        if include_all or (config is not None and (config.include_dominant_freq or config.include_dominant_freq_magnitude)):
            dom_freq = self._compute_dominant_frequency(power, H, W)
            if include_all or (config is not None and config.include_dominant_freq):
                features['dominant_freq_x'] = dom_freq['freq_x']
                features['dominant_freq_y'] = dom_freq['freq_y']
            if include_all or (config is not None and config.include_dominant_freq_magnitude):
                features['dominant_freq_magnitude'] = dom_freq['magnitude']

        # Spectral centroids
        if include_all or (config is not None and (config.include_spectral_centroid_x or config.include_spectral_centroid_y or config.include_spectral_bandwidth)):
            centroids = self._compute_spectral_centroids(power, H, W)
            if include_all or (config is not None and config.include_spectral_centroid_x):
                features['spectral_centroid_x'] = centroids['centroid_x']
            if include_all or (config is not None and config.include_spectral_centroid_y):
                features['spectral_centroid_y'] = centroids['centroid_y']
            if include_all or (config is not None and config.include_spectral_bandwidth):
                features['spectral_bandwidth'] = centroids['bandwidth']

        # Spectral ratios
        if include_all or (config is not None and config.include_low_freq_ratio):
            ratios = self._compute_frequency_ratios(power, H, W)
            features['low_freq_ratio'] = ratios['low']
            features['mid_freq_ratio'] = ratios['mid']
            features['high_freq_ratio'] = ratios['high']

        # Spectral flatness
        if include_all or (config is not None and config.include_spectral_flatness):
            features['spectral_flatness'] = self._compute_spectral_flatness(power)

        # Spectral entropy (NEW)
        if include_all or (config is not None and getattr(config, 'include_spectral_entropy', False)):
            features['spectral_entropy'] = self._compute_spectral_entropy(power)

        # Spectral rolloff
        if include_all or (config is not None and config.include_spectral_rolloff):
            features['spectral_rolloff'] = self._compute_spectral_rolloff(power)

        # Spectral anisotropy
        if include_all or (config is not None and config.include_spectral_anisotropy):
            aniso = self._compute_spectral_anisotropy(power, H, W)
            features['spectral_anisotropy'] = aniso

        # Harmonic content (detects nonlinearity via harmonic generation)
        if include_all or (config is not None and getattr(config, 'include_harmonic_content', False)):
            harmonics = self._compute_harmonic_content(power, H, W)
            features['harmonic_ratio_2f'] = harmonics['harmonic_ratio_2f']
            features['harmonic_ratio_3f'] = harmonics['harmonic_ratio_3f']
            features['total_harmonic_distortion'] = harmonics['total_harmonic_distortion']
            features['fundamental_purity'] = harmonics['fundamental_purity']

        # Orthogonal spectral features (NEW in v3.2)
        if include_all or (config is not None and getattr(config, 'include_wavelet_features', False)):
            wavelet = self._extract_wavelet_features(fields_flat)
            features.update(wavelet)

        # Reshape all features back
        for name, feat in features.items():
            if feat.ndim == 2:  # [NT, C]
                if has_realizations:
                    features[name] = feat.reshape(N, M, T, C)
                else:
                    features[name] = feat.reshape(N, T, C)

        # Apply adaptive outlier clipping to prevent extreme values
        for name in features:
            features[name] = self.ops.outlier_clip(features[name], iqr_multiplier=10.0)

        return features

    # =========================================================================
    # Power Spectrum Features
    # =========================================================================

    def _compute_power_spectrum_features(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int,
        num_scales: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute power spectrum features across multiple frequency bands.

        Divides frequency space into logarithmically-spaced bands and
        computes statistics per band.

        Args:
            power: FFT power spectrum [NT, C, H, W//2+1]
            H, W: Spatial dimensions
            num_scales: Number of frequency bands

        Returns:
            Dict with keys like "fft_power_scale_0_mean", etc.
        """
        NT, C, fft_H, fft_W = power.shape
        features = {}

        # Create radial frequency map
        # fftfreq returns normalized frequencies in cycles per sample (range -0.5 to 0.5)
        # rfftfreq returns normalized frequencies for real FFT (range 0 to 0.5)
        freq_y = torch.fft.fftfreq(H, d=1.0, device=power.device)[:, None]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=power.device)[None, :]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2) * min(H, W)

        # Use actual maximum frequency from the grid (accounts for diagonal)
        max_freq = freq_radial.max().item()

        # Sqrt-spaced frequency bands for balanced distribution
        # Start just above DC (0.5) to avoid extremely small first band
        freq_edges = torch.linspace(
            math.sqrt(0.5),
            math.sqrt(max_freq),
            num_scales + 1,
            device=power.device
        ) ** 2  # Square to get back to frequency space

        # Compute features for each frequency band
        for scale_idx in range(num_scales):
            low_freq = freq_edges[scale_idx]
            high_freq = freq_edges[scale_idx + 1]

            # Mask for this frequency band
            mask = (freq_radial >= low_freq) & (freq_radial < high_freq)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W//2+1]

            # Extract power in this band
            band_power = power * mask  # [NT, C, H, W//2+1]

            # Mean power in band
            mean_power = band_power.sum(dim=(-2, -1)) / (mask.sum() + 1e-8)
            features[f'fft_power_scale_{scale_idx}_mean'] = mean_power

            # Max power in band (per-bin normalized for grid-size independence)
            max_power = band_power.amax(dim=(-2, -1)) / (mask.sum() + 1e-8)
            max_power = torch.clamp(max_power, max=1e6)  # Prevent overflow
            features[f'fft_power_scale_{scale_idx}_max'] = max_power

            # Std of power in band (normalized by sqrt(N) for grid-size independence)
            power_flat = band_power.flatten(start_dim=2)  # [NT, C, H*W]
            std_power = power_flat.std(dim=2) / torch.sqrt(mask.sum() + 1e-8)
            features[f'fft_power_scale_{scale_idx}_std'] = std_power

        return features

    # =========================================================================
    # Dominant Frequencies
    # =========================================================================

    def _compute_dominant_frequency(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dominant frequency (peak in power spectrum).

        Returns frequency coordinates and magnitude at peak.
        """
        NT, C, fft_H, fft_W = power.shape

        # Find peak in power spectrum (per channel)
        power_flat = power.flatten(start_dim=2)  # [NT, C, H*(W//2+1)]
        peak_idx = self.ops.soft_argmax(power_flat, dim=2)  # [NT, C] (float)

        # Convert float flat index to (y, x) coordinates
        peak_y = peak_idx / fft_W  # float division for continuous coords
        peak_x = peak_idx - torch.floor(peak_y) * fft_W

        # Convert to frequency values (normalized by grid size)
        freq_y = peak_y / H
        freq_x = peak_x / W

        # Get magnitude at peak using weighted sum (differentiable)
        weights = torch.softmax(power_flat * 10.0, dim=2)  # sharpen around peak
        magnitude = (power_flat * weights).sum(dim=2)

        return {
            'freq_x': freq_x,
            'freq_y': freq_y,
            'magnitude': magnitude
        }

    # =========================================================================
    # Spectral Centroids
    # =========================================================================

    def _compute_spectral_centroids(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spectral centroids (power-weighted frequency centers).

        Analogous to center of mass in frequency space.
        """
        NT, C, fft_H, fft_W = power.shape

        # Create frequency grids shaped for broadcasting with [NT, C, H, W//2+1]
        freq_y = torch.fft.fftfreq(H, d=1.0, device=power.device)[None, None, :, None]  # [1, 1, H, 1]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=power.device)[None, None, None, :]  # [1, 1, 1, W//2+1]

        # Total power (for normalization)
        total_power = power.sum(dim=(-2, -1), keepdim=True)  # [NT, C, 1, 1]
        total_power_2d = total_power.squeeze(-1).squeeze(-1)  # [NT, C]

        # Power-weighted frequency
        centroid_y = self.ops.safe_div((power * freq_y).sum(dim=(-2, -1)), total_power_2d)
        centroid_x = self.ops.safe_div((power * freq_x).sum(dim=(-2, -1)), total_power_2d)

        # Spectral bandwidth (spread around centroid)
        # freq_y and freq_x are already shaped for broadcasting: [1, 1, H, 1] and [1, 1, 1, W//2+1]
        # centroid_y and centroid_x have shape [NT, C]
        centroid_y_expanded = centroid_y[:, :, None, None]  # [NT, C, 1, 1]
        centroid_x_expanded = centroid_x[:, :, None, None]  # [NT, C, 1, 1]

        deviation_y = (freq_y - centroid_y_expanded) ** 2  # [NT, C, H, 1]
        deviation_x = (freq_x - centroid_x_expanded) ** 2  # [NT, C, 1, W//2+1]

        variance = self.ops.safe_div((power * (deviation_y + deviation_x)).sum(dim=(-2, -1)), total_power_2d)
        bandwidth = torch.sqrt(variance + 1e-8)

        return {
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bandwidth': bandwidth
        }

    # =========================================================================
    # Frequency Ratios
    # =========================================================================

    def _compute_frequency_ratios(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy ratios in low/mid/high frequency bands.

        Adaptively scales with grid size.
        """
        # Create radial frequency map
        freq_y = torch.fft.fftfreq(H, d=1.0, device=power.device)[:, None]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=power.device)[None, :]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)

        # Define frequency band cutoffs (adaptive to grid size)
        max_freq = min(H, W) / 2
        cutoff_low = max_freq / 8  # Low frequency cutoff
        cutoff_high = max_freq / 4  # High frequency cutoff

        # Masks for each band
        low_mask = freq_radial < cutoff_low
        mid_mask = (freq_radial >= cutoff_low) & (freq_radial < cutoff_high)
        high_mask = freq_radial >= cutoff_high

        # Energy in each band
        low_energy = (power * low_mask).sum(dim=(-2, -1))
        mid_energy = (power * mid_mask).sum(dim=(-2, -1))
        high_energy = (power * high_mask).sum(dim=(-2, -1))
        total_energy = power.sum(dim=(-2, -1)) + 1e-8

        # Ratios
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        return {
            'low': low_ratio,
            'mid': mid_ratio,
            'high': high_ratio
        }

    # =========================================================================
    # Spectral Flatness and Rolloff
    # =========================================================================

    def _compute_spectral_flatness(self, power: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral flatness (tonality measure).

        Flatness = geometric_mean / arithmetic_mean
        Close to 1 for noise-like spectra, close to 0 for tonal.
        """
        # Flatten spatial dimensions
        power_flat = power.flatten(start_dim=2)  # [NT, C, H*W]

        # Geometric mean (using log for numerical stability)
        log_power = self.ops.safe_log(power_flat, eps=1e-10)
        geometric_mean = torch.exp(log_power.mean(dim=2))

        # Arithmetic mean
        arithmetic_mean = power_flat.mean(dim=2)

        flatness = self.ops.safe_div(geometric_mean, arithmetic_mean)

        return flatness

    def _compute_spectral_entropy(self, power: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral entropy (Shannon entropy of normalized PSD).

        Measures spectral complexity: high entropy = noise-like, low entropy = tonal.
        Complements spectral flatness with information-theoretic perspective.

        Args:
            power: FFT power spectrum [NT, C, H, W//2+1]

        Returns:
            [NT, C] spectral entropy values
        """
        # Flatten spatial dimensions
        power_flat = power.flatten(start_dim=2)  # [NT, C, H*W]

        # Normalize to probability distribution
        eps = 1e-10
        power_sum = power_flat.sum(dim=2, keepdim=True) + eps
        psd_norm = power_flat / power_sum  # [NT, C, H*W]

        # Shannon entropy: H = -sum(p * log(p))
        spectral_entropy = -(psd_norm * self.ops.safe_log(psd_norm, eps=eps)).sum(dim=2)  # [NT, C]

        return spectral_entropy

    def _compute_spectral_rolloff(
        self,
        power: torch.Tensor,
        percentile: float = 0.85
    ) -> torch.Tensor:
        """
        Compute spectral rolloff frequency.

        Frequency below which `percentile` (default 85%) of power is contained.
        """
        # Flatten and sort power spectrum
        power_flat = power.flatten(start_dim=2)  # [NT, C, H*W]

        # Cumulative power
        sorted_power, _ = torch.sort(power_flat, dim=2, descending=True)
        cumsum = torch.cumsum(sorted_power, dim=2)
        total = cumsum[:, :, -1:]

        # Find index where cumsum exceeds percentile * total
        threshold = percentile * total
        # Use soft_argmax on a signal that peaks at the rolloff point
        rolloff_signal = self.ops.soft_step(cumsum, threshold)
        rolloff_idx = self.ops.soft_argmax(rolloff_signal, dim=2)

        # Normalize by total number of frequency bins
        rolloff_freq = rolloff_idx / power_flat.shape[2]

        return rolloff_freq

    # =========================================================================
    # Anisotropy
    # =========================================================================

    def _compute_spectral_anisotropy(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Compute spectral anisotropy (directional power imbalance).

        Compares power in horizontal vs vertical frequency bands.
        """
        # Sum power along x and y directions
        power_x = power.sum(dim=2)  # Sum over y-frequencies: [NT, C, W//2+1]
        power_y = power.sum(dim=3)  # Sum over x-frequencies: [NT, C, H]

        # Total power in each direction
        total_power_x = power_x.sum(dim=2)
        total_power_y = power_y.sum(dim=2)

        # Anisotropy ratio
        anisotropy = self.ops.safe_div(total_power_x, total_power_y)

        return anisotropy

    # =========================================================================
    # Harmonic Content
    # =========================================================================

    # =========================================================================
    # Orthogonal Spectral Features (NEW in v3.2)
    # =========================================================================

    def _extract_wavelet_features(self, fields_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract wavelet decomposition features (orthogonal to FFT).

        Uses Discrete Wavelet Transform (DWT) for multi-resolution analysis.
        Wavelets are localized in both space and frequency (FFT is global).

        Args:
            fields_flat: [NT, C, H, W] input fields

        Returns:
            Dictionary with 15 features per channel:
                - Approximation coefficients (low-freq): 3D [mean, std, energy]
                - Detail coefficients (high-freq directional): 9D [H/V/D × mean/std/energy]
                - Wavelet entropy: 3D [H/V/D]

        Note:
            Uses a simple Haar wavelet implemented via average pooling for
            differentiability and GPU efficiency (no external dependencies).
        """
        NT, C, H, W = fields_flat.shape
        features = {}

        # Simple Haar wavelet via average pooling (differentiable, GPU-native)
        # Approximation: 2x2 average (low-pass)
        # Details: differences from average (high-pass)

        # Ensure even dimensions for clean 2x2 pooling
        if H % 2 == 1:
            fields_flat = F.pad(fields_flat, (0, 0, 0, 1), mode='replicate')
            H = H + 1
        if W % 2 == 1:
            fields_flat = F.pad(fields_flat, (0, 1, 0, 0), mode='replicate')
            W = W + 1

        # Reshape to extract 2x2 blocks: [NT, C, H/2, 2, W/2, 2]
        blocks = fields_flat.reshape(NT, C, H // 2, 2, W // 2, 2)

        # Extract 4 components from each 2x2 block
        # LL (low-low): approximation
        # LH (low-high): horizontal details
        # HL (high-low): vertical details
        # HH (high-high): diagonal details
        LL = blocks.mean(dim=(3, 5))  # [NT, C, H/2, W/2]
        LH = (blocks[:, :, :, 0, :, :] - blocks[:, :, :, 1, :, :]).mean(dim=-1)  # Horizontal detail
        HL = (blocks[:, :, :, :, :, 0] - blocks[:, :, :, :, :, 1]).mean(dim=-2)  # Vertical detail
        HH = (blocks[:, :, :, 0, :, 0] - blocks[:, :, :, 1, :, 1])  # Diagonal detail

        # --- Approximation coefficients (low-freq) ---
        features['wavelet_approx_mean'] = LL.mean(dim=(-2, -1))  # [NT, C]
        features['wavelet_approx_std'] = LL.std(dim=(-2, -1))  # [NT, C]
        features['wavelet_approx_energy'] = (LL ** 2).sum(dim=(-2, -1)) / (H * W / 4)  # Normalized

        # --- Detail coefficients (high-freq directional) ---
        for name, detail in [('horizontal', LH), ('vertical', HL), ('diagonal', HH)]:
            features[f'wavelet_{name}_mean'] = detail.mean(dim=(-2, -1))  # [NT, C]
            features[f'wavelet_{name}_std'] = detail.std(dim=(-2, -1))  # [NT, C]
            features[f'wavelet_{name}_energy'] = (detail ** 2).sum(dim=(-2, -1)) / (H * W / 4)  # [NT, C]

        # --- Wavelet entropy (regularity measure) ---
        # Shannon entropy of wavelet coefficient magnitudes
        for name, detail in [('horizontal', LH), ('vertical', HL), ('diagonal', HH)]:
            detail_abs = detail.abs()
            # Normalize to probability distribution
            detail_sum = detail_abs.sum(dim=(-2, -1), keepdim=True) + 1e-10
            detail_prob = detail_abs / detail_sum  # [NT, C, H/2, W/2]
            # Entropy: -sum(p * log(p))
            entropy = -(detail_prob * self.ops.safe_log(detail_prob, eps=1e-10)).sum(dim=(-2, -1))  # [NT, C]
            features[f'wavelet_entropy_{name}'] = entropy

        return features

    def _compute_harmonic_content(
        self,
        power: torch.Tensor,  # [NT, C, H, W//2+1]
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute harmonic content features (detects nonlinearity).

        Analyzes energy at harmonic multiples of the fundamental frequency.
        Nonlinear operators generate harmonics (2f, 3f, ...) from fundamental.

        Features:
        - harmonic_ratio_2f: Power at 2× fundamental / fundamental power
        - harmonic_ratio_3f: Power at 3× fundamental / fundamental power
        - total_harmonic_distortion: THD = sqrt(P_2f² + P_3f²) / P_f
        - fundamental_purity: Fundamental power / total power

        Args:
            power: FFT power spectrum [NT, C, H, W//2+1]
            H, W: Spatial dimensions

        Returns:
            Dictionary with 4 harmonic features [NT, C]
        """
        NT, C, fft_H, fft_W = power.shape

        # Create radial frequency grid
        freq_y = torch.fft.fftfreq(H, d=1.0, device=power.device)[:, None]
        freq_x = torch.fft.rfftfreq(W, d=1.0, device=power.device)[None, :]
        freq_radial = torch.sqrt(freq_y ** 2 + freq_x ** 2)

        # Find dominant (fundamental) frequency using soft argmax for differentiability
        power_flat = power.reshape(NT * C, fft_H, fft_W)
        power_flat_2d = power_flat.reshape(NT * C, -1)  # [NT*C, fft_H*fft_W]

        # Soft argmax gives continuous index
        soft_idx = self.ops.soft_argmax(power_flat_2d, dim=1)  # [NT*C] (float)
        # Convert float flat index to (y, x) coordinates
        soft_idx_y = soft_idx / fft_W
        soft_idx_x = soft_idx - torch.floor(soft_idx_y) * fft_W

        # Fundamental frequency from soft coordinates
        freq_y_flat = freq_y.flatten()  # [fft_H]
        freq_x_flat = freq_x.flatten()  # [fft_W]
        # Use linear interpolation for differentiable frequency lookup
        fund_freq_y_idx = soft_idx_y.clamp(0, fft_H - 1)
        fund_freq_x_idx = soft_idx_x.clamp(0, fft_W - 1)
        # Floor/ceil for interpolation
        fy_lo = torch.floor(fund_freq_y_idx).long().clamp(0, fft_H - 1)
        fy_hi = (fy_lo + 1).clamp(0, fft_H - 1)
        fx_lo = torch.floor(fund_freq_x_idx).long().clamp(0, fft_W - 1)
        fx_hi = (fx_lo + 1).clamp(0, fft_W - 1)
        fy_frac = fund_freq_y_idx - fy_lo.float()
        fx_frac = fund_freq_x_idx - fx_lo.float()
        fund_freq_y_val = freq_y_flat[fy_lo] * (1 - fy_frac) + freq_y_flat[fy_hi] * fy_frac
        fund_freq_x_val = freq_x_flat[fx_lo] * (1 - fx_frac) + freq_x_flat[fx_hi] * fx_frac
        fund_freq_radial = torch.sqrt(fund_freq_y_val ** 2 + fund_freq_x_val ** 2 + 1e-10)  # [NT*C]

        # Extract power at fundamental using softmax-weighted sum (differentiable)
        weights = torch.softmax(power_flat_2d * 10.0, dim=1)
        fund_power = (power_flat_2d * weights).sum(dim=1)  # [NT*C]

        # For each sample, find power near 2f and 3f harmonics
        # Use soft Gaussian annulus around harmonic frequency
        tolerance_sigma = 0.1  # Controls width of annulus

        # 2nd harmonic (2f)
        freq_2f = 2.0 * fund_freq_radial.unsqueeze(1).unsqueeze(2)  # [NT*C, 1, 1]
        dist_2f = (freq_radial - freq_2f).abs() / (freq_2f.abs() + 1e-8)
        soft_mask_2f = torch.exp(-0.5 * (dist_2f / tolerance_sigma) ** 2)
        power_2f = self.ops.safe_div(
            (power_flat * soft_mask_2f).sum(dim=(-2, -1)),
            soft_mask_2f.sum(dim=(-2, -1)),
        )  # [NT*C]

        # 3rd harmonic (3f)
        freq_3f = 3.0 * fund_freq_radial.unsqueeze(1).unsqueeze(2)  # [NT*C, 1, 1]
        dist_3f = (freq_radial - freq_3f).abs() / (freq_3f.abs() + 1e-8)
        soft_mask_3f = torch.exp(-0.5 * (dist_3f / tolerance_sigma) ** 2)
        power_3f = self.ops.safe_div(
            (power_flat * soft_mask_3f).sum(dim=(-2, -1)),
            soft_mask_3f.sum(dim=(-2, -1)),
        )  # [NT*C]

        # Total power
        total_power = power_flat.sum(dim=(-2, -1))  # [NT*C]

        # Compute harmonic ratios
        harmonic_ratio_2f = self.ops.safe_div(power_2f, fund_power)
        harmonic_ratio_3f = self.ops.safe_div(power_3f, fund_power)

        # Total Harmonic Distortion (THD)
        thd = self.ops.safe_div(torch.sqrt(power_2f ** 2 + power_3f ** 2 + 1e-10), fund_power)

        # Fundamental purity
        fundamental_purity = self.ops.safe_div(fund_power, total_power)

        # Reshape back to [NT, C]
        harmonic_ratio_2f = harmonic_ratio_2f.reshape(NT, C)
        harmonic_ratio_3f = harmonic_ratio_3f.reshape(NT, C)
        thd = thd.reshape(NT, C)
        fundamental_purity = fundamental_purity.reshape(NT, C)

        return {
            'harmonic_ratio_2f': harmonic_ratio_2f,
            'harmonic_ratio_3f': harmonic_ratio_3f,
            'total_harmonic_distortion': thd,
            'fundamental_purity': fundamental_purity,
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
            methods: Aggregation methods

        Returns:
            Aggregated features dict
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
                    if feat.ndim == 3:
                        aggregated[agg_name] = feat[:, -1, :]
                    else:
                        aggregated[agg_name] = feat[:, :, -1, :]

        return aggregated
