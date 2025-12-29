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
    from spinlock.features.sdf.config import SDFSpectralConfig


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

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize spectral feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W] or [N, T, C, H, W]
        config: Optional['SDFSpectralConfig'] = None,
        num_scales: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Extract spectral features from fields.

        Args:
            fields: Input fields [N, M, T, C, H, W] or [N, T, C, H, W]
            config: Optional SDFSpectralConfig for feature selection
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

        # Reshape all features back
        for name, feat in features.items():
            if feat.ndim == 2:  # [NT, C]
                if has_realizations:
                    features[name] = feat.reshape(N, M, T, C)
                else:
                    features[name] = feat.reshape(N, T, C)

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

            # Max power in band
            max_power = band_power.amax(dim=(-2, -1))
            features[f'fft_power_scale_{scale_idx}_max'] = max_power

            # Std of power in band
            power_flat = band_power.flatten(start_dim=2)  # [NT, C, H*W]
            std_power = power_flat.std(dim=2)
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
        peak_idx = torch.argmax(power_flat, dim=2)  # [NT, C]

        # Convert flat index to (y, x) coordinates
        peak_y = peak_idx // fft_W
        peak_x = peak_idx % fft_W

        # Convert to frequency values (normalized by grid size)
        freq_y = peak_y.float() / H
        freq_x = peak_x.float() / W

        # Get magnitude at peak
        magnitude = power_flat.gather(2, peak_idx.unsqueeze(2)).squeeze(2)

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
        total_power = power.sum(dim=(-2, -1), keepdim=True) + 1e-8  # [NT, C, 1, 1]

        # Power-weighted frequency
        centroid_y = (power * freq_y).sum(dim=(-2, -1)) / total_power.squeeze(-1).squeeze(-1)
        centroid_x = (power * freq_x).sum(dim=(-2, -1)) / total_power.squeeze(-1).squeeze(-1)

        # Spectral bandwidth (spread around centroid)
        # freq_y and freq_x are already shaped for broadcasting: [1, 1, H, 1] and [1, 1, 1, W//2+1]
        # centroid_y and centroid_x have shape [NT, C]
        centroid_y_expanded = centroid_y[:, :, None, None]  # [NT, C, 1, 1]
        centroid_x_expanded = centroid_x[:, :, None, None]  # [NT, C, 1, 1]

        deviation_y = (freq_y - centroid_y_expanded) ** 2  # [NT, C, H, 1]
        deviation_x = (freq_x - centroid_x_expanded) ** 2  # [NT, C, 1, W//2+1]

        variance = (power * (deviation_y + deviation_x)).sum(dim=(-2, -1)) / total_power.squeeze(-1).squeeze(-1)
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
        log_power = torch.log(power_flat + 1e-10)
        geometric_mean = torch.exp(log_power.mean(dim=2))

        # Arithmetic mean
        arithmetic_mean = power_flat.mean(dim=2) + 1e-10

        flatness = geometric_mean / arithmetic_mean

        return flatness

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
        rolloff_idx = (cumsum >= threshold).int().argmax(dim=2)

        # Normalize by total number of frequency bins
        rolloff_freq = rolloff_idx.float() / power_flat.shape[2]

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
        anisotropy = total_power_x / (total_power_y + 1e-8)

        return anisotropy

    # =========================================================================
    # Harmonic Content
    # =========================================================================

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

        # Find dominant (fundamental) frequency per sample/channel
        power_flat = power.reshape(NT * C, fft_H, fft_W)
        max_idx = power_flat.reshape(NT * C, -1).argmax(dim=1)
        max_idx_y = max_idx // fft_W
        max_idx_x = max_idx % fft_W

        # Fundamental frequency
        fund_freq_y = freq_y.flatten()[max_idx_y]
        fund_freq_x = freq_x.flatten()[max_idx_x]
        fund_freq_radial = torch.sqrt(fund_freq_y ** 2 + fund_freq_x ** 2)  # [NT*C]

        # Extract power at fundamental (peak power by definition)
        fund_power = power_flat[torch.arange(NT * C), max_idx_y, max_idx_x]  # [NT*C]

        # For each sample, find power near 2f and 3f harmonics
        # Use annulus around harmonic frequency (±10% tolerance)
        tolerance = 0.1

        # 2nd harmonic (2f)
        freq_2f = 2.0 * fund_freq_radial.unsqueeze(1).unsqueeze(2)  # [NT*C, 1, 1]
        mask_2f = (freq_radial >= freq_2f * (1 - tolerance)) & (freq_radial <= freq_2f * (1 + tolerance))
        power_2f = (power_flat * mask_2f).sum(dim=(-2, -1)) / (mask_2f.sum(dim=(-2, -1)) + 1e-8)  # [NT*C]

        # 3rd harmonic (3f)
        freq_3f = 3.0 * fund_freq_radial.unsqueeze(1).unsqueeze(2)  # [NT*C, 1, 1]
        mask_3f = (freq_radial >= freq_3f * (1 - tolerance)) & (freq_radial <= freq_3f * (1 + tolerance))
        power_3f = (power_flat * mask_3f).sum(dim=(-2, -1)) / (mask_3f.sum(dim=(-2, -1)) + 1e-8)  # [NT*C]

        # Total power
        total_power = power_flat.sum(dim=(-2, -1))  # [NT*C]

        # Compute harmonic ratios
        harmonic_ratio_2f = power_2f / (fund_power + 1e-8)  # [NT*C]
        harmonic_ratio_3f = power_3f / (fund_power + 1e-8)  # [NT*C]

        # Total Harmonic Distortion (THD)
        thd = torch.sqrt(power_2f ** 2 + power_3f ** 2) / (fund_power + 1e-8)  # [NT*C]

        # Fundamental purity
        fundamental_purity = fund_power / (total_power + 1e-8)  # [NT*C]

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
