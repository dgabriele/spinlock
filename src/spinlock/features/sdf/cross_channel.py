"""
Cross-channel interaction feature extraction.

Extracts per-timestep features measuring channel coupling structure:
- Correlation matrix eigendecomposition (effective dimensionality)
- Pairwise correlation statistics (fallback summary)
- Cross-spectral coherence (phase-locked coupling, optional)
- Mutual information (nonlinear coupling, optional)

Optimized for Mid-C operators (5-16 channels), degrades gracefully for High-C (32+).

Example:
    >>> extractor = CrossChannelFeatureExtractor(device='cuda')
    >>> fields = torch.randn(32, 10, 100, 8, 128, 128, device='cuda')  # 8 channels
    >>> features = extractor.extract(fields)  # Dict of per-timestep features
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.sdf.config import SDFCrossChannelConfig


class CrossChannelFeatureExtractor:
    """
    Extract cross-channel interaction features from 2D fields.

    Measures channel coupling at each timestep using correlation spectra,
    coherence, and mutual information. Designed for Mid-C operators (5-16 channels).

    Operates on fields [N, M, T, C, H, W] and computes per-timestep features.

    Example:
        >>> extractor = CrossChannelFeatureExtractor(device='cuda')
        >>> fields = torch.randn(8, 10, 50, 6, 128, 128, device='cuda')
        >>> features = extractor.extract(fields)
        >>> # Returns dict with ~10 features shaped [N, M, T, C] or [N, M, T]
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize cross-channel feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        fields: torch.Tensor,  # [N, M, T, C, H, W] or [N, T, C, H, W]
        config: Optional['SDFCrossChannelConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract cross-channel features from fields.

        Args:
            fields: Input fields [N, M, T, C, H, W] or [N, T, C, H, W]
                N = batch size
                M = num realizations (optional)
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SDFCrossChannelConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors
            Each tensor has shape [N, M, T] or [N, T] (scalar per timestep)
            Returns NaN for C=1 (no cross-channel interactions possible)
        """
        # Handle [N,M,T,C,H,W], [N,T,C,H,W], or [Batch,C,H,W] inputs
        is_batched_input = False
        if fields.ndim == 6:
            N, M, T, C, H, W = fields.shape
            has_realizations = True
        elif fields.ndim == 5:
            N, T, C, H, W = fields.shape
            M = 1
            fields = fields.unsqueeze(1)  # Add M dimension
            has_realizations = False
        elif fields.ndim == 4:
            # Batched input [Batch, C, H, W] from orchestrator
            # Treat as [Batch, 1, 1, C, H, W] for consistency
            Batch, C, H, W = fields.shape
            N = Batch
            M = 1
            T = 1
            fields = fields.reshape(Batch, 1, 1, C, H, W)
            has_realizations = False
            is_batched_input = True  # Flag to return [Batch] shape
        else:
            raise ValueError(f"Unexpected input shape: {fields.shape}. Expected [N,M,T,C,H,W], [N,T,C,H,W], or [Batch,C,H,W]")

        # Edge case: C=1 (no cross-channel interactions possible)
        if C == 1:
            return self._return_nan_features(N, M, T, fields.device, config, is_batched_input)

        features = {}

        # Use config to determine which features to extract
        if config is None:
            include_all = True
        else:
            include_all = False

        # Reshape to [N*M*T, C, H, W] for batched computation
        NMT = N * M * T
        fields_flat = fields.reshape(NMT, C, H, W)

        # Always include: Correlation matrix eigendecomposition
        if include_all or (config and config.include_eigen_values):
            eigen_features = self._compute_correlation_eigen(fields_flat, config)
            features.update(eigen_features)

        # Always include: Pairwise correlation statistics
        if include_all or (config and (config.include_corr_mean or config.include_corr_max)):
            corr_stats = self._compute_pairwise_correlation_stats(fields_flat, config)
            features.update(corr_stats)

        # Optional: Cross-spectral coherence (expensive)
        if config is not None and config.include_coherence:
            # Skip for High-C operators (too expensive)
            if C <= config.max_channels_for_full_corr:
                coherence_features = self._compute_cross_spectral_coherence(fields_flat, config)
                features.update(coherence_features)

        # Optional: Mutual information (expensive, nonlinear coupling)
        if config is not None and config.include_mutual_info:
            # Skip for High-C operators
            if C <= config.max_channels_for_full_corr:
                mi_features = self._compute_mutual_information(fields_flat, config)
                features.update(mi_features)

        # Optional: Conditional mutual information (very expensive, higher-order dependencies)
        if config is not None and getattr(config, 'include_conditional_mi', False):
            # Skip for High-C operators and require at least 3 channels
            if C >= 3 and C <= config.max_channels_for_full_corr:
                cmi_features = self._compute_conditional_mutual_information(fields_flat, config)
                features.update(cmi_features)

        # Reshape all features back to [N, M, T] or [N, T]
        # Exception: For batched input from orchestrator, keep [Batch] shape
        for name, feat in features.items():
            if feat.ndim == 1:  # [NMT]
                if is_batched_input:
                    # Keep as [Batch] for orchestrator
                    pass  # Already in correct shape
                elif has_realizations:
                    features[name] = feat.reshape(N, M, T)
                else:
                    features[name] = feat.reshape(N, T)

        return features

    def _return_nan_features(
        self,
        N: int,
        M: int,
        T: int,
        device: torch.device,
        config: Optional['SDFCrossChannelConfig'] = None,
        is_batched_input: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Return NaN features for C=1 edge case.

        Args:
            N: Batch size (or number of samples)
            M: Number of realizations
            T: Number of timesteps
            device: Device for tensors
            config: Optional config
            is_batched_input: If True, return [Batch] shape instead of [N, M, T]

        Returns:
            Dictionary of NaN features
        """
        if is_batched_input:
            nan_val = torch.full((N,), float('nan'), device=device)
        else:
            nan_val = torch.full((N, M, T), float('nan'), device=device)
        features = {}

        # Eigenvalue features
        num_eigen = config.num_eigen_top if config else 3
        for i in range(1, num_eigen + 1):
            features[f'cross_channel_eigen_top_{i}'] = nan_val.clone()
        features['cross_channel_eigen_trace'] = nan_val.clone()
        features['cross_channel_condition_number'] = nan_val.clone()
        features['cross_channel_participation_ratio'] = nan_val.clone()

        # Correlation statistics
        features['cross_channel_corr_mean'] = nan_val.clone()
        features['cross_channel_corr_max'] = nan_val.clone()
        features['cross_channel_corr_min'] = nan_val.clone()
        features['cross_channel_corr_std'] = nan_val.clone()

        # Optional coherence features
        if config is not None and config.include_coherence:
            for band in config.coherence_freq_bands:
                features[f'cross_spectral_coherence_{band}'] = nan_val.clone()
            features['cross_spectral_coherence_max'] = nan_val.clone()
            features['cross_spectral_coherence_peak_freq'] = nan_val.clone()

        # Optional MI features
        if config is not None and config.include_mutual_info:
            features['cross_channel_mi_mean'] = nan_val.clone()
            features['cross_channel_mi_max'] = nan_val.clone()

        return features

    # =========================================================================
    # Correlation Matrix Eigendecomposition
    # =========================================================================

    def _compute_correlation_eigen(
        self,
        fields: torch.Tensor,  # [NMT, C, H, W]
        config: Optional['SDFCrossChannelConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute correlation matrix eigendecomposition features.

        Extracts eigenvalues to characterize effective dimensionality and
        coupling structure.

        Args:
            fields: Fields [NMT, C, H, W]
            config: Optional config

        Returns:
            Dictionary with eigenvalue-based features [NMT]
        """
        NMT, C, H, W = fields.shape

        # Flatten spatial dimensions: [NMT, C, H*W]
        fields_spatial_flat = fields.reshape(NMT, C, H * W)

        # Standardize per channel (zero mean, unit variance)
        mean = fields_spatial_flat.mean(dim=2, keepdim=True)  # [NMT, C, 1]
        std = fields_spatial_flat.std(dim=2, keepdim=True) + 1e-8  # [NMT, C, 1]
        fields_normalized = (fields_spatial_flat - mean) / std

        # Compute correlation matrix via matrix multiplication
        # corr[i,j] = (1/n) * sum(x_i * x_j) where x_i, x_j are standardized
        corr = torch.bmm(
            fields_normalized,
            fields_normalized.transpose(1, 2)
        ) / (H * W)  # [NMT, C, C]

        # Eigendecomposition (symmetric matrices)
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues = torch.linalg.eigvalsh(corr)  # [NMT, C]

        # Flip to descending order (largest first)
        eigenvalues = eigenvalues.flip(dims=[-1])  # [NMT, C]

        features = {}

        # Top-k eigenvalues
        num_eigen_top = config.num_eigen_top if config else 3
        num_eigen_top = min(num_eigen_top, C)  # Don't exceed C
        for i in range(num_eigen_top):
            features[f'cross_channel_eigen_top_{i+1}'] = eigenvalues[:, i]

        # Trace (sum of eigenvalues, should equal C for correlation matrix)
        features['cross_channel_eigen_trace'] = eigenvalues.sum(dim=1)

        # Condition number (λ_max / λ_min)
        lambda_max = eigenvalues[:, 0]  # Largest eigenvalue
        lambda_min = eigenvalues[:, -1] + 1e-8  # Smallest eigenvalue
        features['cross_channel_condition_number'] = lambda_max / lambda_min

        # Participation ratio: (Σλ_i)² / Σ(λ_i²)
        # Effective dimensionality measure
        sum_lambda = eigenvalues.sum(dim=1)
        sum_lambda_sq = (eigenvalues ** 2).sum(dim=1)
        features['cross_channel_participation_ratio'] = (sum_lambda ** 2) / (sum_lambda_sq + 1e-8)

        return features

    # =========================================================================
    # Pairwise Correlation Statistics
    # =========================================================================

    def _compute_pairwise_correlation_stats(
        self,
        fields: torch.Tensor,  # [NMT, C, H, W]
        config: Optional['SDFCrossChannelConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pairwise correlation statistics.

        Fallback summary for High-C operators or when full eigendecomposition
        is too expensive.

        Args:
            fields: Fields [NMT, C, H, W]
            config: Optional config

        Returns:
            Dictionary with correlation summary statistics [NMT]
        """
        NMT, C, H, W = fields.shape

        # Flatten spatial dimensions: [NMT, C, H*W]
        fields_spatial_flat = fields.reshape(NMT, C, H * W)

        # Standardize per channel
        mean = fields_spatial_flat.mean(dim=2, keepdim=True)
        std = fields_spatial_flat.std(dim=2, keepdim=True) + 1e-8
        fields_normalized = (fields_spatial_flat - mean) / std

        # Compute correlation matrix
        corr = torch.bmm(
            fields_normalized,
            fields_normalized.transpose(1, 2)
        ) / (H * W)  # [NMT, C, C]

        # Extract upper triangle (exclude diagonal)
        # Use torch.triu_indices for efficient indexing
        triu_indices = torch.triu_indices(C, C, offset=1, device=fields.device)
        pairwise_corr = corr[:, triu_indices[0], triu_indices[1]]  # [NMT, C*(C-1)/2]

        features = {}

        # Mean pairwise correlation
        if config is None or config.include_corr_mean:
            features['cross_channel_corr_mean'] = pairwise_corr.mean(dim=1)

        # Max pairwise correlation
        if config is None or config.include_corr_max:
            features['cross_channel_corr_max'] = pairwise_corr.amax(dim=1)

        # Min pairwise correlation
        if config is None or config.include_corr_min:
            features['cross_channel_corr_min'] = pairwise_corr.amin(dim=1)

        # Std of pairwise correlations
        if config is None or config.include_corr_std:
            features['cross_channel_corr_std'] = pairwise_corr.std(dim=1)

        return features

    # =========================================================================
    # Cross-Spectral Coherence (Optional)
    # =========================================================================

    def _compute_cross_spectral_coherence(
        self,
        fields: torch.Tensor,  # [NMT, C, H, W]
        config: 'SDFCrossChannelConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-spectral coherence.

        Measures phase-locked coupling between channels in frequency domain.
        Reveals oscillatory synchronization patterns independent of amplitude.

        Args:
            fields: Fields [NMT, C, H, W]
            config: Config (required for frequency bands)

        Returns:
            Dictionary with coherence features [NMT]
        """
        NMT, C, H, W = fields.shape

        # Compute 2D FFT per channel
        fft_result = torch.fft.rfft2(fields, dim=(-2, -1), norm='ortho')  # [NMT, C, H, W//2+1]
        power = torch.abs(fft_result) ** 2  # [NMT, C, H, W//2+1]

        # Define frequency bands (low, mid, high)
        freq_h = H // 2 + 1
        freq_w = W // 2 + 1

        # Band definitions (approximate thirds of frequency space)
        bands = {
            'low': (0, freq_h // 3, 0, freq_w // 3),
            'mid': (freq_h // 3, 2 * freq_h // 3, freq_w // 3, 2 * freq_w // 3),
            'high': (2 * freq_h // 3, freq_h, 2 * freq_w // 3, freq_w)
        }

        features = {}

        # Compute coherence for each frequency band
        for band_name in config.coherence_freq_bands:
            if band_name not in bands:
                continue

            h_start, h_end, w_start, w_end = bands[band_name]

            # Extract band
            fft_band = fft_result[:, :, h_start:h_end, w_start:w_end]
            power_band = power[:, :, h_start:h_end, w_start:w_end]

            # Compute pairwise coherence in this band
            coherence_values = []
            for i in range(C):
                for j in range(i + 1, C):
                    # Cross-spectral density
                    cross_spec = fft_band[:, i] * torch.conj(fft_band[:, j])

                    # Coherence: |S_ij|² / (S_ii * S_jj)
                    coherence = torch.abs(cross_spec) ** 2 / (
                        power_band[:, i] * power_band[:, j] + 1e-8
                    )

                    # Average over frequencies in this band
                    coherence_mean = coherence.mean(dim=(-2, -1))  # [NMT]
                    coherence_values.append(coherence_mean)

            # Stack and average across all pairs
            if coherence_values:
                coherence_stack = torch.stack(coherence_values, dim=1)  # [NMT, num_pairs]
                features[f'cross_spectral_coherence_{band_name}'] = coherence_stack.mean(dim=1)

        # Max coherence across all frequencies and pairs
        all_coherences = []
        for i in range(C):
            for j in range(i + 1, C):
                cross_spec = fft_result[:, i] * torch.conj(fft_result[:, j])
                coherence = torch.abs(cross_spec) ** 2 / (
                    power[:, i] * power[:, j] + 1e-8
                )
                all_coherences.append(coherence)

        if all_coherences:
            all_coherences_stack = torch.stack(all_coherences, dim=1)  # [NMT, num_pairs, H, W//2+1]
            features['cross_spectral_coherence_max'] = all_coherences_stack.amax(dim=(1, 2, 3))

            # Peak frequency (where max coherence occurs)
            # Flatten and find argmax
            all_coherences_flat = all_coherences_stack.reshape(NMT, -1)
            peak_idx = all_coherences_flat.argmax(dim=1)
            # Normalize to [0, 1]
            features['cross_spectral_coherence_peak_freq'] = peak_idx.float() / all_coherences_flat.shape[1]

        return features

    # =========================================================================
    # Mutual Information (Optional)
    # =========================================================================

    def _compute_mutual_information(
        self,
        fields: torch.Tensor,  # [NMT, C, H, W]
        config: 'SDFCrossChannelConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute mutual information between channel pairs.

        Uses histogram-based approximation with coarse binning for speed.
        Captures nonlinear dependencies beyond correlation.

        Args:
            fields: Fields [NMT, C, H, W]
            config: Config (required for bin count)

        Returns:
            Dictionary with MI features [NMT]
        """
        NMT, C, H, W = fields.shape

        num_bins = config.mi_num_bins

        # Flatten spatial dimensions: [NMT, C, H*W]
        fields_flat = fields.reshape(NMT, C, H * W)

        # Batched computation: Process all samples and pairs at once
        # Normalize all channels to [0, 1] per sample
        c_min = fields_flat.min(dim=2, keepdim=True).values  # [NMT, C, 1]
        c_max = fields_flat.max(dim=2, keepdim=True).values  # [NMT, C, 1]
        fields_norm = (fields_flat - c_min) / (c_max - c_min + 1e-8)  # [NMT, C, H*W]

        # Discretize to bins
        fields_discrete = torch.floor(fields_norm * (num_bins - 1)).long()
        fields_discrete = torch.clamp(fields_discrete, 0, num_bins - 1)  # [NMT, C, H*W]

        # Compute MI for all channel pairs
        num_pairs = C * (C - 1) // 2
        mi_tensor = torch.zeros(NMT, num_pairs, device=fields.device)

        pair_idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                # Extract channels for all samples
                c_i = fields_discrete[:, i, :]  # [NMT, H*W]
                c_j = fields_discrete[:, j, :]  # [NMT, H*W]

                # Batched 2D histogram computation using bincount
                for n in range(NMT):
                    # Linear indexing for 2D histogram
                    indices = c_i[n] * num_bins + c_j[n]  # [H*W]
                    joint_hist = torch.bincount(indices, minlength=num_bins * num_bins)
                    joint_hist = joint_hist.reshape(num_bins, num_bins).float()

                    # Normalize to probabilities
                    joint_prob = joint_hist / (H * W)  # [num_bins, num_bins]

                    # Marginal probabilities
                    prob_i = joint_prob.sum(dim=1)  # [num_bins]
                    prob_j = joint_prob.sum(dim=0)  # [num_bins]

                    # Mutual information (vectorized)
                    # I(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
                    outer_prod = prob_i.unsqueeze(1) * prob_j.unsqueeze(0)  # [num_bins, num_bins]
                    mask = joint_prob > 1e-10
                    mi = torch.where(
                        mask,
                        joint_prob * torch.log(joint_prob / (outer_prod + 1e-10)),
                        torch.zeros_like(joint_prob)
                    ).sum()

                    mi_tensor[n, pair_idx] = mi

                pair_idx += 1

        features = {}

        # Mean MI across all channel pairs
        features['cross_channel_mi_mean'] = mi_tensor.mean(dim=1)

        # Max MI across all channel pairs
        features['cross_channel_mi_max'] = mi_tensor.amax(dim=1)

        return features

    def _compute_conditional_mutual_information(
        self,
        fields: torch.Tensor,  # [NMT, C, H, W]
        config: 'SDFCrossChannelConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute conditional mutual information between channel pairs given a third channel.

        CMI I(X;Y|Z) measures how much information X and Y share beyond what Z provides.
        Detects higher-order dependencies not captured by pairwise MI.

        Formula: I(X;Y|Z) = Σ p(x,y,z) log(p(x,y|z) / (p(x|z)p(y|z)))

        Args:
            fields: Fields [NMT, C, H, W]
            config: Config (required for bin count)

        Returns:
            Dictionary with CMI features [NMT]
            - cmi_mean: Mean CMI across all triplets
            - cmi_max: Maximum CMI
        """
        NMT, C, H, W = fields.shape

        num_bins = config.mi_num_bins

        # Flatten spatial dimensions: [NMT, C, H*W]
        fields_flat = fields.reshape(NMT, C, H * W)

        # Batched computation: Normalize and discretize once for all channels
        c_min = fields_flat.min(dim=2, keepdim=True).values  # [NMT, C, 1]
        c_max = fields_flat.max(dim=2, keepdim=True).values  # [NMT, C, 1]
        fields_norm = (fields_flat - c_min) / (c_max - c_min + 1e-8)  # [NMT, C, H*W]

        # Discretize to bins
        fields_discrete = torch.floor(fields_norm * (num_bins - 1)).long()
        fields_discrete = torch.clamp(fields_discrete, 0, num_bins - 1)  # [NMT, C, H*W]

        # Compute CMI for all triplets
        num_triplets = C * (C - 1) // 2 * (C - 2)  # All (i,j,k) where i<j and k!=i,j
        cmi_tensor = torch.zeros(NMT, num_triplets, device=fields.device)

        triplet_idx = 0
        for i in range(C):
            for j in range(i + 1, C):
                for k in range(C):
                    if k == i or k == j:
                        continue

                    # Extract channels for all samples
                    c_i = fields_discrete[:, i, :]  # [NMT, H*W]
                    c_j = fields_discrete[:, j, :]  # [NMT, H*W]
                    c_k = fields_discrete[:, k, :]  # [NMT, H*W]

                    # Batched 3D histogram computation
                    for n in range(NMT):
                        # Linear indexing for 3D histogram
                        indices = (c_i[n] * num_bins + c_j[n]) * num_bins + c_k[n]  # [H*W]
                        joint_hist = torch.bincount(indices, minlength=num_bins ** 3)
                        joint_hist = joint_hist.reshape(num_bins, num_bins, num_bins).float()

                        # Normalize to probabilities
                        joint_prob = joint_hist / (H * W)  # [num_bins, num_bins, num_bins]

                        # Marginal probabilities
                        prob_k = joint_prob.sum(dim=(0, 1))  # p(z) [num_bins]
                        prob_ik = joint_prob.sum(dim=1)      # p(x,z) [num_bins, num_bins]
                        prob_jk = joint_prob.sum(dim=0)      # p(y,z) [num_bins, num_bins]

                        # CMI computation (vectorized)
                        # I(X;Y|Z) = Σ p(x,y,z) log(p(x,y,z)p(z) / (p(x,z)p(y,z)))
                        numerator = joint_prob * prob_k.view(1, 1, -1)  # [num_bins, num_bins, num_bins]
                        denominator = prob_ik.unsqueeze(1) * prob_jk.unsqueeze(0)  # [num_bins, num_bins, num_bins]

                        mask = joint_prob > 1e-10
                        cmi = torch.where(
                            mask,
                            joint_prob * torch.log((numerator + 1e-10) / (denominator + 1e-10)),
                            torch.zeros_like(joint_prob)
                        ).sum()

                        cmi_tensor[n, triplet_idx] = cmi

                    triplet_idx += 1

        features = {}

        # Mean CMI across all triplets
        features['cross_channel_cmi_mean'] = cmi_tensor.mean(dim=1)

        # Max CMI across all triplets
        features['cross_channel_cmi_max'] = cmi_tensor.amax(dim=1)

        return features
