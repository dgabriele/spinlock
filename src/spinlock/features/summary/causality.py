"""
Causality/directionality feature extraction.

Extracts trajectory-level features measuring temporal causality and information flow:
- Time-lagged correlation asymmetry (directional coupling)
- Prediction error ratios (forward vs backward predictability)
- Time-irreversibility (third-order moment asymmetry)
- Spatial information flow (gradient-based propagation)
- Transfer entropy (optional, nonlinear causality)
- Granger causality (optional, predictive causality)

Focus on fast approximations that capture relative directionality rather than
absolute causality. Full KDE-based methods are excluded for performance.

Example:
    >>> extractor = CausalityFeatureExtractor(device='cuda')
    >>> trajectories = torch.randn(32, 10, 100, 3, 128, 128, device='cuda')
    >>> features = extractor.extract(trajectories)  # Dict of per-trajectory features
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, TYPE_CHECKING, Literal
import numpy as np

if TYPE_CHECKING:
    from spinlock.features.summary.config import SummaryCausalityConfig


class CausalityFeatureExtractor:
    """
    Extract causality/directionality features from trajectories.

    Measures temporal causality and information flow using fast approximations
    optimized for operator characterization (not rigorous causal inference).

    Operates on full trajectories [N, M, T, C, H, W] and computes
    trajectory-level directional summaries.

    Example:
        >>> extractor = CausalityFeatureExtractor(device='cuda')
        >>> trajectories = torch.randn(8, 10, 50, 3, 128, 128, device='cuda')
        >>> features = extractor.extract(trajectories)
        >>> # Returns dict with ~6-10 features shaped [N, M, C] or [N, M]
    """

    def __init__(self, device: torch.device = torch.device('cuda')):
        """
        Initialize causality feature extractor.

        Args:
            device: Computation device (cuda or cpu)
        """
        self.device = device

    def extract(
        self,
        trajectories: torch.Tensor,  # [N, M, T, C, H, W]
        config: Optional['SummaryCausalityConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract causality/directionality features from trajectories.

        Args:
            trajectories: Full trajectories [N, M, T, C, H, W]
                N = batch size
                M = num realizations
                T = num timesteps
                C = num channels
                H, W = spatial dimensions
            config: Optional SummaryCausalityConfig for feature selection

        Returns:
            Dictionary mapping feature names to tensors [N, M, C] or [N, M]
            One value per realization (trajectory-level features)
            Returns NaN for T=1 (no temporal dynamics to analyze)
        """
        N, M, T, C, H, W = trajectories.shape

        # Handle single-timestep data (causality features undefined)
        if T == 1:
            return self._return_nan_features(N, M, C, trajectories.device, config)

        # Handle single-channel data (cross-channel causality undefined)
        if C == 1:
            return self._return_nan_features(N, M, C, trajectories.device, config)

        # Handle very short time series (limited reliability)
        if T < 3:
            # Can only compute lag=1 features
            max_lag = 1
        else:
            max_lag = config.max_lag_correlation if config else 3
            max_lag = min(max_lag, T - 1)  # Can't exceed T-1

        features = {}

        # Use config to determine which features to extract
        if config is None:
            include_all_fast = True
            complexity_level = "fast"
        else:
            include_all_fast = False
            complexity_level = config.complexity_level

        # Compute spatially-averaged time series for efficiency
        # Most causality features operate on global time series
        time_series = trajectories.mean(dim=(-2, -1))  # [N, M, T, C]

        # ===== LEVEL 1: FAST FEATURES (Always included) =====

        # Time-lagged correlation asymmetry (always computed)
        lag_corr_features = self._compute_lagged_correlation_asymmetry(
            time_series, max_lag=max_lag
        )
        features.update(lag_corr_features)

        # Prediction error ratios (always computed)
        pred_lag = config.max_lag_prediction if config else 2
        pred_error_features = self._compute_prediction_error_ratio(
            time_series, max_lag=pred_lag
        )
        features.update(pred_error_features)

        # Time-irreversibility (optional, default enabled)
        if include_all_fast or (config is None or config.include_time_irreversibility):
            time_irrev_features = self._compute_time_irreversibility(time_series)
            features.update(time_irrev_features)

        # Spatial information flow (optional, default enabled)
        if include_all_fast or (config is None or config.include_spatial_flow):
            spatial_flow_features = self._compute_spatial_information_flow(trajectories)
            features.update(spatial_flow_features)

        # ===== LEVEL 2: MEDIUM FEATURES (Optional) =====

        if complexity_level in ["medium", "full"]:
            # Transfer entropy (binned, coarse approximation)
            if config and config.include_transfer_entropy:
                te_features = self._compute_transfer_entropy(time_series, config)
                features.update(te_features)

            # Granger causality (simplified AR models)
            if config and config.include_granger_causality:
                granger_features = self._compute_granger_causality(time_series, config)
                features.update(granger_features)

        return features

    def _return_nan_features(
        self,
        N: int,
        M: int,
        C: int,
        device: torch.device,
        config: Optional['SummaryCausalityConfig'] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Return NaN features for T=1 edge case.

        Args:
            N: Batch size
            M: Number of realizations
            C: Number of channels
            device: Device for tensors
            config: Optional config

        Returns:
            Dictionary of NaN features
        """
        nan_features_c = torch.full((N, M, C), float('nan'), device=device)
        nan_features_scalar = torch.full((N, M), float('nan'), device=device)
        features = {}

        # Level 1: Fast features
        max_lag = config.max_lag_correlation if config else 3
        for lag in range(1, max_lag + 1):
            features[f'causality_lag_corr_asymmetry_mean_lag{lag}'] = nan_features_scalar.clone()
            features[f'causality_lag_corr_asymmetry_max_lag{lag}'] = nan_features_scalar.clone()

        max_pred_lag = config.max_lag_prediction if config else 2
        for lag in range(1, max_pred_lag + 1):
            features[f'causality_pred_error_ratio_lag{lag}'] = nan_features_c.clone()
            features[f'causality_pred_error_diff_lag{lag}'] = nan_features_c.clone()

        features['causality_time_irreversibility'] = nan_features_c.clone()
        features['causality_time_asymmetry_index'] = nan_features_c.clone()

        features['causality_spatial_flow_magnitude'] = nan_features_c.clone()
        features['causality_spatial_flow_anisotropy'] = nan_features_c.clone()

        # Level 2: Medium features (if enabled)
        if config is not None and config.include_transfer_entropy:
            features['causality_transfer_entropy_mean'] = nan_features_scalar.clone()
            features['causality_transfer_entropy_max'] = nan_features_scalar.clone()
            features['causality_transfer_entropy_asymmetry'] = nan_features_scalar.clone()

        if config is not None and config.include_granger_causality:
            features['causality_granger_score_mean'] = nan_features_scalar.clone()
            features['causality_granger_score_max'] = nan_features_scalar.clone()
            features['causality_granger_score_asymmetry'] = nan_features_scalar.clone()

        return features

    # =========================================================================
    # Level 1: Fast Features
    # =========================================================================

    def _compute_lagged_correlation_asymmetry(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        max_lag: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Compute time-lagged cross-correlation asymmetry.

        Measures directional asymmetry in correlations:
        asymmetry(τ) = corr(x(t), y(t+τ)) - corr(x(t+τ), y(t))

        Positive asymmetry suggests x → y causality.

        Args:
            time_series: Time series [N, M, T, C]
            max_lag: Maximum lag to compute

        Returns:
            Dictionary with asymmetry features for each lag [N, M]
            (averaged across channel pairs)
        """
        N, M, T, C = time_series.shape

        features = {}

        # For each lag, compute forward and backward correlations
        for lag in range(1, max_lag + 1):
            if T <= lag:
                # Not enough data for this lag
                nan_val = torch.full((N, M), float('nan'), device=time_series.device)
                features[f'causality_lag_corr_asymmetry_mean_lag{lag}'] = nan_val
                features[f'causality_lag_corr_asymmetry_max_lag{lag}'] = nan_val
                continue

            # Batched computation: Process all channel pairs at once using einsum
            # Prepare time series for forward and backward correlations
            x_past = time_series[:, :, :-lag, :]  # [N, M, T-lag, C]
            y_future = time_series[:, :, lag:, :]  # [N, M, T-lag, C]
            x_future = time_series[:, :, lag:, :]  # [N, M, T-lag, C]
            y_past = time_series[:, :, :-lag, :]  # [N, M, T-lag, C]

            # Center time series
            x_past_centered = x_past - x_past.mean(dim=2, keepdim=True)  # [N, M, T-lag, C]
            y_future_centered = y_future - y_future.mean(dim=2, keepdim=True)
            x_future_centered = x_future - x_future.mean(dim=2, keepdim=True)
            y_past_centered = y_past - y_past.mean(dim=2, keepdim=True)

            # Compute all pairwise correlations at once using einsum
            # Forward: corr(x_i(t), y_j(t+τ)) for all i,j pairs
            cov_forward = torch.einsum('nmti,nmtj->nmij', x_past_centered, y_future_centered) / (T - lag)  # [N, M, C, C]
            std_x_past = x_past.std(dim=2, keepdim=True)  # [N, M, 1, C]
            std_y_future = y_future.std(dim=2, keepdim=True)  # [N, M, 1, C]
            corr_forward = cov_forward / (std_x_past.squeeze(2).unsqueeze(3) * std_y_future.squeeze(2).unsqueeze(2) + 1e-8)  # [N, M, C, C]

            # Backward: corr(x_i(t+τ), y_j(t)) for all i,j pairs
            cov_backward = torch.einsum('nmti,nmtj->nmij', x_future_centered, y_past_centered) / (T - lag)  # [N, M, C, C]
            std_x_future = x_future.std(dim=2, keepdim=True)  # [N, M, 1, C]
            std_y_past = y_past.std(dim=2, keepdim=True)  # [N, M, 1, C]
            corr_backward = cov_backward / (std_x_future.squeeze(2).unsqueeze(3) * std_y_past.squeeze(2).unsqueeze(2) + 1e-8)  # [N, M, C, C]

            # Asymmetry for all pairs
            asymmetry_matrix = corr_forward - corr_backward  # [N, M, C, C]

            # Mask out diagonal (self-correlation) and flatten to get all off-diagonal pairs
            mask = ~torch.eye(C, dtype=torch.bool, device=time_series.device)  # [C, C]
            asymmetries_stack = asymmetry_matrix[:, :, mask]  # [N, M, C*(C-1)]

            # Mean asymmetry
            features[f'causality_lag_corr_asymmetry_mean_lag{lag}'] = asymmetries_stack.mean(dim=-1)

            # Max absolute asymmetry
            features[f'causality_lag_corr_asymmetry_max_lag{lag}'] = asymmetries_stack.abs().amax(dim=-1)

        return features

    def _compute_correlation(
        self,
        x: torch.Tensor,  # [N, M, T]
        y: torch.Tensor   # [N, M, T]
    ) -> torch.Tensor:
        """
        Compute Pearson correlation between x and y time series.

        Args:
            x: Time series [N, M, T]
            y: Time series [N, M, T]

        Returns:
            Correlation [N, M]
        """
        # Center
        x_centered = x - x.mean(dim=2, keepdim=True)
        y_centered = y - y.mean(dim=2, keepdim=True)

        # Correlation = cov(x, y) / (std(x) * std(y))
        cov = (x_centered * y_centered).mean(dim=2)
        std_x = x.std(dim=2) + 1e-8
        std_y = y.std(dim=2) + 1e-8

        corr = cov / (std_x * std_y)

        return corr

    def _compute_prediction_error_ratio(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        max_lag: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Compute forward vs backward prediction error ratios.

        Uses simple persistence model:
        - Forward: predict x(t+τ) from x(t)
        - Backward: predict x(t-τ) from x(t)

        Args:
            time_series: Time series [N, M, T, C]
            max_lag: Maximum lag to compute

        Returns:
            Dictionary with error ratio and difference features [N, M, C]
        """
        N, M, T, C = time_series.shape

        features = {}

        for lag in range(1, max_lag + 1):
            if T <= lag:
                # Not enough data
                nan_val = torch.full((N, M, C), float('nan'), device=time_series.device)
                features[f'causality_pred_error_ratio_lag{lag}'] = nan_val
                features[f'causality_pred_error_diff_lag{lag}'] = nan_val
                continue

            # Forward prediction error: |x(t+τ) - x(t)|
            forward_pred = time_series[:, :, :-lag, :]  # Predict from past
            forward_target = time_series[:, :, lag:, :]
            forward_error = (forward_target - forward_pred).abs().mean(dim=2)  # [N, M, C]

            # Backward prediction error: |x(t-τ) - x(t)|
            backward_pred = time_series[:, :, lag:, :]  # Predict from future
            backward_target = time_series[:, :, :-lag, :]
            backward_error = (backward_target - backward_pred).abs().mean(dim=2)  # [N, M, C]

            # Ratio (>1 means harder to predict forward = forward causality stronger)
            ratio = forward_error / (backward_error + 1e-8)
            features[f'causality_pred_error_ratio_lag{lag}'] = ratio

            # Difference (positive means forward harder = forward causality)
            diff = forward_error - backward_error
            features[f'causality_pred_error_diff_lag{lag}'] = diff

        return features

    def _compute_time_irreversibility(
        self,
        time_series: torch.Tensor  # [N, M, T, C]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute time-irreversibility measures.

        Uses third-order moment asymmetry:
        TI = <(x(t+1) - x(t))³> - <(x(t) - x(t-1))³>

        If TI ≠ 0, the process is time-irreversible.

        Args:
            time_series: Time series [N, M, T, C]

        Returns:
            Dictionary with time-irreversibility features [N, M, C]
        """
        # Forward differences: x(t+1) - x(t)
        forward_diff = time_series[:, :, 1:, :] - time_series[:, :, :-1, :]  # [N, M, T-1, C]

        # Backward differences: x(t) - x(t-1)
        # This is just forward_diff, so we compare forward_diff with itself shifted
        # Actually, for proper asymmetry, we need:
        # Forward: x(t+1) - x(t) for t=0..T-2
        # Backward: x(t) - x(t-1) for t=1..T-1
        # These are the same array! So we need to think differently.

        # Let me redefine:
        # Forward process: differences going forward in time
        # Backward process: differences going backward in time (reverse the series)

        # Forward third moment: <(x(t+1) - x(t))³>
        forward_third_moment = (forward_diff ** 3).mean(dim=2)  # [N, M, C]

        # For backward, we reverse the time series and compute differences
        time_series_reversed = time_series.flip(dims=[2])  # Reverse time
        backward_diff = time_series_reversed[:, :, 1:, :] - time_series_reversed[:, :, :-1, :]
        backward_third_moment = (backward_diff ** 3).mean(dim=2)  # [N, M, C]

        # Time irreversibility statistic
        time_irreversibility = forward_third_moment - backward_third_moment

        # Normalized asymmetry index (divide by variance to make scale-invariant)
        variance = time_series.var(dim=2) + 1e-8
        time_asymmetry_index = time_irreversibility / variance

        return {
            'causality_time_irreversibility': time_irreversibility,
            'causality_time_asymmetry_index': time_asymmetry_index
        }

    def _compute_spatial_information_flow(
        self,
        trajectories: torch.Tensor  # [N, M, T, C, H, W]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial information flow patterns.

        Measures correlation between spatial gradients at time t and
        field values at time t+1 to infer propagation direction.

        Args:
            trajectories: Full trajectories [N, M, T, C, H, W]

        Returns:
            Dictionary with spatial flow features [N, M, C]
        """
        N, M, T, C, H, W = trajectories.shape

        if T == 1:
            # No temporal evolution
            nan_val = torch.full((N, M, C), float('nan'), device=trajectories.device)
            return {
                'causality_spatial_flow_magnitude': nan_val,
                'causality_spatial_flow_anisotropy': nan_val
            }

        # Compute spatial gradients at t
        # Reshape for batch gradient computation
        fields_t = trajectories[:, :, :-1, :, :, :].reshape(N * M * (T-1), C, H, W)

        # Gradients (central differences, circular boundary)
        grad_x = (torch.roll(fields_t, shifts=-1, dims=3) -
                  torch.roll(fields_t, shifts=1, dims=3)) / 2.0
        grad_y = (torch.roll(fields_t, shifts=-1, dims=2) -
                  torch.roll(fields_t, shifts=1, dims=2)) / 2.0

        # Field values at t+1
        fields_t_plus_1 = trajectories[:, :, 1:, :, :, :].reshape(N * M * (T-1), C, H, W)

        # Correlation between gradient and future field (averaged over space)
        grad_x_flat = grad_x.reshape(N * M * (T-1), C, H * W)
        grad_y_flat = grad_y.reshape(N * M * (T-1), C, H * W)
        fields_future_flat = fields_t_plus_1.reshape(N * M * (T-1), C, H * W)

        # Correlation: cov(grad, future) / (std(grad) * std(future))
        # Compute per-channel correlations
        flow_x = []
        flow_y = []

        for c in range(C):
            gx = grad_x_flat[:, c, :]  # [NM(T-1), HW]
            gy = grad_y_flat[:, c, :]
            ff = fields_future_flat[:, c, :]

            # Correlation
            corr_x = self._compute_spatial_correlation(gx, ff)  # [NM(T-1)]
            corr_y = self._compute_spatial_correlation(gy, ff)

            flow_x.append(corr_x)
            flow_y.append(corr_y)

        # Stack: [NM(T-1), C]
        flow_x_stack = torch.stack(flow_x, dim=1)
        flow_y_stack = torch.stack(flow_y, dim=1)

        # Flow magnitude: sqrt(flow_x² + flow_y²)
        flow_magnitude = torch.sqrt(flow_x_stack ** 2 + flow_y_stack ** 2 + 1e-8)

        # Anisotropy: |flow_x| / |flow_y| (directional preference)
        anisotropy = flow_x_stack.abs() / (flow_y_stack.abs() + 1e-8)

        # Average over time and reshape to [N, M, C]
        flow_magnitude = flow_magnitude.reshape(N, M, T-1, C).mean(dim=2)
        anisotropy = anisotropy.reshape(N, M, T-1, C).mean(dim=2)

        return {
            'causality_spatial_flow_magnitude': flow_magnitude,
            'causality_spatial_flow_anisotropy': anisotropy
        }

    def _compute_spatial_correlation(
        self,
        x: torch.Tensor,  # [NMT, HW]
        y: torch.Tensor   # [NMT, HW]
    ) -> torch.Tensor:
        """
        Compute correlation along spatial dimension.

        Args:
            x: Tensor [NMT, HW]
            y: Tensor [NMT, HW]

        Returns:
            Correlation [NMT]
        """
        # Center
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)

        # Correlation
        cov = (x_centered * y_centered).mean(dim=1)
        std_x = x.std(dim=1) + 1e-8
        std_y = y.std(dim=1) + 1e-8

        corr = cov / (std_x * std_y)

        return corr

    # =========================================================================
    # Level 2: Medium Features (Optional)
    # =========================================================================

    def _compute_transfer_entropy(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        config: 'SummaryCausalityConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute binned transfer entropy.

        Transfer entropy: TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

        Uses coarse binning for speed.

        Args:
            time_series: Time series [N, M, T, C]
            config: Config (required for bin count and lag)

        Returns:
            Dictionary with TE features [N, M]
            (averaged across channel pairs)
        """
        N, M, T, C = time_series.shape

        num_bins = config.te_num_bins
        lag = config.te_max_lag

        if T <= lag:
            # Not enough data
            nan_val = torch.full((N, M), float('nan'), device=time_series.device)
            return {
                'causality_transfer_entropy_mean': nan_val,
                'causality_transfer_entropy_max': nan_val,
                'causality_transfer_entropy_asymmetry': nan_val
            }

        # Batched computation: Process all channel pairs and samples at once
        # Discretize all channels
        time_series_discrete = self._discretize_to_bins_batched(time_series, num_bins)  # [N, M, T, C]

        # Prepare lagged sequences for all channels
        y_t = time_series_discrete[:, :, lag:, :]  # [N, M, T-lag, C]
        y_t_minus_1 = time_series_discrete[:, :, lag-1:-1, :]  # [N, M, T-lag, C]
        x_t_minus_1 = time_series_discrete[:, :, lag-1:-1, :]  # [N, M, T-lag, C]

        # Compute TE for all pairs at once
        te_matrix = torch.zeros(N, M, C, C, device=time_series.device)

        for i in range(C):
            for j in range(C):
                if i == j:
                    continue

                # Extract specific channel pair for all samples
                y_curr = y_t[:, :, :, j]  # [N, M, T-lag]
                y_prev = y_t_minus_1[:, :, :, j]  # [N, M, T-lag]
                x_prev = x_t_minus_1[:, :, :, i]  # [N, M, T-lag]

                # Batched conditional entropy computation
                h_y_given_y = self._conditional_entropy_binned_batched(y_curr, y_prev, num_bins)  # [N, M]
                h_y_given_both = self._conditional_entropy_2d_binned_batched(
                    y_curr, y_prev, x_prev, num_bins
                )  # [N, M]

                # TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
                te_matrix[:, :, i, j] = h_y_given_y - h_y_given_both

        # Mask out diagonal and flatten
        mask = ~torch.eye(C, dtype=torch.bool, device=time_series.device)
        te_stack = te_matrix[:, :, mask]  # [N, M, C*(C-1)]

        # Mean TE across pairs
        te_mean = te_stack.mean(dim=2)

        # Max TE across pairs
        te_max = te_stack.amax(dim=2)

        # Asymmetry: TE(X→Y) - TE(Y→X) averaged across pairs
        # This requires pairing up the TE values correctly
        # For simplicity, use std as a proxy for asymmetry
        te_asymmetry = te_stack.std(dim=2)

        return {
            'causality_transfer_entropy_mean': te_mean,
            'causality_transfer_entropy_max': te_max,
            'causality_transfer_entropy_asymmetry': te_asymmetry
        }

    def _discretize_to_bins(
        self,
        x: torch.Tensor,  # [N, M, T]
        num_bins: int
    ) -> torch.Tensor:
        """
        Discretize continuous values to bins.

        Args:
            x: Continuous values [N, M, T]
            num_bins: Number of bins

        Returns:
            Discrete bin indices [N, M, T] (long tensor)
        """
        # Normalize to [0, 1] per sample
        x_min = x.amin(dim=2, keepdim=True)
        x_max = x.amax(dim=2, keepdim=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        # Discretize
        x_discrete = torch.floor(x_norm * (num_bins - 1)).long()
        x_discrete = torch.clamp(x_discrete, 0, num_bins - 1)

        return x_discrete

    def _discretize_to_bins_batched(
        self,
        x: torch.Tensor,  # [N, M, T, C]
        num_bins: int
    ) -> torch.Tensor:
        """
        Batched discretization for all channels.

        Args:
            x: Continuous values [N, M, T, C]
            num_bins: Number of bins

        Returns:
            Discrete bin indices [N, M, T, C] (long tensor)
        """
        # Normalize to [0, 1] per sample and channel
        x_min = x.amin(dim=2, keepdim=True)  # [N, M, 1, C]
        x_max = x.amax(dim=2, keepdim=True)  # [N, M, 1, C]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        # Discretize
        x_discrete = torch.floor(x_norm * (num_bins - 1)).long()
        x_discrete = torch.clamp(x_discrete, 0, num_bins - 1)

        return x_discrete

    def _conditional_entropy_binned_batched(
        self,
        y: torch.Tensor,  # [N, M, T] (discrete bins)
        x: torch.Tensor,  # [N, M, T] (discrete bins)
        num_bins: int
    ) -> torch.Tensor:
        """
        Batched conditional entropy H(Y|X) for all samples.

        Args:
            y: Target variable (discrete) [N, M, T]
            x: Conditioning variable (discrete) [N, M, T]
            num_bins: Number of bins

        Returns:
            Conditional entropy [N, M]
        """
        N, M, T = y.shape

        # Vectorized 2D histogram using scatter_add
        # Flatten batch dimensions
        y_flat = y.reshape(N * M, T)  # [NM, T]
        x_flat = x.reshape(N * M, T)  # [NM, T]

        # Compute joint histograms for all samples
        joint_hist = torch.zeros(N * M, num_bins, num_bins, device=y.device)

        for nm in range(N * M):
            # Linear indexing for 2D histogram
            indices = x_flat[nm] * num_bins + y_flat[nm]  # [T]
            joint_hist[nm] = torch.bincount(indices, minlength=num_bins * num_bins).reshape(num_bins, num_bins).float()

        # Joint and marginal probabilities
        joint_prob = joint_hist / T  # [NM, num_bins, num_bins]
        marginal_x = joint_prob.sum(dim=2)  # [NM, num_bins]

        # H(Y|X) = -Σ_x,y p(x,y) log(p(y|x))
        # Conditional probability: p(y|x) = p(x,y) / p(x)
        p_y_given_x = joint_prob / (marginal_x.unsqueeze(2) + 1e-10)  # [NM, num_bins, num_bins]

        # Entropy computation (vectorized)
        # Only include terms where joint_prob > 0
        mask = joint_prob > 1e-10
        entropy = -torch.where(
            mask,
            joint_prob * torch.log(p_y_given_x + 1e-10),
            torch.zeros_like(joint_prob)
        ).sum(dim=(1, 2))  # [NM]

        return entropy.reshape(N, M)

    def _conditional_entropy_2d_binned_batched(
        self,
        z: torch.Tensor,  # [N, M, T] (discrete bins)
        x: torch.Tensor,  # [N, M, T] (discrete bins)
        y: torch.Tensor,  # [N, M, T] (discrete bins)
        num_bins: int
    ) -> torch.Tensor:
        """
        Batched conditional entropy H(Z|X,Y) for all samples.

        Args:
            z: Target variable (discrete) [N, M, T]
            x: Conditioning variable 1 (discrete) [N, M, T]
            y: Conditioning variable 2 (discrete) [N, M, T]
            num_bins: Number of bins

        Returns:
            Conditional entropy [N, M]
        """
        N, M, T = z.shape

        # Flatten batch dimensions
        z_flat = z.reshape(N * M, T)  # [NM, T]
        x_flat = x.reshape(N * M, T)  # [NM, T]
        y_flat = y.reshape(N * M, T)  # [NM, T]

        # Compute 3D joint histograms for all samples
        joint_hist = torch.zeros(N * M, num_bins, num_bins, num_bins, device=z.device)

        for nm in range(N * M):
            # Linear indexing for 3D histogram
            indices = (x_flat[nm] * num_bins + y_flat[nm]) * num_bins + z_flat[nm]  # [T]
            joint_hist[nm] = torch.bincount(
                indices, minlength=num_bins ** 3
            ).reshape(num_bins, num_bins, num_bins).float()

        # Probabilities
        joint_prob = joint_hist / T  # [NM, num_bins, num_bins, num_bins]
        marginal_xy = joint_prob.sum(dim=3)  # [NM, num_bins, num_bins]

        # H(Z|X,Y) = -Σ_{x,y,z} p(x,y,z) log(p(z|x,y))
        # Conditional probability: p(z|x,y) = p(x,y,z) / p(x,y)
        p_z_given_xy = joint_prob / (marginal_xy.unsqueeze(3) + 1e-10)  # [NM, num_bins, num_bins, num_bins]

        # Entropy computation (vectorized)
        mask = joint_prob > 1e-10
        entropy = -torch.where(
            mask,
            joint_prob * torch.log(p_z_given_xy + 1e-10),
            torch.zeros_like(joint_prob)
        ).sum(dim=(1, 2, 3))  # [NM]

        return entropy.reshape(N, M)

    def _conditional_entropy_binned(
        self,
        y: torch.Tensor,  # [T] (discrete bins)
        x: torch.Tensor,  # [T] (discrete bins)
        num_bins: int
    ) -> float:
        """
        Compute conditional entropy H(Y|X) for binned data.

        Args:
            y: Target variable (discrete) [T]
            x: Conditioning variable (discrete) [T]
            num_bins: Number of bins

        Returns:
            Conditional entropy (scalar)
        """
        T = y.shape[0]

        # Joint histogram
        joint_hist = torch.zeros(num_bins, num_bins, device=y.device)
        for t in range(T):
            joint_hist[x[t], y[t]] += 1

        # Joint and marginal probabilities
        joint_prob = joint_hist / T
        marginal_x = joint_prob.sum(dim=1)

        # H(Y|X) = -Σ_x p(x) Σ_y p(y|x) log p(y|x)
        h = 0.0
        for i in range(num_bins):
            if marginal_x[i] > 1e-10:
                for j in range(num_bins):
                    if joint_prob[i, j] > 1e-10:
                        p_y_given_x = joint_prob[i, j] / marginal_x[i]
                        h += joint_prob[i, j] * torch.log(p_y_given_x)

        return -h.item()

    def _conditional_entropy_2d_binned(
        self,
        z: torch.Tensor,  # [T] (discrete bins)
        x: torch.Tensor,  # [T] (discrete bins)
        y: torch.Tensor,  # [T] (discrete bins)
        num_bins: int
    ) -> float:
        """
        Compute conditional entropy H(Z|X,Y) for binned data.

        Args:
            z: Target variable (discrete) [T]
            x: Conditioning variable 1 (discrete) [T]
            y: Conditioning variable 2 (discrete) [T]
            num_bins: Number of bins

        Returns:
            Conditional entropy (scalar)
        """
        T = z.shape[0]

        # 3D joint histogram
        joint_hist = torch.zeros(num_bins, num_bins, num_bins, device=z.device)
        for t in range(T):
            joint_hist[x[t], y[t], z[t]] += 1

        # Probabilities
        joint_prob = joint_hist / T
        marginal_xy = joint_prob.sum(dim=2)

        # H(Z|X,Y) = -Σ_{x,y} p(x,y) Σ_z p(z|x,y) log p(z|x,y)
        h = 0.0
        for i in range(num_bins):
            for j in range(num_bins):
                if marginal_xy[i, j] > 1e-10:
                    for k in range(num_bins):
                        if joint_prob[i, j, k] > 1e-10:
                            p_z_given_xy = joint_prob[i, j, k] / marginal_xy[i, j]
                            h += joint_prob[i, j, k] * torch.log(p_z_given_xy)

        return -h.item()

    def _compute_granger_causality(
        self,
        time_series: torch.Tensor,  # [N, M, T, C]
        config: 'SummaryCausalityConfig'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute simplified Granger causality.

        Tests whether X helps predict Y beyond Y's own history using
        low-order autoregressive models.

        Args:
            time_series: Time series [N, M, T, C]
            config: Config (required for AR order)

        Returns:
            Dictionary with Granger causality features [N, M]
            (averaged across channel pairs)
        """
        N, M, T, C = time_series.shape

        ar_order = config.granger_ar_order

        if T <= ar_order + 1:
            # Not enough data
            nan_val = torch.full((N, M), float('nan'), device=time_series.device)
            return {
                'causality_granger_score_mean': nan_val,
                'causality_granger_score_max': nan_val,
                'causality_granger_asymmetry': nan_val
            }

        granger_scores = []

        # Compute Granger causality for each channel pair
        for i in range(C):
            for j in range(C):
                if i == j:
                    continue

                # Extract channels
                x = time_series[:, :, :, i]  # [N, M, T]
                y = time_series[:, :, :, j]  # [N, M, T]

                # Granger test: Does X improve prediction of Y?
                # Simplified: compare prediction errors

                # Restricted model: Y(t) ~ Y(t-1), ..., Y(t-p)
                # Full model: Y(t) ~ Y(t-1), ..., Y(t-p), X(t-1), ..., X(t-p)

                # For simplicity, use order p=1 or p=2
                # Predict Y(t) from Y(t-1) and optionally X(t-1)

                y_target = y[:, :, ar_order:]  # [N, M, T-p]

                # Restricted prediction: use only Y history
                y_past = y[:, :, ar_order-1:-1]  # [N, M, T-p]
                error_restricted = (y_target - y_past).pow(2).mean(dim=2)  # [N, M]

                # Full prediction: use Y and X history
                # Simple model: Y(t) ≈ a*Y(t-1) + b*X(t-1)
                # Use least squares fit (approximate for batched computation)
                # For speed, use simple weighted average
                x_past = x[:, :, ar_order-1:-1]  # [N, M, T-p]

                # Weighted prediction (0.5 each for simplicity)
                full_pred = 0.5 * y_past + 0.5 * x_past
                error_full = (y_target - full_pred).pow(2).mean(dim=2)  # [N, M]

                # Granger score: reduction in error
                # Positive score means X improves prediction
                granger_score = (error_restricted - error_full) / (error_restricted + 1e-8)
                granger_scores.append(granger_score)

        # Stack and aggregate
        granger_stack = torch.stack(granger_scores, dim=2)  # [N, M, num_pairs]

        # Mean Granger score
        granger_mean = granger_stack.mean(dim=2)

        # Max Granger score
        granger_max = granger_stack.amax(dim=2)

        # Asymmetry (std of scores as proxy)
        granger_asymmetry = granger_stack.std(dim=2)

        return {
            'causality_granger_score_mean': granger_mean,
            'causality_granger_score_max': granger_max,
            'causality_granger_asymmetry': granger_asymmetry
        }
