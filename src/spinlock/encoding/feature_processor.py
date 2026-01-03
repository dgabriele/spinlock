"""Feature post-processing pipeline for VQ-VAE training.

Ported from unisim.system.features.processor (100% generic, domain-agnostic).

Applies cleaning steps before VQ-VAE training:
1. Remove zero-variance AND extreme-variance features
2. Deduplicate highly correlated features (corr > threshold)
3. Handle NaNs (median replacement)
4. Cap outliers using adaptive method (percentile, IQR, or MAD)

This ensures clean, numerically stable features for categorical VQ-VAE training.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Feature post-processing pipeline.

    Applies all cleaning steps from production system:
    1. Remove zero-variance AND extreme-variance features
    2. Deduplicate highly correlated features
    3. Handle NaNs (median replacement)
    4. Cap outliers using adaptive method (percentile, IQR, or MAD)
    """

    def __init__(
        self,
        variance_threshold: float = 1e-10,
        max_variance_threshold: Optional[float] = 1e10,
        max_cv_threshold: float = 100.0,
        deduplicate_threshold: float = 0.99,
        use_intelligent_dedup: bool = True,
        outlier_method: str = "percentile",
        percentile_range: Tuple[float, float] = (0.5, 99.5),
        iqr_multiplier: float = 1.5,
        mad_threshold: float = 3.0,
        verbose: bool = False,
    ):
        """Initialize processor with intelligent adaptive filtering.

        Args:
            variance_threshold: Remove features with std below this threshold
            max_variance_threshold: Remove features with variance above this threshold (extreme variance)
            max_cv_threshold: Maximum coefficient of variation (CV = std/mean) - filters unstable features
            deduplicate_threshold: Remove features with correlation above this threshold
            use_intelligent_dedup: If True, keep more informative feature from correlated pairs
            outlier_method: Method for outlier capping - "percentile" (adaptive), "iqr" (boxplot), or "mad" (legacy)
            percentile_range: (lower, upper) percentiles for clipping (e.g., (0.5, 99.5) clips 0.5% each end)
            iqr_multiplier: IQR multiplier for boxplot-style outlier detection (1.5 = mild, 3.0 = extreme)
            mad_threshold: MAD multiplier for outlier capping (legacy method, 3.0 = aggressive, 5.0 = moderate)
            verbose: Print detailed cleaning statistics
        """
        self.variance_threshold = variance_threshold
        self.max_variance_threshold = max_variance_threshold
        self.max_cv_threshold = max_cv_threshold
        self.deduplicate_threshold = deduplicate_threshold
        self.use_intelligent_dedup = use_intelligent_dedup
        self.outlier_method = outlier_method
        self.percentile_range = percentile_range
        self.iqr_multiplier = iqr_multiplier
        self.mad_threshold = mad_threshold
        self.verbose = verbose

    def clean(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
        """Apply all post-processing steps.

        Args:
            features: Raw features [N, D]
            feature_names: Original feature names [D] (optional)

        Returns:
            Tuple of (cleaned_features [N, D'], feature_mask [D], cleaned_feature_names [D'])
            where feature_mask indicates which original features were kept,
            and cleaned_feature_names are the names after all cleaning steps (None if input None)
        """
        if features.shape[0] == 0:
            raise ValueError("Cannot process empty feature array")

        original_dim = features.shape[1]

        if self.verbose:
            logger.info("\n" + "=" * 70)
            logger.info("FEATURE CLEANING PIPELINE")
            logger.info("=" * 70)
            logger.info(f"Input: {features.shape[0]} samples × {original_dim} features")

        # Validate feature names if provided
        if feature_names is not None:
            if len(feature_names) != features.shape[1]:
                raise ValueError(
                    f"Feature names length {len(feature_names)} != "
                    f"features dim {features.shape[1]}"
                )
            current_names = np.array(feature_names)
        else:
            current_names = None

        # 1. Remove zero-variance features
        features, variance_mask = self._remove_zero_variance(features)
        if current_names is not None:
            current_names = current_names[variance_mask]

        if self.verbose:
            kept = variance_mask.sum()
            removed = (~variance_mask).sum()
            if self.max_variance_threshold is not None:
                logger.info(f"\n1. Variance filtering (std < {self.variance_threshold} OR var > {self.max_variance_threshold:.1e}):")
            else:
                logger.info(f"\n1. Zero-variance removal (std < {self.variance_threshold}):")
            logger.info(f"   Kept: {kept}/{original_dim}")
            logger.info(f"   Removed: {removed}")

        # 2. Deduplicate highly correlated features
        features, dedup_mask = self._deduplicate(features)
        if current_names is not None:
            current_names = current_names[dedup_mask]

        if self.verbose:
            kept = dedup_mask.sum()
            removed = (~dedup_mask).sum()
            logger.info(f"\n2. Deduplication (|corr| > {self.deduplicate_threshold}):")
            logger.info(f"   Kept: {kept}")
            logger.info(f"   Removed: {removed}")

        # 3. Handle NaNs
        nan_stats = self._get_nan_stats(features)
        features = self._handle_nans(features)

        if self.verbose:
            logger.info(f"\n3. NaN handling (median replacement):")
            logger.info(f"   Features with NaN: {nan_stats['num_features_with_nan']}")
            logger.info(f"   Total NaN values: {nan_stats['total_nan_count']}")
            if nan_stats['num_features_with_nan'] > 0:
                logger.info(f"   NaN rate: {100*nan_stats['nan_fraction']:.2f}%")

        # 4. Cap outliers
        outlier_stats = self._get_outlier_stats_before(features)
        features = self._cap_outliers(features)

        if self.verbose:
            if self.outlier_method == "percentile":
                logger.info(f"\n4. Outlier capping (percentile: {self.percentile_range[0]}-{self.percentile_range[1]}%):")
            elif self.outlier_method == "iqr":
                logger.info(f"\n4. Outlier capping (IQR multiplier = {self.iqr_multiplier}):")
            else:  # mad
                logger.info(f"\n4. Outlier capping (MAD threshold = {self.mad_threshold}):")
            logger.info(f"   Features with outliers: {outlier_stats['num_features_with_outliers']}")
            logger.info(f"   Total outliers capped: {outlier_stats['total_outliers']}")

        # Combine masks
        combined_mask = variance_mask.copy()
        combined_mask[variance_mask] = dedup_mask

        # Convert names back to list
        final_names = current_names.tolist() if current_names is not None else None

        if self.verbose:
            logger.info(f"\nFinal: {features.shape[0]} samples × {features.shape[1]} features")
            logger.info(f"Feature reduction: {original_dim} → {features.shape[1]} ({100*features.shape[1]/original_dim:.1f}%)")
            logger.info("=" * 70 + "\n")

        return features, combined_mask, final_names

    def _remove_zero_variance(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove features with std < threshold OR unstable variance (using CV).

        Uses intelligent distribution-aware filtering:
        - Zero variance: std < threshold
        - Extreme variance: var > max_threshold AND CV > max_cv_threshold
        - CV (Coefficient of Variation) = std / |mean| measures relative variability

        Args:
            features: Features [N, D]

        Returns:
            Tuple of (filtered_features [N, D'], mask [D])
        """
        stds = np.nanstd(features, axis=0)
        vars = np.var(features, axis=0)
        means = np.mean(features, axis=0)

        # Coefficient of Variation: std / |mean| (handles scale differences)
        cv = stds / (np.abs(means) + 1e-10)

        # Start with all features
        mask = np.ones(features.shape[1], dtype=bool)

        # Remove zero variance
        mask &= stds > self.variance_threshold

        # Remove extreme variance ONLY if also extreme CV (intelligent filtering)
        if self.max_variance_threshold is not None:
            # Extreme variance is OK if CV is reasonable (proportional to scale)
            extreme_variance = vars > self.max_variance_threshold
            extreme_cv = cv > self.max_cv_threshold
            mask &= ~(extreme_variance & extreme_cv)

        if not mask.any():
            raise ValueError("All features have zero variance!")

        if self.verbose:
            zero_var = (stds <= self.variance_threshold).sum()
            extreme_both = (extreme_variance & extreme_cv).sum() if self.max_variance_threshold else 0
            logger.info(f"   Zero-variance (std <= {self.variance_threshold}): {zero_var}")
            if self.max_variance_threshold:
                logger.info(f"   Extreme-variance AND extreme-CV (var>{self.max_variance_threshold:.1e}, CV>{self.max_cv_threshold}): {extreme_both}")

        return features[:, mask], mask

    def _deduplicate(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove highly correlated features using intelligent selection.

        If use_intelligent_dedup=True: Keeps more informative feature from correlated pairs
        Otherwise: Keeps first occurrence (legacy behavior)

        Args:
            features: Features [N, D]

        Returns:
            Tuple of (deduplicated_features [N, D'], mask [D])
        """
        if features.shape[1] <= 1:
            # Can't compute correlation with single feature
            return features, np.ones(features.shape[1], dtype=bool)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(features.T)

        # Handle NaN correlation (constant features)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        n_features = features.shape[1]
        to_remove = set()

        # Compute informativeness scores for intelligent deduplication
        if self.use_intelligent_dedup:
            informativeness = self._compute_informativeness_scores(features)
        else:
            informativeness = None

        # Find highly correlated pairs
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j not in to_remove and abs(corr_matrix[i, j]) > self.deduplicate_threshold:
                    # Intelligent: Remove less informative feature
                    if informativeness is not None:
                        if informativeness[i] < informativeness[j]:
                            to_remove.add(i)
                            break  # i is removed, move to next i
                        else:
                            to_remove.add(j)
                    else:
                        # Legacy: Remove second feature
                        to_remove.add(j)

        # Create mask
        keep_mask = np.ones(n_features, dtype=bool)
        keep_mask[list(to_remove)] = False

        return features[:, keep_mask], keep_mask

    def _compute_informativeness_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute informativeness score for each feature.

        Higher score = more informative feature (better to keep).

        Score combines:
        1. Entropy (diversity of values)
        2. IQR (spread/range)
        3. Outlier cleanliness (fewer outliers is better)

        Args:
            features: Features [N, D]

        Returns:
            Informativeness scores [D]
        """
        n_features = features.shape[1]
        scores = np.zeros(n_features)

        for feat_idx in range(n_features):
            feat_values = features[:, feat_idx]

            # 1. Entropy (diversity) - use histogram-based estimate
            hist, _ = np.histogram(feat_values, bins=min(50, len(feat_values) // 20))
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))

            # 2. IQR (spread)
            q75, q25 = np.percentile(feat_values, [75, 25])
            iqr = q75 - q25

            # 3. Outlier ratio (cleanliness - lower is better)
            # Use IQR method for outlier detection
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers = ((feat_values < lower_bound) | (feat_values > upper_bound)).sum()
            outlier_ratio = outliers / len(feat_values)

            # Composite score: entropy * iqr * (1 - outlier_ratio)
            # Higher entropy, higher IQR, lower outlier_ratio → better feature
            scores[feat_idx] = entropy * (iqr + 1e-10) * (1 - outlier_ratio)

        return scores

    def _handle_nans(self, features: np.ndarray) -> np.ndarray:
        """Replace NaNs with feature median.

        Args:
            features: Features [N, D]

        Returns:
            Features with NaNs replaced [N, D]
        """
        nan_mask = np.isnan(features)

        if not nan_mask.any():
            return features

        features_clean = features.copy()

        for feat_idx in range(features.shape[1]):
            if nan_mask[:, feat_idx].any():
                median = np.nanmedian(features[:, feat_idx])
                features_clean[nan_mask[:, feat_idx], feat_idx] = median

        return features_clean

    def _cap_outliers(self, features: np.ndarray) -> np.ndarray:
        """Cap outliers using selected method (percentile, IQR, or MAD).

        Dispatches to the appropriate method based on self.outlier_method.

        Args:
            features: Features [N, D]

        Returns:
            Features with outliers capped [N, D]
        """
        if self.outlier_method == "percentile":
            return self._cap_outliers_percentile(features)
        elif self.outlier_method == "iqr":
            return self._cap_outliers_iqr(features)
        elif self.outlier_method == "mad":
            return self._cap_outliers_mad(features)
        else:
            raise ValueError(f"Unknown outlier_method: {self.outlier_method}")

    def _cap_outliers_percentile(self, features: np.ndarray) -> np.ndarray:
        """Cap outliers using percentile-based clipping (ADAPTIVE).

        Clips each feature at specified percentiles (e.g., 0.5% and 99.5%).
        This is truly adaptive - clips the same proportion of outliers for ALL features
        regardless of their distribution.

        Args:
            features: Features [N, D]

        Returns:
            Features with outliers clipped [N, D]
        """
        features_clipped = features.copy()
        lower_pct, upper_pct = self.percentile_range

        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]

            # Compute percentiles
            low = np.percentile(feat_values, lower_pct)
            high = np.percentile(feat_values, upper_pct)

            # Clip
            features_clipped[:, feat_idx] = np.clip(feat_values, low, high)

        return features_clipped

    def _cap_outliers_iqr(self, features: np.ndarray) -> np.ndarray:
        """Cap outliers using IQR-based detection (ADAPTIVE, boxplot-style).

        Uses Q1 - multiplier*IQR and Q3 + multiplier*IQR as bounds.
        Standard multiplier: 1.5 (mild outliers), 3.0 (extreme outliers only).

        Args:
            features: Features [N, D]

        Returns:
            Features with outliers clipped [N, D]
        """
        features_clipped = features.copy()

        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]

            # Compute quartiles and IQR
            q1 = np.percentile(feat_values, 25)
            q3 = np.percentile(feat_values, 75)
            iqr = q3 - q1

            # Compute bounds
            low = q1 - self.iqr_multiplier * iqr
            high = q3 + self.iqr_multiplier * iqr

            # Clip
            features_clipped[:, feat_idx] = np.clip(feat_values, low, high)

        return features_clipped

    def _cap_outliers_mad(self, features: np.ndarray) -> np.ndarray:
        """Cap outliers using MAD (Median Absolute Deviation) - LEGACY METHOD.

        MAD is a robust measure of variability that is resistant to extreme outliers.
        Formula: MAD = median(|x - median(x)|)
        Outliers are capped at: median ± threshold * MAD

        Args:
            features: Features [N, D]

        Returns:
            Features with outliers capped [N, D]
        """
        features_capped = features.copy()

        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]

            # Compute median and MAD
            median = np.median(feat_values)
            mad = np.median(np.abs(feat_values - median))

            # Handle zero MAD (all values identical or near-identical)
            if mad < 1e-10:
                # Use a small multiple of the median as a fallback
                # This prevents division by zero and allows some variation
                mad = max(abs(median) * 0.01, 1e-6)

            # Cap at median ± threshold * MAD
            low = median - self.mad_threshold * mad
            high = median + self.mad_threshold * mad
            features_capped[:, feat_idx] = np.clip(feat_values, low, high)

        return features_capped

    # =========================================================================
    # Statistics for verbose reporting
    # =========================================================================

    def _get_nan_stats(self, features: np.ndarray) -> dict:
        """Get NaN statistics for reporting."""
        nan_mask = np.isnan(features)
        total_nan = nan_mask.sum()
        features_with_nan = nan_mask.any(axis=0).sum()

        return {
            'total_nan_count': int(total_nan),
            'num_features_with_nan': int(features_with_nan),
            'nan_fraction': float(total_nan) / features.size if features.size > 0 else 0.0,
        }

    def _get_outlier_stats_before(self, features: np.ndarray) -> dict:
        """Get outlier statistics before capping (for reporting)."""
        total_outliers = 0
        features_with_outliers = 0

        for feat_idx in range(features.shape[1]):
            feat_values = features[:, feat_idx]
            median = np.median(feat_values)
            mad = np.median(np.abs(feat_values - median))

            if mad < 1e-10:
                mad = max(abs(median) * 0.01, 1e-6)

            low = median - self.mad_threshold * mad
            high = median + self.mad_threshold * mad

            outliers = ((feat_values < low) | (feat_values > high)).sum()
            total_outliers += outliers

            if outliers > 0:
                features_with_outliers += 1

        return {
            'total_outliers': int(total_outliers),
            'num_features_with_outliers': int(features_with_outliers),
        }
