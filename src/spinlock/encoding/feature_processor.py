"""Feature post-processing pipeline for VQ-VAE training.

Ported from unisim.system.features.processor (100% generic, domain-agnostic).

Applies cleaning steps before VQ-VAE training:
1. Remove zero-variance features (std < threshold)
2. Deduplicate highly correlated features (corr > threshold)
3. Handle NaNs (median replacement)
4. Cap outliers using MAD (Median Absolute Deviation)

This ensures clean, numerically stable features for categorical VQ-VAE training.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Feature post-processing pipeline.

    Applies all cleaning steps from production system:
    1. Remove zero-variance features
    2. Deduplicate highly correlated features
    3. Handle NaNs (median replacement)
    4. Cap outliers using MAD (Median Absolute Deviation)
    """

    def __init__(
        self,
        variance_threshold: float = 1e-8,
        deduplicate_threshold: float = 0.99,
        mad_threshold: float = 5.0,
        verbose: bool = False,
    ):
        """Initialize processor.

        Args:
            variance_threshold: Remove features with std below this threshold
            deduplicate_threshold: Remove features with correlation above this threshold
            mad_threshold: MAD multiplier for outlier capping (values beyond median ± mad_threshold * MAD are capped)
                          Typical values: 3.0 (aggressive), 5.0 (moderate), 7.0 (conservative)
            verbose: Print detailed cleaning statistics
        """
        self.variance_threshold = variance_threshold
        self.deduplicate_threshold = deduplicate_threshold
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
        """Remove features with std < threshold.

        Args:
            features: Features [N, D]

        Returns:
            Tuple of (filtered_features [N, D'], mask [D])
        """
        stds = np.nanstd(features, axis=0)
        mask = stds > self.variance_threshold

        if not mask.any():
            raise ValueError("All features have zero variance!")

        return features[:, mask], mask

    def _deduplicate(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove highly correlated features.

        Keeps first occurrence of correlated pair.

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

        # Find highly correlated pairs
        for i in range(n_features):
            if i in to_remove:
                continue
            for j in range(i + 1, n_features):
                if j not in to_remove and abs(corr_matrix[i, j]) > self.deduplicate_threshold:
                    to_remove.add(j)

        # Create mask
        keep_mask = np.ones(n_features, dtype=bool)
        keep_mask[list(to_remove)] = False

        return features[:, keep_mask], keep_mask

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
        """Cap outliers using MAD (Median Absolute Deviation).

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
