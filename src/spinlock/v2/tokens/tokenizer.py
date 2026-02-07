"""High-level VQ Tokenizer interface.

Provides user-friendly API for training and using VQ-VAE tokenizers
for trajectory encoding.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union

import torch
import numpy as np

from spinlock.v2.data import SpinlockDataset
from spinlock.features.grouping import create_grouper, GroupingConfig
from spinlock.encoding.normalization import (
    compute_normalization_stats,
    apply_standard_normalization,
    NormalizationStats,
)

from .config import TokenizerConfig
from .model import JointHierarchicalVQVAE
from .trainer import VQTokenizerTrainer
from .checkpoint import load_checkpoint, verify_pretrained_cnn

logger = logging.getLogger(__name__)


class VQTokenizer:
    """High-level interface for VQ-VAE tokenization.

    Provides simple train() and tokenize() methods for trajectory encoding.

    **Realization Handling Philosophy**:
    The VQ-VAE tokenizer produces operator-level discrete tokens by aggregating
    across M stochastic realizations per operator. This design aligns with:

      1. **MNO Conditioning**: Downstream MNO receives [B, num_tokens] per sample
         (one token set per operator), NOT [B, M, num_tokens] per realization.

      2. **Distribution Encoding**: Tokens represent aggregate operator behavior
         across stochastic runs, capturing characteristic dynamics while abstracting
         away realization-specific noise.

      3. **Decoder Architecture**: Operates on aggregated features [B, D_agg]
         at the operator level, not per-realization [B, M, D].

    **Dataset Schema Requirements**:
    - /inputs/fields: [N, M, C, H, W] with M=3 realizations at axis=1
    - /features/temporal: [N, T, D] (pre-aggregated across realizations)
    - /features/initial/aggregated: [N, D] (pre-aggregated across realizations)

    If dataset format changes, update realization axis handling in _extract_features().

    Example:
        >>> # Training
        >>> config = TokenizerConfig(...)
        >>> tokenizer = VQTokenizer(config)
        >>> tokenizer.train(dataset, output_dir="checkpoints/")
        >>>
        >>> # Inference
        >>> tokenizer = VQTokenizer.from_checkpoint("checkpoints/best.pt")
        >>> tokens = tokenizer.tokenize(trajectories)

    Args:
        config: Complete tokenizer configuration
        model: Optional pre-initialized model (for inference)
        group_indices: Optional pre-computed group indices
    """

    def __init__(
        self,
        config: TokenizerConfig,
        model: Optional[JointHierarchicalVQVAE] = None,
        group_indices: Optional[Dict[str, list]] = None,
    ):
        self.config = config
        self.model = model
        self.group_indices = group_indices
        self.normalization_stats = None

    def train(
        self,
        dataset: Union[SpinlockDataset, str, Path],
        output_dir: Union[str, Path] = "checkpoints",
        checkpoint_prefix: str = "vq_tokenizer",
    ) -> Dict[str, Any]:
        """Train VQ tokenizer on dataset.

        Args:
            dataset: SpinlockDataset or path to dataset file
            output_dir: Directory to save checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames

        Returns:
            Training history dict
        """
        output_dir = Path(output_dir)

        # Load dataset if path provided
        if isinstance(dataset, (str, Path)):
            logger.info(f"Loading dataset from {dataset}")
            dataset = SpinlockDataset.from_file(str(dataset))

        # Extract features
        logger.info("Extracting features from dataset")
        features = self._extract_features(dataset)

        # Perform feature grouping if config specifies it
        if self.config.grouping is not None and self.group_indices is None:
            logger.info("Performing feature grouping")
            self.group_indices = self._perform_grouping(features, dataset)
        elif self.group_indices is None:
            raise ValueError(
                "group_indices must be provided or config.grouping must be set"
            )

        # Normalize features if configured (AFTER grouping, as it needs group_indices)
        if self.config.normalization.method != "none":
            features = self._normalize_features(features)

        # Verify CNN pretraining if needed
        if "initial" in features:
            self._verify_cnn_pretraining()

        # Detect input dimensions from features
        temporal_input_dim = None
        initial_input_dim = None

        if features.get('temporal') is not None:
            temporal_input_dim = features['temporal'].shape[2]  # [N, T, D]
            logger.info(f"Detected temporal input dim: {temporal_input_dim}")

        if features.get('initial_manual') is not None:
            initial_input_dim = features['initial_manual'].shape[1]  # [N, D]
            logger.info(f"Detected initial input dim: {initial_input_dim}")

        # Create model
        logger.info("Creating VQ-VAE model")
        self.model = JointHierarchicalVQVAE(
            self.config,
            self.group_indices,
            temporal_input_dim=temporal_input_dim,
            initial_input_dim=initial_input_dim,
        )

        # Create trainer
        logger.info("Creating trainer")
        trainer = VQTokenizerTrainer(
            self.model,
            self.config,
            self.group_indices,
            self.normalization_stats,
        )

        # Train
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        history = trainer.train(
            temporal_features=features.get("temporal"),
            initial_manual=features.get("initial_manual"),
            initial_raw=features.get("initial_raw"),
            temporal_mask=features.get("temporal_mask"),
            temporal_lengths=features.get("temporal_lengths"),
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
        )

        logger.info("Training complete")
        return history

    def tokenize(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize trajectories to discrete codes.

        Args:
            temporal_features: Temporal sequences [B, T, D_t] (optional)
            initial_manual: Manual initial features [B, D_i] (optional)
            initial_raw: Raw initial conditions [B, C, H, W] (optional)
            temporal_mask: Validity mask for temporal [B, T] (optional)
            temporal_lengths: Actual sequence lengths [B] (optional)

        Returns:
            Dict mapping "family_category_Ll" → token indices [B]

        Example:
            >>> tokens = tokenizer.tokenize(temporal_features=x_t, initial_raw=x_i)
            >>> # tokens = {
            >>> #   "temporal_group_1_L0": [B],
            >>> #   "temporal_group_1_L1": [B],
            >>> #   "initial_group_1_L0": [B],
            >>> #   ...
            >>> # }
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train or load from checkpoint first.")

        self.model.eval()

        with torch.no_grad():
            tokens = self.model.encode(
                temporal_features=temporal_features,
                initial_manual=initial_manual,
                initial_raw=initial_raw,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
            )

        return tokens

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path]) -> "VQTokenizer":
        """Load tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Initialized VQTokenizer instance

        Example:
            >>> tokenizer = VQTokenizer.from_checkpoint("checkpoints/best.pt")
            >>> tokens = tokenizer.tokenize(trajectories)
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path)

        config = checkpoint['config']
        group_indices = checkpoint['group_indices']
        normalization_stats = checkpoint.get('normalization_stats')

        # Create model
        model = JointHierarchicalVQVAE(config, group_indices)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create tokenizer
        tokenizer = cls(config, model=model, group_indices=group_indices)
        tokenizer.normalization_stats = normalization_stats

        logger.info("Checkpoint loaded successfully")
        return tokenizer

    def _extract_features(
        self, dataset: SpinlockDataset
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Extract and prepare features from dataset.

        **Realization Handling**:
        The dataset contains M=3 stochastic realizations per operator. Features
        are aggregated to operator level [N] before training because:

          1. VQ-VAE tokens represent operator behavior (not individual trajectories)
          2. Downstream MNO expects one token set per operator for conditioning
          3. Decoder architecture operates on aggregated features [B, D]

        Temporal and initial manual features are pre-aggregated in the dataset.
        Raw ICs preserve the realization dimension and are aggregated here via mean.

        Args:
            dataset: SpinlockDataset with shapes:
                - inputs: [N, M, C, H, W] where M=3 realizations
                - features.temporal: [N, T, D] (pre-aggregated)
                - features.initial: [N, D] (pre-aggregated)

        Returns:
            Dict with operator-level features (no M dimension):
                - temporal: [N, T, D] if available
                - initial_manual: [N, D] if available
                - initial_raw: [N, C, H, W] if available (aggregated from [N, M, C, H, W])
                - temporal_mask: [N, T] if variable_length enabled
                - temporal_lengths: [N] if variable_length enabled
        """
        features = {}

        with dataset.open():
            # Temporal features
            if hasattr(dataset.features, 'temporal'):
                temporal = dataset.features.temporal.load_all()  # [N, T, D_t]
                features['temporal'] = torch.from_numpy(temporal).float()

                # Variable-length support
                if self.config.encoder.temporal.variable_length:
                    # Create mask (assume all timesteps valid for now)
                    # In practice, this should come from dataset metadata
                    N, T, _ = temporal.shape
                    features['temporal_mask'] = torch.ones(N, T, dtype=torch.bool)
                    features['temporal_lengths'] = torch.full((N,), T, dtype=torch.long)
                else:
                    features['temporal_mask'] = None
                    features['temporal_lengths'] = None
            else:
                features['temporal'] = None
                features['temporal_mask'] = None
                features['temporal_lengths'] = None

            # Initial features
            if hasattr(dataset.features, 'initial'):
                if self.config.encoder.initial.variant == "hybrid":
                    # Manual features
                    initial_manual = dataset.features.initial.load_all()  # [N, D_i]
                    features['initial_manual'] = torch.from_numpy(initial_manual).float()

                    # Raw ICs from /inputs/fields
                    if dataset.inputs is not None:
                        # Load from /inputs/fields [N, M, C, H, W]
                        #
                        # IMPORTANT: We aggregate across M realizations to obtain operator-level
                        # initial conditions for training. This aligns with:
                        #   1. Temporal features (pre-aggregated in dataset: [N, T, D])
                        #   2. Initial manual features (pre-aggregated: [N, D])
                        #   3. Downstream MNO architecture (expects one token set per operator)
                        #
                        # Aggregation method: Mean across M dimension (axis=1)
                        #   - Captures central tendency of stochastic IC distribution
                        #   - Consistent with feature extraction pipeline
                        #   - Preserves channel structure [C] and spatial dimensions [H, W]
                        ics = dataset.inputs.load_all()

                        # Average over realizations M dimension and keep channels
                        if ics.ndim == 5:  # [N, M, C, H, W]
                            ics = ics.mean(axis=1)  # [N, C, H, W]
                        elif ics.ndim == 4:  # [N, C, H, W]
                            pass  # Already in correct format
                        elif ics.ndim == 3:  # [N, H, W]
                            ics = ics[:, None, :, :]  # [N, 1, H, W]

                        features['initial_raw'] = torch.from_numpy(ics).float()
                        logger.info(f"Loaded raw ICs with shape: {features['initial_raw'].shape}")
                    else:
                        logger.warning(
                            "Initial hybrid mode requires raw ICs but inputs not found in dataset"
                        )
                        features['initial_raw'] = None
                else:
                    # CNN-only mode
                    features['initial_manual'] = None
                    if dataset.inputs is not None:
                        # Load and aggregate raw ICs (same logic as hybrid mode)
                        # See comments in hybrid mode section for aggregation rationale
                        ics = dataset.inputs.load_all()

                        # Average over realizations M dimension and keep channels
                        if ics.ndim == 5:  # [N, M, C, H, W]
                            ics = ics.mean(axis=1)  # [N, C, H, W]
                        elif ics.ndim == 4:  # [N, C, H, W]
                            pass  # Already in correct format
                        elif ics.ndim == 3:  # [N, H, W]
                            ics = ics[:, None, :, :]  # [N, 1, H, W]

                        features['initial_raw'] = torch.from_numpy(ics).float()
                        logger.info(f"Loaded raw ICs with shape: {features['initial_raw'].shape}")
                    else:
                        features['initial_raw'] = None
            else:
                features['initial_manual'] = None
                features['initial_raw'] = None

        # Validate operator-level shapes (no M dimension)
        N = None
        for name, feat in features.items():
            if feat is not None and name not in ['temporal_mask', 'temporal_lengths']:
                if N is None:
                    N = feat.shape[0]
                else:
                    assert feat.shape[0] == N, (
                        f"Feature {name} has inconsistent batch size: "
                        f"expected {N}, got {feat.shape[0]}"
                    )
                # Ensure no realization dimension (should be aggregated)
                # Expected shapes: 2D (batch, feat), 3D (batch, time, feat), 4D (batch, channel, height, width)
                assert feat.ndim in [2, 3, 4], (
                    f"Feature {name} has unexpected ndim={feat.ndim}. "
                    f"Expected 2 (batch, feat), 3 (batch, time, feat), "
                    f"or 4 (batch, channel, height, width)"
                )

        if N is not None:
            logger.info(f"Validated features: N={N} operators (aggregated across M realizations)")

        return features

    def _normalize_features(
        self, features: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize features according to config.

        Applies normalization to temporal and initial_manual features based on
        config.normalization.method. Does NOT normalize initial_raw (CNNs handle
        their own normalization).

        Normalization modes:
          - "per_category": Normalize each group independently (default)
          - "global": Apply global normalization across all features per family
          - "none": Return features unchanged

        Temporal features are aggregated over time for stats computation, then
        normalization is broadcast across the time dimension.

        Args:
            features: Dict with keys:
                - temporal: [N, T, D_t] (optional)
                - initial_manual: [N, D_i] (optional)
                - initial_raw: [N, C, H, W] (not normalized, passed through)
                - temporal_mask: [N, T] (not normalized)
                - temporal_lengths: [N] (not normalized)

        Returns:
            Normalized feature dict with same structure

        Side effects:
            Sets self.normalization_stats to Dict[str, NormalizationStats]
        """
        # Early exit for "none" mode
        if self.config.normalization.method == "none":
            self.normalization_stats = None
            return features

        # Initialize output dict and stats storage
        normalized = {}
        stats_dict = {}

        # Normalize temporal features
        if features.get('temporal') is not None:
            normalized['temporal'], temporal_stats = self._normalize_temporal_features(
                features['temporal']
            )
            stats_dict.update(temporal_stats)

        # Normalize initial_manual features
        if features.get('initial_manual') is not None:
            normalized['initial_manual'], initial_stats = self._normalize_initial_features(
                features['initial_manual']
            )
            stats_dict.update(initial_stats)

        # Pass through features that are not normalized
        normalized['initial_raw'] = features.get('initial_raw')  # CNN handles its own norm
        normalized['temporal_mask'] = features.get('temporal_mask')
        normalized['temporal_lengths'] = features.get('temporal_lengths')

        # Store stats and log completion
        self.normalization_stats = stats_dict if stats_dict else None
        logger.info(
            f"Feature normalization complete "
            f"(method={self.config.normalization.method}, "
            f"categories={len(stats_dict)})"
        )

        return normalized

    def _normalize_temporal_features(
        self, temporal: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, NormalizationStats]]:
        """Normalize temporal features [N, T, D_t].

        Args:
            temporal: Temporal features [N, T, D_t]

        Returns:
            Tuple of (normalized_temporal, stats_dict)
        """
        # Aggregate over time for stats computation
        temporal_agg = temporal.mean(dim=1)  # [N, D_t]

        # Get temporal categories from group_indices
        temporal_categories = self._filter_categories_by_prefix('temporal_')

        # Apply normalization based on method
        if self.config.normalization.method == "per_category":
            return self._normalize_per_category(
                temporal, temporal_agg, temporal_categories
            )
        elif self.config.normalization.method == "global":
            return self._normalize_global(
                temporal, temporal_agg, temporal_categories
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization.method}")

    def _normalize_initial_features(
        self, initial: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, NormalizationStats]]:
        """Normalize initial_manual features [N, D_i].

        Args:
            initial: Initial manual features [N, D_i]

        Returns:
            Tuple of (normalized_initial, stats_dict)
        """
        # Get initial categories from group_indices
        initial_categories = self._filter_categories_by_prefix('initial_')

        # Apply normalization based on method
        # For 2D features, temporal_agg is the same as the input
        if self.config.normalization.method == "per_category":
            return self._normalize_per_category(
                initial, initial, initial_categories
            )
        elif self.config.normalization.method == "global":
            return self._normalize_global(
                initial, initial, initial_categories
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization.method}")

    def _filter_categories_by_prefix(self, prefix: str) -> Dict[str, list]:
        """Filter group_indices by category prefix.

        Args:
            prefix: Category prefix (e.g., 'temporal_', 'initial_')

        Returns:
            Dict mapping category name → feature indices
        """
        return {
            k: v for k, v in self.group_indices.items()
            if k.startswith(prefix)
        }

    def _normalize_per_category(
        self,
        features: torch.Tensor,
        features_for_stats: torch.Tensor,
        categories: Dict[str, list]
    ) -> tuple[torch.Tensor, Dict[str, NormalizationStats]]:
        """Apply per-category normalization.

        Each category is normalized independently to mean=0, std=1.

        Args:
            features: Full features to normalize [N, ...] (e.g., [N, T, D] or [N, D])
            features_for_stats: Features to compute stats from [N, D]
            categories: Dict mapping category → feature indices

        Returns:
            Tuple of (normalized_features, stats_dict)
        """
        normalized = features.clone()
        stats_dict = {}

        for category, indices in categories.items():
            # Skip empty categories
            if len(indices) == 0:
                logger.warning(f"Skipping empty category: {category}")
                continue

            # Compute stats on aggregated features
            cat_features = features_for_stats[:, indices]
            stats = compute_normalization_stats(cat_features)

            # Apply normalization (handles broadcasting for temporal [N, T, D])
            if features.ndim == 3:  # Temporal [N, T, D]
                normalized[:, :, indices] = apply_standard_normalization(
                    features[:, :, indices], stats
                )
            else:  # Initial [N, D]
                normalized[:, indices] = apply_standard_normalization(
                    features[:, indices], stats
                )

            # Apply clipping if configured
            if self.config.normalization.clip_std_multiplier is not None:
                clip_val = self.config.normalization.clip_std_multiplier
                if features.ndim == 3:
                    normalized[:, :, indices] = torch.clamp(
                        normalized[:, :, indices], -clip_val, clip_val
                    )
                else:
                    normalized[:, indices] = torch.clamp(
                        normalized[:, indices], -clip_val, clip_val
                    )

            stats_dict[category] = stats

        return normalized, stats_dict

    def _normalize_global(
        self,
        features: torch.Tensor,
        features_for_stats: torch.Tensor,
        categories: Dict[str, list]
    ) -> tuple[torch.Tensor, Dict[str, NormalizationStats]]:
        """Apply global normalization.

        All features in the family share the same global statistics.

        Args:
            features: Full features to normalize [N, ...] (e.g., [N, T, D] or [N, D])
            features_for_stats: Features to compute stats from [N, D]
            categories: Dict mapping category → feature indices

        Returns:
            Tuple of (normalized_features, stats_dict)
        """
        # Compute global stats across all features
        global_stats = compute_normalization_stats(features_for_stats)

        # Apply global normalization
        normalized = apply_standard_normalization(features, global_stats)

        # Apply clipping if configured
        if self.config.normalization.clip_std_multiplier is not None:
            clip_val = self.config.normalization.clip_std_multiplier
            normalized = torch.clamp(normalized, -clip_val, clip_val)

        # Store per-category slices for checkpoint compatibility
        stats_dict = {}
        for category, indices in categories.items():
            if len(indices) > 0:
                stats_dict[category] = NormalizationStats(
                    mean=global_stats.mean[indices],
                    std=global_stats.std[indices]
                )

        return normalized, stats_dict

    def _perform_grouping(
        self, features: Dict[str, Optional[torch.Tensor]], dataset: SpinlockDataset
    ) -> Dict[str, list]:
        """Perform automatic feature grouping.

        Args:
            features: Extracted features
            dataset: Source dataset

        Returns:
            Dict mapping family_category → feature indices
        """
        group_indices = {}

        # Temporal grouping
        if features.get('temporal') is not None:
            temporal = features['temporal']

            # Aggregate over time for grouping (mean)
            temporal_agg = temporal.mean(dim=1).numpy()  # [N, D_t]

            # Get feature names
            temporal_names = [f"temporal_{i}" for i in range(temporal_agg.shape[1])]

            # Create grouper
            grouper = create_grouper("temporal", config=self.config.grouping)

            # Group features
            result = grouper.group_features(temporal_agg, temporal_names)

            # Convert to expected format
            for group_name, group in result.groups.items():
                group_indices[f"temporal_{group_name}"] = group.feature_indices

        # Initial grouping
        if features.get('initial_manual') is not None:
            initial = features['initial_manual'].numpy()  # [N, D_i]

            # Get feature names
            initial_names = [f"initial_{i}" for i in range(initial.shape[1])]

            # Create grouper
            grouper = create_grouper("initial", config=self.config.grouping)

            # Group features
            result = grouper.group_features(initial, initial_names)

            # Convert to expected format
            for group_name, group in result.groups.items():
                group_indices[f"initial_{group_name}"] = group.feature_indices

        logger.info(f"Feature grouping complete: {len(group_indices)} groups")
        return group_indices

    def _verify_cnn_pretraining(self):
        """Verify CNN pretraining if configured."""
        if self.config.encoder.initial.variant == "hybrid":
            pretrained_path = self.config.encoder.initial.pretrained_cnn_path

            if pretrained_path is not None:
                # Verify checkpoint exists and is valid
                verify_pretrained_cnn(
                    Path(pretrained_path),
                    self.config.encoder.initial.cnn_embedding_dim,
                )
            else:
                logger.warning(
                    "Initial hybrid encoder without CNN pretraining. "
                    "For better results, pretrain the CNN first using:\n"
                    f"  poetry run spinlock pretrain-initial-features-cnn \\\n"
                    f"    --dataset <dataset_path> \\\n"
                    f"    --embedding-dim {self.config.encoder.initial.cnn_embedding_dim} \\\n"
                    f"    --output <pretrained_cnn_path>"
                )
