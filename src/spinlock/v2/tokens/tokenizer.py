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

from .config import TokenizerConfig
from .model import JointHierarchicalVQVAE
from .trainer import VQTokenizerTrainer
from .checkpoint import load_checkpoint, verify_pretrained_cnn

logger = logging.getLogger(__name__)


class VQTokenizer:
    """High-level interface for VQ-VAE tokenization.

    Provides simple train() and tokenize() methods for trajectory encoding.

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

        # Verify CNN pretraining if needed
        if "initial" in features:
            self._verify_cnn_pretraining()

        # Create model
        logger.info("Creating VQ-VAE model")
        self.model = JointHierarchicalVQVAE(self.config, self.group_indices)

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

        Args:
            dataset: Input dataset

        Returns:
            Dict with keys: temporal, initial_manual, initial_raw, temporal_mask, temporal_lengths
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

                    # Raw ICs (if available)
                    if hasattr(dataset, 'initial_conditions'):
                        ics = dataset.initial_conditions.load_all()  # [N, H, W]
                        # Add channel dimension if needed
                        if ics.ndim == 3:
                            ics = ics[:, None, :, :]  # [N, 1, H, W]
                        features['initial_raw'] = torch.from_numpy(ics).float()
                    else:
                        logger.warning(
                            "Initial hybrid mode requires raw ICs but none found in dataset"
                        )
                        features['initial_raw'] = None
                else:
                    # CNN-only mode
                    features['initial_manual'] = None
                    if hasattr(dataset, 'initial_conditions'):
                        ics = dataset.initial_conditions.load_all()
                        if ics.ndim == 3:
                            ics = ics[:, None, :, :]
                        features['initial_raw'] = torch.from_numpy(ics).float()
                    else:
                        features['initial_raw'] = None
            else:
                features['initial_manual'] = None
                features['initial_raw'] = None

        # Normalize features if configured
        if self.config.normalization.method != "none":
            features = self._normalize_features(features)

        return features

    def _normalize_features(
        self, features: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalize features according to config.

        Args:
            features: Feature dict

        Returns:
            Normalized feature dict
        """
        # Placeholder for normalization logic
        # In practice, this would compute per-category or global statistics
        # and apply normalization

        # For now, just pass through
        # TODO: Implement proper normalization
        logger.warning("Feature normalization not yet implemented")

        return features

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
