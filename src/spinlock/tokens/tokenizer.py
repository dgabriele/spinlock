"""High-level VQ Tokenizer interface.

Provides user-friendly API for training and using VQ-VAE tokenizers
for trajectory encoding.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple

import torch
import numpy as np

from spinlock.data import SpinlockDataset
from spinlock.features.grouping import create_grouper, GroupingConfig
from spinlock.encoding.normalization import (
    compute_normalization_stats,
    apply_standard_normalization,
    NormalizationStats,
)
from spinlock.encoding.feature_processor import FeatureProcessor

from .config import TokenizerConfig
from .model import JointHierarchicalVQVAE
from .trainer import VQTokenizerTrainer
from .checkpoint import load_checkpoint, verify_pretrained_cnn

logger = logging.getLogger(__name__)


class VQTokenizer:
    """High-level interface for VQ-VAE tokenization.

    Provides simple train() and tokenize() methods for trajectory encoding.

    **Multi-Codebook Architecture - CRITICAL UNDERSTANDING**:
    VQTokenizer trains 30+ codebooks SIMULTANEOUSLY (not a single codebook).
    Each rollout is encoded as a SET of tokens (one from each codebook).

    **Token Diversity Interpretation**:
    Individual codebook utilization metrics (e.g., "10% utilization") are
    MISLEADING in isolation. What matters is COMBINATORIAL DIVERSITY:

      - 30 codebooks × 10% utilization per codebook = massive diversity
      - If each codebook uses 3/28 codes (10%), combinatorial space = 3^30 ≈ 2×10^14
      - Diversity metric: unique token SET combinations across rollouts
      - Similarity metric: Jaccard similarity between token sets

    DO NOT interpret low per-codebook utilization as poor diversity!
    The joint distribution of token sets is what provides semantic richness.

    Example token set for one rollout:
        {
          'theta_group_1_L0': 5,
          'theta_group_1_L1': 12,
          'initial_manual_features_L0': 3,
          'temporal_group_1_L0': 8,
          'temporal_group_2_L0': 15,
          ... (30+ tokens total)
        }

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
        self.feature_metadata = None  # Will be populated during training

    def _resume_training(
        self,
        resume_from: Path,
        trainer: VQTokenizerTrainer,
    ) -> int:
        """Load checkpoint and restore model + trainer state for resume.

        Args:
            resume_from: Path to checkpoint file
            trainer: Trainer to restore optimizer/scheduler/tracking state into

        Returns:
            start_epoch for the training loop
        """
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_checkpoint(resume_from)

        # Restore model weights
        self.model.load_state_dict(ckpt.model_state_dict)
        logger.info("Restored model weights")

        # Restore trainer state (optimizer, scheduler, loss EMA, tracking)
        return trainer.resume_state(ckpt)

    def train(
        self,
        dataset: Union[SpinlockDataset, str, Path],
        output_dir: Union[str, Path] = "checkpoints",
        checkpoint_prefix: str = "vq_tokenizer",
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Train VQ tokenizer on dataset.

        Args:
            dataset: SpinlockDataset or path to dataset file
            output_dir: Directory to save checkpoints
            checkpoint_prefix: Prefix for checkpoint filenames
            resume_from: Path to checkpoint to resume from (optional)

        Returns:
            Training history dict
        """
        output_dir = Path(output_dir)

        # Load dataset if path provided
        if isinstance(dataset, (str, Path)):
            logger.info(f"Loading dataset from {dataset}")
            dataset = SpinlockDataset.from_file(str(dataset))

        # ── Learned mode: entirely different data/grouping/training path ──
        if self.config.feature_source == "learned":
            return self._train_learned_mode(dataset, output_dir, checkpoint_prefix, resume_from)

        # ── Manual mode (default): hand-crafted features ──
        # Extract features
        logger.info("Extracting features from dataset")
        features = self._extract_features(dataset)

        # Clean features if configured (BEFORE grouping for pre-categorization)
        if (self.config.feature_cleaning is not None and
            self.config.feature_cleaning.enabled and
            self.config.feature_cleaning.pre_categorization):
            logger.info("Cleaning features (pre-categorization)")
            features = self.clean_features(features)

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
            D_t = features['temporal'].shape[2]  # [N, T, D]
            _pca_method = (
                self.config.grouping is not None
                and getattr(self.config.grouping, "method", "correlation")
                in ("pca_striped", "opq")
            )
            # Per-group encoders receive cat([mean, std]) → 2*D_t features
            temporal_input_dim = 2 * D_t if _pca_method else D_t
            logger.info(
                f"Detected temporal input dim: {temporal_input_dim}"
                + (" (mean+std concat)" if _pca_method else "")
            )

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

        # Wire PCA/OPQ rotation into model buffers (per-group encoder path)
        if hasattr(self, "_temporal_linear_transform") and self._temporal_linear_transform is not None:
            self.model.set_temporal_rotation(self._temporal_linear_transform)
            logger.info("Temporal rotation wired into model buffers.")

        # Create trainer
        logger.info("Creating trainer")
        trainer = VQTokenizerTrainer(
            self.model,
            self.config,
            self.group_indices,
            self.normalization_stats,
            self.feature_metadata,
        )

        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from:
            start_epoch = self._resume_training(resume_from, trainer)

        # Train
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        history = trainer.train(
            temporal_features=features.get("temporal"),
            initial_manual=features.get("initial_manual"),
            initial_raw=features.get("initial_raw"),
            theta_features=features.get("theta"),
            temporal_mask=features.get("temporal_mask"),
            temporal_lengths=features.get("temporal_lengths"),
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            start_epoch=start_epoch,
        )

        logger.info("Training complete")
        return history

    def tokenize(
        self,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        initial_raw: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        temporal_raw: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize trajectories to discrete codes.

        Args:
            temporal_features: Temporal sequences [B, T, D_t] (optional, manual mode)
            initial_manual: Manual initial features [B, D_i] (optional)
            initial_raw: Raw initial conditions [B, C, H, W] (optional)
            theta_features: Operator parameters [B, param_dim] (optional)
            temporal_mask: Validity mask for temporal [B, T] (optional)
            temporal_lengths: Actual sequence lengths [B] (optional)
            temporal_raw: Raw trajectories [B, T, C, H, W] (required for learned mode)

        Returns:
            Dict mapping "family_category_Ll" → token indices [B]

        Example:
            >>> tokens = tokenizer.tokenize(temporal_features=x_t, initial_raw=x_i, theta_features=theta)
            >>> # tokens = {
            >>> #   "temporal_group_1_L0": [B],
            >>> #   "temporal_group_1_L1": [B],
            >>> #   "initial_group_1_L0": [B],
            >>> #   "theta_group_1_L0": [B],
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
                theta_features=theta_features,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
                temporal_raw=temporal_raw,
            )

        return tokens

    def decode(
        self,
        tokens: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode discrete tokens back to actual (theta, ICs).

        Requires inverse models to be loaded. Returns proper operator parameters
        and spatial initial conditions suitable for CNOReplayer.

        Args:
            tokens: Dict mapping "family_category_Ll" → token indices [B]

        Returns:
            Tuple of (theta [B, 14], u0 [B, 3, 64, 64])

        Raises:
            ValueError: If inverse models are not loaded
            ValueError: If required families (theta, initial) are missing

        Example:
            >>> tokenizer = VQTokenizer.from_checkpoint(
            ...     "checkpoints/best.pt",
            ...     theta_inverse_path="checkpoints/theta_inverse.pt",
            ...     initial_inverse_path="checkpoints/initial_inverse.pt",
            ... )
            >>> theta, u0 = tokenizer.decode(tokens)
            >>> # theta: [B, 14] in [0,1], u0: [B, 3, 64, 64]
        """
        if self.model is None:
            raise ValueError("Model not initialized. Load from checkpoint first.")

        # Verify required families exist
        if "theta" not in self.model.families or "initial" not in self.model.families:
            raise ValueError(
                f"Decode requires both 'theta' and 'initial' families. "
                f"Found families: {self.model.families}"
            )

        self.model.eval()

        with torch.no_grad():
            # Prepare tokens
            device = next(self.model.parameters()).device
            tokens = self._move_tokens_to_device(tokens, device)

            # Extract embeddings and reconstruct
            quantized = self._extract_embeddings(tokens)
            reconstructed = self.model.decoder(quantized)

            # Split by family and apply inverse decoders
            theta, u0 = self._split_and_decode(reconstructed)

        return theta, u0

    def _move_tokens_to_device(
        self, tokens: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Move tokens to device."""
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
            for k, v in tokens.items()
        }

    def _extract_embeddings(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings from codebooks."""
        embeddings = []

        for key in sorted(tokens.keys()):
            if key not in self.model.quantizers:
                logger.warning(f"Skipping unknown quantizer: {key}")
                continue

            quantizer = self.model.quantizers[key]
            embedding = quantizer.embedding(tokens[key])
            embeddings.append(embedding)

        if not embeddings:
            raise ValueError("No valid tokens found")

        return torch.cat(embeddings, dim=1)

    def _split_and_decode(
        self, reconstructed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split reconstructed features and apply inverse decoders.

        Returns:
            Tuple of (theta [B, 14], u0 [B, 3, 64, 64])
        """
        # Determine family order and offsets
        # Order: temporal, initial, theta (as constructed in model)
        offset = 0

        # Skip temporal if present
        if "temporal" in self.model.families:
            offset += self.model.temporal_dim

        # Extract and decode initial
        initial_encoded = self._extract_family_features(
            reconstructed, offset, self.model.initial_dim
        )
        u0 = self._decode_initial(initial_encoded)
        offset += self.model.initial_dim

        # Extract and decode theta
        theta_encoded = self._extract_family_features(
            reconstructed, offset, self.model.theta_dim
        )
        theta = self._decode_theta(theta_encoded)

        return theta, u0

    def _extract_family_features(
        self, reconstructed: torch.Tensor, offset: int, dim: int
    ) -> torch.Tensor:
        """Extract features for a family."""
        return reconstructed[:, offset:offset + dim]

    def _decode_theta(self, theta_encoded: torch.Tensor) -> torch.Tensor:
        """Decode theta from encoded representation."""
        if hasattr(self.model, 'theta_inverse') and self.model.theta_inverse is not None:
            return self.model.theta_inverse(theta_encoded)

        # No fallback: inverse model is required for proper decoding
        raise ValueError(
            "Theta inverse model not loaded. Train and load the inverse model using:\n"
            "  1. Train: poetry run python scripts/train_theta_inverse.py ...\n"
            "  2. Load: tokenizer.load_theta_inverse('checkpoints/theta_inverse.pt')\n"
            "Or load at initialization: VQTokenizer.from_checkpoint(..., theta_inverse_path=...)"
        )

    def _decode_initial(self, initial_encoded: torch.Tensor) -> torch.Tensor:
        """Decode initial conditions from encoded representation."""
        if hasattr(self.model, 'initial_inverse') and self.model.initial_inverse is not None:
            return self.model.initial_inverse(initial_encoded)

        # No fallback: inverse model is required for proper decoding
        raise ValueError(
            "Initial inverse model not loaded. Train and load the inverse model using:\n"
            "  1. Train: poetry run python scripts/train_initial_inverse.py ...\n"
            "  2. Load: tokenizer.load_initial_inverse('checkpoints/initial_inverse.pt')\n"
            "Or load at initialization: VQTokenizer.from_checkpoint(..., initial_inverse_path=...)"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        theta_inverse_path: Optional[Union[str, Path]] = None,
        initial_inverse_path: Optional[Union[str, Path]] = None,
    ) -> "VQTokenizer":
        """Load tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            theta_inverse_path: Optional path to trained ThetaInverseMLP checkpoint
            initial_inverse_path: Optional path to trained InitialInverseCNN checkpoint

        Returns:
            Initialized VQTokenizer instance

        Example:
            >>> tokenizer = VQTokenizer.from_checkpoint("checkpoints/best.pt")
            >>> tokens = tokenizer.tokenize(trajectories)
            >>>
            >>> # With inverse models
            >>> tokenizer = VQTokenizer.from_checkpoint(
            ...     "checkpoints/best.pt",
            ...     theta_inverse_path="checkpoints/theta_inverse.pt",
            ...     initial_inverse_path="checkpoints/initial_inverse.pt",
            ... )
            >>> theta, u0, _ = tokenizer.decode(tokens)
        """
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path)

        # Type-safe access via Pydantic schema
        config = checkpoint.config
        group_indices = checkpoint.group_indices
        normalization_stats = checkpoint.normalization_stats
        temporal_input_dim = checkpoint.temporal_input_dim
        initial_input_dim = checkpoint.initial_input_dim

        # Create model with saved dimensions
        model = JointHierarchicalVQVAE(
            config,
            group_indices,
            temporal_input_dim=temporal_input_dim,
            initial_input_dim=initial_input_dim,
        )
        model.load_state_dict(checkpoint.model_state_dict)
        model.eval()

        # Create tokenizer
        tokenizer = cls(config, model=model, group_indices=group_indices)
        tokenizer.normalization_stats = normalization_stats
        tokenizer.feature_metadata = checkpoint.feature_metadata  # Load feature metadata (v2.1+)

        # Load inverse models if provided
        if theta_inverse_path is not None:
            tokenizer.load_theta_inverse(theta_inverse_path)

        if initial_inverse_path is not None:
            tokenizer.load_initial_inverse(initial_inverse_path)

        logger.info("Checkpoint loaded successfully")
        return tokenizer

    def _train_learned_mode(
        self,
        dataset: SpinlockDataset,
        output_dir: Path,
        checkpoint_prefix: str,
        resume_from: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Training path for learned CNN temporal features.

        Bypasses the entire manual feature extraction/cleaning/grouping pipeline.
        Instead:
        1. Loads params lazily + ICs lazily from dataset
        2. Creates sequential group indices (CNN output slices)
        3. Creates CNOReplayer for on-the-fly trajectory generation
        4. Trains with per-batch trajectory generation via lazy Dataset

        Uses lazy HDF5 IC loading to avoid OOM on large datasets (e.g. 500K
        samples at 73 GB). Per-batch HDF5 reads add ~10-50ms latency, which
        is negligible vs. the ~1s Lenia simulation per batch.
        """
        logger.info("LEARNED MODE: CNN temporal features with on-the-fly trajectories")

        # Prepare data: lazy dataset, sequential groups, replayer
        learned_dataset, replayer = self._prepare_learned_mode_data(dataset)

        # Detect initial_input_dim from introspected dataset dimensions
        initial_input_dim = None
        if self.config.encoder.initial.variant == "hybrid":
            initial_input_dim = dataset.initial_feature_dim or 0

        # Create model (temporal_input_dim not needed — CNN sets its own dims)
        logger.info("Creating VQ-VAE model (learned mode)")
        self.model = JointHierarchicalVQVAE(
            self.config,
            self.group_indices,
            temporal_input_dim=None,  # Not used in learned mode
            initial_input_dim=initial_input_dim,
        )

        # Create trainer with replayer
        logger.info(f"Creating trainer with {type(replayer).__name__ if replayer else 'no'} replayer")
        trainer = VQTokenizerTrainer(
            self.model,
            self.config,
            self.group_indices,
            self.normalization_stats,
            self.feature_metadata,
            replayer=replayer,
        )

        # Resume from checkpoint if requested
        start_epoch = 0
        if resume_from:
            start_epoch = self._resume_training(resume_from, trainer)

        # Train — pass lazy dataset directly (no tensors in RAM)
        logger.info(f"Starting learned-mode training for {self.config.training.num_epochs} epochs")
        history = trainer.train(
            dataset=learned_dataset,
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            start_epoch=start_epoch,
        )

        # Clean up lazy HDF5 handle
        learned_dataset.close_lazy()

        logger.info("Learned-mode training complete")
        return history

    def _prepare_learned_mode_data(
        self,
        dataset: SpinlockDataset,
    ) -> tuple:
        """Prepare data for learned CNN temporal feature mode.

        Creates a lazy SpinlockDataset that reads ICs per-sample from HDF5
        instead of loading all into RAM. Params are loaded eagerly (small).
        Creates sequential group indices and builds a replayer for on-the-fly
        trajectory generation.

        Args:
            dataset: SpinlockDataset (introspection-only, already opened by CLI)

        Returns:
            Tuple of (learned_dataset, replayer):
                learned_dataset: SpinlockDataset with lazy_ics=True
                replayer: Replayer instance for trajectory generation
        """
        # ── Create lazy dataset (ICs read per-sample, params loaded eagerly) ──
        total = dataset.raw_input_shape[0]
        # Respect --max-samples CLI override (smoke test mode)
        max_override = getattr(dataset, '_max_samples_override', None)
        if max_override is not None:
            total = min(total, max_override)
            logger.info(f"Dataset capped to {total} samples (--max-samples override)")
        learned_dataset = SpinlockDataset(
            str(dataset.file_path),
            max_samples=total,
            realization_mode="mean",
            lazy_ics=True,
        )
        logger.info(
            f"Lazy dataset created: {learned_dataset.n_samples} samples, "
            f"ICs read per-sample from HDF5 (realization_mode='mean')"
        )

        # ── Auto-detect dimensions from introspected metadata ──
        learned_cfg = self.config.encoder.temporal.learned
        if learned_cfg is None:
            raise ValueError(
                "feature_source='learned' requires encoder.temporal.learned config"
            )

        num_channels = dataset.num_channels
        if num_channels is not None and learned_cfg.in_channels is None:
            learned_cfg.in_channels = num_channels
            logger.info(f"Auto-detected in_channels={learned_cfg.in_channels} from dataset metadata")

        param_dim = dataset.theta_param_dim

        # ── Create group indices ──
        num_groups = learned_cfg.num_groups
        self.group_indices = {}

        if learned_cfg.architecture == "pyramid_first":
            # Pyramid-first mode: grouping is learned (not index-based).
            # Indices are placeholders — the LearnedGroupProjection handles
            # assignment. Each group gets D_group = encoder.embedding_dim.
            d_group = self.config.encoder.embedding_dim  # 64
            for i in range(num_groups):
                self.group_indices[f"temporal_group_{i}"] = list(
                    range(i * d_group, (i + 1) * d_group)
                )
        else:
            # Legacy per_frame mode: sequential slicing of CNN output
            group_dim = learned_cfg.embedding_dim // num_groups
            for i in range(num_groups):
                self.group_indices[f"temporal_group_{i}"] = list(
                    range(i * group_dim, (i + 1) * group_dim)
                )

        # Add theta and initial groups if data available
        # In temporal-only mode, skip theta/initial VQ groups (they're decoded
        # from temporal tokens via aux heads), but still auto-detect dimensions
        # needed for aux head creation.
        if param_dim is not None:
            # Auto-detect theta param_dim in config (needed for aux head dim)
            if self.config.encoder.theta is not None and self.config.encoder.theta.param_dim is None:
                self.config.encoder.theta.param_dim = param_dim

            if not self.config.temporal_only:
                theta_cfg = self.config.encoder.theta
                if theta_cfg is not None and theta_cfg.variant == "direct":
                    for i in range(param_dim):
                        self.group_indices[f"theta_param_{i}"] = [i]
                else:
                    self.group_indices["theta_group_0"] = list(range(param_dim))

        if num_channels is not None:
            # Auto-detect in_channels (needed for aux head creation)
            if self.config.encoder.initial.in_channels is None:
                self.config.encoder.initial.in_channels = num_channels

            if not self.config.temporal_only and self.config.encoder.initial.variant == "cnn":
                cnn_dim = self.config.encoder.initial.cnn_embedding_dim
                self.group_indices["initial_group_0"] = list(range(cnn_dim))

        logger.info(f"Sequential group indices: {len(self.group_indices)} groups")

        # Create trajectory replayer (auto-detect operator type from config)
        replayer = self._create_replayer()

        # Auto-detect generation_timesteps from dataset if not specified
        if self.config.generation_timesteps is None:
            ts = dataset.temporal_feature_dim  # Check if temporal features exist
            if dataset._dimension_cache.get('temporal_timesteps') is not None:
                self.config.generation_timesteps = dataset._dimension_cache['temporal_timesteps']
                logger.info(
                    f"Auto-detected generation_timesteps={self.config.generation_timesteps}"
                )
            else:
                self.config.generation_timesteps = 64  # Sensible default
                logger.info(
                    f"Using default generation_timesteps={self.config.generation_timesteps}"
                )

        return learned_dataset, replayer

    def _create_replayer(self):
        """Create trajectory replayer from config, auto-detecting operator type.

        Resolution order:
            1. generation_config_path (preferred — auto-detects operator type)
            2. cno_config_path (backward compat — always creates CNOReplayer)
            3. None (no replayer — trajectories must be in dataset)
        """
        import yaml as _yaml

        config_path = self.config.generation_config_path or self.config.cno_config_path
        if config_path is None:
            logger.warning(
                "No generation_config_path or cno_config_path — "
                "trajectories must be provided externally"
            )
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Auto-detect operator type from generation config
        with open(config_path) as f:
            gen_config = _yaml.safe_load(f)
        operator_type = gen_config.get("simulation", {}).get("operator_type")

        match operator_type:
            case "lenia":
                from spinlock.lenia.replay_adapter import LeniaReplayAdapter
                replayer = LeniaReplayAdapter.from_config(config_path, device=device)
            case "cnn" | "u_afno":
                from spinlock.mno.cno_replay import CNOReplayer
                replayer = CNOReplayer.from_config(
                    config_path,
                    device=device,
                    cache_size=self.config.replayer_cache_size,
                )
            case "qbm":
                raise NotImplementedError(
                    f"QBM replay adapter not yet implemented. "
                    f"QBM trajectories must be stored in the dataset."
                )
            case _:
                raise NotImplementedError(
                    f"No replay adapter for operator_type='{operator_type}'. "
                    f"Supported types: cnn, u_afno, lenia. "
                    f"QBM requires stored trajectories."
                )

        logger.info(
            f"Created {type(replayer).__name__} for operator_type='{operator_type}' "
            f"from {config_path}"
        )
        return replayer

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

            # Theta parameters (operator parameters)
            # Note: These are stored at /parameters/params as [N, 14] in [0,1] unit hypercube
            if dataset.parameters is not None and dataset.parameters.params is not None:
                params = dataset.parameters.params.load_all()  # [N, 14]
                features['theta'] = torch.from_numpy(params).float()
                logger.info(f"Loaded theta parameters with shape: {features['theta'].shape}")
            else:
                features['theta'] = None
                # Only warn if theta is needed (check if config will use it)
                # We'll validate in model forward pass if actually needed

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

    def clean_features(
        self, features: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Clean features using FeatureProcessor.

        Removes zero-variance features, deduplicates highly correlated features,
        handles NaNs, and caps outliers. Only processes temporal features for now.

        Args:
            features: Dict with keys:
                - temporal: [N, T, D_t] (will be cleaned)
                - initial_manual: [N, D_i] (passed through)
                - initial_raw: [N, C, H, W] (passed through)
                - temporal_mask: [N, T] (passed through)
                - temporal_lengths: [N] (passed through)
                - theta: [N, param_dim] (passed through)

        Returns:
            Cleaned feature dict with reduced temporal dimension
        """
        if self.config.feature_cleaning is None or not self.config.feature_cleaning.enabled:
            return features

        cleaned = {}

        # Clean temporal features
        if features.get('temporal') is not None:
            temporal = features['temporal']  # [N, T, D_t]
            N, T, D_t = temporal.shape

            # Aggregate over time for cleaning (use mean per trajectory)
            temporal_agg = temporal.mean(dim=1)  # [N, D_t]

            # Initialize FeatureProcessor
            processor = FeatureProcessor(
                variance_threshold=self.config.feature_cleaning.variance_threshold,
                deduplicate_threshold=self.config.feature_cleaning.deduplicate_threshold,
                use_intelligent_dedup=self.config.feature_cleaning.use_intelligent_dedup,
                outlier_method=self.config.feature_cleaning.outlier_method,
                percentile_range=self.config.feature_cleaning.percentile_range,
                verbose=self.config.verbose,
            )

            # Build semantic feature names from the orchestrator if possible.
            # This requires num_channels to reconstruct the feature registry.
            # Detect from initial_raw [N, C, H, W] or fall back to generic.
            original_feature_names = self._build_semantic_feature_names(
                D_t, features
            )

            # Clean features
            temporal_cleaned_np, feature_mask, cleaned_feature_names, cleaning_report = processor.clean(
                temporal_agg.numpy(),
                feature_names=original_feature_names,
            )

            # Convert back to tensor
            temporal_cleaned_agg = torch.from_numpy(temporal_cleaned_np).float()

            # Apply mask to full temporal tensor
            temporal_cleaned = temporal[:, :, feature_mask]  # [N, T, D_t']

            cleaned['temporal'] = temporal_cleaned

            # Build feature metadata from cleaning report
            self._build_feature_metadata(
                cleaning_report=cleaning_report,
                feature_mask=feature_mask,
                original_feature_names=original_feature_names,
                cleaned_feature_names=cleaned_feature_names,
            )

            logger.info(
                f"Feature cleaning complete: "
                f"{D_t} → {temporal_cleaned.shape[2]} features "
                f"({D_t - temporal_cleaned.shape[2]} removed)"
            )
        else:
            cleaned['temporal'] = None

        # Pass through other features
        cleaned['initial_manual'] = features.get('initial_manual')
        cleaned['initial_raw'] = features.get('initial_raw')
        cleaned['temporal_mask'] = features.get('temporal_mask')
        cleaned['temporal_lengths'] = features.get('temporal_lengths')
        cleaned['theta'] = features.get('theta')

        return cleaned

    def _build_semantic_feature_names(
        self,
        D_t: int,
        features: Dict[str, Optional[torch.Tensor]],
    ) -> list:
        """Build semantic feature names for temporal features.

        Uses TemporalFeatureOrchestrator.get_all_feature_names() to produce
        names like 'spatial_mean_ch0', 'spectral_entropy_ch1' instead of
        'temporal_feature_0'. Falls back to generic names if the orchestrator
        is unavailable or produces a dimension mismatch.

        Args:
            D_t: Number of temporal features (raw, before cleaning).
            features: Full feature dict (used to detect num_channels from
                initial_raw shape [N, C, H, W]).

        Returns:
            List of D_t feature names.
        """
        # Detect num_channels from initial_raw [N, C, H, W]
        initial_raw = features.get('initial_raw')
        if initial_raw is not None and initial_raw.dim() == 4:
            num_channels = initial_raw.shape[1]
        else:
            logger.warning(
                "Cannot detect num_channels from initial_raw — "
                "falling back to generic feature names"
            )
            return [f"temporal_feature_{i}" for i in range(D_t)]

        try:
            from spinlock.features.temporal.config import TemporalFeatureConfig
            from spinlock.features.temporal.extractors import (
                TemporalFeatureOrchestrator,
            )

            # Create a temporary orchestrator on CPU for name generation.
            # Use default TemporalFeatureConfig (which has quantum.enabled=True)
            # to match the dataset generation pipeline (MNOFeatureExtractor).
            feature_config = TemporalFeatureConfig()
            orchestrator = TemporalFeatureOrchestrator(
                device=torch.device('cpu'),
                config=feature_config,
            )
            names = orchestrator.get_all_feature_names(num_channels)

            if len(names) == D_t:
                logger.info(
                    f"Semantic feature names: {len(names)} names for "
                    f"{num_channels}-channel data "
                    f"(e.g. {names[0]}, {names[-1]})"
                )
                return names
            else:
                logger.warning(
                    f"Feature name count mismatch: orchestrator produced "
                    f"{len(names)} but data has {D_t}. "
                    f"Falling back to generic names."
                )
                return [f"temporal_feature_{i}" for i in range(D_t)]

        except Exception as e:
            logger.warning(
                f"Failed to build semantic feature names: {e}. "
                f"Falling back to generic names."
            )
            return [f"temporal_feature_{i}" for i in range(D_t)]

    def _build_feature_metadata(
        self,
        cleaning_report: dict,
        feature_mask: np.ndarray,
        original_feature_names: list,
        cleaned_feature_names: list,
    ):
        """Build FeatureMetadata from cleaning report.

        Args:
            cleaning_report: Detailed cleaning report from FeatureProcessor.clean()
            feature_mask: Boolean mask indicating which features were kept
            original_feature_names: Names of all original features
            cleaned_feature_names: Names of features after cleaning
        """
        from .checkpoint import (
            FeatureMetadata,
            FeatureFamilyMetadata,
            CleaningOperation,
        )

        # Build CleaningOperation objects from report
        operations = []
        for op in cleaning_report['operations']:
            operations.append(
                CleaningOperation(
                    operation_type=op['operation_type'],
                    features_removed=op['features_removed'],
                    features_kept=op['features_kept'],
                    parameters=op['parameters'],
                    num_removed=op['num_removed'],
                    num_kept=op['num_kept'],
                )
            )

        # Get kept and removed indices
        kept_indices = np.where(feature_mask)[0].tolist()
        removed_indices = np.where(~feature_mask)[0].tolist()

        # Build FeatureFamilyMetadata for temporal features
        temporal_family = FeatureFamilyMetadata(
            family_name='temporal',
            original_feature_count=len(original_feature_names),
            cleaned_feature_count=len(cleaned_feature_names),
            original_feature_names=original_feature_names,
            kept_feature_indices=kept_indices,
            kept_feature_names=cleaned_feature_names,
            removed_feature_indices=removed_indices,
            removed_feature_names=[original_feature_names[i] for i in removed_indices],
            cleaning_operations=operations,
        )

        # Build FeatureMetadata (categories will be added later during grouping)
        self.feature_metadata = FeatureMetadata(
            families={'temporal': temporal_family},
            categories={},  # Will be populated during grouping
            total_original_features=len(original_feature_names),
            total_cleaned_features=len(cleaned_feature_names),
            total_features_removed=len(removed_indices),
            cleaning_config=self.config.feature_cleaning.model_dump() if self.config.feature_cleaning else None,
            grouping_config=None,  # Will be set during grouping
        )

        logger.info(f"Feature metadata built: {len(cleaned_feature_names)} features kept from {len(original_feature_names)} original")

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

        # Normalize theta features
        if features.get('theta') is not None:
            normalized['theta'], theta_stats = self._normalize_theta_features(
                features['theta']
            )
            stats_dict.update(theta_stats)

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

    def _normalize_theta_features(
        self, theta: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, NormalizationStats]]:
        """Normalize theta (parameter) features [N, param_dim].

        Args:
            theta: Operator parameters [N, param_dim] in [0,1] range

        Returns:
            Tuple of (normalized_theta, stats_dict)
        """
        # Get theta categories from group_indices
        theta_categories = self._filter_categories_by_prefix('theta_')

        # Apply normalization based on method
        # For 2D features, no temporal aggregation needed
        if self.config.normalization.method == "per_category":
            return self._normalize_per_category(
                theta, theta, theta_categories
            )
        elif self.config.normalization.method == "global":
            return self._normalize_global(
                theta, theta, theta_categories
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
        _use_mean_std = False  # set True in temporal branch for pca_striped/opq

        # Temporal grouping
        if features.get('temporal') is not None:
            temporal = features['temporal']

            # Aggregate over time for grouping.
            # For pca_striped/opq: cat([mean, std]) so PCA sees both the
            # trajectory's mean level AND its dynamical amplitude.  Mean alone
            # is near rank-1 for physics datasets (all trajectories share similar
            # time-averaged behavior); std captures per-feature oscillation depth,
            # which varies meaningfully with the 9 system parameters.
            # For correlation-based grouping: mean only (legacy behaviour).
            _use_mean_std = (
                self.config.grouping is not None
                and getattr(self.config.grouping, "method", "correlation")
                in ("pca_striped", "opq")
            )
            if _use_mean_std:
                t_mean = temporal.mean(dim=1)          # [N, D_t]
                t_std  = temporal.std(dim=1)           # [N, D_t]
                temporal_agg = torch.cat([t_mean, t_std], dim=1).numpy()  # [N, 2*D_t]
                logger.info(
                    f"Temporal grouping input: cat([mean, std]) → {temporal_agg.shape} "
                    f"(mean alone was near rank-1)"
                )
            else:
                temporal_agg = temporal.mean(dim=1).numpy()  # [N, D_t]

            # Get feature names matching the grouping input dimension.
            # For pca_striped/opq the input is cat([mean, std]) → 2*D_t, so
            # semantic names (length D_t) don't match; use generic indexed names.
            D_agg = temporal_agg.shape[1]
            if (not _use_mean_std
                    and self.feature_metadata is not None
                    and 'temporal' in self.feature_metadata.families):
                temporal_names = self.feature_metadata.families['temporal'].kept_feature_names
            else:
                temporal_names = [f"temporal_{i}" for i in range(D_agg)]

            # Create grouper
            grouper = create_grouper("temporal", config=self.config.grouping)

            # Group features
            result = grouper.group_features(temporal_agg, temporal_names)

            # Store PCA/OPQ rotation transform for wiring into model later
            if result.linear_transform is not None:
                self._temporal_linear_transform = result.linear_transform
                logger.info(
                    f"Temporal linear transform stored "
                    f"(shape: {result.linear_transform.components.shape})"
                )

            # Convert to expected format
            for group_name, group in result.groups.items():
                group_indices[f"temporal_{group_name}"] = group.feature_indices

        # Initial grouping
        if features.get('initial_manual') is not None:
            initial = features['initial_manual'].numpy()  # [N, D_i]

            # Get feature names
            initial_names = [f"initial_{i}" for i in range(initial.shape[1])]

            # Create grouper — use family-specific override if present, else
            # factory falls back to InitialGroupingConfig defaults.
            grouper = create_grouper("initial", config=self.config.grouping.initial)

            # Group features
            result = grouper.group_features(initial, initial_names)

            # Convert to expected format
            for group_name, group in result.groups.items():
                group_indices[f"initial_{group_name}"] = group.feature_indices

        # Theta grouping
        if features.get('theta') is not None:
            theta = features['theta'].numpy()  # [N, param_dim]

            # Get feature names
            theta_names = [f"theta_{i}" for i in range(theta.shape[1])]

            # Create grouper — use family-specific override if present, else
            # factory falls back to ThetaGroupingConfig defaults (single group).
            grouper = create_grouper("theta", config=self.config.grouping.theta)

            # Group features
            result = grouper.group_features(theta, theta_names)

            # Convert to expected format
            for group_name, group in result.groups.items():
                group_indices[f"theta_{group_name}"] = group.feature_indices

        logger.info(f"Feature grouping complete: {len(group_indices)} groups")

        # Populate category metadata if feature_metadata exists
        if self.feature_metadata is not None:
            self._populate_category_metadata(
                group_indices,
                features,
                temporal_groups_are_pc_indices=_use_mean_std,
            )

        return group_indices

    def _populate_category_metadata(
        self,
        group_indices: Dict[str, list],
        features: Dict[str, Optional[torch.Tensor]],
        temporal_groups_are_pc_indices: bool = False,
    ):
        """Populate category metadata in feature_metadata.

        Args:
            group_indices: Dict mapping category_name → feature indices
            features: Dict of features after cleaning
            temporal_groups_are_pc_indices: True when grouping method is pca_striped
                or opq. In that case, temporal group feature_indices are indices into
                the rotated PC space (0..2*D_t-1), not into the cleaned feature list
                (0..D_t-1), so we use generic PC names rather than semantic names.
        """
        from .checkpoint import CategoryMetadata

        # Get cleaned feature names from the temporal family metadata
        if 'temporal' in self.feature_metadata.families:
            temporal_family = self.feature_metadata.families['temporal']
            cleaned_feature_names = temporal_family.kept_feature_names
            kept_indices = temporal_family.kept_feature_indices

            # Build mapping from cleaned index to original index
            cleaned_to_original = {i: orig_idx for i, orig_idx in enumerate(kept_indices)}
        else:
            # Fallback if no temporal family (shouldn't happen after clean_features)
            temporal = features.get('temporal')
            if temporal is not None:
                D_t = temporal.shape[2]
                cleaned_feature_names = self._build_semantic_feature_names(
                    D_t, features
                )
                cleaned_to_original = {i: i for i in range(D_t)}
            else:
                cleaned_feature_names = []
                cleaned_to_original = {}

        # Build CategoryMetadata for each group
        categories = {}
        for category_name, feature_indices in group_indices.items():
            is_temporal = category_name.startswith("temporal_")
            if is_temporal and temporal_groups_are_pc_indices:
                # Temporal group indices are positions in the rotated PC space
                # (0..2*D_t-1), not in the cleaned feature list (0..D_t-1).
                category_feature_names = [f"pc_{idx}" for idx in feature_indices]
                original_indices = list(feature_indices)
            else:
                category_feature_names = [
                    cleaned_feature_names[idx] for idx in feature_indices
                ]
                original_indices = [
                    cleaned_to_original.get(idx, idx) for idx in feature_indices
                ]

            categories[category_name] = CategoryMetadata(
                category_name=category_name,
                feature_indices=feature_indices,
                feature_names=category_feature_names,
                original_indices=original_indices,
                num_features=len(feature_indices),
            )

        # Update feature_metadata
        self.feature_metadata.categories = categories
        self.feature_metadata.grouping_config = (
            self.config.grouping.model_dump() if self.config.grouping else None
        )

        logger.info(f"Category metadata populated: {len(categories)} categories")

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

    def load_theta_inverse(self, checkpoint_path: Union[str, Path]):
        """Load trained ThetaInverseMLP model.

        Args:
            checkpoint_path: Path to trained theta inverse checkpoint

        Example:
            >>> tokenizer.load_theta_inverse("checkpoints/theta_inverse.pt")
        """
        from .inverse_models import ThetaInverseMLP

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading theta inverse from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Create model with saved dimensions
        model = ThetaInverseMLP(
            encoded_dim=checkpoint['encoded_dim'],
            param_dim=checkpoint['param_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Attach to VQTokenizer model
        self.model.theta_inverse = model

        logger.info(
            f"Theta inverse loaded: {checkpoint['encoded_dim']} → {checkpoint['param_dim']} "
            f"(Val MSE: {checkpoint['val_mse']:.6f})"
        )

    def load_initial_inverse(self, checkpoint_path: Union[str, Path]):
        """Load trained InitialInverseCNN model.

        Args:
            checkpoint_path: Path to trained initial inverse checkpoint

        Example:
            >>> tokenizer.load_initial_inverse("checkpoints/initial_inverse.pt")
        """
        from .inverse_models import InitialInverseCNN

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading initial inverse from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Create model with saved dimensions
        model = InitialInverseCNN(
            encoded_dim=checkpoint['encoded_dim'],
            channels=checkpoint['channels'],
            spatial_size=checkpoint['spatial_size'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Attach to VQTokenizer model
        self.model.initial_inverse = model

        logger.info(
            f"Initial inverse loaded: {checkpoint['encoded_dim']} → "
            f"[{checkpoint['channels']}, {checkpoint['spatial_size']}, {checkpoint['spatial_size']}] "
            f"(Val MSE: {checkpoint['val_mse']:.6f})"
        )
