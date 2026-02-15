"""Token synthesis self-play pipeline.

Orchestrates the explore-refine loop:
1. EXPLORE: Generate novel tokens → decode → QBM rollout → retokenize → score → queue
2. REFINE: Pop high-surprisal items → train diffusion model on physically grounded tokens

Single-GPU constraint: models are swapped between CPU and GPU when switching modes.
"""

import gc
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from spinlock.execution.memory import MemoryManager
from spinlock.tokens.schema import TokenSchema
from spinlock.tokens.tokenizer import VQTokenizer

from spinlock.experimental.token_synthesis.config import TokenSynthesisConfig
from spinlock.experimental.token_synthesis.priority_queue import SurprisalPriorityQueue
from spinlock.experimental.token_synthesis.scheduler import Mode, ModeScheduler
from spinlock.experimental.token_synthesis.verification import (
    CounterfactualSurprisal,
    SurprisalComputer,
)

logger = logging.getLogger(__name__)


class SynthesisVerificationPipeline:
    """Main orchestrator for token synthesis self-play.

    Composes:
    - DiscreteD3PM + DenoisingNetwork (diffusion generation/refinement)
    - VQTokenizer (encode/decode between tokens and features)
    - QBMReplayer (physics simulation)
    - SurprisalComputer (verification scoring)
    - SurprisalPriorityQueue (novelty-ranked buffer)
    - ModeScheduler (explore/refine switching)

    All families and feature dimensions are discovered at runtime from the
    VQTokenizer — no hardcoded assumptions about data schema.

    Args:
        config: TokenSynthesisConfig with all checkpoint paths and hyperparameters
    """

    def __init__(self, config: TokenSynthesisConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Models (loaded during initialize)
        self._diffusion = None
        self._denoiser = None
        self._tokenizer = None
        self._replayer = None

        # Schema and feature resolution
        self._schema: Optional[TokenSchema] = None
        self._feature_resolvers: Dict[str, str] = {}
        self._temporal_extractor = None
        self._temporal_feature_mask: Optional[np.ndarray] = None  # 247→152 cleaning mask
        self._temporal_skip_extractors: Set[str] = set()  # Sub-extractors to skip (fully masked)
        self._initial_extractor = None  # IC feature extractor for initial_manual

        # Components (created during initialize)
        self._surprisal_computer = None
        self._queue = None
        self._scheduler = None
        self._optimizer = None

        # Replay buffer for refinement diversity
        self._replay_buffer: list = []

        # Counterfactual surprisal scorer (optional, replaces Jaccard-based scoring)
        self._counterfactual = None

        # Experience buffer for ADAPT phase (raw features from EXPLORE)
        self._experience_buffer: List[Dict[str, np.ndarray]] = []
        # Cached original training features for ADAPT anti-forgetting
        self._original_features_cache: Optional[Dict[str, torch.Tensor]] = None
        # ADAPT cycle counter for anchor cache refresh
        self._adapt_cycle_count: int = 0

        # Explore step counter (for deterministic rollout seeding)
        self._global_explore_step: int = 0

        # Metrics
        self._metrics: Dict[str, List] = defaultdict(list)
        # Per-cycle explore Jaccard means (for learned priority delta computation)
        self._cycle_jaccards: Dict[int, List[float]] = defaultdict(list)

    def initialize(self) -> None:
        """Load all models, discover schema, create pipeline components."""
        logger.info("Initializing synthesis pipeline...")

        self._load_diffusion_model()
        self._load_tokenizer()
        self._load_replayer()
        self._resolve_token_schema()
        self._resolve_feature_extractors()
        self._create_components()

        logger.info(
            f"Pipeline ready: {len(self._schema.keys)} token keys, "
            f"families={self._schema.families}"
        )

    def _load_diffusion_model(self) -> None:
        """Load DiscreteD3PM and DenoisingNetwork from checkpoint."""
        from spinlock.experimental.diffusion.models import (
            DenoisingNetwork,
            DiscreteD3PM,
            DiffusionSchedule,
        )

        checkpoint_path = self.config.checkpoints.diffusion_checkpoint
        logger.info(f"Loading diffusion model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Reconstruct diffusion model from checkpoint config
        config = checkpoint['config']

        # Extract vocab sizes from D3PM transition matrices (authoritative source)
        # transition_matrices.<key> has shape [T, V_eff, V_eff]
        # For uniform: V_eff = vocab_size
        # For absorbing: V_eff = vocab_size + 1 (extra mask token)
        diffusion_state = checkpoint['diffusion_state_dict']
        diffusion_config = config.diffusion if hasattr(config, 'diffusion') else config
        transition_type = getattr(diffusion_config, 'transition_type', 'uniform')

        vocab_sizes = {}
        for param_key, param in diffusion_state.items():
            if param_key.startswith('transition_matrices.'):
                token_key = param_key[len('transition_matrices.'):]
                V_eff = param.shape[1]  # [T, V_eff, V_eff]
                # Absorbing state stores V+1 in transition matrices;
                # subtract 1 to get the base vocab size expected by DiscreteD3PM
                if transition_type == "absorbing":
                    V_eff -= 1
                vocab_sizes[token_key] = V_eff

        # Build category_level_info from discovered keys
        category_level_info = {}
        for key in vocab_sizes:
            parsed = TokenSchema.parse_key(key)
            category_level_info[key] = {
                'family': parsed.family,
                'category': parsed.category,
                'level': parsed.level,
            }

        # Reconstruct schedule
        schedule = DiffusionSchedule(
            num_timesteps=getattr(diffusion_config, 'num_timesteps', 50),
            beta_start=getattr(diffusion_config, 'beta_start', 0.0001),
            beta_end=getattr(diffusion_config, 'beta_end', 0.02),
            schedule_type=getattr(diffusion_config, 'schedule_type', 'cosine'),
        )

        # Create models
        beta_scaling = getattr(diffusion_config, 'beta_scaling', 'uniform')
        self._diffusion = DiscreteD3PM(
            vocab_sizes, schedule, category_level_info,
            transition_type=transition_type,
            beta_scaling=beta_scaling,
        )
        self._diffusion.load_state_dict(checkpoint['diffusion_state_dict'])

        model_config = config.model if hasattr(config, 'model') else config
        self._denoiser = DenoisingNetwork(
            vocab_sizes=vocab_sizes,
            category_level_info=category_level_info,
            hidden_dim=getattr(model_config, 'hidden_dim', 256),
            num_layers=getattr(model_config, 'num_layers', 6),
            num_heads=getattr(model_config, 'num_heads', 8),
            dropout=getattr(model_config, 'dropout', 0.1),
            use_hierarchical_guidance=getattr(model_config, 'use_hierarchical_guidance', True),
            hierarchical_guidance_weight=getattr(model_config, 'hierarchical_guidance_weight', 0.1),
            guidance_mode=getattr(model_config, 'hierarchical_guidance_mode', 'global'),
            transition_type=transition_type,
        )
        self._denoiser.load_state_dict(checkpoint['denoiser_state_dict'])

        logger.info(
            f"Diffusion model loaded: {len(vocab_sizes)} categories, "
            f"T={schedule.num_timesteps}"
        )

    def _load_tokenizer(self) -> None:
        """Load VQTokenizer with inverse models for decode.

        If external inverse model paths are provided, they override the
        built-in inverse heads. Otherwise, the V2 tokenizer's own trained
        inverse heads (from roundtrip loss) are used.
        """
        logger.info(f"Loading VQTokenizer from {self.config.checkpoints.vqvae_checkpoint}")

        kwargs = {}
        if self.config.checkpoints.theta_inverse_path:
            kwargs['theta_inverse_path'] = self.config.checkpoints.theta_inverse_path
        if self.config.checkpoints.initial_inverse_path:
            kwargs['initial_inverse_path'] = self.config.checkpoints.initial_inverse_path

        self._tokenizer = VQTokenizer.from_checkpoint(
            self.config.checkpoints.vqvae_checkpoint,
            **kwargs,
        )

    def _load_replayer(self) -> None:
        """Load QBMReplayer from substrate config."""
        from spinlock.qbm.replayer import QBMReplayer

        logger.info(
            f"Loading QBMReplayer from {self.config.checkpoints.qbm_substrate_config}"
        )
        self._replayer = QBMReplayer.from_config(
            str(self.config.checkpoints.qbm_substrate_config),
            device=str(self.device),
        )

    def _resolve_token_schema(self) -> None:
        """Discover token schema from VQTokenizer via shared TokenSchema class."""
        self._schema = TokenSchema.from_tokenizer(self._tokenizer)

    def _resolve_feature_extractors(self) -> None:
        """For each active family, determine how to produce features from rollout.

        - 'theta': Direct passthrough (decoded theta params)
        - 'initial': decoded u0 as initial_raw
        - 'temporal': Temporal feature extractor on rollout
        - Any other family: Log warning, skip (graceful degradation)
        """
        self._feature_resolvers = {}

        for family in self._schema.families:
            match family:
                case 'theta':
                    self._feature_resolvers[family] = 'passthrough'
                case 'initial':
                    self._feature_resolvers[family] = 'initial'
                    self._initial_extractor = self._load_initial_extractor()
                case 'temporal':
                    self._feature_resolvers[family] = 'temporal'
                    self._temporal_extractor = self._load_temporal_extractor()
                    self._load_temporal_feature_mask()
                    self._compute_skip_extractors()
                    self._compute_reduced_feature_mask()
                case _:
                    logger.warning(
                        f"Unknown family '{family}' — tokens will be "
                        f"generated but not verified"
                    )
                    self._feature_resolvers[family] = 'skip'

        logger.info(f"Feature resolvers: {self._feature_resolvers}")

    def _load_initial_extractor(self):
        """Load initial condition feature extractor for computing initial_manual.

        The InitialHybridEncoder in the tokenizer requires both:
        - initial_raw: [B, C, H, W] raw ICs (from inverse decode)
        - initial_manual: [B, D_manual] hand-crafted IC features

        We extract initial_manual from u0 using the same extractor type that was
        used during dataset generation (statistical features).
        """
        from spinlock.features.initial.ic_feature_extractors import (
            InitialConditionsFeatureExtractor,
        )

        extractor = InitialConditionsFeatureExtractor(device=str(self.device))
        logger.info("Initial condition feature extractor loaded (for initial_manual)")
        return extractor

    def _load_temporal_extractor(self):
        """Load temporal feature extractor matching the tokenizer's training config.

        The orchestrator must produce the same raw feature count that the tokenizer
        was trained on. For QBM data (2-channel wavefunctions), this means quantum
        features must be enabled, matching the dataset generation config.

        The orchestrator's config is validated against the tokenizer's feature_metadata
        to ensure dimension compatibility.
        """
        from spinlock.features.temporal.config import TemporalFeatureConfig
        from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator

        # Default config enables quantum features (QuantumConfig.enabled=True)
        config = TemporalFeatureConfig()
        extractor = TemporalFeatureOrchestrator(device=self.device, config=config)

        # Validate: raw output dimension must match tokenizer's expected input
        if (
            hasattr(self._tokenizer, 'feature_metadata')
            and self._tokenizer.feature_metadata is not None
            and 'temporal' in self._tokenizer.feature_metadata.families
        ):
            expected_raw_dim = self._tokenizer.feature_metadata.families[
                'temporal'
            ].original_feature_count

            # Probe actual dimension with dummy data matching QBM (2-channel)
            actual_dim = extractor.get_actual_per_timestep_dim(num_channels=2)

            if actual_dim != expected_raw_dim:
                raise ValueError(
                    f"Temporal extractor dimension mismatch: "
                    f"extractor produces {actual_dim} features but tokenizer "
                    f"expects {expected_raw_dim} raw features. "
                    f"The feature extraction config must match what was used "
                    f"during tokenizer training."
                )

            logger.info(
                f"Temporal feature extractor validated: {actual_dim} raw features "
                f"→ {self._tokenizer.feature_metadata.families['temporal'].cleaned_feature_count} "
                f"after cleaning"
            )
        else:
            logger.warning(
                "Tokenizer has no feature_metadata — cannot validate temporal "
                "extractor dimensions. Feature cleaning will be skipped."
            )

        return extractor

    def _load_temporal_feature_mask(self) -> None:
        """Load temporal feature cleaning mask from tokenizer's feature_metadata.

        The tokenizer was trained on cleaned features (e.g. 247→152), so the
        synthesis pipeline must apply the same cleaning after extraction.
        This mirrors the pretokenization CLI's _apply_feature_cleaning() logic.
        """
        if (
            hasattr(self._tokenizer, 'feature_metadata')
            and self._tokenizer.feature_metadata is not None
            and 'temporal' in self._tokenizer.feature_metadata.families
        ):
            temporal_meta = self._tokenizer.feature_metadata.families['temporal']
            self._temporal_feature_mask = np.array(temporal_meta.kept_feature_indices)
            logger.info(
                f"Temporal feature mask loaded: {temporal_meta.original_feature_count} "
                f"→ {temporal_meta.cleaned_feature_count} features "
                f"({temporal_meta.original_feature_count - temporal_meta.cleaned_feature_count} removed)"
            )
        else:
            logger.warning(
                "No temporal feature mask available — raw features will be "
                "passed directly to tokenizer (may cause dimension mismatch "
                "if tokenizer was trained with feature cleaning)"
            )

    def _compute_skip_extractors(self) -> None:
        """Determine which sub-extractors produce ONLY masked-out features.

        Analyzes the tokenizer's kept_feature_indices against each sub-extractor's
        index range. Sub-extractors whose entire output range has zero overlap
        with the kept indices are added to ``_temporal_skip_extractors`` and will
        be skipped during extraction, saving compute (e.g., eigendecomposition
        in CrossChannel, quantum state calculations).
        """
        if self._temporal_feature_mask is None or self._temporal_extractor is None:
            return

        kept_set = set(self._temporal_feature_mask.tolist())
        # QBM data has 2 channels — get feature ranges for that
        feature_ranges = self._temporal_extractor.get_feature_ranges(num_channels=2)

        for name, idx_range in feature_ranges.items():
            range_set = set(idx_range)
            if range_set.isdisjoint(kept_set):
                self._temporal_skip_extractors.add(name)
                logger.info(
                    f"Skipping sub-extractor '{name}': all {len(idx_range)} "
                    f"features masked (indices {idx_range.start}-{idx_range.stop - 1})"
                )

        if self._temporal_skip_extractors:
            logger.info(
                f"Feature extraction optimization: skipping {self._temporal_skip_extractors}, "
                f"keeping {set(feature_ranges.keys()) - self._temporal_skip_extractors}"
            )

    def _compute_reduced_feature_mask(self) -> None:
        """Remap cleaning mask indices after skipping sub-extractors.

        When sub-extractors are skipped, the raw output has fewer features
        (e.g., 237 instead of 247 if cross-channel 10D is skipped). The
        cleaning mask indices must be adjusted to account for the removed
        dimensions.

        Indices falling within a skipped range are removed (they were masked
        anyway — that's why the extractor was skipped). Indices after a
        skipped range are shifted down by the range's width.
        """
        if not self._temporal_skip_extractors or self._temporal_feature_mask is None:
            return

        feature_ranges = self._temporal_extractor.get_feature_ranges(num_channels=2)

        # Collect all skipped ranges, sorted by start index
        skipped_ranges = sorted(
            [feature_ranges[name] for name in self._temporal_skip_extractors],
            key=lambda r: r.start,
        )

        # Build a cumulative offset: how many indices are removed before each position
        # For each original index, compute how much to subtract
        original_mask = self._temporal_feature_mask.tolist()
        skipped_indices = set()
        for r in skipped_ranges:
            skipped_indices.update(range(r.start, r.stop))

        remapped = []
        for idx in original_mask:
            if idx in skipped_indices:
                # This index was in a skipped range (should not happen since
                # we only skip extractors whose ranges are disjoint from kept)
                continue
            # Count how many skipped indices are before this index
            offset = sum(1 for si in skipped_indices if si < idx)
            remapped.append(idx - offset)

        old_len = len(self._temporal_feature_mask)
        self._temporal_feature_mask = np.array(remapped, dtype=np.int64)

        logger.info(
            f"Remapped temporal feature mask: {old_len} → {len(self._temporal_feature_mask)} "
            f"indices (offset by {len(skipped_indices)} skipped features)"
        )

    def _apply_temporal_feature_cleaning(
        self, temporal: torch.Tensor,
    ) -> torch.Tensor:
        """Apply feature cleaning mask to raw temporal features.

        Args:
            temporal: [B, T, D_raw] raw temporal features from orchestrator

        Returns:
            [B, T, D_cleaned] cleaned features matching tokenizer input dims
        """
        if self._temporal_feature_mask is not None:
            # Index select along the last dimension: [B, T, D_raw] → [B, T, D_cleaned]
            mask = torch.tensor(
                self._temporal_feature_mask, device=temporal.device, dtype=torch.long,
            )
            temporal = temporal.index_select(-1, mask)
        return temporal

    def _create_components(self) -> None:
        """Create verification, queue, scheduler, and optional counterfactual components."""
        vocab_sizes = self._schema.vocab_sizes

        self._surprisal_computer = SurprisalComputer(
            config=self.config.surprisal,
            vocab_sizes=vocab_sizes,
        )

        self._queue = SurprisalPriorityQueue(
            config=self.config.priority,
            vocab_sizes=vocab_sizes,
        )

        self._scheduler = ModeScheduler(
            config=self.config.scheduler,
            queue=self._queue,
        )

        if self.config.counterfactual is not None:
            self._counterfactual = CounterfactualSurprisal(
                config=self.config.counterfactual,
                d3pm=self._diffusion,
                denoiser=self._denoiser,
                schema=self._schema,
                device=self.device,
            )
            logger.info(
                f"Counterfactual surprisal enabled: "
                f"mask_fraction={self.config.counterfactual.mask_fraction}, "
                f"category_mask_prob={self.config.counterfactual.category_mask_prob}"
            )

    # ─── Mode Switching ───────────────────────────────────────────

    def _enter_mode(self, mode: Mode) -> None:
        """Prepare GPU for the given mode.

        EXPLORE: All models in inference mode on GPU.
        REFINE: Tokenizer offloaded to CPU, denoiser in training mode.
        """
        device = self.device

        match mode:
            case Mode.EXPLORE:
                with MemoryManager.managed_memory(device):
                    self._diffusion.to(device)
                    self._denoiser.to(device)
                    if self._tokenizer.model is not None:
                        self._tokenizer.model.to(device)
                    MemoryManager.optimize_for_inference(self._diffusion)
                    MemoryManager.optimize_for_inference(self._denoiser)
                    if self._tokenizer.model is not None:
                        MemoryManager.optimize_for_inference(self._tokenizer.model)

                logger.info("Entered EXPLORE mode (all models on GPU, inference)")

            case Mode.REFINE:
                with MemoryManager.managed_memory(device):
                    # Offload tokenizer to free GPU memory for gradients
                    if self._tokenizer.model is not None:
                        self._tokenizer.model.to('cpu')

                    self._denoiser.to(device)
                    self._denoiser.train()
                    for p in self._denoiser.parameters():
                        p.requires_grad = True

                    self._optimizer = AdamW(
                        self._denoiser.parameters(),
                        lr=self.config.refinement.learning_rate,
                    )

                logger.info("Entered REFINE mode (denoiser training, tokenizer on CPU)")

    # ─── ADAPT Phase (VQTokenizer Refinement) ──────────────────────

    def _adapt_tokenizer(self) -> Dict[str, float]:
        """Periodic VQTokenizer refinement on accumulated experiences.

        Fine-tunes the VQTokenizer on a mix of original training data
        (anti-forgetting) and newly discovered experiences from EXPLORE.
        Then retokenizes the priority queue, replay buffer, and probe
        buffer with the updated codebooks.

        Returns:
            Dict with ADAPT metrics
        """
        from spinlock.tokens.trainer import VQTokenizerTrainer

        metrics: Dict[str, float] = {}
        n_experiences = len(self._experience_buffer)
        logger.info(f"ADAPT: Starting with {n_experiences} accumulated experiences")

        # Move tokenizer to GPU and re-enable gradients for training
        # (optimize_for_inference sets requires_grad=False during EXPLORE)
        self._tokenizer.model.to(self.device)

        # Snapshot codebook state for drift measurement (after .to(device)
        # so old_state and new_state are on the same device)
        old_state = {
            k: v.clone()
            for k, v in self._tokenizer.model.state_dict().items()
            if 'embedding.weight' in k
        }
        self._tokenizer.model.train()
        for p in self._tokenizer.model.parameters():
            p.requires_grad = True

        # 1. Measure losses BEFORE fine-tuning (using tokenizer's own loss config)
        losses_before = self._compute_tokenizer_losses()
        metrics['adapt/total_loss_before'] = losses_before['total']
        metrics['adapt/recon_loss_before'] = losses_before['reconstruction']
        metrics['adapt/roundtrip_loss_before'] = losses_before['roundtrip']

        # 2. Prepare combined features (discoveries + original data)
        combined = self._prepare_adapt_features()
        if combined is None:
            logger.warning("ADAPT: Failed to prepare features, skipping")
            return metrics

        # 3. Fine-tune VQTokenizer
        adapt_config = self._create_adapt_training_config()
        trainer = VQTokenizerTrainer(
            model=self._tokenizer.model,
            config=adapt_config,
            group_indices=self._tokenizer.group_indices,
            normalization_stats=self._tokenizer.normalization_stats,
            feature_metadata=self._tokenizer.feature_metadata,
        )

        # Override optimizer LR with ADAPT-specific LR
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = self.config.adapt.learning_rate

        history = trainer.train_on_features(
            num_epochs=self.config.adapt.num_epochs,
            **combined,
        )
        metrics['adapt/final_train_loss'] = history['train_losses'][-1] if history['train_losses'] else 0.0

        # 4. Measure losses AFTER fine-tuning
        losses_after = self._compute_tokenizer_losses()
        metrics['adapt/total_loss_after'] = losses_after['total']
        metrics['adapt/recon_loss_after'] = losses_after['reconstruction']
        metrics['adapt/roundtrip_loss_after'] = losses_after['roundtrip']
        metrics['adapt/total_improvement'] = losses_before['total'] - losses_after['total']
        metrics['adapt/recon_improvement'] = losses_before['reconstruction'] - losses_after['reconstruction']
        metrics['adapt/roundtrip_improvement'] = losses_before['roundtrip'] - losses_after['roundtrip']

        # 5. Measure codebook drift
        new_state = {
            k: v.clone()
            for k, v in self._tokenizer.model.state_dict().items()
            if 'embedding.weight' in k
        }
        drift = self._compute_codebook_drift(old_state, new_state)
        metrics['adapt/codebook_drift'] = drift

        # 6. Retokenize cascade
        # Queue MUST be retokenized (used for verification/scoring with the
        # updated codebook). Replay buffer retokenization is optional:
        # replay items represent the denoiser's training history and retokenizing
        # them can destabilize training by shifting labels mid-optimization.
        self._tokenizer.model.eval()
        queue_changed = self._retokenize_queue()
        if self.config.adapt.retokenize_replay:
            replay_changed = self._retokenize_replay_buffer()
        else:
            replay_changed = 0
            logger.info("Skipping replay buffer retokenization (retokenize_replay=False)")
        metrics['adapt/queue_tokens_changed'] = queue_changed
        metrics['adapt/replay_tokens_changed'] = replay_changed
        metrics['adapt/experiences_used'] = n_experiences

        # 7. Clear experience buffer
        self._experience_buffer.clear()

        # 8. Offload tokenizer back to CPU (REFINE mode expects this)
        self._tokenizer.model.to('cpu')

        logger.info(
            f"ADAPT complete: "
            f"total {losses_before['total']:.4f} → {losses_after['total']:.4f} "
            f"(Δ={losses_before['total'] - losses_after['total']:+.4f}), "
            f"roundtrip {losses_before['roundtrip']:.4f} → {losses_after['roundtrip']:.4f} "
            f"(Δ={losses_before['roundtrip'] - losses_after['roundtrip']:+.4f}), "
            f"drift={drift:.6f}, "
            f"retokenized: queue={queue_changed}, replay={replay_changed}"
        )

        return metrics

    def _compute_tokenizer_losses(self) -> Dict[str, float]:
        """Compute VQTokenizer losses on experience buffer using its own loss config.

        Evaluates using the same loss function the tokenizer was trained with,
        including roundtrip consistency loss (the dominant training signal).

        Returns:
            Dict with 'total', 'reconstruction', 'roundtrip' losses
        """
        from spinlock.tokens.losses import VQVAELoss

        if not self._experience_buffer:
            return {'total': 0.0, 'reconstruction': 0.0, 'roundtrip': 0.0}

        # Use the tokenizer's own loss config
        loss_fn = VQVAELoss(self._tokenizer.config.loss)

        # Use up to 64 items for a quick measurement
        sample_size = min(64, len(self._experience_buffer))
        sample = self._experience_buffer[:sample_size]
        stacked = self._stack_feature_dicts(sample)

        self._tokenizer.model.eval()
        with torch.no_grad():
            features_on_device = {k: v.to(self.device) for k, v in stacked.items()}
            outputs = self._tokenizer.model(**features_on_device)

            # Extract category embeddings (same as trainer._extract_category_embeddings)
            category_embeddings = {}
            if 'encodings' in outputs:
                for cat, enc in outputs['encodings'].items():
                    category_embeddings[cat] = enc

            codebooks = {
                key: quantizer.embedding.weight
                for key, quantizer in self._tokenizer.model.quantizers.items()
            }

            losses = loss_fn(
                original=outputs['original_encoded'],
                reconstructed=outputs['reconstructed'],
                vq_loss=outputs['vq_loss'],
                category_embeddings=category_embeddings,
                quantized_vectors=outputs.get('encodings'),
                token_indices=outputs.get('token_indices'),
                codebooks=codebooks,
                latent_vectors=outputs.get('latents'),
                model=self._tokenizer.model,
                tokens=outputs.get('token_indices'),
                decoded=outputs.get('decoded'),
                initial_manual=features_on_device.get('initial_manual'),
            )

        result = {
            'total': losses['total'].item(),
            'reconstruction': losses['reconstruction'].item(),
        }
        if 'roundtrip/total' in losses:
            result['roundtrip'] = losses['roundtrip/total']
        else:
            result['roundtrip'] = 0.0

        return result

    def _prepare_adapt_features(self) -> Optional[Dict[str, torch.Tensor]]:
        """Combine experience buffer with original training data subsample.

        Returns:
            Dict of combined feature tensors ready for trainer, or None
        """
        # Stack discovered features
        discovered = self._stack_feature_dicts(self._experience_buffer)

        # Load original training data for anti-forgetting mixing.
        # original_mix_ratio = fraction of COMBINED batch that should be original
        # e.g., ratio=0.8 with 40 discoveries → 160 original samples (80/20 mix)
        original = None
        n_discovery = len(self._experience_buffer)
        if (
            self.config.adapt.original_features_path is not None
            and self.config.adapt.original_mix_ratio > 0
            and n_discovery > 0
        ):
            # Load and cache original features on first use (HDF5 I/O is slow)
            if self._original_features_cache is None:
                self._original_features_cache = self._load_original_features_subsample(
                    self.config.adapt.anchor_cache_size
                )
                self._adapt_cycle_count = 0

            # Re-randomize cache periodically for broader coverage
            self._adapt_cycle_count += 1
            if (
                self._adapt_cycle_count > 1
                and self._adapt_cycle_count % self.config.adapt.anchor_refresh_interval == 0
            ):
                logger.info("Refreshing anchor cache with new random samples")
                self._original_features_cache = self._load_original_features_subsample(
                    self.config.adapt.anchor_cache_size
                )

            if self._original_features_cache is not None:
                ratio = self.config.adapt.original_mix_ratio
                n_original = max(1, int(n_discovery * ratio / (1 - ratio)))
                n_cached = next(iter(self._original_features_cache.values())).shape[0]
                n_use = min(n_original, n_cached)
                # Random subsample from cache
                perm = torch.randperm(n_cached)[:n_use]
                original = {k: v[perm] for k, v in self._original_features_cache.items()}

        if original is not None:
            # Concatenate along batch dimension
            combined = {}
            for key in discovered:
                if key in original:
                    combined[key] = torch.cat([discovered[key], original[key]], dim=0)
                else:
                    combined[key] = discovered[key]
        else:
            combined = discovered

        logger.info(
            f"ADAPT features: {combined.get('temporal_features', combined.get('theta_features')).shape[0]} total "
            f"({len(self._experience_buffer)} discovered"
            f"{f' + {original[list(original.keys())[0]].shape[0]} original' if original else ''})"
        )

        return combined

    def _load_original_features_subsample(
        self, n_samples: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load a random subsample of original training features from HDF5.

        Supports both flat format (keys like 'temporal_features') and the
        QBM dataset format (nested groups like 'features/temporal/features').
        Applies temporal cleaning mask if available.

        Args:
            n_samples: Number of original training samples to load
        """
        import h5py

        path = self.config.adapt.original_features_path

        # Mapping from ADAPT feature keys to possible HDF5 paths
        KEY_PATHS = {
            'temporal_features': ['temporal_features', 'features/temporal/features'],
            'initial_manual': ['initial_manual', 'features/initial/aggregated/features'],
            'theta_features': ['theta_features', 'parameters/params'],
            'initial_raw': ['initial_raw', 'inputs/fields'],
        }

        try:
            with h5py.File(path, 'r') as f:
                n_total = None
                result = {}
                indices = None

                for key, paths in KEY_PATHS.items():
                    for hdf5_path in paths:
                        if hdf5_path in f:
                            ds = f[hdf5_path]
                            if n_total is None:
                                n_total = ds.shape[0]
                                n_actual = min(n_samples, n_total)
                                indices = np.sort(
                                    np.random.choice(n_total, n_actual, replace=False)
                                )
                            data = ds[indices]

                            # Handle initial_raw: [N, M, C, H, W] → [N, C, H, W]
                            # (take first realization to match tokenizer expectations)
                            if key == 'initial_raw' and data.ndim == 5:
                                data = data[:, 0]  # [N, C, H, W]

                            result[key] = torch.tensor(data, dtype=torch.float32)
                            break

                # Apply temporal cleaning mask (raw 247 → cleaned 152)
                if 'temporal_features' in result and self._temporal_feature_mask is not None:
                    raw_dim = result['temporal_features'].shape[-1]
                    clean_dim = len(self._temporal_feature_mask)
                    if raw_dim != clean_dim:
                        mask_idx = torch.tensor(
                            self._temporal_feature_mask, dtype=torch.long,
                        )
                        result['temporal_features'] = result['temporal_features'].index_select(
                            -1, mask_idx,
                        )
                        logger.info(
                            f"ADAPT: Applied temporal cleaning mask: {raw_dim} → "
                            f"{result['temporal_features'].shape[-1]} features"
                        )

                if result:
                    n_loaded = indices.shape[0] if indices is not None else 0
                    logger.info(
                        f"ADAPT: Loaded {n_loaded} original features from {path} "
                        f"(keys: {list(result.keys())})"
                    )
                return result if result else None

        except Exception as e:
            logger.warning(f"ADAPT: Failed to load original features: {e}")
            return None

    def _stack_feature_dicts(
        self, items: List[Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        """Stack a list of per-item feature dicts into batch tensors.

        Args:
            items: List of {feature_name: np.ndarray} dicts

        Returns:
            Dict of {feature_name: torch.Tensor[N, ...]}
        """
        if not items:
            return {}

        keys = items[0].keys()
        result = {}
        for k in keys:
            arrays = [item[k] for item in items]
            result[k] = torch.tensor(np.stack(arrays), dtype=torch.float32)
        return result

    def _create_adapt_training_config(self):
        """Create a TokenizerConfig variant for short ADAPT fine-tuning.

        Reuses the tokenizer's original config but overrides training
        parameters for a lighter, gentler fine-tuning pass.
        """
        import copy
        config = copy.deepcopy(self._tokenizer.config)
        config.training.num_epochs = self.config.adapt.num_epochs
        config.training.learning_rate = self.config.adapt.learning_rate
        # Disable early stopping and dead code resets for short fine-tuning
        config.training.early_stopping_patience = self.config.adapt.num_epochs + 1
        config.training.dead_code_reset_interval = self.config.adapt.num_epochs + 1
        config.verbose = False
        return config

    def _compute_codebook_drift(
        self,
        old_state: Dict[str, torch.Tensor],
        new_state: Dict[str, torch.Tensor],
    ) -> float:
        """Average L2 distance between old and new codebook embeddings."""
        total_drift = 0.0
        n_codebooks = 0
        for key in old_state:
            if key in new_state:
                drift = (old_state[key] - new_state[key]).norm(dim=1).mean().item()
                total_drift += drift
                n_codebooks += 1
        return total_drift / max(n_codebooks, 1)

    def _retokenize_queue(self) -> int:
        """Retokenize all queue items with updated VQTokenizer codebooks.

        Returns:
            Number of items whose tokens changed
        """
        items = self._queue.get_all_items()
        changed = 0

        for item in items:
            if item.raw_features is None:
                continue

            features = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in item.raw_features.items()
            }
            with torch.no_grad():
                new_tokens = self._tokenizer.tokenize(**features)

            new_token_dict = {k: v.item() for k, v in new_tokens.items()}
            if new_token_dict != item.retokenized_tokens:
                changed += 1
            item.retokenized_tokens = new_token_dict

        self._queue.update_items_in_place(items)
        logger.info(f"ADAPT: Retokenized {len(items)} queue items ({changed} changed)")
        return changed

    def _retokenize_replay_buffer(self) -> int:
        """Retokenize replay buffer items with updated VQTokenizer codebooks.

        Returns:
            Number of replay items whose tokens changed
        """
        changed = 0
        for item in self._replay_buffer:
            if item.raw_features is None:
                continue

            features = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in item.raw_features.items()
            }
            with torch.no_grad():
                new_tokens = self._tokenizer.tokenize(**features)

            new_token_dict = {k: v.item() for k, v in new_tokens.items()}
            if new_token_dict != item.retokenized_tokens:
                changed += 1
            item.retokenized_tokens = new_token_dict

        if self._replay_buffer:
            logger.info(
                f"ADAPT: Retokenized {len(self._replay_buffer)} replay items ({changed} changed)"
            )
        return changed

    # ─── Exploration ──────────────────────────────────────────────

    def _generate_tokens(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Generate novel tokens from noise via diffusion sampling.

        Args:
            batch_size: Number of token sequences to generate

        Returns:
            Dict mapping key → generated token indices [B]
        """
        with torch.no_grad():
            tokens = self._diffusion.sample(
                batch_size=batch_size,
                denoising_network=self._denoiser,
                device=str(self.device),
            )
        return tokens

    def _extract_features_from_rollout(
        self,
        trajectory: torch.Tensor,
        theta: torch.Tensor,
        u0: torch.Tensor,
        ics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract features for each active family from a physical rollout.

        Resolution determined at init time by _resolve_feature_extractors().

        Args:
            trajectory: [B, M, T, C, H, W] raw rollout from QBM
            theta: [B, param_dim] decoded parameters
            u0: [B, C, H, W] decoded initial conditions
            ics: [B, M, C, H, W] per-realization initial conditions from QBM
                 Used to extract initial_manual features matching the dataset
                 generation pipeline's shape normalization (M*C channels).

        Returns:
            Dict matching VQTokenizer.tokenize() kwargs
        """
        features: Dict[str, torch.Tensor] = {}

        for family, strategy in self._feature_resolvers.items():
            match strategy:
                case 'passthrough':
                    features['theta_features'] = theta
                case 'initial':
                    features['initial_raw'] = u0
                    # Extract initial_manual features (required by InitialHybridEncoder)
                    # During dataset generation, ICs [B, M, C, H, W] were shape-normalized
                    # to [B, M*C, H, W] (M and C flattened), producing features for all
                    # realization-channel combinations. We replicate that here.
                    if self._initial_extractor is not None and ics is not None:
                        B, M, C, H, W = ics.shape
                        ics_flattened = ics.reshape(B, M * C, H, W)  # [B, M*C, H, W]
                        with torch.no_grad():
                            initial_manual = self._initial_extractor(ics_flattened)
                        features['initial_manual'] = initial_manual
                case 'temporal':
                    if self._temporal_extractor is not None:
                        temporal = self._temporal_extractor.extract_per_timestep(
                            trajectory,
                            skip_extractors=self._temporal_skip_extractors or None,
                        )
                        # Apply feature cleaning (raw → cleaned)
                        temporal = self._apply_temporal_feature_cleaning(temporal)
                        features['temporal_features'] = temporal
                case 'skip':
                    pass  # Tokens from this family always mismatch

        return features

    def _inpaint_tokens(
        self,
        observed_dict: Dict[str, torch.BoolTensor],
        x_0_dict: Dict[str, torch.Tensor],
        start_step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """D3PM inpainting: re-generate masked positions conditioned on observed context.

        Shared by _correct_tokens() and _frontier_generate_tokens().

        Args:
            observed_dict: {key: [B] bool} — True = keep fixed, False = re-generate
            x_0_dict: {key: [B] int} — clean token values for RePaint conditioning
            start_step: Denoising starts from this step (None = full T)

        Returns:
            Dict mapping key → inpainted token indices [B]
        """
        batch_size = next(iter(x_0_dict.values())).shape[0]
        with torch.no_grad():
            return self._diffusion.sample(
                batch_size=batch_size,
                observed_dict=observed_dict,
                x_0_dict=x_0_dict,
                denoising_network=self._denoiser,
                device=str(self.device),
                start_step=start_step,
            )

    def _correct_tokens(
        self,
        generated: Dict[str, torch.Tensor],
        retokenized: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], int]:
        """Re-generate mismatched tokens via D3PM RePaint inpainting.

        Keys where dreamed == retokenized are marked as observed (kept fixed).
        Keys where they differ are re-generated by the denoiser conditioned
        on the physically-grounded matching context.

        The correction timestep is adaptive: T_correction = T × mismatch_rate × noise_scale.
        Higher mismatch rates get more noise (more freedom to explore), lower rates
        get less (conservative refinement near the retokenized values).

        Args:
            generated: Dict mapping key → token indices [B] (from diffusion sampling)
            retokenized: Dict mapping key → token indices [B] (from QBM roundtrip)

        Returns:
            Tuple of (corrected token dict, start_step used)
        """
        batch_size = next(iter(generated.values())).shape[0]

        # Per-key, per-sample: True where dreamed matches retokenized
        observed_dict = {}
        x_0_dict = {}
        n_mismatched = 0
        n_total = 0
        for key in generated:
            match = (generated[key] == retokenized[key])  # [B] bool
            observed_dict[key] = match
            x_0_dict[key] = retokenized[key]
            n_mismatched += (~match).sum().item()
            n_total += match.numel()

        if n_mismatched == 0:
            return generated, 0  # Perfect match, nothing to correct

        # Adaptive correction timestep: proportional to mismatch rate and model T
        T = self._diffusion.schedule.num_timesteps
        mismatch_rate = n_mismatched / max(n_total, 1)
        noise_scale = self.config.resampling.noise_scale
        start_step = max(1, round(T * mismatch_rate * noise_scale))
        if self.config.resampling.max_correction_steps is not None:
            start_step = min(start_step, self.config.resampling.max_correction_steps)
        start_step = min(start_step, T)  # Cannot exceed model's total timesteps

        logger.debug(
            f"  Correction: {n_mismatched}/{n_total} mismatched "
            f"({mismatch_rate:.1%}), start_step={start_step}/{T}"
        )

        corrected = self._inpaint_tokens(observed_dict, x_0_dict, start_step=start_step)
        return corrected, start_step

    @staticmethod
    def _select_per_sample(
        original: Dict[str, torch.Tensor],
        corrected: Dict[str, torch.Tensor],
        use_corrected: torch.BoolTensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-sample selection between two token/feature dicts.

        For each sample in the batch, picks from ``original`` or ``corrected``
        based on the boolean mask. Works for any per-key tensor shape [B, ...].

        Args:
            original: Dict mapping key → tensor [B, ...]
            corrected: Dict mapping key → tensor [B, ...]
            use_corrected: [B] bool mask, True = use corrected sample
        """
        result = {}
        for key in original:
            a, b = original[key], corrected[key]
            mask = use_corrected.to(a.device)
            for _ in range(a.dim() - 1):
                mask = mask.unsqueeze(-1)
            result[key] = torch.where(mask.expand_as(a), b, a)
        return result

    def _process_generated_tokens(
        self, generated_tokens: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Common pipeline tail: decode → QBM → retokenize → correct → score → queue.

        Shared by _explore_step() (unconditional) and _frontier_explore_step()
        (frontier-conditioned). Handles resampling, surprisal computation,
        ADAPT experience collection, and queue insertion.

        Args:
            generated_tokens: Dict mapping key → token indices [B]

        Returns:
            Dict of metrics for this step
        """
        batch_size = next(iter(generated_tokens.values())).shape[0]

        # 1. Decode tokens to (theta, u0)
        theta, u0 = self._tokenizer.decode(generated_tokens)

        # 2. Run QBM simulation (1 rollout for physical grounding)
        # Seed rollout for deterministic ICs: same theta → same dynamics → stable retokenization.
        rollout_seed = self.config.seed + self._global_explore_step * 1000
        theta_np = theta.cpu().numpy()
        trajectory, ics = self._replayer.rollout_batch(
            theta_np,
            num_realizations=self.config.rollout.num_realizations,
            num_timesteps=self.config.rollout.num_timesteps,
            seed=rollout_seed,
            return_ics=True,
        )  # trajectory: [B, M, T, C, H, W], ics: [B, M, C, H, W]

        # 3. Extract features per active family
        features = self._extract_features_from_rollout(trajectory, theta, u0, ics=ics)

        # 4. Retokenize (single pass — always needed for queue storage)
        with torch.no_grad():
            retokenized = self._tokenizer.tokenize(**features)

        # 4b. Physically-guided resampling with per-sample keep-best gate.
        correction_metrics = {}
        if self.config.resampling is not None:
            # Pre-correction Jaccard per sample [B]
            pre_jaccard = self._surprisal_computer.compute_jaccard(
                generated_tokens, retokenized,
            )

            # Save pre-correction state
            tokens_pre = generated_tokens
            retokenized_pre = retokenized
            features_pre = features
            theta_pre = theta

            # Correct: inpaint mismatched keys conditioned on matching context
            corrected_tokens, start_step = self._correct_tokens(generated_tokens, retokenized)

            # Re-decode corrected tokens → second QBM roundtrip
            # Same seed as initial rollout → same ICs → fair Jaccard comparison
            theta_c, u0_c = self._tokenizer.decode(corrected_tokens)
            theta_c_np = theta_c.cpu().numpy()
            trajectory_c, ics_c = self._replayer.rollout_batch(
                theta_c_np,
                num_realizations=self.config.rollout.num_realizations,
                num_timesteps=self.config.rollout.num_timesteps,
                seed=rollout_seed,
                return_ics=True,
            )
            features_post = self._extract_features_from_rollout(
                trajectory_c, theta_c, u0_c, ics=ics_c,
            )
            with torch.no_grad():
                retokenized_post = self._tokenizer.tokenize(**features_post)

            # Post-correction Jaccard per sample [B]
            post_jaccard = self._surprisal_computer.compute_jaccard(
                corrected_tokens, retokenized_post,
            )

            # Per-sample keep-best gate: only accept corrections that improve Jaccard
            use_corrected = post_jaccard >= pre_jaccard  # [B]
            n_accepted = use_corrected.sum().item()

            generated_tokens = self._select_per_sample(
                tokens_pre, corrected_tokens, use_corrected,
            )
            retokenized = self._select_per_sample(
                retokenized_pre, retokenized_post, use_corrected,
            )
            features = self._select_per_sample(
                features_pre, features_post, use_corrected,
            )
            theta = torch.where(
                use_corrected.to(theta_pre.device).unsqueeze(-1),
                theta_c, theta_pre,
            )
            theta_np = theta.cpu().numpy()

            T = self._diffusion.schedule.num_timesteps
            correction_metrics = {
                'correction_jaccard_before': pre_jaccard.mean().item(),
                'correction_jaccard_after': post_jaccard.mean().item(),
                'correction_accepted': n_accepted,
                'correction_rejected': batch_size - n_accepted,
                'correction_start_step': start_step,
            }
            logger.info(
                f"  Correction: jaccard {pre_jaccard.mean().item():.3f} → "
                f"{post_jaccard.mean().item():.3f}, "
                f"accepted {n_accepted}/{batch_size}, "
                f"T_corr={start_step}/{T}"
            )

        # 5. Compute surprisal (counterfactual or legacy Jaccard-based)
        if self._counterfactual is not None:
            surprisal = self._counterfactual.compute(retokenized)
            variance = torch.zeros(batch_size)
        elif self.config.surprisal.verification_samples > 1:
            # Legacy: multi-QBM variance estimation
            retokenized_samples = [retokenized]
            for k in range(1, self.config.surprisal.verification_samples):
                traj_k, ics_k = self._replayer.rollout_batch(
                    theta_np,
                    num_realizations=self.config.rollout.num_realizations,
                    num_timesteps=self.config.rollout.num_timesteps,
                    seed=self.config.seed + k + 1,
                    return_ics=True,
                )
                feats_k = self._extract_features_from_rollout(traj_k, theta, u0, ics=ics_k)
                with torch.no_grad():
                    retok_k = self._tokenizer.tokenize(**feats_k)
                retokenized_samples.append(retok_k)

            surprisal, variance = self._surprisal_computer.verify_with_multiple_samples(
                generated_tokens, retokenized_samples,
            )
        else:
            surprisal = self._surprisal_computer.compute_surprisal(
                generated_tokens, retokenized,
            )
            variance = torch.zeros_like(surprisal)

        # 6. Compute Jaccard for monitoring (always, regardless of scoring method)
        jaccard = self._surprisal_computer.compute_jaccard(generated_tokens, retokenized)

        # 6b. Accumulate raw features for ADAPT phase
        if self.config.adapt.enabled:
            for i in range(batch_size):
                if len(self._experience_buffer) < self.config.adapt.max_experiences:
                    item_features = {
                        k: v[i].cpu().numpy() for k, v in features.items()
                    }
                    self._experience_buffer.append(item_features)

        # 7. Queue items (with raw_features for ADAPT retokenization)
        raw_features_list = None
        if self.config.adapt.enabled:
            raw_features_list = [
                {k: v[i].cpu().numpy() for k, v in features.items()}
                for i in range(batch_size)
            ]
        num_queued = self._queue.push_batch(
            generated_tokens, retokenized, surprisal, theta,
            raw_features_list=raw_features_list,
        )

        metrics = {
            'avg_surprisal': surprisal.mean().item(),
            'avg_jaccard': jaccard.mean().item(),
            'avg_variance': variance.mean().item(),
            'num_queued': num_queued,
            'queue_size': self._queue.size,
            'queue_fill': self._queue.fill_fraction,
            **correction_metrics,
        }
        if self._counterfactual is not None:
            metrics['scoring_method'] = 'counterfactual'

        logger.info(
            f"  Explore: surprisal={metrics['avg_surprisal']:.3f}, "
            f"jaccard={metrics['avg_jaccard']:.3f}, "
            f"queued={num_queued}/{batch_size}, "
            f"queue={self._queue.size}/{self.config.priority.queue_capacity}"
        )

        self._global_explore_step += 1
        return metrics

    def _explore_step(self) -> Dict[str, float]:
        """Unconditional exploration: generate tokens from noise, then process.

        Returns:
            Dict of metrics for this step
        """
        batch_size = self.config.generation.batch_size
        generated_tokens = self._generate_tokens(batch_size)
        return self._process_generated_tokens(generated_tokens)

    def _frontier_generate_tokens(
        self, batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Generate tokens via frontier-conditioned inpainting.

        Takes high-surprisal items from the priority queue, masks a subset
        of their categories, and uses D3PM inpainting to re-generate those
        categories conditioned on the physically-grounded context.

        Args:
            batch_size: Number of token sequences to generate

        Returns:
            Dict mapping key → generated token indices [B]
        """
        cfg = self.config.frontier

        # 1. Sample seeds from queue (non-destructive)
        seeds = self._queue.sample_batch(batch_size)
        if not seeds:
            # Fallback to unconditional if queue is empty
            return self._generate_tokens(batch_size)

        # Pad with repeated seeds if queue has fewer items than batch_size
        while len(seeds) < batch_size:
            seeds.append(seeds[random.randint(0, len(seeds) - 1)])

        # 2. Batch retokenized tokens into Dict[str, Tensor[B]]
        all_keys = sorted(seeds[0].retokenized_tokens.keys())
        x_0_dict = {}
        for key in all_keys:
            x_0_dict[key] = torch.tensor(
                [s.retokenized_tokens[key] for s in seeds],
                dtype=torch.long, device=self.device,
            )

        # 3. Build category-level masks
        # Discover unique categories from schema
        categories = self._schema.unique_categories()  # List[(family, category)]
        cat_to_keys = {
            (fam, cat): self._schema.keys_for_category(fam, cat)
            for fam, cat in categories
        }

        # Per-sample mask building
        observed_dict = {key: torch.ones(batch_size, dtype=torch.bool, device=self.device)
                         for key in all_keys}

        for i, seed in enumerate(seeds):
            cats_to_mask = set()

            # 3a. Mismatch-based masking: mask categories where generated != retokenized
            if cfg.use_mismatch_mask:
                for fam, cat in categories:
                    cat_keys = cat_to_keys[(fam, cat)]
                    for key in cat_keys:
                        gen_val = seed.generated_tokens.get(key)
                        retok_val = seed.retokenized_tokens.get(key)
                        if gen_val is not None and retok_val is not None and gen_val != retok_val:
                            cats_to_mask.add((fam, cat))
                            break  # One mismatch in category is enough

            # 3b. Add random extra categories
            remaining = [c for c in categories if c not in cats_to_mask]
            n_extra = min(cfg.extra_categories_to_mask, len(remaining))
            if n_extra > 0:
                cats_to_mask.update(random.sample(remaining, n_extra))

            # 3c. Cap total masked categories
            if len(cats_to_mask) > cfg.max_categories_to_mask:
                cats_to_mask = set(random.sample(sorted(cats_to_mask), cfg.max_categories_to_mask))

            # Apply mask: set observed=False for all keys in masked categories
            for fam, cat in cats_to_mask:
                for key in cat_to_keys[(fam, cat)]:
                    if key in observed_dict:
                        observed_dict[key][i] = False

        # 4. Inpaint: re-generate masked positions from full T
        generated = self._inpaint_tokens(observed_dict, x_0_dict, start_step=None)

        n_masked_keys = sum(
            (~v).any().item() for v in observed_dict.values()
        )
        logger.debug(
            f"  Frontier: {n_masked_keys}/{len(all_keys)} keys have masked positions"
        )

        return generated

    def _frontier_explore_step(self) -> Dict[str, float]:
        """Frontier-conditioned exploration: inpaint queue seeds, then process.

        Returns:
            Dict of metrics for this step
        """
        batch_size = self.config.generation.batch_size
        generated_tokens = self._frontier_generate_tokens(batch_size)
        return self._process_generated_tokens(generated_tokens)

    # ─── Refinement ───────────────────────────────────────────────

    def _compute_refinement_loss(
        self,
        predicted_logits: Dict[str, torch.Tensor],
        target_tokens: Dict[str, torch.Tensor],
        target_mask: Dict[str, torch.BoolTensor],
    ) -> torch.Tensor:
        """Per-category-level cross-entropy on target positions.

        Replicates the DiffusionTrainer._compute_loss pattern.

        Args:
            predicted_logits: Dict mapping key → logits [B, V]
            target_tokens: Dict mapping key → true tokens [B]
            target_mask: Dict mapping key → mask [B]

        Returns:
            Scalar loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_count = 0

        for key in predicted_logits.keys():
            logits = predicted_logits[key]     # [B, V]
            targets = target_tokens[key]       # [B]
            mask = target_mask[key]            # [B]

            loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
            masked_loss = (loss * mask.float()).sum()
            count = mask.sum().item()

            total_loss = total_loss + masked_loss
            total_count += count

        if total_count > 0:
            total_loss = total_loss / total_count

        return total_loss

    def _refine_epoch(self) -> Dict[str, float]:
        """One refinement epoch: pop queue (or sample replay buffer) → train denoiser.

        Training data sources (in priority order):
        1. Fresh items from the priority queue (highest-surprisal first)
        2. Random samples from the replay buffer (when queue is empty)

        Returns:
            Dict of refinement metrics
        """
        # Try queue first; fall back to replay buffer
        items = self._queue.pop_batch(self.config.refinement.batch_size)
        source = 'queue'

        if not items and self._replay_buffer:
            # Queue exhausted — sample from replay buffer for continued training
            sample_size = min(
                self.config.refinement.batch_size,
                len(self._replay_buffer),
            )
            indices = np.random.choice(
                len(self._replay_buffer), size=sample_size, replace=False,
            )
            items = [self._replay_buffer[i] for i in indices]
            source = 'replay'

        if not items:
            return {'refine_loss': 0.0, 'refine_samples': 0}

        # Build training batch from retokenized tokens (= physically grounded truth)
        # Clip token values to D3PM's vocabulary range per category.
        # Retokenized tokens may exceed D3PM's vocab_size (trained on max+1 from
        # pretokenized data) when the VQ-VAE assigns novel codebook entries.
        # These out-of-range tokens represent true novelty discoveries.
        d3pm_vocab = self._diffusion.vocab_sizes  # Dict[str, int]
        keys = list(items[0].retokenized_tokens.keys())
        tokens = {}
        num_clipped = 0
        for k in keys:
            raw = torch.tensor(
                [item.retokenized_tokens[k] for item in items],
                device=self.device,
                dtype=torch.long,
            )
            max_valid = d3pm_vocab[k] - 1
            clip_mask = raw > max_valid
            num_clipped += clip_mask.sum().item()
            tokens[k] = raw.clamp(max=max_valid)

        if num_clipped > 0:
            logger.debug(
                f"Clipped {num_clipped} out-of-range token values to D3PM vocab bounds"
            )

        # All positions are targets (unconditional training)
        target_mask = {
            k: torch.ones(len(items), device=self.device, dtype=torch.bool)
            for k in keys
        }

        # Standard diffusion training step
        t = torch.randint(
            0, self._diffusion.schedule.num_timesteps,
            (len(items),), device=self.device,
        )
        # Forward process with per-sample timesteps
        noisy_tokens, _ = self._diffusion.forward_process(
            tokens, t, mask_dict=target_mask,
        )

        # Predict clean tokens
        predicted_logits = self._denoiser(noisy_tokens, t)

        # Compute loss
        loss = self._compute_refinement_loss(predicted_logits, tokens, target_mask)

        # Compute per-item losses for learned priority training (detached)
        # Sum cross-entropy across categories per item, then average
        if source == 'queue' and self._queue._head is not None:
            with torch.no_grad():
                per_item = torch.zeros(len(items), device=self.device)
                for key in predicted_logits.keys():
                    logits_k = predicted_logits[key]       # [B, V]
                    targets_k = tokens[key]                # [B]
                    mask_k = target_mask[key]               # [B]
                    ce = F.cross_entropy(logits_k, targets_k, reduction='none')
                    per_item += ce * mask_k.float()
                per_item /= max(len(predicted_logits), 1)
                per_item_losses = per_item.cpu().tolist()
            self._queue.record_refined_features(items, per_item_losses)

        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._denoiser.parameters(),
            self.config.refinement.gradient_clip_norm,
        )
        self._optimizer.step()

        # Add fresh queue items to replay buffer (not replay samples — avoid duplicates)
        if source == 'queue':
            self._replay_buffer.extend(items)
            if len(self._replay_buffer) > self.config.refinement.replay_buffer_size:
                self._replay_buffer = self._replay_buffer[
                    -self.config.refinement.replay_buffer_size:
                ]

        metrics = {
            'refine_loss': loss.item(),
            'refine_samples': len(items),
            'refine_source': source,
            'replay_buffer_size': len(self._replay_buffer),
        }

        logger.info(
            f"  Refine [{source}]: loss={loss.item():.4f}, "
            f"samples={len(items)}, "
            f"replay_buffer={len(self._replay_buffer)}"
        )

        return metrics

    # ─── Learned Priority ────────────────────────────────────────

    def _compute_cycle_improvement(self, cycle: int) -> Optional[float]:
        """Compute Jaccard improvement delta for this cycle vs previous.

        Compares the mean Jaccard similarity of explore steps in cycle N
        against cycle N-1. A positive delta means the diffusion model is
        generating tokens that better match physical grounding.

        Uses ``_cycle_jaccards`` (populated during explore) rather than
        indexing into the ragged ``_metrics`` dict-of-lists.

        Args:
            cycle: Current cycle number

        Returns:
            Improvement delta, or None if insufficient data
        """
        if cycle == 0:
            return None

        prev_jaccards = self._cycle_jaccards.get(cycle - 1, [])
        curr_jaccards = self._cycle_jaccards.get(cycle, [])

        if not prev_jaccards or not curr_jaccards:
            return None

        prev_mean = sum(prev_jaccards) / len(prev_jaccards)
        curr_mean = sum(curr_jaccards) / len(curr_jaccards)
        return curr_mean - prev_mean

    def _train_and_log_priority_head(self, cycle: int) -> None:
        """Train learned priority head and log weights at cycle boundary.

        Called at the end of each cycle in the main loop. Computes the
        improvement delta, trains the head, and logs the learned weights.
        """
        delta = self._compute_cycle_improvement(cycle)
        if delta is not None:
            loss = self._queue.train_priority_head(delta, cycle)

            # Log delta
            self._metrics['improvement_delta'].append(delta)
            self._metrics['improvement_delta_cycle'].append(cycle)

            if loss is not None:
                logger.info(
                    f"  Priority head trained: delta={delta:+.4f}, "
                    f"loss={loss:.6f}"
                )

        # Log learned weights (even during warmup for comparison)
        weights = self._queue.get_learned_weights()
        if weights is not None:
            w_alpha, w_delta, w_epsilon, w_gamma, w_bias = weights
            self._metrics['learned_alpha'].append(w_alpha)
            self._metrics['learned_delta'].append(w_delta)
            self._metrics['learned_epsilon'].append(w_epsilon)
            self._metrics['learned_gamma'].append(w_gamma)
            self._metrics['learned_bias'].append(w_bias)
            self._metrics['learned_weights_cycle'].append(cycle)

            active = "LEARNED" if self._queue._head_active else "FIXED"
            logger.info(
                f"  Priority weights [{active}]: "
                f"\u03b1(sur)={w_alpha:.3f}, "
                f"\u03b4(rt)={w_delta:.3f}, \u03b5(sg)={w_epsilon:.3f}, "
                f"\u03b3(rar)={w_gamma:.3f}, bias={w_bias:.4f}"
            )

    # ─── Logging & Checkpointing ──────────────────────────────────

    def _log_metrics(
        self,
        cycle: int,
        mode: Mode,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Record metrics for analysis."""
        entry = {
            'cycle': cycle,
            'mode': mode.value,
            'step': step,
            'timestamp': time.time(),
            **metrics,
        }
        for key, value in entry.items():
            self._metrics[key].append(value)

    def _save_cycle_checkpoint(self, cycle: int) -> None:
        """Save pipeline state at the end of a cycle."""
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save denoiser and priority head (the models that change)
        save_dict = {
            'cycle': cycle,
            'denoiser_state_dict': self._denoiser.state_dict(),
            'queue_size': self._queue.size,
            'metrics': dict(self._metrics),
            'experience_buffer_size': len(self._experience_buffer),
        }
        if self._queue._head is not None:
            save_dict['priority_head_state_dict'] = self._queue._head.state_dict()
            save_dict['priority_head_active'] = self._queue._head_active

        path = checkpoint_dir / f"denoiser_cycle_{cycle}.pt"
        torch.save(save_dict, path)

        # Save metrics as JSON for easy analysis
        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(dict(self._metrics), f, indent=2, default=str)

        logger.info(f"Cycle {cycle} checkpoint saved to {path}")

    # ─── Main Loop ────────────────────────────────────────────────

    # ─── Frontier Scheduling ────────────────────────────────────

    def _frontier_steps_this_cycle(self, cycle: int) -> int:
        """Linearly interpolate frontier ratio across cycles.

        Returns the number of explore steps that should be frontier-conditioned
        for the given cycle. At cycle 0 this is ratio_start * explore_steps;
        at the final cycle it's ratio_end * explore_steps.
        """
        cfg = self.config.frontier
        if cfg is None or not cfg.enabled:
            return 0
        progress = cycle / max(self.config.scheduler.max_cycles - 1, 1)
        ratio = cfg.ratio_start + (cfg.ratio_end - cfg.ratio_start) * progress
        return round(ratio * self.config.scheduler.explore_steps)

    def _can_do_frontier(self) -> bool:
        """Check if frontier exploration is possible right now."""
        cfg = self.config.frontier
        return (cfg is not None and cfg.enabled
                and self._queue.size >= cfg.min_queue_for_frontier)

    def run(self) -> Dict[str, List]:
        """Execute the full explore-refine self-play loop.

        Returns:
            Dict of metrics lists, keyed by metric name
        """
        torch.manual_seed(self.config.seed)
        self.initialize()

        learned_info = ""
        if self.config.priority.learned:
            learned_info = (
                f", learned_priority=ON (warmup={self.config.priority.warmup_cycles}, "
                f"lr={self.config.priority.priority_lr})"
            )
        else:
            learned_info = ", learned_priority=OFF"

        logger.info(
            f"Starting self-play: {self.config.scheduler.max_cycles} cycles, "
            f"explore={self.config.scheduler.explore_steps} steps, "
            f"refine={self.config.scheduler.refine_epochs} epochs"
            f"{learned_info}"
        )

        while not self._scheduler.is_complete:
            mode = self._scheduler.current_mode
            cycle = self._scheduler.current_cycle

            logger.info(
                f"=== Cycle {cycle} | Mode: {mode.value.upper()} | "
                f"Queue: {self._queue.size}/{self.config.priority.queue_capacity} ==="
            )

            self._enter_mode(mode)

            match mode:
                case Mode.EXPLORE:
                    step = 0
                    n_explore = self.config.scheduler.explore_steps
                    n_frontier = self._frontier_steps_this_cycle(cycle)
                    # Unconditional steps first, then frontier steps.
                    # Unconditional builds queue diversity; frontier refines it.
                    while self._scheduler.get_explore_steps_remaining() > 0:
                        if step >= (n_explore - n_frontier) and self._can_do_frontier():
                            metrics = self._frontier_explore_step()
                            metrics['explore_mode'] = 'frontier'
                        else:
                            metrics = self._explore_step()
                            metrics['explore_mode'] = 'unconditional'
                        self._log_metrics(cycle, mode, step, metrics)
                        self._cycle_jaccards[cycle].append(metrics['avg_jaccard'])
                        self._scheduler.step()
                        step += 1
                        if self._scheduler.should_switch():
                            break

                case Mode.REFINE:
                    if self._queue.size >= self.config.priority.min_queue_for_refinement:
                        step = 0
                        while self._scheduler.get_refine_epochs_remaining() > 0:
                            metrics = self._refine_epoch()
                            self._log_metrics(cycle, mode, step, metrics)
                            self._scheduler.step()
                            step += 1
                    else:
                        logger.info(
                            f"Queue too small for refinement: "
                            f"{self._queue.size} < {self.config.priority.min_queue_for_refinement}"
                        )

            # Train learned priority head at cycle boundary (after explore+refine)
            if mode == Mode.REFINE:
                self._train_and_log_priority_head(cycle)

                # === ADAPT Phase (periodic VQTokenizer refinement) ===
                if (
                    self.config.adapt.enabled
                    and (cycle + 1) % self.config.adapt.frequency == 0
                    and len(self._experience_buffer) >= self.config.adapt.min_experiences
                ):
                    logger.info(f"=== ADAPT Phase (cycle {cycle}) ===")
                    adapt_metrics = self._adapt_tokenizer()
                    for k, v in adapt_metrics.items():
                        self._metrics[k].append(v)
                    self._metrics['adapt_cycle'].append(cycle)

            self._scheduler.advance()
            self._save_cycle_checkpoint(cycle)

        # Log final learned weights
        weights = self._queue.get_learned_weights()
        if weights is not None:
            w_alpha, w_delta, w_epsilon, w_gamma, w_bias = weights
            logger.info(
                f"Final learned priority weights: "
                f"\u03b1(sur)={w_alpha:.3f}, "
                f"\u03b4(rt)={w_delta:.3f}, \u03b5(sg)={w_epsilon:.3f}, "
                f"\u03b3(rar)={w_gamma:.3f}, bias={w_bias:.4f}"
            )

        logger.info("Self-play complete!")
        logger.info(
            f"Final queue: {self._queue.size} items, "
            f"replay buffer: {len(self._replay_buffer)} items"
        )

        return dict(self._metrics)
