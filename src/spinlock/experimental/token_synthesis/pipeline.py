"""Token synthesis self-play pipeline.

Orchestrates the explore-refine loop:
1. EXPLORE: Generate novel tokens → decode → QBM rollout → retokenize → score → queue
2. REFINE: Pop high-surprisal items → train diffusion model on physically grounded tokens

Single-GPU constraint: models are swapped between CPU and GPU when switching modes.
"""

import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from spinlock.experimental.token_synthesis.verification import SurprisalComputer

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

        # Components (created during initialize)
        self._surprisal_computer = None
        self._queue = None
        self._scheduler = None
        self._optimizer = None

        # Replay buffer for refinement diversity
        self._replay_buffer: list = []

        # Metrics
        self._metrics: Dict[str, List] = defaultdict(list)

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

        # Extract vocab sizes from denoiser state dict
        # The denoiser stores per-category embedding layers: "embeddings.<key>.weight"
        denoiser_state = checkpoint['denoiser_state_dict']
        vocab_sizes = {}
        for param_key, param in denoiser_state.items():
            if param_key.startswith('embeddings.') and param_key.endswith('.weight'):
                # "embeddings.temporal_group_1_L0.weight" → "temporal_group_1_L0"
                token_key = param_key[len('embeddings.'):-len('.weight')]
                vocab_sizes[token_key] = param.shape[0]

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
        diffusion_config = config.diffusion if hasattr(config, 'diffusion') else config
        schedule = DiffusionSchedule(
            num_timesteps=getattr(diffusion_config, 'num_timesteps', 50),
            beta_start=getattr(diffusion_config, 'beta_start', 0.0001),
            beta_end=getattr(diffusion_config, 'beta_end', 0.02),
            schedule_type=getattr(diffusion_config, 'schedule_type', 'cosine'),
        )

        # Create models
        self._diffusion = DiscreteD3PM(vocab_sizes, schedule, category_level_info)
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
        )
        self._denoiser.load_state_dict(checkpoint['denoiser_state_dict'])

        logger.info(
            f"Diffusion model loaded: {len(vocab_sizes)} categories, "
            f"T={schedule.num_timesteps}"
        )

    def _load_tokenizer(self) -> None:
        """Load VQTokenizer with inverse models for decode."""
        logger.info(f"Loading VQTokenizer from {self.config.checkpoints.vqvae_checkpoint}")

        self._tokenizer = VQTokenizer.from_checkpoint(
            self.config.checkpoints.vqvae_checkpoint,
            theta_inverse_path=self.config.checkpoints.theta_inverse_path,
            initial_inverse_path=self.config.checkpoints.initial_inverse_path,
        )

    def _load_replayer(self) -> None:
        """Load QBMReplayer from substrate config."""
        from spinlock.qbm.replayer import QBMReplayer

        logger.info(
            f"Loading QBMReplayer from {self.config.checkpoints.qbm_substrate_config}"
        )
        self._replayer = QBMReplayer.from_config(
            str(self.config.checkpoints.qbm_substrate_config)
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
                case 'temporal':
                    self._feature_resolvers[family] = 'temporal'
                    self._temporal_extractor = self._load_temporal_extractor()
                case _:
                    logger.warning(
                        f"Unknown family '{family}' — tokens will be "
                        f"generated but not verified"
                    )
                    self._feature_resolvers[family] = 'skip'

        logger.info(f"Feature resolvers: {self._feature_resolvers}")

    def _load_temporal_extractor(self):
        """Load temporal feature extractor matching dataset generation config."""
        from spinlock.features.temporal.extractors import TemporalFeatureOrchestrator

        extractor = TemporalFeatureOrchestrator(device=self.device)
        logger.info("Temporal feature extractor loaded")
        return extractor

    def _create_components(self) -> None:
        """Create verification, queue, and scheduler components."""
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
    ) -> Dict[str, torch.Tensor]:
        """Extract features for each active family from a physical rollout.

        Resolution determined at init time by _resolve_feature_extractors().

        Args:
            trajectory: [B, M, T, C, H, W] raw rollout from QBM
            theta: [B, param_dim] decoded parameters
            u0: [B, C, H, W] decoded initial conditions

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
                case 'temporal':
                    if self._temporal_extractor is not None:
                        temporal = self._temporal_extractor.extract_per_timestep(
                            trajectory
                        )
                        features['temporal_features'] = temporal
                case 'skip':
                    pass  # Tokens from this family always mismatch

        return features

    def _explore_step(self) -> Dict[str, float]:
        """One exploration batch: generate → decode → rollout → retokenize → verify → queue.

        Returns:
            Dict of metrics for this step
        """
        batch_size = self.config.generation.batch_size

        # 1. Generate novel tokens from noise
        generated_tokens = self._generate_tokens(batch_size)

        # 2. Decode tokens to (theta, u0)
        theta, u0 = self._tokenizer.decode(generated_tokens)

        # 3. Run QBM simulation
        # QBMReplayer expects numpy [B, param_dim] in [0,1]
        theta_np = theta.cpu().numpy()
        trajectory = self._replayer.rollout_batch(
            theta_np,
            num_realizations=self.config.rollout.num_realizations,
            num_timesteps=self.config.rollout.num_timesteps,
        )  # [B, M, T, C, H, W] torch tensor

        # 4. Extract features per active family
        features = self._extract_features_from_rollout(trajectory, theta, u0)

        # 5. Retokenize (with multiple samples for variance estimation)
        if self.config.surprisal.verification_samples > 1:
            retokenized_samples = []
            for k in range(self.config.surprisal.verification_samples):
                # Re-run QBM with different seed for stochastic variation
                traj_k = self._replayer.rollout_batch(
                    theta_np,
                    num_realizations=self.config.rollout.num_realizations,
                    num_timesteps=self.config.rollout.num_timesteps,
                    seed=self.config.seed + k + 1,
                )
                feats_k = self._extract_features_from_rollout(traj_k, theta, u0)
                with torch.no_grad():
                    retok_k = self._tokenizer.tokenize(**feats_k)
                retokenized_samples.append(retok_k)

            mean_surprisal, variance = self._surprisal_computer.verify_with_multiple_samples(
                generated_tokens, retokenized_samples,
            )
            # Use mean retokenized for queue storage
            retokenized = retokenized_samples[0]
            surprisal = mean_surprisal
        else:
            with torch.no_grad():
                retokenized = self._tokenizer.tokenize(**features)
            surprisal = self._surprisal_computer.compute_surprisal(
                generated_tokens, retokenized,
            )
            variance = torch.zeros_like(surprisal)

        # 6. Compute Jaccard for logging
        jaccard = self._surprisal_computer.compute_jaccard(generated_tokens, retokenized)

        # 7. Queue high-surprisal items
        num_queued = self._queue.push_batch(
            generated_tokens, retokenized, surprisal, theta,
        )

        metrics = {
            'avg_surprisal': surprisal.mean().item(),
            'avg_jaccard': jaccard.mean().item(),
            'avg_variance': variance.mean().item(),
            'num_queued': num_queued,
            'queue_size': self._queue.size,
            'queue_fill': self._queue.fill_fraction,
        }

        logger.info(
            f"  Explore: surprisal={metrics['avg_surprisal']:.3f}, "
            f"jaccard={metrics['avg_jaccard']:.3f}, "
            f"queued={num_queued}/{batch_size}, "
            f"queue={self._queue.size}/{self.config.priority.queue_capacity}"
        )

        return metrics

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
        """One refinement epoch: pop queue → build batch → train denoiser.

        Returns:
            Dict of refinement metrics
        """
        items = self._queue.pop_batch(self.config.refinement.batch_size)
        if not items:
            return {'refine_loss': 0.0, 'refine_samples': 0}

        # Build training batch from retokenized tokens (= physically grounded truth)
        keys = list(items[0].retokenized_tokens.keys())
        tokens = {
            k: torch.tensor(
                [item.retokenized_tokens[k] for item in items],
                device=self.device,
                dtype=torch.long,
            )
            for k in keys
        }
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
        # Forward process needs a scalar timestep — use the first element
        # (batched timestep handling varies by implementation)
        noisy_tokens, _ = self._diffusion.forward_process(
            tokens, t[0].item(), mask_dict=target_mask,
        )

        # Predict clean tokens
        predicted_logits = self._denoiser(noisy_tokens, t)

        # Compute loss
        loss = self._compute_refinement_loss(predicted_logits, tokens, target_mask)

        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self._denoiser.parameters(),
            self.config.refinement.gradient_clip_norm,
        )
        self._optimizer.step()

        # Replay buffer for diversity
        self._replay_buffer.extend(items)
        if len(self._replay_buffer) > self.config.refinement.replay_buffer_size:
            self._replay_buffer = self._replay_buffer[
                -self.config.refinement.replay_buffer_size:
            ]

        metrics = {
            'refine_loss': loss.item(),
            'refine_samples': len(items),
            'replay_buffer_size': len(self._replay_buffer),
        }

        logger.info(
            f"  Refine: loss={loss.item():.4f}, "
            f"samples={len(items)}, "
            f"replay_buffer={len(self._replay_buffer)}"
        )

        return metrics

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

        # Save denoiser (the only model that changes)
        path = checkpoint_dir / f"denoiser_cycle_{cycle}.pt"
        torch.save({
            'cycle': cycle,
            'denoiser_state_dict': self._denoiser.state_dict(),
            'queue_size': self._queue.size,
            'metrics': dict(self._metrics),
        }, path)

        # Save metrics as JSON for easy analysis
        metrics_path = self.config.output_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(dict(self._metrics), f, indent=2, default=str)

        logger.info(f"Cycle {cycle} checkpoint saved to {path}")

    # ─── Main Loop ────────────────────────────────────────────────

    def run(self) -> Dict[str, List]:
        """Execute the full explore-refine self-play loop.

        Returns:
            Dict of metrics lists, keyed by metric name
        """
        torch.manual_seed(self.config.seed)
        self.initialize()

        logger.info(
            f"Starting self-play: {self.config.scheduler.max_cycles} cycles, "
            f"explore={self.config.scheduler.explore_steps} steps, "
            f"refine={self.config.scheduler.refine_epochs} epochs"
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
                    while self._scheduler.get_explore_steps_remaining() > 0:
                        metrics = self._explore_step()
                        self._log_metrics(cycle, mode, step, metrics)
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

            self._scheduler.advance()
            self._save_cycle_checkpoint(cycle)

        logger.info("Self-play complete!")
        logger.info(
            f"Final queue: {self._queue.size} items, "
            f"replay buffer: {len(self._replay_buffer)} items"
        )

        return dict(self._metrics)
