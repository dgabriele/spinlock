"""Conversational agent: query → token targets → exploration.

Top-level orchestrator that connects natural language queries to the
token synthesis pipeline. All methods return both structured results
AND natural language responses via the ConversationManager.

Usage:
    agent = ConversationalAgent(config, pipeline)
    agent.initialize()

    response = agent.chat("What does turbulence look like?")
    print(response.text)     # Natural language response
    print(response.results)  # Structured ExplorationResult list

    response = agent.chat("Make it decay faster")
    print(response.text)     # References previous context
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from spinlock.experimental.llm_agent.config import AgentConfig
from spinlock.experimental.llm_agent.gpu_manager import LLMGpuManager

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    """A single exploration candidate with its evaluation."""

    tokens_dict: Dict[str, int]
    description: str
    features: Optional[Dict[str, float]] = None
    score: Optional[float] = None
    feedback: Optional[str] = None


@dataclass
class ChatResponse:
    """Agent response: natural language + structured data."""

    text: str
    results: List[ExplorationResult] = field(default_factory=list)
    intent: str = "explore"
    scores: Optional[List[float]] = None
    alignment_metrics: Optional[Dict[str, float]] = None
    background_iterations: int = 0


class ConversationalAgent:
    """Chatbot interface to token synthesis exploration.

    Classifies user intent (explore/refine/ask) and dispatches to
    the appropriate pipeline methods. Maintains conversation history
    for multi-turn dialogue, and accumulates (token, description)
    pairs for ongoing alignment training.

    Args:
        config: AgentConfig with LLM, alignment, and evaluator settings.
        pipeline: SynthesisVerificationPipeline instance (already initialized).
    """

    def __init__(self, config: AgentConfig, pipeline: Any):
        self.config = config
        self.pipeline = pipeline

        # Components (created during initialize)
        self.alignment = None       # RosettaAlignment
        self.trainer = None         # AlignmentTrainer
        self.descriptor = None      # TemplateDescriptor or HybridDescriptor
        self.evaluator = None       # RLAIFEvaluator
        self.conversation = None    # ConversationManager

        # LLM components
        self._llm = None
        self._tokenizer = None
        self._token_embedder = None

        # State
        self._last_query = None
        self._last_query_embedding = None
        self._last_results: List[ExplorationResult] = []
        self._initialized = False

        # Background exploration (continuous mode)
        self._stop_event = threading.Event()
        self._error_event = threading.Event()
        self._error_message: Optional[str] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._accumulated_results: List[ExplorationResult] = []
        self._results_lock = threading.Lock()
        self._background_iteration_count = 0

        # GPU memory management
        self._gpu_manager: Optional[LLMGpuManager] = None

        # Temporal feature name mapping (built during initialize)
        self._temporal_feature_names: Optional[List[str]] = None

    def initialize(self) -> None:
        """Load LLM, create LoRA adapter, build all components.

        This is separate from __init__ because it involves heavy I/O
        (downloading/loading model weights) and GPU allocation.
        """
        if self._initialized:
            logger.warning("ConversationalAgent already initialized")
            return

        logger.info("Initializing ConversationalAgent...")

        # Import heavy dependencies here (optional install)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "LLM agent requires 'transformers' and 'peft' packages. "
                "Install with: poetry install --with llm"
            ) from e

        from spinlock.experimental.llm_agent.alignment import (
            AlignmentTrainer,
            RosettaAlignment,
        )
        from spinlock.experimental.llm_agent.conversation import ConversationManager
        from spinlock.experimental.llm_agent.descriptor import (
            HybridDescriptor,
            TemplateDescriptor,
        )
        from spinlock.experimental.llm_agent.embedder import TokenEmbedder
        from spinlock.experimental.llm_agent.evaluator import RLAIFEvaluator

        llm_cfg = self.config.llm
        align_cfg = self.config.alignment

        # 1. Load frozen LLM + apply LoRA
        logger.info(f"Loading LLM: {llm_cfg.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(llm_cfg.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._llm = AutoModelForCausalLM.from_pretrained(
            llm_cfg.model_name,
            torch_dtype=torch.float16,
            device_map=llm_cfg.device,
        )

        # Apply LoRA adapters
        lora_config = LoraConfig(
            r=llm_cfg.lora_rank,
            lora_alpha=llm_cfg.lora_alpha,
            target_modules=llm_cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._llm = get_peft_model(self._llm, lora_config)
        trainable = sum(p.numel() for p in self._llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._llm.parameters())
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100*trainable/total:.2f}%)"
        )

        # 2. Auto-detect denoiser dimensions
        denoiser = self.pipeline._denoiser
        if self.config.denoiser_hidden_dim is None:
            self.config.denoiser_hidden_dim = denoiser.hidden_dim
        if self.config.num_token_positions is None:
            self.config.num_token_positions = denoiser.num_tokens

        # 3. Build TokenEmbedder
        denoiser.eval()
        for p in denoiser.parameters():
            p.requires_grad_(False)
        self._token_embedder = TokenEmbedder(
            denoiser, shared_dim=align_cfg.shared_dim
        )
        device = next(denoiser.parameters()).device
        self._token_embedder.pool.to(device)
        self._token_embedder.project.to(device)

        # 4. Build alignment model
        self.alignment = RosettaAlignment(
            token_embedder=self._token_embedder,
            llm_model=self._llm,
            llm_tokenizer=self._tokenizer,
            config=align_cfg,
        )
        self.trainer = AlignmentTrainer(self.alignment, align_cfg)

        # 5. Build descriptor
        template = TemplateDescriptor()
        if align_cfg.description_mode == "template":
            self.descriptor = template
        else:
            self.descriptor = HybridDescriptor(
                template_descriptor=template,
                llm=self._llm,
                tokenizer=self._tokenizer,
            )

        # 6. Build evaluator (optional)
        if self.config.evaluator.enabled:
            self.evaluator = RLAIFEvaluator(
                llm=self._llm,
                tokenizer=self._tokenizer,
                config=self.config.evaluator,
            )

        # 7. Build conversation manager
        self.conversation = ConversationManager(
            llm=self._llm,
            tokenizer=self._tokenizer,
            system_prompt=self.config.system_prompt,
        )

        # 8. Build GPU manager for LLM ↔ CPU swapping
        self._gpu_manager = LLMGpuManager(
            self._llm, llm_cfg.device, self.config.memory_mode
        )
        # In swap mode, offload LLM immediately — exploration owns the GPU
        if self.config.memory_mode == "swap":
            self._gpu_manager.offload_llm()
            logger.info("Swap mode: LLM offloaded to CPU after init")

        # 9. Read temporal feature names from tokenizer checkpoint metadata
        self._temporal_feature_names = self._read_feature_names_from_checkpoint()

        self._initialized = True
        logger.info("ConversationalAgent initialized successfully")

    def chat(self, message: str) -> ChatResponse:
        """Unified conversational interface.

        Classifies intent (explore/refine/ask) and dispatches:
        - Exploration: embed query → generate → QBM → score → respond
        - Refinement: adjust from previous query → re-explore → respond
        - Question: retrieve relevant context → LLM summarizes → respond

        If continuous exploration is enabled, a background thread continues
        exploring after the initial response is returned. Accumulated results
        from a previous background run are merged into the current response.

        Args:
            message: User's natural language message.

        Returns:
            ChatResponse with text, structured results, and metadata.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before chat()")

        # 1. Stop any running background exploration and collect its results
        bg_iterations = self._background_iteration_count
        self._stop_background()
        background_results = self._drain_accumulated()

        # 2. Classify intent
        intent = self._classify_intent(message)
        logger.info(f"Intent: {intent} | Message: {message[:80]}")

        # 3. Run initial exploration with LLM on GPU for scoring + response
        with self._gpu_manager.llm_on_gpu():
            if intent == "explore":
                results = self._explore(message)
            elif intent == "refine":
                results = self._refine(message)
            else:
                results = []

            # Merge background findings if relevant
            if background_results and intent in ("explore", "refine"):
                results = self._merge_results(results, background_results)

            # Build grounding context and compose response
            context = self._build_context(intent, results)
            text = self.conversation.compose_response(message, context)

            # Maybe train alignment if enough pairs accumulated
            alignment_metrics = self._maybe_train_alignment()

        # 4. Start background exploration for this query
        if self.config.continuous.enabled and intent in ("explore", "refine"):
            self._start_background(message, seed_results=results[:4])

        return ChatResponse(
            text=text,
            results=results,
            intent=intent,
            scores=[r.score for r in results if r.score is not None] or None,
            alignment_metrics=alignment_metrics,
            background_iterations=bg_iterations,
        )

    def _classify_intent(self, message: str) -> str:
        """Classify user intent: 'explore', 'refine', or 'ask'.

        Uses keyword heuristics first, falling back to LLM classification
        for ambiguous messages.
        """
        msg_lower = message.lower()

        # Refinement keywords (require prior context)
        refine_keywords = [
            "more", "less", "faster", "slower", "stronger", "weaker",
            "again", "similar", "like that", "but", "instead", "adjust",
            "tweak", "modify", "change",
        ]
        if self._last_results and any(kw in msg_lower for kw in refine_keywords):
            return "refine"

        # Question keywords
        ask_keywords = [
            "what", "how", "why", "explain", "describe", "tell me about",
            "summary", "so far", "found", "history", "patterns",
        ]
        if any(kw in msg_lower for kw in ask_keywords) and not any(
            action in msg_lower for action in ["find", "show", "generate", "create", "explore"]
        ):
            return "ask"

        return "explore"

    def _explore(
        self, query: str, num_candidates: int = 16
    ) -> List[ExplorationResult]:
        """Directed exploration pipeline.

        1. Generate candidate token sequences via diffusion
        2. Run QBM simulation on each
        3. Describe physical behavior
        4. Score against query (if evaluator enabled)
        5. Accumulate training pairs
        """
        self._last_query = query
        results = []

        # Generate via the synthesis pipeline's generation method
        batch_size = min(
            num_candidates, self.pipeline.config.generation.batch_size
        )
        generated_tokens = self.pipeline._generate_tokens(batch_size)

        # Decode → QBM → features (reuse pipeline infrastructure)
        theta, u0 = self.pipeline._tokenizer.decode(generated_tokens)
        theta_np = theta.cpu().numpy()
        trajectory, ics = self.pipeline._replayer.rollout_batch(
            theta_np,
            num_realizations=self.pipeline.config.rollout.num_realizations,
            num_timesteps=self.pipeline.config.rollout.num_timesteps,
            return_ics=True,
        )
        features_batch = self.pipeline._extract_features_from_rollout(
            trajectory, theta, u0, ics=ics
        )

        # Retokenize for grounded token sequences
        with torch.no_grad():
            retokenized = self.pipeline._tokenizer.tokenize(**features_batch)

        # Per-sample processing
        descriptions = []
        for i in range(batch_size):
            # Extract per-sample features as a flat dict
            sample_features = self._extract_sample_features(features_batch, i)
            description = self.descriptor.describe(sample_features)
            descriptions.append(description)

            # Extract per-sample token dict
            sample_tokens = {k: v[i].item() for k, v in retokenized.items()}

            result = ExplorationResult(
                tokens_dict=sample_tokens,
                description=description,
                features=sample_features,
            )
            results.append(result)

            # Accumulate training pair
            sample_tokens_tensor = {k: v[i:i+1] for k, v in retokenized.items()}
            self.trainer.accumulate(sample_tokens_tensor, description)

        # RLAIF scoring (if enabled)
        if self.evaluator is not None:
            scores = self.evaluator.score_batch(query, descriptions)
            for result, score in zip(results, scores):
                result.score = score

            # Sort by score (best first)
            results.sort(key=lambda r: r.score or 0.0, reverse=True)

        # Cache for refinement
        self._last_results = results

        # Embed query for later retrieval
        with torch.no_grad():
            self._last_query_embedding = self.alignment.encode_text([query])[0]

        logger.info(
            f"Explored {batch_size} candidates, "
            f"{sum(1 for r in results if r.score and r.score >= self.config.evaluator.reward_threshold)} matches"
        )

        return results

    def _refine(self, feedback: str) -> List[ExplorationResult]:
        """Refine previous exploration based on user feedback.

        Uses frontier-style inpainting: take the best previous results,
        mask some categories, and re-generate conditioned on context +
        the new feedback.
        """
        if not self._last_results:
            logger.warning("No previous results to refine, falling back to explore")
            return self._explore(feedback)

        # Use top results from last round as seeds
        top_results = self._last_results[:4]
        combined_query = f"{self._last_query}. Additionally: {feedback}"

        return self._explore(combined_query, num_candidates=16)

    def _build_context(
        self, intent: str, results: List[ExplorationResult]
    ) -> Dict[str, Any]:
        """Build grounding context for response composition."""
        context: Dict[str, Any] = {"intent": intent}

        if results:
            context["num_candidates"] = len(results)
            context["descriptions"] = [r.description for r in results[:5]]

            if any(r.score is not None for r in results):
                context["scores"] = [r.score for r in results[:5]]
                threshold = self.config.evaluator.reward_threshold
                context["num_matches"] = sum(
                    1 for r in results if r.score is not None and r.score >= threshold
                )

            # Feature highlights from best result
            if results[0].features:
                highlights = self._feature_highlights(results[0].features)
                if highlights:
                    context["feature_highlights"] = highlights

        # Queue summary
        if hasattr(self.pipeline, '_queue') and self.pipeline._queue is not None:
            context["queue_summary"] = (
                f"{self.pipeline._queue.size} items, "
                f"alignment buffer: {self.trainer.buffer_size} pairs"
            )

        return context

    @staticmethod
    def _feature_highlights(features: Dict[str, float]) -> str:
        """Extract notable feature values for context grounding."""
        highlights = []
        for key, val in sorted(features.items()):
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                continue
            # Report extreme values
            if abs(val) > 0.8 or abs(val) < 0.05:
                highlights.append(f"{key}={val:.3f}")
            if len(highlights) >= 5:
                break
        return ", ".join(highlights)

    def _read_feature_names_from_checkpoint(self) -> List[str]:
        """Read semantic feature names from the VQTokenizer checkpoint metadata.

        The tokenizer's feature_metadata (populated during training) stores
        the kept_feature_names with semantic names like 'spatial_mean_ch0',
        'spectral_entropy_ch1', etc. These are used by the TemplateDescriptor
        for pattern-based natural language descriptions.

        Returns:
            List of feature names, or empty list if metadata unavailable.
        """
        tokenizer = getattr(self.pipeline, '_tokenizer', None)
        if tokenizer is None:
            logger.warning("No tokenizer on pipeline — no feature names available")
            return []

        metadata = getattr(tokenizer, 'feature_metadata', None)
        if metadata is None:
            logger.warning(
                "Tokenizer has no feature_metadata — "
                "retrain with semantic names to enable rich descriptions"
            )
            return []

        families = getattr(metadata, 'families', {})
        if 'temporal' not in families:
            logger.warning("No 'temporal' family in feature metadata")
            return []

        names = families['temporal'].kept_feature_names
        if names and not names[0].startswith('temporal_feature_'):
            logger.info(
                f"Loaded {len(names)} semantic feature names from checkpoint "
                f"(e.g. {names[0]}, {names[-1]})"
            )
        else:
            logger.info(
                f"Loaded {len(names)} feature names from checkpoint "
                f"(generic — retrain tokenizer for semantic names)"
            )
        return names

    def _extract_sample_features(
        self, features_batch: Dict[str, Any], index: int
    ) -> Dict[str, float]:
        """Extract a single sample's features from batched feature tensors.

        Handles the key feature families from the pipeline:
        - temporal_features: [B, T, D] → time-averaged, named per feature
        - theta_features: [B, 9] → named Sobol parameters
        - initial_manual: [B, D] → summarized
        """
        sample = {}

        for key, val in features_batch.items():
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                if isinstance(val, (int, float)):
                    sample[key] = float(val)
                continue

            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            if val.ndim < 1 or val.shape[0] <= index:
                continue

            item = val[index]  # Remove batch dim

            if key == 'temporal_features' and item.ndim == 2:
                # [T, D] → time-average → [D], then assign named features
                time_avg = item.mean(dim=0)  # [D]
                names = self._temporal_feature_names or []
                for i, v in enumerate(time_avg):
                    feat_name = names[i] if i < len(names) else f"temporal_{i}"
                    sample[feat_name] = float(v.item())

            elif key == 'theta_features' and item.ndim == 1:
                # [9] Sobol parameters
                theta_names = [
                    'theta_sigma', 'theta_epsilon', 'theta_coupling',
                    'theta_mass', 'theta_damping', 'theta_driving',
                    'theta_nonlinearity', 'theta_boundary', 'theta_scale',
                ]
                for i, v in enumerate(item):
                    name = theta_names[i] if i < len(theta_names) else f"theta_{i}"
                    sample[name] = float(v.item())

            elif key == 'initial_raw':
                # Skip raw ICs — not useful for descriptions
                continue

            elif key == 'initial_manual' and item.ndim == 1:
                # Summarize initial condition features
                sample['initial_energy'] = float(item.mean().item())
                sample['initial_variance'] = float(item.var().item())

            else:
                # Fallback: scalar or mean
                sample[key] = float(item.mean().item())

        return sample

    def _maybe_train_alignment(self) -> Optional[Dict[str, float]]:
        """Train alignment model if enough pairs have accumulated."""
        if not self.trainer.should_train():
            return None

        logger.info("Triggering alignment training...")
        epoch_metrics = self.trainer.train()
        if epoch_metrics:
            return epoch_metrics[-1]  # return final epoch metrics
        return None

    # ------------------------------------------------------------------
    # Background exploration (continuous mode)
    # ------------------------------------------------------------------

    def _start_background(
        self, query: str, seed_results: List[ExplorationResult]
    ) -> None:
        """Start background exploration thread.

        The thread runs _background_loop, which generates candidates
        using only the denoiser + QBM (no LLM calls). The LLM is
        offloaded in swap mode to free GPU memory for exploration.

        Args:
            query: The user query to explore.
            seed_results: Top results from initial batch for frontier seeding.
        """
        self._stop_event.clear()
        self._error_event.clear()
        self._error_message = None
        self._background_iteration_count = 0
        with self._results_lock:
            self._accumulated_results.clear()

        # In swap mode, offload LLM before background thread starts
        self._gpu_manager.offload_llm()

        self._worker_thread = threading.Thread(
            target=self._background_loop,
            args=(query, seed_results),
            daemon=True,
            name="AgentExplorer",
        )
        self._worker_thread.start()
        logger.info("Background exploration started")

    def _stop_background(self) -> None:
        """Stop background exploration thread if running.

        Blocks until the current batch finishes (up to 10s timeout).
        Reports any errors that occurred during background exploration.
        """
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return

        self._stop_event.set()
        self._worker_thread.join(timeout=10.0)

        if self._worker_thread.is_alive():
            logger.warning("Background thread did not stop within timeout")

        if self._error_event.is_set():
            logger.warning(
                f"Background exploration error: {self._error_message}"
            )

        logger.info(
            f"Background exploration stopped after "
            f"{self._background_iteration_count} iterations"
        )

    def _drain_accumulated(self) -> List[ExplorationResult]:
        """Drain all accumulated background results (thread-safe)."""
        with self._results_lock:
            results = list(self._accumulated_results)
            self._accumulated_results.clear()
        return results

    def _background_loop(
        self, query: str, seed_results: List[ExplorationResult]
    ) -> None:
        """Background thread: keep exploring until stopped.

        Runs headless exploration batches (no LLM calls). Periodically
        loads the LLM for batch scoring if the evaluator is enabled.

        Args:
            query: User query to explore around.
            seed_results: Initial frontier seeds for refinement.
        """
        try:
            cfg = self.config.continuous
            iteration = 0

            while (
                not self._stop_event.is_set()
                and iteration < cfg.max_iterations
            ):
                # GPU: denoiser + QBM own the GPU (LLM offloaded in swap)
                batch_results = self._explore_batch_headless(
                    query, seed_results
                )

                # Accumulate results (thread-safe)
                with self._results_lock:
                    self._accumulated_results.extend(batch_results)

                # Accumulate alignment pairs
                for r in batch_results:
                    if r.description:
                        sample_t = {
                            k: torch.tensor([v])
                            for k, v in r.tokens_dict.items()
                        }
                        self.trainer.accumulate(sample_t, r.description)

                # Update seeds for frontier refinement
                if cfg.use_frontier_refinement and batch_results:
                    seed_results = self._best_results(4)

                iteration += 1
                self._background_iteration_count = iteration

                # Periodic LLM scoring (batch results, load LLM, score, offload)
                if (
                    self.evaluator is not None
                    and iteration % cfg.score_batch_size == 0
                    and not self._stop_event.is_set()
                ):
                    self._background_score_batch(query)

            logger.debug(
                f"Background loop finished: {iteration} iterations"
            )

        except Exception as e:
            self._error_message = str(e)
            self._error_event.set()
            logger.error(f"Background exploration error: {e}")

    def _explore_batch_headless(
        self,
        query: str,
        seed_results: List[ExplorationResult],
    ) -> List[ExplorationResult]:
        """One exploration batch WITHOUT LLM calls.

        In swap mode, the LLM is on CPU during this method. Only uses:
        denoiser (generate), tokenizer (decode/retokenize), QBM replayer
        (rollout), and template descriptor (no LLM needed).

        Args:
            query: User query (unused here, kept for future guided generation).
            seed_results: Frontier seeds for guided generation.

        Returns:
            List of ExplorationResult with score=None (scored later).
        """
        batch_size = min(16, self.pipeline.config.generation.batch_size)

        # Generate tokens (unconditional for now; frontier seeding is future)
        generated_tokens = self.pipeline._generate_tokens(batch_size)

        # Decode → QBM → features
        theta, u0 = self.pipeline._tokenizer.decode(generated_tokens)
        theta_np = theta.cpu().numpy()
        trajectory, ics = self.pipeline._replayer.rollout_batch(
            theta_np,
            num_realizations=self.pipeline.config.rollout.num_realizations,
            num_timesteps=self.pipeline.config.rollout.num_timesteps,
            return_ics=True,
        )
        features_batch = self.pipeline._extract_features_from_rollout(
            trajectory, theta, u0, ics=ics
        )

        # Retokenize
        with torch.no_grad():
            retokenized = self.pipeline._tokenizer.tokenize(**features_batch)

        # Build results with template descriptions only (no LLM)
        results = []
        for i in range(batch_size):
            sample_features = self._extract_sample_features(features_batch, i)
            # TemplateDescriptor.describe() is pure computation — no LLM
            description = self.descriptor.describe(sample_features)
            sample_tokens = {k: v[i].item() for k, v in retokenized.items()}

            results.append(
                ExplorationResult(
                    tokens_dict=sample_tokens,
                    description=description,
                    features=sample_features,
                    score=None,  # Scored later in batch
                )
            )

        return results

    def _background_score_batch(self, query: str) -> None:
        """Score accumulated results using the LLM evaluator.

        In swap mode, loads LLM to GPU, scores all unscored results,
        then offloads LLM back to CPU. Called periodically from the
        background loop.
        """
        with self._results_lock:
            unscored = [
                r for r in self._accumulated_results if r.score is None
            ]
        if not unscored:
            return

        descriptions = [r.description for r in unscored]

        with self._gpu_manager.llm_on_gpu():
            scores = self.evaluator.score_batch(query, descriptions)

        # Update scores (thread-safe: we only write to individual result objects)
        for result, score in zip(unscored, scores):
            result.score = score

        logger.debug(f"Background scored {len(unscored)} results")

    def _best_results(self, n: int) -> List[ExplorationResult]:
        """Get the top-n accumulated results by score (or most recent if unscored)."""
        with self._results_lock:
            all_results = list(self._accumulated_results)

        scored = [r for r in all_results if r.score is not None]
        if scored:
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[:n]
        # Fall back to most recent if nothing scored yet
        return all_results[-n:] if all_results else []

    @staticmethod
    def _merge_results(
        foreground: List[ExplorationResult],
        background: List[ExplorationResult],
    ) -> List[ExplorationResult]:
        """Merge foreground and background results, preferring higher scores.

        Combines both lists and sorts by score (best first). Unscored
        results sort after scored ones.
        """
        combined = foreground + background

        # Partition into scored and unscored
        scored = [r for r in combined if r.score is not None]
        unscored = [r for r in combined if r.score is None]

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored + unscored

    def fit_descriptor(self, features_list: List[Dict[str, float]]) -> None:
        """Fit the template descriptor from dataset statistics.

        Should be called before chat() if using template or hybrid mode.
        Can use features from the synthesis pipeline's training data.

        Args:
            features_list: List of feature dicts from training samples.
        """
        if hasattr(self.descriptor, 'fit'):
            self.descriptor.fit(features_list)
        elif hasattr(self.descriptor, 'template'):
            self.descriptor.template.fit(features_list)
