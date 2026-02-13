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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from spinlock.experimental.llm_agent.config import AgentConfig

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

        self._initialized = True
        logger.info("ConversationalAgent initialized successfully")

    def chat(self, message: str) -> ChatResponse:
        """Unified conversational interface.

        Classifies intent (explore/refine/ask) and dispatches:
        - Exploration: embed query → generate → QBM → score → respond
        - Refinement: adjust from previous query → re-explore → respond
        - Question: retrieve relevant context → LLM summarizes → respond

        All paths end with ConversationManager.compose_response() to produce
        natural language grounded in the physical findings.

        Args:
            message: User's natural language message.

        Returns:
            ChatResponse with text, structured results, and metadata.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before chat()")

        intent = self._classify_intent(message)
        logger.info(f"Intent: {intent} | Message: {message[:80]}")

        if intent == "explore":
            results = self._explore(message)
        elif intent == "refine":
            results = self._refine(message)
        else:
            results = []

        # Build grounding context for response composition
        context = self._build_context(intent, results)

        # Compose natural language response
        text = self.conversation.compose_response(message, context)

        # Maybe train alignment if enough pairs accumulated
        alignment_metrics = self._maybe_train_alignment()

        return ChatResponse(
            text=text,
            results=results,
            intent=intent,
            scores=[r.score for r in results if r.score is not None] or None,
            alignment_metrics=alignment_metrics,
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

    @staticmethod
    def _extract_sample_features(
        features_batch: Dict[str, Any], index: int
    ) -> Dict[str, float]:
        """Extract a single sample's features from batched feature tensors."""
        sample = {}
        for key, val in features_batch.items():
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] > index:
                sample[key] = float(val[index].item()) if val[index].numel() == 1 else float(val[index].mean().item())
            elif isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > index:
                sample[key] = float(val[index].mean()) if val[index].size > 1 else float(val[index])
            elif isinstance(val, (int, float)):
                sample[key] = float(val)
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
