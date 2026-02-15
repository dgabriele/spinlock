"""Autonomous alignment self-play trainer.

Bootstraps the Rosetta contrastive alignment (token space ↔ English space)
without human interaction. Uses the LLM to generate diverse exploration
queries, runs headless exploration (denoiser + QBM), scores results via
RLAIF, accumulates high-quality pairs, and trains the alignment model.

Lifecycle per cycle:
    DISCOVER:
      Phase 1 — LLM on GPU: generate exploration queries
      Phase 2 — LLM offloaded: headless exploration per query
      Phase 3 — LLM on GPU: score results, accumulate pairs
    LEARN:
      Phase 4 — LLM on GPU: train alignment (InfoNCE) if buffer ≥ threshold

GPU Memory Flow (swap mode):
    Phase 1: LLM ~2.2GB on GPU
    Phase 2: denoiser+QBM ~0.6GB on GPU, LLM on CPU
    Phase 3: LLM ~2.2GB on GPU
    Phase 4: LLM ~2.2GB + denoiser ~0.1GB on GPU
    Peak: ~2.4GB — fits in 5-6GB GPU budget
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from spinlock.experimental.llm_agent.config import AutonomousTrainingConfig
from spinlock.experimental.llm_agent.query_generator import (
    DiscoverySummary,
    QueryCategory,
    QueryGenerator,
    QueryResult,
)

logger = logging.getLogger(__name__)


class AutonomousAlignmentTrainer:
    """Autonomous DISCOVER/LEARN loop for Rosetta alignment bootstrap.

    Takes a fully initialized ConversationalAgent and runs the self-play
    loop. Reuses the agent's components (denoiser, QBM, LLM, alignment
    trainer, evaluator, descriptor, gpu_manager) rather than constructing
    its own.

    Args:
        agent: Initialized ConversationalAgent with all components ready.
        config: AutonomousTrainingConfig with loop parameters.
    """

    def __init__(self, agent: Any, config: AutonomousTrainingConfig):
        self.agent = agent
        self.config = config

        # Resolve enabled categories
        if config.enabled_categories:
            self._categories = [
                QueryCategory(c) for c in config.enabled_categories
            ]
        else:
            self._categories = list(QueryCategory)

        # Build query generator (needs LLM + tokenizer from agent)
        self._query_generator = QueryGenerator(
            llm=agent._llm,
            tokenizer=agent._tokenizer,
            categories=self._categories,
            max_history=config.discovery_history_size,
        )

        # Metrics history
        self._metrics: List[Dict[str, Any]] = []

        # Ensure output directory exists
        self._output_dir = Path(config.output_dir)

    def run(self) -> Dict[str, List]:
        """Execute the full autonomous training loop.

        Returns:
            Dict mapping metric name → list of per-cycle values.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self._output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        logger.info(
            f"Starting autonomous alignment training: "
            f"{self.config.num_cycles} cycles, "
            f"{self.config.queries_per_cycle} queries/cycle, "
            f"{self.config.candidates_per_query} candidates/query, "
            f"categories={[c.value for c in self._categories]}"
        )

        # Warm-up: fit the template descriptor with feature statistics
        # so descriptions are natural language instead of raw numbers
        self._warmup_descriptor()

        for cycle in range(self.config.num_cycles):
            logger.info(f"{'='*60}")
            logger.info(f"Cycle {cycle + 1}/{self.config.num_cycles}")
            logger.info(f"{'='*60}")

            cycle_metrics = self._run_cycle(cycle)
            self._metrics.append(cycle_metrics)

            # Log cycle summary
            self._log_cycle_summary(cycle, cycle_metrics)

            # Checkpoint
            if (cycle + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(cycle + 1, checkpoint_dir)

        # Final checkpoint
        self._save_checkpoint(self.config.num_cycles, checkpoint_dir)

        # Save full metrics
        metrics_path = self._output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_path}")

        return self._aggregate_metrics()

    def _run_cycle(self, cycle_index: int) -> Dict[str, Any]:
        """Execute one DISCOVER → LEARN cycle.

        Args:
            cycle_index: Zero-based cycle number.

        Returns:
            Dict of cycle metrics.
        """
        metrics: Dict[str, Any] = {"cycle": cycle_index}

        # ── DISCOVER ──────────────────────────────────────────────
        discover_start = time.time()

        # Phase 1: Generate queries (LLM on GPU)
        queries = self._phase_generate_queries()
        metrics["queries_generated"] = len(queries)

        # Phase 2: Headless exploration (LLM offloaded)
        all_results = self._phase_explore(queries)
        metrics["candidates_explored"] = sum(len(r) for r in all_results.values())

        # Phase 3: Score + accumulate (LLM on GPU)
        scoring_metrics = self._phase_score_and_accumulate(queries, all_results)
        metrics.update(scoring_metrics)

        metrics["discover_time_s"] = time.time() - discover_start

        # ── LEARN ─────────────────────────────────────────────────
        learn_start = time.time()

        # Phase 4: Train alignment if buffer is large enough
        learn_metrics = self._phase_train_alignment()
        metrics.update(learn_metrics)

        metrics["learn_time_s"] = time.time() - learn_start

        return metrics

    def _phase_generate_queries(self) -> List[QueryResult]:
        """Phase 1: Generate exploration queries with LLM on GPU."""
        with self.agent._gpu_manager.llm_on_gpu():
            queries = self._query_generator.generate_batch(
                self.config.queries_per_cycle
            )
        logger.info(f"Phase 1: Generated {len(queries)} queries")
        return queries

    def _phase_explore(
        self, queries: List[QueryResult]
    ) -> Dict[str, List]:
        """Phase 2: Headless exploration per query (LLM offloaded).

        Uses the agent's _explore_batch_headless() which only needs
        denoiser + QBM on GPU (no LLM calls).

        Returns:
            Dict mapping query text → list of ExplorationResult.
        """
        all_results: Dict[str, List] = {}

        for i, query in enumerate(queries):
            logger.debug(
                f"Phase 2: Exploring query {i+1}/{len(queries)}: "
                f"{query.text[:60]}..."
            )
            results = self.agent._explore_batch_headless(
                query.text,
                seed_results=[],
            )
            all_results[query.text] = results

        total = sum(len(r) for r in all_results.values())
        logger.info(f"Phase 2: Explored {total} candidates across {len(queries)} queries")
        return all_results

    def _phase_score_and_accumulate(
        self,
        queries: List[QueryResult],
        all_results: Dict[str, List],
    ) -> Dict[str, Any]:
        """Phase 3: Score results and accumulate training pairs (LLM on GPU).

        Returns:
            Dict of scoring metrics.
        """
        metrics: Dict[str, Any] = {}
        total_scored = 0
        total_matches = 0
        total_pairs = 0
        all_scores: List[float] = []
        category_scores: Dict[str, List[float]] = {}

        threshold = self.agent.config.evaluator.reward_threshold

        with self.agent._gpu_manager.llm_on_gpu():
            for query in queries:
                results = all_results.get(query.text, [])
                if not results:
                    continue

                descriptions = [r.description for r in results]
                scores = self.agent.evaluator.score_batch(query.text, descriptions)

                for result, score in zip(results, scores):
                    result.score = score
                    all_scores.append(score)
                    total_scored += 1

                    # Track per-category scores
                    cat_key = query.category.value
                    category_scores.setdefault(cat_key, []).append(score)

                    # Accumulate training pair if score qualifies
                    should_accumulate = (
                        self.config.accumulate_all or score >= threshold
                    )
                    if should_accumulate and result.description:
                        sample_tokens = {
                            k: torch.tensor([v])
                            for k, v in result.tokens_dict.items()
                        }
                        self.agent.trainer.accumulate(
                            sample_tokens, result.description
                        )
                        total_pairs += 1

                    if score >= threshold:
                        total_matches += 1

                        # Record high-scoring discovery for query context
                        self._query_generator.add_discovery(
                            DiscoverySummary(
                                query=query.text,
                                description=result.description,
                                score=score,
                                category=query.category.value,
                            )
                        )

        metrics["candidates_scored"] = total_scored
        metrics["mean_score"] = (
            sum(all_scores) / len(all_scores) if all_scores else 0.0
        )
        metrics["matches_found"] = total_matches
        metrics["pairs_accumulated"] = total_pairs
        metrics["buffer_size"] = self.agent.trainer.buffer_size

        # Per-category mean scores
        for cat_key, scores in category_scores.items():
            metrics[f"mean_score/{cat_key}"] = (
                sum(scores) / len(scores) if scores else 0.0
            )

        logger.info(
            f"Phase 3: Scored {total_scored}, "
            f"matches={total_matches}, "
            f"pairs accumulated={total_pairs}, "
            f"buffer={self.agent.trainer.buffer_size}"
        )
        return metrics

    def _phase_train_alignment(self) -> Dict[str, Any]:
        """Phase 4: Train alignment model if enough pairs accumulated.

        Returns:
            Dict of training metrics (empty if training skipped).
        """
        metrics: Dict[str, Any] = {}

        if not self.agent.trainer.should_train():
            logger.info(
                f"Phase 4: Skipped (buffer {self.agent.trainer.buffer_size} "
                f"< threshold {self.agent.config.alignment.accumulation_threshold})"
            )
            metrics["alignment_loss"] = None
            metrics["alignment_acc_t2e"] = None
            metrics["alignment_acc_e2t"] = None
            return metrics

        logger.info("Phase 4: Training alignment...")
        with self.agent._gpu_manager.llm_on_gpu():
            epoch_metrics = self.agent.trainer.train()

        if epoch_metrics:
            final = epoch_metrics[-1]
            metrics["alignment_loss"] = final.get("loss")
            metrics["alignment_acc_t2e"] = final.get("accuracy_t2e")
            metrics["alignment_acc_e2t"] = final.get("accuracy_e2t")
            logger.info(
                f"Phase 4: loss={final.get('loss', 0):.4f} "
                f"acc_t2e={final.get('accuracy_t2e', 0):.3f} "
                f"acc_e2t={final.get('accuracy_e2t', 0):.3f}"
            )
        else:
            metrics["alignment_loss"] = None
            metrics["alignment_acc_t2e"] = None
            metrics["alignment_acc_e2t"] = None

        return metrics

    def _warmup_descriptor(self, num_batches: int = 4) -> None:
        """Fit the template descriptor with feature statistics.

        Runs a few headless exploration batches to collect feature vectors,
        then calls descriptor.fit(). Without this, the descriptor produces
        raw feature dumps ("spatial_mean=0.123") instead of natural language
        ("an energetic system with chaotic dynamics"), which makes RLAIF
        scoring meaningless.
        """
        logger.info(f"Warming up descriptor with {num_batches} exploration batches...")
        all_features = []

        for i in range(num_batches):
            results = self.agent._explore_batch_headless(
                query="warm-up batch",
                seed_results=[],
            )
            for r in results:
                if r.features:
                    all_features.append(r.features)

        if all_features:
            self.agent.fit_descriptor(all_features)
            logger.info(
                f"Descriptor fitted with {len(all_features)} feature samples"
            )
        else:
            logger.warning("No features collected during warm-up")

    def _log_cycle_summary(
        self, cycle: int, metrics: Dict[str, Any]
    ) -> None:
        """Log a summary of cycle results."""
        parts = [f"Cycle {cycle + 1} summary:"]
        parts.append(f"  queries={metrics.get('queries_generated', 0)}")
        parts.append(f"  explored={metrics.get('candidates_explored', 0)}")
        parts.append(f"  scored={metrics.get('candidates_scored', 0)}")
        parts.append(f"  matches={metrics.get('matches_found', 0)}")
        parts.append(f"  mean_score={metrics.get('mean_score', 0):.2f}")
        parts.append(f"  pairs={metrics.get('pairs_accumulated', 0)}")
        parts.append(f"  buffer={metrics.get('buffer_size', 0)}")

        loss = metrics.get("alignment_loss")
        if loss is not None:
            parts.append(f"  loss={loss:.4f}")
            parts.append(f"  acc_t2e={metrics.get('alignment_acc_t2e', 0):.3f}")
            parts.append(f"  acc_e2t={metrics.get('alignment_acc_e2t', 0):.3f}")
        else:
            parts.append("  alignment: not trained yet")

        t_discover = metrics.get("discover_time_s", 0)
        t_learn = metrics.get("learn_time_s", 0)
        parts.append(f"  time: discover={t_discover:.1f}s learn={t_learn:.1f}s")

        logger.info("\n".join(parts))

    def _save_checkpoint(self, cycle: int, checkpoint_dir: Path) -> None:
        """Save alignment model, query state, and metrics."""
        # Alignment model (trainable params only)
        alignment_path = checkpoint_dir / f"alignment_cycle_{cycle}.pt"
        trainable_state = {
            k: v
            for k, v in self.agent.alignment.state_dict().items()
            if any(
                p.requires_grad
                for p in self.agent.alignment.parameters()
            )
        }
        # Save all state_dict entries — during load, non-trainable weights
        # are ignored by using strict=False
        torch.save(self.agent.alignment.state_dict(), alignment_path)
        logger.info(f"Checkpoint: alignment saved to {alignment_path}")

        # Query generator state
        query_state_path = checkpoint_dir / f"query_state_cycle_{cycle}.json"
        with open(query_state_path, "w") as f:
            json.dump(self._query_generator.get_state(), f, indent=2, default=str)

        # Metrics snapshot
        metrics_path = self._output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)

        logger.info(f"Checkpoint saved at cycle {cycle}")

    def _aggregate_metrics(self) -> Dict[str, List]:
        """Aggregate per-cycle metrics into lists for plotting."""
        if not self._metrics:
            return {}

        keys = set()
        for m in self._metrics:
            keys.update(m.keys())
        keys.discard("cycle")

        result: Dict[str, List] = {}
        for key in sorted(keys):
            result[key] = [m.get(key) for m in self._metrics]

        return result
