"""RLAIF: LLM-as-judge for scoring rollout-vs-query match.

Given a user query ("find something turbulent") and a candidate
rollout's feature description, asks the LLM to rate the match 0-10.

High-scoring items become:
1. Positive training signal for the alignment model
2. Seeds for frontier-conditioned exploration (mask & resample)
3. Entries in the priority queue with boosted priority

Low-scoring items become negative examples for contrastive training.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from spinlock.experimental.llm_agent.config import EvaluatorConfig

logger = logging.getLogger(__name__)


class RLAIFEvaluator:
    """Reinforcement Learning from AI Feedback.

    Uses the LLM (same frozen+LoRA model as alignment) to judge how
    well a physics description matches a user's qualitative query.

    The scoring prompt is templated and produces a 0-10 numeric rating.
    Batch scoring amortizes tokenization and forward pass costs.

    Args:
        llm: HuggingFace causal LM (frozen + LoRA).
        tokenizer: Corresponding tokenizer.
        config: EvaluatorConfig with prompt template and thresholds.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        config: EvaluatorConfig,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.config = config

    def score(self, query: str, description: str) -> float:
        """Rate match between a query and a physics description.

        Args:
            query: User's qualitative target (e.g., "turbulent and decaying").
            description: Auto-generated physics description of a rollout.

        Returns:
            Float score 0-10. Returns 0.0 on parsing failure.
        """
        scores = self.score_batch(query, [description])
        return scores[0]

    def score_batch(self, query: str, descriptions: List[str]) -> List[float]:
        """Batch scoring for efficiency.

        Args:
            query: User's qualitative target.
            descriptions: List of physics descriptions to score.

        Returns:
            List of float scores 0-10, one per description.
        """
        prompts = [
            self.config.score_prompt_template.format(
                query=query, description=desc
            )
            for desc in descriptions
        ]

        scores = []
        for prompt in prompts:
            score = self._generate_score(prompt)
            scores.append(score)

        return scores

    def _generate_score(self, prompt: str) -> float:
        """Generate a numeric score from a scoring prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,  # greedy for deterministic scores
                temperature=1.0,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return self._parse_score(response)

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numeric score from LLM response text.

        Handles formats like "7", "7.5", "7/10", "Score: 8".
        Returns 0.0 if no number can be parsed.
        """
        # Try to find a number (possibly decimal) in the response
        match = re.search(r'(\d+(?:\.\d+)?)', response)
        if match:
            val = float(match.group(1))
            return min(max(val, 0.0), 10.0)  # clamp to [0, 10]
        return 0.0

    def is_match(self, score: float) -> bool:
        """Check if a score exceeds the reward threshold."""
        return score >= self.config.reward_threshold

    def generate_feedback(
        self, query: str, description: str, score: float
    ) -> str:
        """Generate qualitative feedback explaining why the score is what it is.

        Useful for debugging alignment quality and for conversational
        responses that explain the agent's reasoning.

        Args:
            query: User's qualitative target.
            description: Physics description of the candidate.
            score: Numeric score already assigned.

        Returns:
            Natural language explanation.
        """
        prompt = (
            f"A user asked for '{query}'. A physics simulation produced: "
            f"{description}. This was rated {score:.1f}/10 as a match. "
            f"Briefly explain why in one sentence."
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        feedback = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Take first sentence
        sentence_end = re.search(r'[.!?]', feedback)
        if sentence_end:
            feedback = feedback[:sentence_end.end()]

        return feedback

    def select_top_k(
        self,
        query: str,
        descriptions: List[str],
        k: Optional[int] = None,
    ) -> List[int]:
        """Score all descriptions and return indices of top-k matches.

        Args:
            query: User's qualitative target.
            descriptions: List of candidate descriptions.
            k: Number of top items to return (default: config.top_k_refinement).

        Returns:
            Indices into descriptions, sorted by score (highest first).
        """
        if k is None:
            k = self.config.top_k_refinement

        scores = self.score_batch(query, descriptions)
        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        return scored_indices[:k]
