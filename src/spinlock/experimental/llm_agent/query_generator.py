"""Query generation for autonomous alignment training.

Generates diverse exploration queries across a taxonomy of categories:
physics-targeted, abstract-physical, philosophical, consciousness, and
contrastive. The LLM produces novel queries seeded by category examples
and recent discovery context, enabling the self-play loop to explore
broadly across the token space.

The taxonomy is deliberately broad — it tests whether the Rosetta
alignment can map between the LLM's semantic understanding (which
includes abstract concepts like "symmetry breaking") and the grounded
QBM token space (which encodes energy, entropy, Lyapunov exponents).
"""

import enum
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

logger = logging.getLogger(__name__)


class QueryCategory(enum.Enum):
    """Taxonomy of query categories for diverse exploration."""

    PHYSICS_TARGETED = "physics_targeted"
    ABSTRACT_PHYSICAL = "abstract_physical"
    PHILOSOPHICAL = "philosophical"
    CONSCIOUSNESS = "consciousness"
    CONTRASTIVE = "contrastive"


# Seed prompts per category — used as few-shot examples in the LLM prompt
# and as fallbacks when the LLM fails to produce enough novel queries.
SEED_PROMPTS: Dict[QueryCategory, List[str]] = {
    QueryCategory.PHYSICS_TARGETED: [
        "Find a system with high Lyapunov exponent",
        "Show me low-entropy spatially ordered dynamics",
        "Find rapid energy dissipation with spectral narrowing",
        "A system near the edge of chaos with moderate entropy",
        "Find coherent multi-channel oscillations with slow decay",
        "Show me high-purity quantum states with broadband spectra",
        "A system with maximal spectral entropy but low spatial variance",
    ],
    QueryCategory.ABSTRACT_PHYSICAL: [
        "What does the onset of turbulence look like?",
        "Show me spontaneous symmetry breaking",
        "What does thermalization look like?",
        "Find a phase transition between order and disorder",
        "Show me resonance and energy transfer between modes",
        "What does decoherence look like in a dynamical system?",
    ],
    QueryCategory.PHILOSOPHICAL: [
        "What is the simplest system that appears complex?",
        "Something that looks purposeful but isn't",
        "A system that remembers its past",
        "Motion that tells a story",
        "The boundary between alive and mechanical",
        "Find something that looks like emergence",
        "A system that creates and destroys structure simultaneously",
    ],
    QueryCategory.CONSCIOUSNESS: [
        "What does integrated information look like in dynamical systems?",
        "Find a system that appears to observe itself",
        "Self-referential computation",
        "Dynamics that distinguish self from environment",
        "A system with internal models of its own states",
        "Find something that looks like attention or selective focus",
    ],
    QueryCategory.CONTRASTIVE: [
        "Something as different as possible from the previous result",
        "The opposite of chaos",
        "The most boring system you can find",
        "Find the maximally simple counterpart to a complex system",
        "Something that violates the pattern we've seen so far",
    ],
}

CATEGORY_DESCRIPTIONS: Dict[QueryCategory, str] = {
    QueryCategory.PHYSICS_TARGETED: (
        "Queries targeting specific physical observables: energy levels, entropy, "
        "Lyapunov exponents, spectral structure, coherence, quantum purity. "
        "Be quantitatively specific."
    ),
    QueryCategory.ABSTRACT_PHYSICAL: (
        "Queries about physical phenomena described qualitatively: turbulence, "
        "symmetry breaking, thermalization, phase transitions. Connect abstract "
        "concepts to observable dynamics."
    ),
    QueryCategory.PHILOSOPHICAL: (
        "Queries exploring philosophical aspects of dynamical systems: complexity, "
        "purpose, memory, narrative, emergence. These test whether the alignment "
        "can bridge metaphorical and physical descriptions."
    ),
    QueryCategory.CONSCIOUSNESS: (
        "Queries about consciousness-like properties in dynamical systems: "
        "integrated information, self-reference, attention, self-environment "
        "boundaries. Inspired by IIT and enactivism."
    ),
    QueryCategory.CONTRASTIVE: (
        "Queries that explicitly contrast with previous discoveries or request "
        "opposites, extremes, or violations of observed patterns. These push "
        "exploration to under-sampled regions of token space."
    ),
}

QUERY_GENERATION_TEMPLATE = """\
You are generating exploration queries for a physics simulation system.
The system generates quantum-inspired dynamical systems described by features like:
energy, entropy, Lyapunov exponents, spectral structure, coherence, quantum purity.

Category: {category_description}

Example queries in this category:
{seed_examples}

{discovery_context}

Generate {num_queries} novel, diverse queries in this category.
Each query should be a single sentence on its own line.
Do NOT repeat the examples above. Be creative and specific.

Queries:
"""


@dataclass
class DiscoverySummary:
    """Summary of a high-scoring discovery for query context feedback."""

    query: str
    description: str
    score: float
    category: str


@dataclass
class QueryResult:
    """A generated query with its metadata."""

    text: str
    category: QueryCategory


class QueryGenerator:
    """Generates diverse exploration queries using an LLM.

    Uses a taxonomy of 5 query categories with seed examples. The LLM
    generates novel queries conditioned on category descriptions, seed
    examples, and recent high-scoring discoveries.

    Args:
        llm: HuggingFace causal LM (frozen + LoRA).
        tokenizer: Corresponding tokenizer.
        categories: Subset of categories to use (None = all).
        max_history: Max discoveries to keep for context.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        categories: Optional[List[QueryCategory]] = None,
        max_history: int = 10,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.categories = categories or list(QueryCategory)
        self.max_history = max_history

        self._used_queries: Set[str] = set()
        self._discovery_history: List[DiscoverySummary] = []
        self._category_index = 0  # Round-robin counter

    def generate_batch(
        self, num_queries: int, category: Optional[QueryCategory] = None
    ) -> List[QueryResult]:
        """Generate a batch of novel exploration queries.

        Picks a category (round-robin if not specified), builds an LLM
        prompt with seed examples + discovery context, generates queries,
        deduplicates, and falls back to unused seeds if needed.

        Args:
            num_queries: Number of queries to generate.
            category: Specific category, or None for round-robin.

        Returns:
            List of QueryResult with text and category metadata.
        """
        if category is None:
            category = self._next_category()

        # Build prompt
        seeds = SEED_PROMPTS[category]
        seed_sample = random.sample(seeds, min(5, len(seeds)))
        seed_text = "\n".join(f"- {s}" for s in seed_sample)

        discovery_context = self._build_discovery_context(category)

        prompt = QUERY_GENERATION_TEMPLATE.format(
            category_description=CATEGORY_DESCRIPTIONS[category],
            seed_examples=seed_text,
            discovery_context=discovery_context,
            num_queries=num_queries,
        )

        # Generate via LLM
        raw_queries = self._generate_queries(prompt, num_queries)

        # Deduplicate against used queries
        novel = []
        for q in raw_queries:
            normalized = q.strip().lower()
            if normalized and normalized not in self._used_queries:
                self._used_queries.add(normalized)
                novel.append(QueryResult(text=q.strip(), category=category))

        # Fallback: fill remaining slots with unused seed prompts
        if len(novel) < num_queries:
            for seed in seeds:
                if seed.strip().lower() not in self._used_queries:
                    self._used_queries.add(seed.strip().lower())
                    novel.append(QueryResult(text=seed, category=category))
                if len(novel) >= num_queries:
                    break

        result = novel[:num_queries]
        logger.debug(
            f"Generated {len(result)} queries for {category.value} "
            f"({len(raw_queries)} raw, {len(novel)} novel)"
        )
        return result

    def add_discovery(self, summary: DiscoverySummary) -> None:
        """Record a high-scoring discovery for future query context.

        Recent discoveries appear in the LLM prompt for query generation,
        enabling the contrastive category to produce meaningful queries
        like "the opposite of what we found for turbulence".

        Args:
            summary: Discovery summary with query, description, score.
        """
        self._discovery_history.append(summary)
        if len(self._discovery_history) > self.max_history:
            self._discovery_history = self._discovery_history[-self.max_history :]

    def get_state(self) -> Dict[str, Any]:
        """Serialize generator state for checkpointing."""
        return {
            "used_queries": list(self._used_queries),
            "discovery_history": [
                {
                    "query": d.query,
                    "description": d.description,
                    "score": d.score,
                    "category": d.category,
                }
                for d in self._discovery_history
            ],
            "category_index": self._category_index,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore generator state from checkpoint."""
        self._used_queries = set(state.get("used_queries", []))
        self._discovery_history = [
            DiscoverySummary(**d) for d in state.get("discovery_history", [])
        ]
        self._category_index = state.get("category_index", 0)

    def _next_category(self) -> QueryCategory:
        """Round-robin category selection."""
        cat = self.categories[self._category_index % len(self.categories)]
        self._category_index += 1
        return cat

    def _build_discovery_context(self, category: QueryCategory) -> str:
        """Build context string from recent discoveries."""
        if not self._discovery_history:
            return ""

        recent = self._discovery_history[-5:]
        lines = ["Recent discoveries from the system:"]
        for d in recent:
            lines.append(
                f"- Query: \"{d.query}\" → {d.description} (score: {d.score:.1f})"
            )

        if category == QueryCategory.CONTRASTIVE:
            lines.append(
                "\nGenerate queries that contrast with or oppose these findings."
            )

        return "\n".join(lines)

    def _generate_queries(self, prompt: str, num_queries: int) -> List[str]:
        """Generate query lines from the LLM.

        Uses high temperature + repetition penalty for diversity.
        Parses output as one query per line.
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=min(num_queries * 30, 256),
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse: one query per line, skip empty/numbered prefix lines
        queries = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Strip numbering like "1. " or "- "
            if line and (line[0].isdigit() or line[0] == "-"):
                line = line.lstrip("0123456789.-) ").strip()
            if line and len(line) > 10:  # Skip very short fragments
                queries.append(line)

        return queries[:num_queries]
