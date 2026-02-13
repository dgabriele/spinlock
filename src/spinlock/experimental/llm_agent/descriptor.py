"""Auto-description of physical behavior from quantitative features.

Generates natural language descriptions of QBM rollout outcomes from
their feature vectors. This is the key to self-supervised alignment:
no human labels are needed because every rollout's physical features
can be automatically described in English.

Two modes:
- TemplateDescriptor: Fast, deterministic, no LLM call. Uses percentile
  thresholds computed from dataset statistics.
- HybridDescriptor: Template base + LLM zero-shot elaboration for richer,
  more varied descriptions.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureMapping:
    """Maps a quantitative feature to qualitative adjectives via thresholds.

    Thresholds are percentile boundaries computed from dataset statistics,
    NOT hardcoded. The descriptor's fit() method populates these.

    Args:
        name: Feature name pattern (matched against feature dict keys).
        adjectives: Ordered list of (adjective, upper_bound) pairs.
            The first adjective whose upper_bound exceeds the value is used.
        prefix: Optional prefix for the description fragment.
    """

    name: str
    adjectives: List[Tuple[str, float]] = field(default_factory=list)
    prefix: str = ""


# Default adjective scales — thresholds are filled by fit()
_DEFAULT_SCALES = {
    "energy": ["quiescent", "low-energy", "moderate-energy", "energetic", "highly energetic"],
    "entropy": ["ordered", "partially ordered", "moderate-entropy", "disordered", "maximally entropic"],
    "chaos": ["stable", "weakly chaotic", "mildly chaotic", "chaotic", "strongly chaotic"],
    "growth": ["decaying", "slowly decaying", "stable-amplitude", "growing", "rapidly growing"],
    "coherence": ["incoherent", "weakly coherent", "partially coherent", "coherent", "highly coherent"],
    "spectral": ["narrowband", "moderate-bandwidth", "broadband", "ultra-broadband"],
}


class TemplateDescriptor:
    """Generate physics descriptions from feature vectors using templates.

    Maps quantitative features to qualitative language via percentile
    thresholds learned from dataset statistics. No LLM calls needed.

    The fit() method computes percentile boundaries from training features,
    following the framework principle of runtime auto-detection rather than
    hardcoded values.

    Usage:
        descriptor = TemplateDescriptor()
        descriptor.fit(training_features)  # learn thresholds from data
        text = descriptor.describe(sample_features)
        # → "An energetic system with chaotic dynamics and decaying amplitude"
    """

    # Feature name patterns → (scale_key, prefix)
    # Patterns are matched against feature dict keys using substring matching
    FEATURE_PATTERNS: List[Tuple[str, str, str]] = [
        # (pattern_substring, scale_key, description_prefix)
        ("spatial_mean", "energy", "energy"),
        ("spatial_variance", "growth", "spatial variation"),
        ("spectral_entropy", "entropy", "spectral complexity"),
        ("spectral_flatness", "spectral", "spectral shape"),
        ("cross_channel_correlation", "coherence", "cross-channel coherence"),
        ("temporal_quantum_von_neumann_entropy", "entropy", "quantum entropy"),
        ("temporal_quantum_purity", "coherence", "quantum purity"),
        ("temporal_quantum_coherence", "coherence", "quantum coherence"),
        ("temporal_stability", "chaos", "dynamical stability"),
        ("temporal_multiscale", "growth", "multi-scale structure"),
    ]

    TEMPLATE = (
        "A {energy_adj} system with {stability_adj} dynamics"
        "{spectral_clause}{coherence_clause}{quantum_clause}."
    )

    def __init__(self) -> None:
        self._thresholds: Dict[str, List[Tuple[str, float]]] = {}
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, features_list: List[Dict[str, float]]) -> "TemplateDescriptor":
        """Compute percentile-based thresholds from training features.

        Args:
            features_list: List of feature dicts, one per sample.
                Each dict maps feature_name → float value.

        Returns:
            self (for chaining)
        """
        if not features_list:
            logger.warning("TemplateDescriptor.fit() called with empty features")
            return self

        # Collect all values per feature key
        all_values: Dict[str, List[float]] = {}
        for features in features_list:
            for key, val in features.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    all_values.setdefault(key, []).append(float(val))

        # Compute percentile boundaries for each matched pattern
        for pattern, scale_key, _ in self.FEATURE_PATTERNS:
            matched_keys = [k for k in all_values if pattern in k]
            if not matched_keys:
                continue

            # Pool values across all matching keys
            pooled = []
            for k in matched_keys:
                pooled.extend(all_values[k])

            if not pooled:
                continue

            arr = np.array(pooled)
            adjectives = _DEFAULT_SCALES.get(scale_key, [])
            n = len(adjectives)
            if n == 0:
                continue

            # Percentile boundaries: split into n equal bins
            percentiles = np.linspace(0, 100, n + 1)[1:-1]
            boundaries = np.percentile(arr, percentiles).tolist()

            self._thresholds[pattern] = list(zip(
                adjectives,
                boundaries + [float("inf")],
            ))

            # Store stats for debugging
            self._feature_stats[pattern] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n_samples": len(pooled),
            }

        self._fitted = True
        logger.info(
            f"TemplateDescriptor fitted with {len(self._thresholds)} feature patterns "
            f"from {len(features_list)} samples"
        )
        return self

    def _classify(self, value: float, pattern: str) -> Optional[str]:
        """Map a value to an adjective using fitted thresholds."""
        thresholds = self._thresholds.get(pattern)
        if thresholds is None:
            return None
        for adjective, upper in thresholds:
            if value <= upper:
                return adjective
        return thresholds[-1][0]  # last adjective if all exceeded

    def _aggregate_feature(
        self, features: Dict[str, float], pattern: str
    ) -> Optional[float]:
        """Average all feature values matching a pattern."""
        vals = [
            v for k, v in features.items()
            if pattern in k and isinstance(v, (int, float)) and np.isfinite(v)
        ]
        if not vals:
            return None
        return float(np.mean(vals))

    def describe(self, features: Dict[str, float]) -> str:
        """Generate a natural language description from a feature vector.

        Args:
            features: Dict mapping feature_name → float value.

        Returns:
            Natural language description string.
        """
        if not self._fitted:
            return self._describe_unfitted(features)

        # Classify each pattern
        classifications: Dict[str, str] = {}
        for pattern, scale_key, prefix in self.FEATURE_PATTERNS:
            val = self._aggregate_feature(features, pattern)
            if val is not None:
                adj = self._classify(val, pattern)
                if adj is not None:
                    classifications[prefix] = adj

        return self._compose(classifications)

    def _describe_unfitted(self, features: Dict[str, float]) -> str:
        """Fallback description when thresholds haven't been fitted."""
        parts = []
        for key, val in sorted(features.items()):
            if isinstance(val, (int, float)) and np.isfinite(val):
                parts.append(f"{key}={val:.3f}")
            if len(parts) >= 10:
                break
        return f"System with features: {', '.join(parts)}"

    def _compose(self, classifications: Dict[str, str]) -> str:
        """Compose a natural language description from classified features."""
        energy_adj = classifications.get("energy", "moderate-energy")
        stability_adj = classifications.get("dynamical stability", "stable")

        # Optional clauses
        spectral = classifications.get("spectral complexity")
        spectral_clause = f", {spectral} spectral structure" if spectral else ""

        coherence = classifications.get("cross-channel coherence")
        coherence_clause = f", {coherence} channels" if coherence else ""

        # Quantum clause: combine available quantum features
        quantum_parts = []
        for prefix in ["quantum entropy", "quantum purity", "quantum coherence"]:
            adj = classifications.get(prefix)
            if adj:
                quantum_parts.append(f"{adj} {prefix.replace('quantum ', '')}")
        quantum_clause = ""
        if quantum_parts:
            quantum_clause = f", and {', '.join(quantum_parts)}"

        return self.TEMPLATE.format(
            energy_adj=energy_adj,
            stability_adj=stability_adj,
            spectral_clause=spectral_clause,
            coherence_clause=coherence_clause,
            quantum_clause=quantum_clause,
        )

    @property
    def feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Dataset statistics computed during fit()."""
        return self._feature_stats


class HybridDescriptor:
    """Template descriptor + LLM zero-shot elaboration.

    1. TemplateDescriptor produces a structured summary
    2. LLM expands it into a richer, more varied description

    This produces more diverse training signal for contrastive alignment
    at the cost of LLM inference per description.

    Args:
        template_descriptor: Fitted TemplateDescriptor instance.
        llm: HuggingFace causal LM (frozen + LoRA).
        tokenizer: Corresponding tokenizer.
        max_new_tokens: Max tokens for LLM elaboration.
    """

    ELABORATION_PROMPT = (
        "Expand this physics summary into a natural one-sentence description "
        "of the system's behavior. Be specific and vivid.\n"
        "Summary: {summary}\n"
        "Description:"
    )

    def __init__(
        self,
        template_descriptor: TemplateDescriptor,
        llm: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int = 64,
    ):
        self.template = template_descriptor
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def describe(self, features: Dict[str, float]) -> str:
        """Generate an LLM-elaborated description.

        Falls back to template-only if LLM is not available.
        """
        summary = self.template.describe(features)

        if self.llm is None or self.tokenizer is None:
            return summary

        return self._elaborate(summary)

    def _elaborate(self, summary: str) -> str:
        """Use LLM to expand a template summary."""
        import torch

        prompt = self.ELABORATION_PROMPT.format(summary=summary)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the new tokens (after the prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        elaboration = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean up: take first sentence only
        elaboration = elaboration.strip()
        sentence_end = re.search(r'[.!?]', elaboration)
        if sentence_end:
            elaboration = elaboration[:sentence_end.end()]

        return elaboration if elaboration else summary
