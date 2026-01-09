"""
Adaptive MSE Response System for VQ-VAE Training.

This module provides tools for detecting and fixing categories with poor reconstruction
quality through compression ratio optimization.

Design Pattern: Strategy pattern for MSE detection heuristics.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class CategoryPerformance:
    """Performance metrics for a single category."""

    category_name: str
    reconstruction_mse: float
    utilization: float
    feature_count: int
    current_ratios: List[float]
    mse_history: List[float]  # Last N epochs


@dataclass
class AdaptationDecision:
    """Decision to adapt a category's compression ratios."""

    category_name: str
    reason: str  # Why adaptation is needed
    current_ratios: List[float]
    candidate_ratios: List[List[float]]  # Sweep candidates
    priority: int  # Higher = more urgent


class MSEDetectionStrategy(ABC):
    """Abstract strategy for detecting problematic categories."""

    @abstractmethod
    def should_adapt(
        self,
        category_perf: CategoryPerformance,
        all_category_perfs: List[CategoryPerformance],
        global_stats: Dict[str, float]
    ) -> Tuple[bool, str]:
        """
        Determine if category needs adaptation.

        Args:
            category_perf: Performance metrics for the category under evaluation
            all_category_perfs: Performance metrics for all categories
            global_stats: Global statistics (avg_mse, max_mse, avg_utilization)

        Returns:
            Tuple of (should_adapt: bool, reason: str)
        """
        pass


class AbsoluteThresholdDetector(MSEDetectionStrategy):
    """Detect categories exceeding absolute MSE threshold."""

    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: Absolute MSE threshold above which adaptation is triggered
        """
        self.threshold = threshold

    def should_adapt(
        self,
        category_perf: CategoryPerformance,
        all_category_perfs: List[CategoryPerformance],
        global_stats: Dict[str, float]
    ) -> Tuple[bool, str]:
        if category_perf.reconstruction_mse > self.threshold:
            return True, (
                f"MSE {category_perf.reconstruction_mse:.4f} exceeds "
                f"threshold {self.threshold}"
            )
        return False, ""


class RelativeThresholdDetector(MSEDetectionStrategy):
    """Detect categories significantly worse than average."""

    def __init__(self, multiplier: float = 2.0):
        """
        Args:
            multiplier: MSE multiplier above average that triggers adaptation
        """
        self.multiplier = multiplier

    def should_adapt(
        self,
        category_perf: CategoryPerformance,
        all_category_perfs: List[CategoryPerformance],
        global_stats: Dict[str, float]
    ) -> Tuple[bool, str]:
        avg_mse = np.mean([c.reconstruction_mse for c in all_category_perfs])
        if category_perf.reconstruction_mse > avg_mse * self.multiplier:
            ratio = category_perf.reconstruction_mse / avg_mse
            return True, (
                f"MSE {category_perf.reconstruction_mse:.4f} is {ratio:.1f}x "
                f"worse than average {avg_mse:.4f}"
            )
        return False, ""


class TrendBasedDetector(MSEDetectionStrategy):
    """Detect categories not improving over N epochs."""

    def __init__(self, window: int = 10, improvement_threshold: float = 0.001):
        """
        Args:
            window: Number of recent epochs to analyze
            improvement_threshold: Minimum improvement required over window
        """
        self.window = window
        self.improvement_threshold = improvement_threshold

    def should_adapt(
        self,
        category_perf: CategoryPerformance,
        all_category_perfs: List[CategoryPerformance],
        global_stats: Dict[str, float]
    ) -> Tuple[bool, str]:
        if len(category_perf.mse_history) < self.window:
            return False, ""

        recent_mse = category_perf.mse_history[-self.window:]
        improvement = recent_mse[0] - recent_mse[-1]

        if improvement < self.improvement_threshold:
            return True, (
                f"MSE stagnated (improvement {improvement:.5f} < "
                f"{self.improvement_threshold} over {self.window} epochs)"
            )
        return False, ""


class CombinedDetector(MSEDetectionStrategy):
    """Combine multiple detection strategies using OR logic."""

    def __init__(
        self,
        absolute_threshold: float = 0.05,
        relative_multiplier: float = 2.0,
        trend_window: int = 10,
        trend_improvement_threshold: float = 0.001
    ):
        """
        Args:
            absolute_threshold: Absolute MSE threshold
            relative_multiplier: Relative MSE multiplier above average
            trend_window: Number of epochs for trend analysis
            trend_improvement_threshold: Minimum improvement required
        """
        self.detectors = [
            AbsoluteThresholdDetector(absolute_threshold),
            RelativeThresholdDetector(relative_multiplier),
            TrendBasedDetector(trend_window, trend_improvement_threshold)
        ]

    def should_adapt(
        self,
        category_perf: CategoryPerformance,
        all_category_perfs: List[CategoryPerformance],
        global_stats: Dict[str, float]
    ) -> Tuple[bool, str]:
        reasons = []
        for detector in self.detectors:
            should, reason = detector.should_adapt(
                category_perf, all_category_perfs, global_stats
            )
            if should:
                reasons.append(reason)

        if reasons:
            return True, "; ".join(reasons)
        return False, ""


class CompressionRatioSweeper:
    """Generate and evaluate compression ratio candidates."""

    def __init__(
        self,
        search_space: List[List[float]] = None,
        num_candidates: int = 5
    ):
        """
        Args:
            search_space: List of candidate ratio triplets, e.g.:
                [[0.3, 0.7, 1.2], [0.4, 1.0, 1.5], [0.5, 1.5, 2.5]]
            num_candidates: If search_space not provided, generate this many candidates
        """
        if search_space is not None:
            self.search_space = search_space
        else:
            # Generate default search space centered around [0.5, 1.0, 1.5]
            self.search_space = self._generate_default_search_space(num_candidates)

    def _generate_default_search_space(self, num_candidates: int) -> List[List[float]]:
        """Generate search space around baseline [0.5, 1.0, 1.5]."""
        candidates = [
            [0.25, 0.75, 2.0],   # Less compression (preserve detail)
            [0.4, 1.0, 1.8],     # Moderate
            [0.5, 1.0, 1.5],     # Baseline
            [0.6, 1.2, 1.8],     # Slight expansion
            [0.3, 0.8, 1.5],     # Aggressive coarse compression
        ]
        return candidates[:num_candidates]

    def generate_candidates(
        self,
        current_ratios: List[float],
        category_perf: CategoryPerformance
    ) -> List[List[float]]:
        """
        Generate candidates tailored to category performance.

        Strategy:
        - If MSE is high, bias toward less compression (more capacity)
        - If utilization is low, bias toward more compression (smaller codebooks)

        Args:
            current_ratios: Current compression ratios for the category
            category_perf: Performance metrics for the category

        Returns:
            List of candidate ratio triplets to evaluate
        """
        candidates = list(self.search_space)

        # Add targeted candidates based on current performance
        if category_perf.reconstruction_mse > 0.05:
            # High MSE → need more capacity
            candidates.insert(0, [r * 1.5 for r in current_ratios])

        if category_perf.utilization < 0.4:
            # Low utilization → try smaller codebooks
            candidates.append([r * 0.7 for r in current_ratios])

        return candidates


class AdaptiveMSEManager:
    """
    Main orchestrator for adaptive MSE response.

    Workflow:
    1. Collect category performance metrics from tuning phase
    2. Detect problematic categories using combined heuristics
    3. Generate compression ratio sweep candidates
    4. Rank categories by priority (worst MSE first)
    5. Return adaptation decisions for downstream execution
    """

    def __init__(
        self,
        detector: MSEDetectionStrategy = None,
        sweeper: CompressionRatioSweeper = None
    ):
        """
        Args:
            detector: MSE detection strategy (defaults to CombinedDetector)
            sweeper: Compression ratio sweeper (defaults to CompressionRatioSweeper)
        """
        self.detector = detector or CombinedDetector()
        self.sweeper = sweeper or CompressionRatioSweeper()

    def analyze_tuning_results(
        self,
        category_metrics: Dict[str, CategoryPerformance]
    ) -> List[AdaptationDecision]:
        """
        Analyze tuning phase results and generate adaptation decisions.

        Args:
            category_metrics: Dict mapping category_name -> CategoryPerformance

        Returns:
            List of AdaptationDecision for problematic categories, sorted by priority
        """
        all_perfs = list(category_metrics.values())
        global_stats = {
            "avg_mse": np.mean([p.reconstruction_mse for p in all_perfs]),
            "max_mse": np.max([p.reconstruction_mse for p in all_perfs]),
            "avg_utilization": np.mean([p.utilization for p in all_perfs])
        }

        decisions = []

        for category_name, perf in category_metrics.items():
            should_adapt, reason = self.detector.should_adapt(
                perf, all_perfs, global_stats
            )

            if should_adapt:
                candidates = self.sweeper.generate_candidates(
                    perf.current_ratios, perf
                )

                # Priority: higher MSE = higher priority
                priority = int(perf.reconstruction_mse * 1000)

                decision = AdaptationDecision(
                    category_name=category_name,
                    reason=reason,
                    current_ratios=perf.current_ratios,
                    candidate_ratios=candidates,
                    priority=priority
                )
                decisions.append(decision)

        # Sort by priority (highest first)
        decisions.sort(key=lambda d: d.priority, reverse=True)

        return decisions
