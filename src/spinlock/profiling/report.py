"""Profiling report generation and analysis."""

import json
import csv
import torch
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


class ProfilingReport:
    """
    Analyze and report feature profiling results.

    Identifies bottlenecks and generates actionable insights from timing data.
    """

    def __init__(self, stats: Dict[str, Dict[str, float]], device: torch.device):
        """
        Initialize profiling report.

        Args:
            stats: Dict mapping feature/category names to timing statistics
            device: torch.device used for profiling
        """
        self.stats = stats
        self.device = device

    def get_sorted_by_time(
        self, top_n: int = None
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Get features/categories sorted by total time (descending).

        Args:
            top_n: Optional limit on number of results

        Returns:
            List of (name, stats_dict) tuples
        """
        sorted_items = sorted(
            self.stats.items(), key=lambda x: x[1]["total_ms"], reverse=True
        )
        return sorted_items[:top_n] if top_n else sorted_items

    def get_category_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Aggregate stats by category.

        Returns:
            Dict mapping category names to aggregate statistics
        """
        category_stats = defaultdict(
            lambda: {"total_ms": 0.0, "num_features": 0, "mean_ms": 0.0}
        )

        for name, stats in self.stats.items():
            category = stats["category"]
            category_stats[category]["total_ms"] += stats["total_ms"]
            category_stats[category]["num_features"] += 1

        # Compute means
        for cat, s in category_stats.items():
            if s["num_features"] > 0:
                s["mean_ms"] = s["total_ms"] / s["num_features"]

        return dict(category_stats)

    def get_bottlenecks(self, threshold_pct: float = 5.0) -> List[Tuple[str, float]]:
        """
        Identify bottlenecks (features taking >threshold% of total time).

        Args:
            threshold_pct: Percentage threshold (default: 5%)

        Returns:
            List of (feature_name, percentage) tuples
        """
        total_time = sum(s["total_ms"] for s in self.stats.values())
        if total_time == 0:
            return []

        bottlenecks = []
        for name, stats in self.stats.items():
            pct = (stats["total_ms"] / total_time) * 100.0
            if pct >= threshold_pct:
                bottlenecks.append((name, pct))

        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)

    def print_summary(self):
        """Print formatted summary to console."""
        print("\n" + "=" * 80)
        print("FEATURE PROFILING REPORT")
        print("=" * 80)

        # Device info
        print(f"\nDevice: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")

        # Total time breakdown
        total_time = sum(s["total_ms"] for s in self.stats.values())
        print(
            f"\nTotal feature extraction time: {total_time:.2f} ms ({total_time/1000:.2f}s)"
        )

        # Category summary
        print("\n" + "-" * 80)
        print("CATEGORY SUMMARY")
        print("-" * 80)
        print(
            f"{'Category':<25} {'Total (ms)':<15} {'Count':<10} {'Mean (ms)':<15} {'% of Total':<12}"
        )
        print("-" * 80)

        category_summary = self.get_category_summary()
        for cat, stats in sorted(
            category_summary.items(), key=lambda x: x[1]["total_ms"], reverse=True
        ):
            pct = (stats["total_ms"] / total_time) * 100.0 if total_time > 0 else 0.0
            print(
                f"{cat:<25} {stats['total_ms']:<15.2f} {stats['num_features']:<10} "
                f"{stats['mean_ms']:<15.2f} {pct:<12.1f}%"
            )

        # Top features/categories
        print("\n" + "-" * 80)
        print("TOP 20 FEATURES BY TOTAL TIME")
        print("-" * 80)
        print(
            f"{'Feature/Category':<40} {'Total (ms)':<15} {'Mean (ms)':<15} {'% of Total':<12}"
        )
        print("-" * 80)

        for name, stats in self.get_sorted_by_time(top_n=20):
            pct = (stats["total_ms"] / total_time) * 100.0 if total_time > 0 else 0.0
            print(
                f"{name:<40} {stats['total_ms']:<15.2f} {stats['mean_ms']:<15.2f} {pct:<12.1f}%"
            )

        # Bottlenecks
        bottlenecks = self.get_bottlenecks(threshold_pct=5.0)
        if bottlenecks:
            print("\n" + "-" * 80)
            print("BOTTLENECKS (>5% of total time)")
            print("-" * 80)
            for name, pct in bottlenecks:
                print(f"  {name:<50} {pct:>6.1f}%")

        # Per-sample cost (useful for scaling estimates)
        total_samples = sum(s.get("total_samples", 0) for s in self.stats.values())
        if total_samples > 0:
            ms_per_sample = total_time / total_samples
            print(f"\nAverage time per sample: {ms_per_sample:.2f} ms")
            print(
                f"Estimated time for 10K samples: {(ms_per_sample * 10000) / 1000 / 60:.2f} min"
            )

        print("=" * 80 + "\n")

    def save_json(self, output_path: Path):
        """
        Save detailed profiling data to JSON.

        Args:
            output_path: Path to output JSON file
        """
        total_time = sum(s["total_ms"] for s in self.stats.values())
        output_data = {
            "device": str(self.device),
            "gpu_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
            "total_time_ms": total_time,
            "category_summary": self.get_category_summary(),
            "detailed_stats": self.stats,
            "bottlenecks": [
                {"name": name, "percentage": pct} for name, pct in self.get_bottlenecks()
            ],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Detailed profiling data saved to: {output_path}")

    def save_csv(self, output_path: Path):
        """
        Save profiling data as CSV for external analysis.

        Args:
            output_path: Path to output CSV file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "feature",
                    "category",
                    "total_ms",
                    "count",
                    "mean_ms",
                    "min_ms",
                    "max_ms",
                    "ms_per_sample",
                    "total_samples",
                ]
            )

            for name, stats in self.get_sorted_by_time():
                writer.writerow(
                    [
                        name,
                        stats["category"],
                        stats["total_ms"],
                        stats["count"],
                        stats["mean_ms"],
                        stats["min_ms"],
                        stats["max_ms"],
                        stats.get("ms_per_sample", 0.0),
                        stats.get("total_samples", 0),
                    ]
                )

        print(f"CSV data saved to: {output_path}")
