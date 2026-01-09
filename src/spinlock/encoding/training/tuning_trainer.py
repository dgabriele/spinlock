"""
Per-Category Tuning Trainer for Adaptive MSE Response.

This module provides tools for running upfront per-category tuning to validate
and optimize compression ratios before full joint VQ-VAE training.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import h5py
import torch

from .adaptive_mse import (
    CategoryPerformance,
    AdaptiveMSEManager,
    AdaptationDecision
)


class PerCategoryTuningTrainer:
    """
    Train each category independently for rapid validation of compression ratios.

    Workflow:
    1. Create temporary single-category VQ-VAE for each category
    2. Train for N epochs (default: 50)
    3. Collect MSE, utilization, and training curves
    4. Return CategoryPerformance for each category
    """

    def __init__(
        self,
        dataset_path: str,
        group_indices: Dict[str, List[int]],
        initial_compression_ratios: Dict[str, List[float]],
        tuning_epochs: int = 50,
        tuning_samples: int = 10000,
        device: str = "cuda"
    ):
        """
        Args:
            dataset_path: Path to HDF5 dataset with features
            group_indices: Dict mapping category_name -> feature indices
            initial_compression_ratios: Initial ratios from auto computation
            tuning_epochs: Epochs per tuning run (default: 50)
            tuning_samples: Samples for tuning (default: 10K)
            device: Training device
        """
        self.dataset_path = Path(dataset_path)
        self.group_indices = group_indices
        self.initial_compression_ratios = initial_compression_ratios
        self.tuning_epochs = tuning_epochs
        self.tuning_samples = tuning_samples
        self.device = device

    def train_single_category(
        self,
        category_name: str,
        compression_ratios: List[float]
    ) -> CategoryPerformance:
        """
        Train VQ-VAE for a single category.

        Creates a temporary single-category model with given compression ratios,
        trains for tuning_epochs, and extracts performance metrics.

        Args:
            category_name: Name of the category to train
            compression_ratios: Compression ratios to use (L0, L1, L2)

        Returns:
            CategoryPerformance with MSE, utilization, and training curves
        """
        from spinlock.encoding.categorical_vqvae import (
            CategoricalVQVAEConfig,
            CategoricalHierarchicalVQVAE
        )
        from spinlock.encoding.training.trainer import VQVAETrainer

        # Create single-category config
        single_category_group_indices = {
            category_name: self.group_indices[category_name]
        }

        # Load features subset for this category
        features = self._load_features_subset(
            category_indices=self.group_indices[category_name]
        )

        input_dim = len(self.group_indices[category_name])

        # Create single-category VQ-VAE
        config = CategoricalVQVAEConfig(
            input_dim=input_dim,
            group_indices=single_category_group_indices,
            group_embedding_dim=256,
            group_hidden_dim=512,
            levels={category_name: []},  # Will be auto-filled
            compression_ratios={category_name: compression_ratios}
        )

        model = CategoricalHierarchicalVQVAE(config).to(self.device)

        # Create in-memory dataset for tuning
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        # Write features to temporary HDF5 file
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('features', data=features)

        try:
            # Train
            trainer = VQVAETrainer(
                model=model,
                dataset_path=tmp_path,
                checkpoint_dir=None,  # Don't save checkpoints
                num_epochs=self.tuning_epochs,
                batch_size=512,
                learning_rate=0.001,
                verbose=False
            )

            history = trainer.train()

            # Extract performance
            final_metrics = history["metrics"][-1]
            mse_key = f"{category_name}/reconstruction_mse"
            util_keys = [
                k for k in final_metrics.keys()
                if category_name in k and "/utilization" in k
            ]

            perf = CategoryPerformance(
                category_name=category_name,
                reconstruction_mse=final_metrics.get(mse_key, 0.0),
                utilization=np.mean([final_metrics[k] for k in util_keys])
                            if util_keys else 0.0,
                feature_count=input_dim,
                current_ratios=compression_ratios,
                mse_history=[
                    m.get(mse_key, 0.0)
                    for m in history["metrics"][-min(10, len(history["metrics"])):]
                ]
            )

            return perf

        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def run_tuning_phase(self) -> Dict[str, CategoryPerformance]:
        """
        Train all categories independently and collect performance metrics.

        Returns:
            Dict mapping category_name -> CategoryPerformance
        """
        results = {}

        for category_name in self.group_indices.keys():
            print(f"Tuning {category_name}...")
            ratios = self.initial_compression_ratios.get(
                category_name,
                [0.5, 1.0, 1.5]  # Fallback
            )
            perf = self.train_single_category(category_name, ratios)
            results[category_name] = perf

            print(f"  MSE: {perf.reconstruction_mse:.4f}, Util: {perf.utilization:.2%}")

        return results

    def run_compression_sweep(
        self,
        category_name: str,
        candidate_ratios: List[List[float]]
    ) -> Tuple[List[float], CategoryPerformance]:
        """
        Evaluate multiple compression ratio candidates and select best.

        Args:
            category_name: Name of the category
            candidate_ratios: List of compression ratio triplets to evaluate

        Returns:
            Tuple of (best_ratios, best_performance)
        """
        print(f"\nRunning compression sweep for {category_name}...")
        print(f"  Evaluating {len(candidate_ratios)} candidates...")

        results = []

        for i, ratios in enumerate(candidate_ratios):
            print(f"  [{i+1}/{len(candidate_ratios)}] Ratios {ratios}")
            perf = self.train_single_category(category_name, ratios)
            results.append((ratios, perf))
            print(f"    MSE: {perf.reconstruction_mse:.4f}")

        # Select best by MSE
        best_ratios, best_perf = min(results, key=lambda x: x[1].reconstruction_mse)

        print(f"  Best ratios: {best_ratios} (MSE: {best_perf.reconstruction_mse:.4f})")

        return best_ratios, best_perf

    def _load_features_subset(self, category_indices: List[int]) -> np.ndarray:
        """
        Load features for specific category indices.

        Args:
            category_indices: Feature column indices for this category

        Returns:
            Features array of shape (tuning_samples, n_features_in_category)
        """
        with h5py.File(self.dataset_path, "r") as f:
            all_features = f["features"][:]
            # Take first tuning_samples rows
            all_features = all_features[:self.tuning_samples]
            # Extract columns for this category
            features = all_features[:, category_indices]
        return features


def run_adaptive_tuning_workflow(
    dataset_path: str,
    group_indices: Dict[str, List[int]],
    initial_compression_ratios: Dict[str, List[float]],
    tuning_epochs: int = 50,
    tuning_samples: int = 10000,
    max_sweeps: int = 3,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Complete adaptive tuning workflow.

    Steps:
    1. Run initial tuning phase for all categories
    2. Detect problematic categories
    3. Run compression sweeps for problematic categories
    4. Return final optimized compression ratios

    Args:
        dataset_path: Path to HDF5 dataset
        group_indices: Dict mapping category_name -> feature indices
        initial_compression_ratios: Initial ratios from auto computation
        tuning_epochs: Epochs per tuning run (default: 50)
        tuning_samples: Samples for tuning (default: 10K)
        max_sweeps: Maximum categories to sweep (default: 3, most problematic)
        device: Training device

    Returns:
        Dict mapping category_name -> optimized compression ratios
    """
    print("=" * 60)
    print("ADAPTIVE MSE TUNING WORKFLOW")
    print("=" * 60)

    # Step 1: Initial tuning phase
    print("\n[Step 1] Initial Tuning Phase (all categories)")
    print(f"  Training each category for {tuning_epochs} epochs on {tuning_samples} samples...")

    tuner = PerCategoryTuningTrainer(
        dataset_path=dataset_path,
        group_indices=group_indices,
        initial_compression_ratios=initial_compression_ratios,
        tuning_epochs=tuning_epochs,
        tuning_samples=tuning_samples,
        device=device
    )

    category_perfs = tuner.run_tuning_phase()

    # Step 2: Detect problematic categories
    print("\n[Step 2] Detecting Problematic Categories")

    manager = AdaptiveMSEManager()
    adaptation_decisions = manager.analyze_tuning_results(category_perfs)

    if not adaptation_decisions:
        print("  ✓ All categories performing well (no adaptation needed)")
        return initial_compression_ratios

    print(f"  ⚠ {len(adaptation_decisions)} categories need adaptation:")
    for decision in adaptation_decisions:
        print(f"    - {decision.category_name}: {decision.reason}")

    # Step 3: Run compression sweeps (limit to top N worst)
    print(f"\n[Step 3] Compression Ratio Sweeps (top {max_sweeps} categories)")

    final_ratios = dict(initial_compression_ratios)

    for i, decision in enumerate(adaptation_decisions[:max_sweeps]):
        print(f"\n  [{i+1}/{min(len(adaptation_decisions), max_sweeps)}] {decision.category_name}")
        print(f"    Current MSE: {category_perfs[decision.category_name].reconstruction_mse:.4f}")
        print(f"    Reason: {decision.reason}")

        best_ratios, best_perf = tuner.run_compression_sweep(
            decision.category_name,
            decision.candidate_ratios
        )

        final_ratios[decision.category_name] = best_ratios

        improvement = (
            category_perfs[decision.category_name].reconstruction_mse -
            best_perf.reconstruction_mse
        )
        print(f"    ✓ Improved MSE by {improvement:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"  Adapted {min(len(adaptation_decisions), max_sweeps)} categories")
    print("\n  Final Compression Ratios:")
    for cat, ratios in final_ratios.items():
        marker = (
            " (ADAPTED)"
            if cat in [d.category_name for d in adaptation_decisions[:max_sweeps]]
            else ""
        )
        print(f"    {cat}: {ratios}{marker}")

    return final_ratios
