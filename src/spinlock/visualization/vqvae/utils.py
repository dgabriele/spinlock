"""Utility functions for VQ-VAE visualization.

Handles loading and processing of VQ-VAE checkpoints, training history,
and normalization statistics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
import torch


@dataclass
class VQVAECheckpointData:
    """Container for all VQ-VAE checkpoint data needed for visualization."""

    # Model configuration
    input_dim: int
    group_embedding_dim: int
    group_hidden_dim: int
    num_categories: int
    num_levels: int

    # Category structure
    group_indices: Dict[str, List[int]]  # category -> feature indices
    category_names: List[str]  # sorted category names
    levels: Dict[str, List[Dict]]  # category -> level configs

    # Feature information
    feature_names: List[str]  # all feature names
    feature_families: Dict[str, List[int]]  # family -> feature indices

    # Normalization statistics
    normalization_stats: Dict[str, Dict[str, np.ndarray]]  # category -> {mean, std}

    # Training history
    train_loss: List[float]
    val_loss: List[float]
    metrics_history: List[Dict[str, float]]  # per-epoch metrics
    final_metrics: Dict[str, float]

    # Model state
    model_state_dict: Optional[Dict] = None
    epoch: int = 0
    best_val_loss: float = 0.0


def load_vqvae_checkpoint(checkpoint_dir: str | Path) -> VQVAECheckpointData:
    """Load all VQ-VAE checkpoint data for visualization.

    Args:
        checkpoint_dir: Path to checkpoint directory containing:
            - final_model.pt or best_model.pt
            - training_history.json
            - normalization_stats.npz

    Returns:
        VQVAECheckpointData with all loaded data
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Prefer final_model.pt (has feature_names), fall back to best_model.pt
    if (checkpoint_dir / "final_model.pt").exists():
        model_path = checkpoint_dir / "final_model.pt"
    elif (checkpoint_dir / "best_model.pt").exists():
        model_path = checkpoint_dir / "best_model.pt"
    else:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Extract config - try model_config first, then config
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("No config found in checkpoint")

    # Extract group indices
    group_indices = config.get("group_indices", checkpoint.get("group_indices", {}))
    # Use insertion order (list()) to match VQ layer ordering, not sorted()
    # This ensures visualizations match model structure
    category_names = list(group_indices.keys())
    num_categories = len(category_names)

    # Extract levels configuration
    levels = config.get("levels", config.get("category_levels", {}))

    # Determine number of levels (assume consistent across categories)
    num_levels = 3  # default
    if levels and category_names:
        first_cat_levels = levels.get(category_names[0], [])
        if isinstance(first_cat_levels, list):
            num_levels = len(first_cat_levels)

    # Extract feature names
    feature_names = checkpoint.get("feature_names", [])
    if not feature_names:
        # Generate generic names if not available
        input_dim = config.get("input_dim", sum(len(v) for v in group_indices.values()))
        feature_names = [f"feature_{i}" for i in range(input_dim)]

    # Parse feature families from names (format: "family::name")
    feature_families: Dict[str, List[int]] = {}
    for i, name in enumerate(feature_names):
        if "::" in name:
            family = name.split("::")[0]
        else:
            family = "unknown"
        if family not in feature_families:
            feature_families[family] = []
        feature_families[family].append(i)

    # Load normalization stats
    norm_stats_path = checkpoint_dir / "normalization_stats.npz"
    normalization_stats: Dict[str, Dict[str, np.ndarray]] = {}
    if norm_stats_path.exists():
        stats = np.load(norm_stats_path, allow_pickle=True)
        for cat in category_names:
            mean_key = f"{cat}_mean"
            std_key = f"{cat}_std"
            if mean_key in stats.files and std_key in stats.files:
                normalization_stats[cat] = {
                    "mean": stats[mean_key],
                    "std": stats[std_key],
                }

    # Load training history
    history_path = checkpoint_dir / "training_history.json"
    train_loss = []
    val_loss = []
    metrics_history = []
    final_metrics = {}

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        metrics_history = history.get("metrics", [])
        final_metrics = history.get("final_metrics", {})
    elif "history" in checkpoint:
        # History embedded in checkpoint
        history = checkpoint["history"]
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        metrics_history = history.get("metrics", [])
        final_metrics = history.get("final_metrics", {})

    # Get epoch and val_loss from checkpoint
    epoch = checkpoint.get("epoch", len(train_loss))
    best_val_loss = checkpoint.get("val_loss", val_loss[-1] if val_loss else 0.0)

    return VQVAECheckpointData(
        input_dim=config.get("input_dim", len(feature_names)),
        group_embedding_dim=config.get("group_embedding_dim", 256),
        group_hidden_dim=config.get("group_hidden_dim", 512),
        num_categories=num_categories,
        num_levels=num_levels,
        group_indices=group_indices,
        category_names=category_names,
        levels=levels,
        feature_names=feature_names,
        feature_families=feature_families,
        normalization_stats=normalization_stats,
        train_loss=train_loss,
        val_loss=val_loss,
        metrics_history=metrics_history,
        final_metrics=final_metrics,
        model_state_dict=checkpoint.get("model_state_dict"),
        epoch=epoch,
        best_val_loss=best_val_loss,
    )


def get_category_feature_mapping(
    data: VQVAECheckpointData,
) -> Dict[str, List[str]]:
    """Get mapping from category names to feature names.

    Args:
        data: Loaded checkpoint data

    Returns:
        Dict mapping category name to list of feature names in that category
    """
    mapping = {}
    for cat in data.category_names:
        indices = data.group_indices[cat]
        names = [data.feature_names[i] for i in indices if i < len(data.feature_names)]
        mapping[cat] = names
    return mapping


def get_abbreviated_feature_name(full_name: str) -> str:
    """Convert full feature name to abbreviated form.

    Examples:
        "summary::summary_0" -> "sum_0"
        "temporal::temporal_5" -> "tmp_5"
        "architecture::architecture_3" -> "arch_3"
    """
    if "::" in full_name:
        family, name = full_name.split("::", 1)
        # Abbreviate family prefix
        family_abbrev = {
            "summary": "sum",
            "temporal": "tmp",
            "architecture": "arch",
        }.get(family, family[:3])

        # Abbreviate feature name
        if name.startswith(family):
            # Remove redundant family prefix in name
            name = name[len(family) :].lstrip("_")

        return f"{family_abbrev}_{name}"
    return full_name


def extract_utilization_matrix(
    data: VQVAECheckpointData,
) -> tuple[np.ndarray, List[str], List[str]]:
    """Extract utilization values as a matrix for heatmap visualization.

    Args:
        data: Loaded checkpoint data

    Returns:
        Tuple of (matrix, category_labels, level_labels)
        - matrix: [num_categories, num_levels] utilization values
        - category_labels: list of category names
        - level_labels: list of level names ("L0", "L1", "L2")
    """
    num_cats = data.num_categories
    num_levels = data.num_levels

    matrix = np.zeros((num_cats, num_levels))
    level_labels = [f"L{i}" for i in range(num_levels)]

    for i, cat in enumerate(data.category_names):
        for level in range(num_levels):
            key = f"{cat}/level_{level}/utilization"
            if key in data.final_metrics:
                # Cap at 1.0 to handle legacy metrics with utilization > 1.0
                matrix[i, level] = min(data.final_metrics[key], 1.0)

    return matrix, data.category_names, level_labels


def extract_utilization_counts(
    data: VQVAECheckpointData,
) -> Dict[str, Dict[int, tuple[int, int]]]:
    """Extract utilization as N/M counts (used codes / codebook size).

    Args:
        data: Loaded checkpoint data

    Returns:
        Dict mapping category name to {level_idx: (used_codes, codebook_size)}
        Example: {"cluster_1": {0: (15, 24), 1: (7, 13), 2: (2, 6)}}
    """
    result: Dict[str, Dict[int, tuple[int, int]]] = {}

    for cat in data.category_names:
        result[cat] = {}
        cat_levels = data.levels.get(cat, [])

        for level_idx in range(data.num_levels):
            # Get utilization from metrics
            util_key = f"{cat}/level_{level_idx}/utilization"
            utilization = data.final_metrics.get(util_key, 0.0)

            # Get codebook size from level config
            if level_idx < len(cat_levels):
                codebook_size = cat_levels[level_idx].get("num_tokens", 0)
            else:
                codebook_size = 0

            # Compute used codes: N = utilization × M (round to nearest int)
            # Cap at codebook_size to handle legacy metrics with utilization > 1.0
            used_codes = round(utilization * codebook_size) if codebook_size > 0 else 0
            used_codes = min(used_codes, codebook_size)  # N cannot exceed M

            result[cat][level_idx] = (used_codes, codebook_size)

    return result


def extract_reconstruction_mse(data: VQVAECheckpointData) -> Dict[str, float]:
    """Extract per-category reconstruction MSE.

    Args:
        data: Loaded checkpoint data

    Returns:
        Dict mapping category name to reconstruction MSE
    """
    mse = {}
    for cat in data.category_names:
        key = f"{cat}/reconstruction_mse"
        if key in data.final_metrics:
            mse[cat] = data.final_metrics[key]
    return mse


def compute_category_semantics(data: VQVAECheckpointData) -> Dict[str, str]:
    """Compute semantic labels for categories based on dominant features.

    Args:
        data: Loaded checkpoint data

    Returns:
        Dict mapping category name to semantic label
    """
    semantics = {}

    for cat in data.category_names:
        indices = data.group_indices[cat]
        if not indices:
            semantics[cat] = "empty"
            continue

        # Count feature families in this category
        family_counts: Dict[str, int] = {}
        for idx in indices:
            if idx < len(data.feature_names):
                name = data.feature_names[idx]
                if "::" in name:
                    family = name.split("::")[0]
                else:
                    family = "unknown"
                family_counts[family] = family_counts.get(family, 0) + 1

        # Use dominant family as semantic label
        if family_counts:
            dominant = max(family_counts, key=lambda k: family_counts[k])
            count = family_counts[dominant]
            total = len(indices)
            if count == total:
                semantics[cat] = dominant
            else:
                pct = int(100 * count / total)
                semantics[cat] = f"{dominant} ({pct}%)"
        else:
            semantics[cat] = "mixed"

    return semantics
