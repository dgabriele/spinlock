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
from matplotlib.colors import LinearSegmentedColormap


def get_utilization_cmap():
    """Get colormap for utilization heatmaps.

    Uses dark gray (#111111) to green gradient.
    Neutral coloring since low utilization isn't necessarily "bad" -
    it may indicate natural capacity discovery.
    """
    return LinearSegmentedColormap.from_list(
        "utilization",
        ["#111111", "#2d5a27", "#4a9c3f", "#6ece5a"],  # dark gray → green
        N=256
    )


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

    # Parse feature families from names (format: "family::name") or infer from checkpoint
    feature_families: Dict[str, List[int]] = {}

    # First, try to get family info from checkpoint's 'families' key
    families_config = checkpoint.get("families", config.get("families", {}))
    if families_config:
        # Config has explicit family definitions (e.g., initial, summary, temporal)
        for family_name in families_config.keys():
            feature_families[family_name.upper()] = []

        total_features = sum(len(v) for v in group_indices.values())
        family_names = list(families_config.keys())
        if family_names:
            for i in range(total_features):
                feature_families[family_names[i % len(family_names)].upper()].append(i)
    else:
        # Try to parse from feature names
        has_family_info = any("::" in name for name in feature_names)
        if has_family_info:
            for i, name in enumerate(feature_names):
                if "::" in name:
                    family = name.split("::")[0]
                else:
                    family = "behavioral"
                if family not in feature_families:
                    feature_families[family] = []
                feature_families[family].append(i)
        else:
            # No family info available - group all as "behavioral"
            # Better than "unknown" for models without ARCHITECTURE
            feature_families["behavioral"] = list(range(len(feature_names)))

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

    # If no history file, try to get final_metrics from checkpoint directly
    if not final_metrics and "metrics" in checkpoint:
        final_metrics = checkpoint["metrics"]

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


def _load_features_from_hdf5(
    dataset_path: Path,
    group_indices: Dict[str, List[int]],
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """Load and concatenate features from HDF5 in group_indices order.

    Supports the spinlock HDF5 structure:
    - /features/summary/aggregated/features [N, 360]
    - /features/temporal/features [N, 256, 63] -> aggregate to [N, D]
    - /features/architecture/aggregated/features [N, 12]
    - /features/initial/aggregated/features [N, 14]

    Args:
        dataset_path: Path to HDF5 dataset
        group_indices: Category -> feature indices mapping
        max_samples: Optional maximum samples to load

    Returns:
        Concatenated feature array [n_samples, total_dim]
    """
    import h5py

    # Compute total feature dimension
    total_dim = sum(len(indices) for indices in group_indices.values())

    with h5py.File(dataset_path, "r") as f:
        # Get features group
        features_grp = f.get("features", f)  # Support both /features/... and /...

        # Determine number of samples from first available dataset
        n_samples = None
        paths_to_check = [
            ("summary", "aggregated", "features"),
            ("temporal", "features"),
            ("architecture", "aggregated", "features"),
            ("initial", "aggregated", "features"),
        ]
        for path in paths_to_check:
            obj = features_grp
            for key in path:
                if key in obj:
                    obj = obj[key]
                else:
                    obj = None
                    break
            if obj is not None and isinstance(obj, h5py.Dataset):
                n_samples = obj.shape[0]
                break

        if n_samples is None:
            raise ValueError("Could not determine dataset size from HDF5 file")

        if max_samples:
            n_samples = min(n_samples, max_samples)

        # Pre-allocate output array
        features = np.zeros((n_samples, total_dim), dtype=np.float32)

        # Load each family's features in a consistent order
        col_offset = 0

        # Summary features (360D) - /features/summary/aggregated/features
        if "summary" in features_grp:
            summary_grp = features_grp["summary"]
            if "aggregated" in summary_grp and "features" in summary_grp["aggregated"]:
                data = summary_grp["aggregated"]["features"][:n_samples]
                features[:, col_offset:col_offset + data.shape[1]] = data
                col_offset += data.shape[1]

        # Temporal features - /features/temporal/features [N, 256, 63]
        # Need to aggregate from 3D to 2D (mean over trajectory dimension)
        if "temporal" in features_grp:
            temporal_grp = features_grp["temporal"]
            if "features" in temporal_grp:
                data = temporal_grp["features"][:n_samples]  # [N, 256, 63]
                # Aggregate: mean over the 256 time steps
                if len(data.shape) == 3:
                    data = data.mean(axis=1)  # [N, 63]
                features[:, col_offset:col_offset + data.shape[1]] = data
                col_offset += data.shape[1]

        # Architecture parameters (12D) - /features/architecture/aggregated/features
        if "architecture" in features_grp:
            arch_grp = features_grp["architecture"]
            if "aggregated" in arch_grp and "features" in arch_grp["aggregated"]:
                data = arch_grp["aggregated"]["features"][:n_samples]
                features[:, col_offset:col_offset + data.shape[1]] = data
                col_offset += data.shape[1]

        # Initial condition features (14D) - /features/initial/aggregated/features
        if "initial" in features_grp:
            initial_grp = features_grp["initial"]
            if "aggregated" in initial_grp and "features" in initial_grp["aggregated"]:
                data = initial_grp["aggregated"]["features"][:n_samples]
                features[:, col_offset:col_offset + data.shape[1]] = data
                col_offset += data.shape[1]

    return features[:, :col_offset]  # Return only the filled columns


def compute_topographic_metrics_from_checkpoint(
    checkpoint_dir: str | Path,
    dataset_path: Optional[str | Path] = None,
    n_samples: int = 1000,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute topographic metrics from a checkpoint.

    Loads precomputed topology metrics if available in the checkpoint.
    For models trained with topo_weight > 0, metrics are recorded during training.

    Args:
        checkpoint_dir: Path to checkpoint directory
        dataset_path: Optional path to dataset HDF5 file (not used currently)
        n_samples: Number of samples for metric computation (not used currently)
        device: Computation device (not used currently)

    Returns:
        Dict with topology metrics:
            - pre_quantization: Input → Latent correlation
            - post_quantization: Latent → Code correlation
            - end_to_end: Input → Code correlation
            - quantization_degradation: Pre - Post

    Note:
        Currently returns approximate values based on training metrics.
        Full topographic recomputation requires the encoder pipeline and
        is not supported in visualization mode.
    """
    import torch
    import json

    checkpoint_dir = Path(checkpoint_dir)

    # Try to load from training history
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        final_metrics = history.get("final_metrics", {})

        # Check if topology metrics were recorded during training
        # (would be stored if we add them to the training loop)
        if "topo_pre_quantization" in final_metrics:
            return {
                "pre_quantization": final_metrics.get("topo_pre_quantization", 0.0),
                "post_quantization": final_metrics.get("topo_post_quantization", 0.0),
                "end_to_end": final_metrics.get("topo_end_to_end", 0.0),
                "quantization_degradation": final_metrics.get("topo_degradation", 0.0),
            }

    # If no precomputed metrics, estimate from training quality
    # This is an approximation based on typical relationships
    checkpoint = torch.load(
        checkpoint_dir / "final_model.pt", map_location="cpu", weights_only=False
    )
    model_config = checkpoint.get("model_config", checkpoint.get("config", {}))

    # Load training history for quality estimate
    train_metrics = {}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        train_metrics = history.get("final_metrics", {})

    quality = train_metrics.get("quality", 0.9)
    utilization = train_metrics.get("utilization", 0.5)

    # Estimate topology preservation based on reconstruction quality
    # Higher quality typically correlates with better topology preservation
    estimated_pre = 0.8 * quality + 0.1  # Pre-quantization usually high
    estimated_post = 0.6 * quality + 0.2  # Post-quantization lower due to discretization
    estimated_e2e = 0.7 * quality + 0.1  # End-to-end in between
    degradation = estimated_pre - estimated_post

    # Mark as estimated
    return {
        "pre_quantization": round(estimated_pre, 4),
        "post_quantization": round(estimated_post, 4),
        "end_to_end": round(estimated_e2e, 4),
        "quantization_degradation": round(degradation, 4),
        "_estimated": True,  # Flag to indicate these are estimates
    }
