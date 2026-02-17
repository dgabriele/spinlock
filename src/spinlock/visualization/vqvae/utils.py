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


def _is_pyramid_model(category_names: List[str]) -> bool:
    """Detect pyramid encoding via temporal_p0, temporal_p1, etc. in category names.

    Args:
        category_names: List of category names from checkpoint

    Returns:
        True if pyramid temporal encoding detected
    """
    return any('temporal_p' in cat for cat in category_names)


def _get_pyramid_levels(category_names: List[str]) -> List[str]:
    """Extract unique pyramid level prefixes from category names.

    Args:
        category_names: List of category names from checkpoint
            (e.g., ['temporal_p0_cluster_1', 'temporal_p0_cluster_2', 'temporal_p1_cluster_1', ...])

    Returns:
        List of unique pyramid level prefixes (e.g., ['temporal_p0', 'temporal_p1', ...])
    """
    pyramid_prefixes = set()
    for cat in category_names:
        if 'temporal_p' in cat:
            # Extract pyramid level prefix (e.g., 'temporal_p0' from 'temporal_p0_cluster_1')
            # Split by underscore and take first two parts
            parts = cat.split('_')
            if len(parts) >= 2:
                # Find the part with 'p' followed by digit
                for i, part in enumerate(parts):
                    if part.startswith('p') and len(part) > 1 and part[1].isdigit():
                        # Join parts up to and including this part
                        prefix = '_'.join(parts[:i+1])
                        pyramid_prefixes.add(prefix)
                        break

    # Sort by pyramid level number (p0, p1, p2, p3)
    sorted_prefixes = sorted(list(pyramid_prefixes), key=lambda x: int(x.split('_p')[1][0]))
    return sorted_prefixes


def _get_variable_length_config(checkpoint: Dict) -> Optional[Dict]:
    """Extract variable-length config from checkpoint model_config.

    Args:
        checkpoint: Loaded checkpoint dictionary

    Returns:
        Variable-length config dict if enabled, None otherwise
    """
    # Extract config - try model_config first, then config
    if "model_config" in checkpoint:
        config = checkpoint["model_config"]
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        return None

    # Navigate to families -> temporal -> encoder_params -> variable_length
    families = config.get("families", {})
    temporal = families.get("temporal", {})
    encoder_params = temporal.get("encoder_params", {})
    vl_config = encoder_params.get("variable_length", {})

    # Check if enabled
    if vl_config.get("enabled", False):
        return vl_config

    return None


def extract_pyramid_metrics(
    final_metrics: Dict[str, float],
    pyramid_levels: List[str],
    category_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Extract per-pyramid-level metrics from final_metrics.

    Args:
        final_metrics: Dictionary of all metrics from checkpoint
        pyramid_levels: List of pyramid level prefixes (e.g., ['temporal_p0', 'temporal_p1', ...])
        category_names: All category names (to find which belong to each pyramid level)

    Returns:
        Dict with structure: {
            'p0': {'utilization': [[u00, u01, u02], [u10, u11, u12], ...], 'mse': [X1, X2, ...], 'categories': [...]},
            'p1': {...}, 'p2': {...}, 'p3': {...}
        }
    """
    pyramid_metrics = {}

    for level_prefix in pyramid_levels:
        # Extract level number (e.g., 'temporal_p0' -> 'p0')
        level_key = level_prefix.split('temporal_')[1]

        # Find all categories that belong to this pyramid level
        level_categories = [cat for cat in category_names if cat.startswith(level_prefix)]

        # Collect metrics across all categories in this level
        all_utilization = []
        all_mse = []

        for cat_name in level_categories:
            # Extract utilization across all VQ levels for this category
            cat_utilization = []
            for vq_level in range(3):  # Assume 3 VQ levels (L0, L1, L2)
                key = f"{cat_name}/level_{vq_level}/utilization"
                if key in final_metrics:
                    cat_utilization.append(min(final_metrics[key], 1.0))  # Cap at 1.0

            if cat_utilization:
                all_utilization.append(cat_utilization)

            # Extract reconstruction MSE
            mse_key = f"{cat_name}/reconstruction_mse"
            if mse_key in final_metrics:
                all_mse.append(final_metrics[mse_key])

        pyramid_metrics[level_key] = {
            'utilization': all_utilization,
            'mse': all_mse,
            'categories': level_categories,
            'avg_mse': float(np.mean(all_mse)) if all_mse else None,
        }

    return pyramid_metrics


def extract_variable_length_metrics(final_metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract variable-length training metrics from final_metrics.

    Args:
        final_metrics: Dictionary of all metrics from checkpoint

    Returns:
        Dict with VL metrics if available: {
            'length_distribution': {16: N, 32: N, ...},
            'per_length_quality': {16: Q, 32: Q, ...},
            'active_levels_mean': X,
            'masking_efficiency': Y
        }
        None if no VL metrics found
    """
    # Check if VL metrics exist in final_metrics
    has_vl_metrics = any(k.startswith('vl_') for k in final_metrics.keys())

    if not has_vl_metrics:
        return None

    vl_metrics = {}

    # Extract length distribution
    if 'vl_length_distribution' in final_metrics:
        vl_metrics['length_distribution'] = final_metrics['vl_length_distribution']

    # Extract per-length quality
    if 'vl_per_length_quality' in final_metrics:
        vl_metrics['per_length_quality'] = final_metrics['vl_per_length_quality']

    # Extract active levels by length
    if 'vl_active_levels_by_length' in final_metrics:
        vl_metrics['active_levels_by_length'] = final_metrics['vl_active_levels_by_length']

    # Extract masking efficiency
    if 'vl_masking_efficiency' in final_metrics:
        vl_metrics['masking_efficiency'] = final_metrics['vl_masking_efficiency']

    return vl_metrics if vl_metrics else None


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
    loss_components_history: List[Dict[str, float]]  # per-epoch loss components
    final_metrics: Dict[str, float]

    # Model state
    model_state_dict: Optional[Dict] = None
    epoch: int = 0
    best_val_loss: float = 0.0

    # Pyramid encoding fields
    is_pyramid: bool = False
    pyramid_levels: List[str] = field(default_factory=list)  # ['temporal_p0', 'temporal_p1', ...]
    pyramid_metrics: Optional[Dict] = None

    # Variable-length fields
    vl_enabled: bool = False
    vl_config: Optional[Dict] = None
    vl_metrics: Optional[Dict] = None


def find_checkpoint_file(checkpoint_dir: Path) -> Path:
    """Find the checkpoint file in a directory.

    Supports multiple naming conventions:
    - final_model.pt / best_model.pt (legacy)
    - vq_tokenizer_final.pt / vq_tokenizer_best.pt (new)

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint file is found
    """
    # Prefer final_model.pt (has feature_names), fall back to best_model.pt
    # Also support vq_tokenizer_* naming convention
    candidates = [
        checkpoint_dir / "final_model.pt",
        checkpoint_dir / "vq_tokenizer_final.pt",
        checkpoint_dir / "best_model.pt",
        checkpoint_dir / "vq_tokenizer_best.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")


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

    # Find checkpoint file (supports multiple naming conventions)
    model_path = find_checkpoint_file(checkpoint_dir)

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

    # Extract levels configuration from model_state_dict (embedding shapes)
    levels = {}
    model_state = checkpoint.get("model_state_dict", {})

    # Parse embedding weight shapes to get codebook sizes
    # Format: quantizers.{category}_L{level}.embedding.weight -> [codebook_size, embedding_dim]
    for key, tensor in model_state.items():
        if "quantizers." in key and ".embedding.weight" in key:
            # Extract category and level from key
            # Example: "quantizers.temporal_group_1_L0.embedding.weight"
            parts = key.split(".")
            if len(parts) >= 2:
                quantizer_name = parts[1]  # "temporal_group_1_L0"
                if "_L" in quantizer_name:
                    cat_and_level = quantizer_name.rsplit("_L", 1)
                    if len(cat_and_level) == 2:
                        category = cat_and_level[0]
                        level_str = cat_and_level[1]
                        try:
                            level_idx = int(level_str)
                            codebook_size = tensor.shape[0]
                            embedding_dim = tensor.shape[1]

                            if category not in levels:
                                levels[category] = []

                            # Ensure levels list is large enough
                            while len(levels[category]) <= level_idx:
                                levels[category].append({})

                            levels[category][level_idx] = {
                                "num_tokens": codebook_size,
                                "embedding_dim": embedding_dim,
                            }
                        except ValueError:
                            pass

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

    # First, try to infer families from group_indices keys
    # Keys follow pattern: "family_group_N" (e.g., "temporal_group_1", "initial_group_2", "theta_group_1")
    detected_families = set()
    for key in group_indices.keys():
        # Split on first underscore to get family prefix
        parts = key.split('_', 1)
        if len(parts) >= 2:
            family = parts[0]  # temporal, initial, theta, etc.
            detected_families.add(family)

    if detected_families:
        # Map features to families based on group_indices
        for family in detected_families:
            feature_families[family] = []

        # Assign feature indices to their families
        for group_name, indices in group_indices.items():
            family = group_name.split('_', 1)[0]
            if family in feature_families:
                feature_families[family].extend(indices)

        # Sort indices within each family
        for family in feature_families:
            feature_families[family] = sorted(feature_families[family])
    else:
        # Fallback: try to get family info from checkpoint's 'families' key
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
    loss_components_history = []
    final_metrics = {}

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        metrics_history = history.get("metrics", [])
        loss_components_history = history.get("loss_components", [])
        final_metrics = history.get("final_metrics", {})
    elif "history" in checkpoint:
        # History embedded in checkpoint
        history = checkpoint["history"]
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        metrics_history = history.get("metrics", [])
        loss_components_history = history.get("loss_components", [])
        final_metrics = history.get("final_metrics", {})
    elif "metadata" in checkpoint and "training_history" in checkpoint["metadata"]:
        # VQTokenizer format: metadata.training_history
        history = checkpoint["metadata"]["training_history"]
        train_loss = history.get("train_losses", [])
        val_loss_sparse = history.get("val_losses", [])
        metrics_history = history.get("val_metrics", [])

        # Align val_loss with train_loss (validation happens every N epochs)
        # Infer validation frequency from lengths
        if len(train_loss) > 0 and len(val_loss_sparse) > 0:
            val_freq = max(1, len(train_loss) // len(val_loss_sparse))
            val_loss = []
            val_idx = 0
            for epoch_idx in range(len(train_loss)):
                if (epoch_idx + 1) % val_freq == 0 and val_idx < len(val_loss_sparse):
                    val_loss.append(val_loss_sparse[val_idx])
                    val_idx += 1
                elif val_loss:
                    # Interpolate or repeat last value
                    val_loss.append(val_loss[-1])
                else:
                    val_loss.append(train_loss[epoch_idx])  # Fallback to train loss
        else:
            val_loss = val_loss_sparse

        # Extract loss components from val_metrics
        loss_components_history = []
        for val_metric in metrics_history:
            components = {
                "reconstruction": val_metric.get("reconstruction", 0.0),
                "vq": val_metric.get("vq", 0.0),
                "orthogonality": val_metric.get("orthogonality", 0.0),
                "informativeness": val_metric.get("informativeness", 0.0),
                "topographic": val_metric.get("topographic", 0.0),
            }
            loss_components_history.append(components)

        final_metrics = history.get("final_metrics", {})

        # If final_metrics doesn't exist, construct from last validation metrics
        if not final_metrics and metrics_history:
            last_val = metrics_history[-1]
            per_quantizer_util = last_val.get("per_quantizer_utilization", {})
            per_category_mse = last_val.get("per_category_mse", {})

            # Reformat to visualization format
            for quantizer_name, utilization in per_quantizer_util.items():
                parts = quantizer_name.rsplit('_L', 1)
                if len(parts) == 2:
                    category = parts[0]
                    level = parts[1]
                    try:
                        level_num = int(level)
                        key = f"{category}/level_{level_num}/utilization"
                        # Convert from percentage (0-100) to fraction (0-1)
                        final_metrics[key] = utilization / 100.0
                    except ValueError:
                        final_metrics[f"{quantizer_name}/utilization"] = utilization / 100.0
                else:
                    final_metrics[f"{quantizer_name}/utilization"] = utilization / 100.0

            # Add per-category MSE
            for category, mse in per_category_mse.items():
                final_metrics[f"{category}/reconstruction_mse"] = mse

    # If no history file, try to get final_metrics from checkpoint directly
    if not final_metrics and "metrics" in checkpoint:
        final_metrics = checkpoint["metrics"]

    # Get epoch and val_loss from checkpoint
    epoch = checkpoint.get("epoch", len(train_loss))
    best_val_loss = checkpoint.get("val_loss", val_loss[-1] if val_loss else 0.0)

    # Detect pyramid encoding
    is_pyramid = _is_pyramid_model(category_names)
    pyramid_levels = _get_pyramid_levels(category_names) if is_pyramid else []
    pyramid_metrics = extract_pyramid_metrics(final_metrics, pyramid_levels, category_names) if is_pyramid else None

    # Detect variable-length mode
    vl_config = _get_variable_length_config(checkpoint)
    vl_enabled = vl_config is not None
    vl_metrics = extract_variable_length_metrics(final_metrics) if vl_enabled else None

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
        loss_components_history=loss_components_history,
        final_metrics=final_metrics,
        model_state_dict=checkpoint.get("model_state_dict"),
        epoch=epoch,
        best_val_loss=best_val_loss,
        # Pyramid fields
        is_pyramid=is_pyramid,
        pyramid_levels=pyramid_levels,
        pyramid_metrics=pyramid_metrics,
        # Variable-length fields
        vl_enabled=vl_enabled,
        vl_config=vl_config,
        vl_metrics=vl_metrics,
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
            "theta": "θ",  # Greek theta symbol
            "initial": "ic",
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

    checkpoint_dir = Path(checkpoint_dir)

    # Try to load metrics from checkpoint directly
    model_path = checkpoint_dir / "best_model.pt"
    if not model_path.exists():
        model_path = checkpoint_dir / "final_model.pt"

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})

        # Check if topo metrics were recorded during training
        if "topo_pre" in metrics and "topo_post" in metrics:
            topo_pre = metrics["topo_pre"]
            topo_post = metrics["topo_post"]
            # End-to-end is approximately pre * post (correlation chain)
            end_to_end = topo_pre * topo_post
            degradation = topo_pre - topo_post

            return {
                "pre_quantization": round(topo_pre, 4),
                "post_quantization": round(topo_post, 4),
                "end_to_end": round(end_to_end, 4),
                "quantization_degradation": round(degradation, 4),
            }

    # If no precomputed metrics, estimate from training quality
    # This is an approximation based on typical relationships
    checkpoint = torch.load(
        checkpoint_dir / "final_model.pt", map_location="cpu", weights_only=False
    )
    model_config = checkpoint.get("model_config", checkpoint.get("config", {}))

    # Load training history for quality estimate
    history_path = checkpoint_dir / "training_history.json"
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


# ============================================================================
# Roundtrip Consistency & Semantic Structure Metrics
# ============================================================================


def extract_roundtrip_metrics(data: VQVAECheckpointData) -> Dict[str, Any]:
    """Extract roundtrip consistency metrics from checkpoint.

    Returns:
        {
            'per_quantizer': Dict[str, float],  # roundtrip/{key} → MSE
            'per_family': {
                'theta': float,     # Average theta roundtrip loss
                'initial': float,   # Average initial roundtrip loss
                'temporal': float,  # Average temporal roundtrip loss
            },
            'overall': float,       # Total roundtrip loss
            'training_curve': List[float],  # Epoch-wise roundtrip from history
        }

    Data sources:
    - data.final_metrics['roundtrip/{quantizer_key}']
    - data.metrics_history[*]['roundtrip']
    """
    # Extract per-quantizer roundtrip losses
    per_quantizer = {}
    for key, value in data.final_metrics.items():
        if key.startswith("roundtrip/"):
            quantizer_name = key.split("roundtrip/", 1)[1]
            per_quantizer[quantizer_name] = value

    # Aggregate by family
    per_family = {}
    for family in data.feature_families.keys():
        family_losses = []
        for quantizer_name, loss in per_quantizer.items():
            # Check if quantizer belongs to this family (e.g., "theta_group_1_L0")
            if quantizer_name.lower().startswith(family.lower()):
                family_losses.append(loss)
        if family_losses:
            per_family[family] = float(np.mean(family_losses))

    # Overall roundtrip loss
    overall = float(np.mean(list(per_quantizer.values()))) if per_quantizer else 0.0

    # Extract training curve if available
    training_curve = []
    for epoch_metrics in data.metrics_history:
        if "roundtrip" in epoch_metrics:
            training_curve.append(epoch_metrics["roundtrip"])
        elif "roundtrip_loss" in epoch_metrics:
            training_curve.append(epoch_metrics["roundtrip_loss"])

    return {
        "per_quantizer": per_quantizer,
        "per_family": per_family,
        "overall": overall,
        "training_curve": training_curve,
    }


def compute_combinatorial_space_size(data: VQVAECheckpointData) -> Dict[str, Any]:
    """Compute size of combinatorial token space.

    Returns:
        {
            'total': int,                   # Product of all N (active codes)
            'per_category': Dict[str, int], # N for each category
            'per_level': {                  # Product across each level
                0: int,  # L0 product
                1: int,  # L1 product
                2: int,  # L2 product
            },
            'log_size': float,              # log10(total) for display
        }

    Calculation:
    - Extract utilization counts N/M per quantizer
    - Compute product of all N values
    """
    utilization_counts = extract_utilization_counts(data)

    # Per-category active codes (sum across levels)
    per_category = {}
    for cat, level_counts in utilization_counts.items():
        # Sum active codes across all levels for this category
        total_active = sum(used for used, _ in level_counts.values())
        per_category[cat] = total_active

    # Per-level products
    per_level = {}
    for level_idx in range(data.num_levels):
        level_product = 1
        for cat in data.category_names:
            if cat in utilization_counts and level_idx in utilization_counts[cat]:
                used_codes, _ = utilization_counts[cat][level_idx]
                # If no codes used, treat as 1 (doesn't contribute to diversity)
                level_product *= max(used_codes, 1)
        per_level[level_idx] = level_product

    # Total combinatorial space: product of all active codes
    total = 1
    for cat in data.category_names:
        cat_product = 1
        if cat in utilization_counts:
            for level_idx in range(data.num_levels):
                if level_idx in utilization_counts[cat]:
                    used_codes, _ = utilization_counts[cat][level_idx]
                    cat_product *= max(used_codes, 1)
        total *= cat_product

    # Log size for display
    import math
    log_size = math.log10(max(total, 1))

    return {
        "total": total,
        "per_category": per_category,
        "per_level": per_level,
        "log_size": log_size,
    }


def analyze_token_frequencies(
    tokenized_dataset_path: Path,
    data: VQVAECheckpointData,
    max_samples: int = 50000,
) -> Dict[str, Any]:
    """Analyze token usage patterns from pretokenized dataset.

    Args:
        tokenized_dataset_path: Path to pretokenized HDF5 (e.g., 50k_tokenized.h5)
        data: Checkpoint data for category structure
        max_samples: Limit analysis to first N samples

    Returns:
        {
            'frequency_distribution': np.ndarray,  # Histogram of pattern counts
            'unique_patterns': int,                # Number of unique tokenizations
            'entropy_per_category': Dict[str, float],  # Shannon entropy
            'top_patterns': List[Tuple[tuple, int]],  # Most common (tokens, count)
            'singleton_patterns': int,             # Patterns appearing once
        }

    Data source:
    - Load tokens from tokenized_dataset['tokens/{quantizer_key}'][:]
    - Compute frequency histogram across all samples
    """
    import h5py
    from collections import Counter

    with h5py.File(tokenized_dataset_path, "r") as f:
        tokens_group = f["tokens"]

        # Load all tokens for each quantizer
        all_tokens = {}
        for cat in data.category_names:
            for level in range(data.num_levels):
                key = f"{cat}_L{level}"
                if key in tokens_group:
                    tokens = tokens_group[key][:max_samples]
                    all_tokens[key] = tokens

        # Create composite patterns (tuple of all token assignments per sample)
        n_samples = min(max_samples, len(next(iter(all_tokens.values()))))
        patterns = []
        for i in range(n_samples):
            pattern = tuple(all_tokens[key][i] for key in sorted(all_tokens.keys()))
            patterns.append(pattern)

        # Count pattern frequencies
        pattern_counts = Counter(patterns)
        unique_patterns = len(pattern_counts)
        singleton_patterns = sum(1 for count in pattern_counts.values() if count == 1)

        # Frequency distribution (how many patterns appear N times)
        frequency_counts = Counter(pattern_counts.values())
        frequency_distribution = np.array(
            [frequency_counts.get(i, 0) for i in range(1, max(frequency_counts.keys()) + 1)]
        )

        # Top patterns
        top_patterns = pattern_counts.most_common(10)

        # Per-category entropy (Shannon entropy of token distribution)
        entropy_per_category = {}
        for cat in data.category_names:
            cat_tokens = []
            for level in range(data.num_levels):
                key = f"{cat}_L{level}"
                if key in all_tokens:
                    cat_tokens.extend(all_tokens[key])

            # Compute Shannon entropy
            if cat_tokens:
                token_counts = Counter(cat_tokens)
                total = sum(token_counts.values())
                probs = np.array([count / total for count in token_counts.values()])
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropy_per_category[cat] = float(entropy)

    return {
        "frequency_distribution": frequency_distribution,
        "unique_patterns": unique_patterns,
        "entropy_per_category": entropy_per_category,
        "top_patterns": top_patterns,
        "singleton_patterns": singleton_patterns,
    }


def compute_token_cooccurrence(
    tokenized_dataset_path: Path,
    category_pairs: List[tuple[str, str]],
    max_samples: int = 50000,
) -> Dict[tuple[str, str], np.ndarray]:
    """Compute co-occurrence matrix for selected category pairs.

    Args:
        tokenized_dataset_path: Path to pretokenized HDF5
        category_pairs: List of (cat1, cat2) to analyze
                       (e.g., [('theta_group_1', 'initial_group_1')])
        max_samples: Limit analysis to first N samples

    Returns:
        Dict mapping (cat1, cat2) to co-occurrence matrix [N1 × N2]

    Shows: How often token pairs appear together (compositional structure)
    """
    import h5py

    cooccurrence_matrices = {}

    with h5py.File(tokenized_dataset_path, "r") as f:
        tokens_group = f["tokens"]

        for cat1, cat2 in category_pairs:
            # Use L0 tokens for simplicity (can extend to all levels)
            key1 = f"{cat1}_L0"
            key2 = f"{cat2}_L0"

            if key1 in tokens_group and key2 in tokens_group:
                tokens1 = tokens_group[key1][:max_samples]
                tokens2 = tokens_group[key2][:max_samples]

                # Determine matrix size
                max_token1 = int(tokens1.max()) + 1
                max_token2 = int(tokens2.max()) + 1

                # Build co-occurrence matrix
                cooccur = np.zeros((max_token1, max_token2), dtype=np.int32)
                for t1, t2 in zip(tokens1, tokens2):
                    cooccur[t1, t2] += 1

                cooccurrence_matrices[(cat1, cat2)] = cooccur

    return cooccurrence_matrices


def extract_level_composition(data: VQVAECheckpointData) -> Dict[int, Dict[str, Any]]:
    """Analyze per-level token composition across categories.

    Returns:
        {
            0: {
                'active_categories': int,  # Categories with L0 tokens
                'total_tokens': int,       # Sum of N across L0
                'avg_utilization': float,  # Mean N/M for L0
            },
            1: {...},
            2: {...},
        }
    """
    utilization_counts = extract_utilization_counts(data)

    composition = {}
    for level_idx in range(data.num_levels):
        active_categories = 0
        total_tokens = 0
        utilizations = []

        for cat in data.category_names:
            if cat in utilization_counts and level_idx in utilization_counts[cat]:
                used_codes, codebook_size = utilization_counts[cat][level_idx]

                if used_codes > 0:
                    active_categories += 1
                    total_tokens += used_codes

                if codebook_size > 0:
                    utilizations.append(used_codes / codebook_size)

        avg_utilization = float(np.mean(utilizations)) if utilizations else 0.0

        composition[level_idx] = {
            "active_categories": active_categories,
            "total_tokens": total_tokens,
            "avg_utilization": avg_utilization,
        }

    return composition


# ============================================================================
# Hierarchical Token Pattern Analysis - Rollout Similarity
# ============================================================================


def extract_rollout_token_sets(
    dataset_path: str | Path,
    family: str = "temporal",
) -> Dict[int, Dict[str, set]]:
    """Extract token sets for each rollout from pretokenized dataset.

    Args:
        dataset_path: Path to HDF5 pretokenized dataset with /tokens/{category}_L{level}
        family: Feature family to analyze ("temporal", "initial", "theta", "all")
                Use "all" to analyze all families together.

    Returns:
        Dict mapping rollout_idx → category → token_set

        Example structure:
        {
            0: {"temporal_group_0_L0": {5, 12}, "temporal_group_0_L1": {3}, ...},
            1: {"temporal_group_0_L0": {7, 15}, "temporal_group_0_L1": {2}, ...},
            ...
        }
    """
    import h5py

    dataset_path = Path(dataset_path)
    rollout_tokens: Dict[int, Dict[str, set]] = {}

    with h5py.File(dataset_path, "r") as f:
        tokens_group = f["tokens"]

        # Find all token keys matching family pattern
        token_keys = []
        for key in tokens_group.keys():
            # Match pattern: "{family}_group_{id}_L{level}"
            if family == "all":
                # Include all token keys
                if "_group_" in key and "_L" in key:
                    token_keys.append(key)
            else:
                # Filter by specific family
                if key.startswith(f"{family}_group_") and "_L" in key:
                    token_keys.append(key)

        if not token_keys:
            raise ValueError(f"No token keys found for family '{family}' in {dataset_path}")

        # Load first key to determine number of samples
        first_key = token_keys[0]
        n_samples = tokens_group[first_key].shape[0]

        # Initialize rollout_tokens structure
        for rollout_idx in range(n_samples):
            rollout_tokens[rollout_idx] = {}

        # Load tokens for each category and populate sets
        for key in token_keys:
            token_array = tokens_group[key][:]  # [N] array

            for rollout_idx, token_id in enumerate(token_array):
                if rollout_idx not in rollout_tokens:
                    rollout_tokens[rollout_idx] = {}

                if key not in rollout_tokens[rollout_idx]:
                    rollout_tokens[rollout_idx][key] = set()

                rollout_tokens[rollout_idx][key].add(int(token_id))

    return rollout_tokens


def flatten_rollout_tokens(
    rollout_tokens: Dict[int, Dict[str, set]]
) -> Dict[int, set]:
    """Flatten token sets by concatenating all categories.

    For similarity computation, we treat each category's tokens as distinct
    by prefixing with category name (e.g., "temporal_group_0_L0_token5").

    Args:
        rollout_tokens: Nested dict from extract_rollout_token_sets()

    Returns:
        Dict mapping rollout_idx → flattened_token_set with globally unique tokens
    """
    flattened: Dict[int, set] = {}

    for rollout_idx, category_tokens in rollout_tokens.items():
        combined = set()

        for category_name, token_set in category_tokens.items():
            # Prefix each token with category name to make globally unique
            for token_id in token_set:
                unique_token = f"{category_name}_token{token_id}"
                combined.add(unique_token)

        flattened[rollout_idx] = combined

    return flattened


def compute_rollout_similarity(
    rollout_tokens_flat: Dict[int, set],
    metric: str = "jaccard",
) -> np.ndarray:
    """Compute pairwise similarity matrix between rollouts.

    Args:
        rollout_tokens_flat: Flattened token sets from flatten_rollout_tokens()
        metric: "jaccard" or "cosine"

    Returns:
        [N, N] similarity matrix where entry (i, j) = similarity(rollout_i, rollout_j)

    Notes:
        - Jaccard: |A ∩ B| / |A ∪ B| (set-based)
        - Cosine: dot(A, B) / (||A|| ||B||) (vector-based, treats tokens as binary features)
    """
    # Get rollout indices in sorted order
    rollout_indices = sorted(rollout_tokens_flat.keys())
    n_rollouts = len(rollout_indices)

    similarity_matrix = np.zeros((n_rollouts, n_rollouts))

    if metric == "jaccard":
        # Compute Jaccard similarity for each pair
        for i, idx_i in enumerate(rollout_indices):
            for j, idx_j in enumerate(rollout_indices):
                set_i = rollout_tokens_flat[idx_i]
                set_j = rollout_tokens_flat[idx_j]

                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)

                    if union > 0:
                        similarity_matrix[i, j] = intersection / union
                    else:
                        similarity_matrix[i, j] = 0.0

    elif metric == "cosine":
        # Build vocabulary of all unique tokens
        all_tokens = set()
        for token_set in rollout_tokens_flat.values():
            all_tokens.update(token_set)

        token_to_idx = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        vocab_size = len(token_to_idx)

        # Convert each rollout to binary vector
        vectors = np.zeros((n_rollouts, vocab_size))
        for i, rollout_idx in enumerate(rollout_indices):
            for token in rollout_tokens_flat[rollout_idx]:
                vectors[i, token_to_idx[token]] = 1

        # Compute cosine similarity using sklearn
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(vectors)

    else:
        raise ValueError(f"Unknown metric: {metric}. Choose 'jaccard' or 'cosine'.")

    return similarity_matrix


def hierarchical_cluster_rollouts(
    similarity_matrix: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """Perform hierarchical clustering on rollouts.

    Args:
        similarity_matrix: [N, N] similarity matrix
        method: Linkage method ("average", "ward", "complete", "single")

    Returns:
        Linkage matrix for scipy.cluster.hierarchy.dendrogram

    Notes:
        - Convert similarity → distance: distance = 1 - similarity
        - Use scipy.cluster.hierarchy.linkage
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix

    # Extract upper triangle as condensed distance matrix
    # scipy linkage expects condensed form (1D array of upper triangle)
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)

    return linkage_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Physics-alignment diagnostics (Data Layer)
# ──────────────────────────────────────────────────────────────────────────────


def load_token_matrix(tokens_h5_path: str) -> tuple:
    """Load all token categories as [N, n_cats] int32 array + sorted key list.

    Args:
        tokens_h5_path: Path to pretokenized HDF5 (tokens group with one dataset per category)

    Returns:
        (tokens [N, n_cats] int32, sorted list of category keys)
    """
    import h5py

    with h5py.File(tokens_h5_path, "r") as f:
        tokens_group = f["tokens"]
        token_keys = sorted(tokens_group.keys())
        n_samples = tokens_group[token_keys[0]].shape[0]

        print(f"  Loading token matrix: {n_samples:,} samples × {len(token_keys)} categories")
        tokens = np.stack(
            [tokens_group[k][:] for k in token_keys], axis=1
        ).astype(np.int32)

    print(f"  ✓ Token matrix shape: {tokens.shape}  ({tokens.nbytes / 1e6:.1f} MB)")
    return tokens, list(token_keys)


def load_physics_params(
    params_h5_path: str, param_names: Optional[List[str]] = None
) -> tuple:
    """Load physical parameters [N, P] float32 + infer/return param names.

    Expects dataset at /parameters/params or /params (falls back in order).

    Args:
        params_h5_path: Path to HDF5 with physical parameters
        param_names: Optional list of parameter names; auto-inferred if None

    Returns:
        (params [N, P] float32, list of param names)
    """
    import h5py

    with h5py.File(params_h5_path, "r") as f:
        # Try common locations for parameter data
        if "parameters" in f and "params" in f["parameters"]:
            raw = f["parameters/params"][:]
            if param_names is None:
                # Try to load names from metadata
                if "parameters" in f and "param_names" in f["parameters"]:
                    param_names = [
                        n.decode() if isinstance(n, bytes) else n
                        for n in f["parameters/param_names"][:]
                    ]
        elif "params" in f:
            raw = f["params"][:]
        else:
            raise ValueError(f"No parameter dataset found in {params_h5_path}")

    params = raw.astype(np.float32)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(params.shape[1])]

    print(f"  ✓ Physics params: {params.shape}  names={param_names[:5]}{'...' if len(param_names) > 5 else ''}")
    return params, param_names


def compute_discrimination_ratios(
    tokens: np.ndarray,
    params: np.ndarray,
    param_names: List[str],
    subsample: int = 5000,
    n_quintiles: int = 5,
) -> Dict[str, float]:
    """Compute discrimination ratio per physical parameter.

    For each parameter, split samples into quintile groups, then compute:
      ratio = mean_between_group_distance / mean_within_group_distance

    ratio < 1.0: token space discriminates samples that differ on this parameter
    ratio ≈ 1.0: token space is indifferent to this parameter

    Uses Hamming distance between token vectors.

    Args:
        tokens: [N, n_cats] int32 token array
        params: [N, P] float32 physical parameters
        param_names: Names for each parameter column
        subsample: Max samples to use (random subsample for speed)
        n_quintiles: Number of quantile groups to split into

    Returns:
        Dict mapping param_name → discrimination ratio
    """
    from sklearn.metrics import pairwise_distances

    N = tokens.shape[0]
    if N > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, subsample, replace=False)
        tokens_sub = tokens[idx]
        params_sub = params[idx]
    else:
        tokens_sub = tokens
        params_sub = params

    print(f"  Computing Hamming distance matrix ({len(tokens_sub):,} × {len(tokens_sub):,})...")
    # Normalize tokens to [0,1] range for hamming distance (hamming = fraction of positions that differ)
    dist_matrix = pairwise_distances(tokens_sub, metric="hamming").astype(np.float32)
    print(f"  ✓ Distance matrix computed")

    ratios = {}
    for p_idx, p_name in enumerate(param_names):
        p_vals = params_sub[:, p_idx]
        # Skip degenerate params
        if np.std(p_vals) < 1e-10:
            ratios[p_name] = 1.0
            continue

        # Assign quintile groups
        quantiles = np.percentile(p_vals, np.linspace(0, 100, n_quintiles + 1))
        groups = np.digitize(p_vals, quantiles[1:-1])  # 0..n_quintiles-1

        within_dists = []
        between_dists = []
        for g in range(n_quintiles):
            mask_g = groups == g
            if mask_g.sum() < 2:
                continue
            # Within-group distances (upper triangle)
            d_block = dist_matrix[np.ix_(mask_g, mask_g)]
            n_g = mask_g.sum()
            triu_idx = np.triu_indices(n_g, k=1)
            within_dists.extend(d_block[triu_idx].tolist())
            # Between-group: this group vs all others
            mask_other = ~mask_g
            if mask_other.sum() > 0:
                d_cross = dist_matrix[np.ix_(mask_g, mask_other)]
                between_dists.extend(d_cross.ravel().tolist())

        mean_within = np.mean(within_dists) if within_dists else 1.0
        mean_between = np.mean(between_dists) if between_dists else 1.0
        ratios[p_name] = float(mean_between / (mean_within + 1e-12))

    return ratios


def compute_category_param_mi(
    tokens: np.ndarray,
    params: np.ndarray,
    param_names: List[str],
    n_bins: int = 5,
) -> tuple:
    """Compute mutual information matrix between token categories and parameters.

    Each token category is discrete (code index); each parameter is binned
    into n_bins equal-width bins before computing MI.

    Args:
        tokens: [N, n_cats] int32
        params: [N, P] float32
        param_names: Names for each parameter column
        n_bins: Number of bins for continuous parameter discretization

    Returns:
        (mi_raw [n_cats, P], mi_norm [n_cats, P]) — raw and normalized MI matrices.
        mi_norm is normalized per-parameter by H(param) so values are in [0, 1].
    """
    from sklearn.metrics import mutual_info_score

    n_cats = tokens.shape[1]
    n_params = params.shape[1]
    mi_raw = np.zeros((n_cats, n_params), dtype=np.float32)

    print(f"  Computing MI matrix: {n_cats} categories × {n_params} parameters...")

    # Bin each parameter
    params_binned = np.zeros_like(params, dtype=np.int32)
    for p in range(n_params):
        p_vals = params[:, p]
        if np.std(p_vals) < 1e-10:
            params_binned[:, p] = 0
        else:
            bins = np.percentile(p_vals, np.linspace(0, 100, n_bins + 1))
            params_binned[:, p] = np.clip(np.digitize(p_vals, bins[1:-1]), 0, n_bins - 1)

    for c in range(n_cats):
        for p in range(n_params):
            mi_raw[c, p] = mutual_info_score(tokens[:, c], params_binned[:, p])

    # Normalize per-parameter by H(param)
    mi_norm = np.zeros_like(mi_raw)
    for p in range(n_params):
        counts = np.bincount(params_binned[:, p], minlength=n_bins)
        probs = counts / counts.sum()
        h_param = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
        if h_param > 1e-10:
            mi_norm[:, p] = mi_raw[:, p] / h_param
        # else: param is constant, MI stays 0

    print(f"  ✓ MI matrix computed. Max normalized MI: {mi_norm.max():.4f}")
    return mi_raw, mi_norm


def compute_knn_param_recovery(
    tokens: np.ndarray,
    params: np.ndarray,
    k: int = 10,
    test_frac: float = 0.2,
) -> np.ndarray:
    """Predict physical parameters from tokens using KNN regression (Hamming distance).

    Tests whether token combinations carry enough information to reconstruct
    the physical parameters that generated the QBM.

    Args:
        tokens: [N, n_cats] int32
        params: [N, P] float32
        k: Number of nearest neighbors
        test_frac: Fraction of data held out for testing

    Returns:
        R² per parameter [P], R²=1 perfect, R²=0 no better than mean
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    N = tokens.shape[0]
    n_params = params.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        tokens.astype(np.float32), params, test_size=test_frac, random_state=42
    )

    print(f"  KNN parameter recovery: train={len(X_train):,}, test={len(X_test):,}, k={k}")

    # Use hamming metric for integer token vectors
    knn = KNeighborsRegressor(n_neighbors=k, metric="hamming", n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    r2_per_param = np.array([r2_score(y_test[:, p], y_pred[:, p]) for p in range(n_params)])
    print(f"  ✓ R² range: [{r2_per_param.min():.3f}, {r2_per_param.max():.3f}]")
    return r2_per_param


def classify_token_categories(
    tokens: np.ndarray,
    mi_norm: np.ndarray,
    mi_threshold: float = 0.05,
) -> List[str]:
    """Classify each token category by its role.

    Categories:
      'collapsed'           — only 1 unique code used (cannot carry any signal)
      'physics-informative' — ≥2 codes AND max normalized MI > mi_threshold
      'conserved'           — ≥2 codes but low MI (physics invariant, not a failure)

    Args:
        tokens: [N, n_cats] int32
        mi_norm: [n_cats, P] normalized MI matrix
        mi_threshold: Min MI to call a category physics-informative

    Returns:
        List of classification strings, one per category
    """
    n_cats = tokens.shape[1]
    classifications = []

    for c in range(n_cats):
        n_unique = len(np.unique(tokens[:, c]))
        max_mi = float(mi_norm[c].max()) if mi_norm.shape[1] > 0 else 0.0

        if n_unique <= 1:
            classifications.append("collapsed")
        elif max_mi > mi_threshold:
            classifications.append("physics-informative")
        else:
            classifications.append("conserved")

    return classifications


def compute_binned_similarity(
    tokenized_h5_path: str,
    n_bins: int = 5000,
    metric: str = "jaccard",
) -> tuple:
    """Load tokens, bin by consecutive groups, compute pairwise similarity matrix.

    Full pipeline replacing the scattered logic in scripts/visualize_binned_jaccard.py.

    For metric='jaccard': builds set union per bin, pairwise Jaccard loop.
    For metric='js': builds frequency distributions per bin, scipy cdist jensenshannon.

    Args:
        tokenized_h5_path: Path to pretokenized HDF5 dataset
        n_bins: Target number of bins (consecutive groups of samples)
        metric: 'jaccard' or 'js'

    Returns:
        (similarity_matrix [n_bins, n_bins] float32, bin_indices list of (start, end) tuples)
    """
    import h5py
    from scipy.spatial.distance import cdist

    if metric not in ("jaccard", "js"):
        raise ValueError(f"metric must be 'jaccard' or 'js', got {metric!r}")

    print(f"Loading tokens from {tokenized_h5_path}...")
    with h5py.File(tokenized_h5_path, "r") as f:
        tokens_group = f["tokens"]
        token_keys = sorted(tokens_group.keys())
        n_samples = tokens_group[token_keys[0]].shape[0]
        print(f"  Found {len(token_keys)} token categories, {n_samples:,} samples")

        if metric == "jaccard":
            # Load as sets of "key_code" strings
            token_sets = []
            for i in range(n_samples):
                sample_tokens = set()
                for key in token_keys:
                    sample_tokens.add(f"{key}_{tokens_group[key][i]}")
                token_sets.append(sample_tokens)
        else:
            # Load as integer array for JS
            arrays = np.stack(
                [tokens_group[k][:] for k in token_keys], axis=1
            ).astype(np.int32)

    # Bin
    bin_size = max(1, n_samples // n_bins)
    print(f"  Bin size: ~{bin_size} samples/bin")

    if metric == "jaccard":
        binned_sets, bin_indices = [], []
        for i in range(0, n_samples, bin_size):
            end_idx = min(i + bin_size, n_samples)
            bin_tokens: set = set()
            for j in range(i, end_idx):
                bin_tokens.update(token_sets[j])
            binned_sets.append(bin_tokens)
            bin_indices.append((i, end_idx))

        actual_n_bins = len(binned_sets)
        print(f"  Computing {actual_n_bins:,}×{actual_n_bins:,} Jaccard similarity matrix...")
        sim = np.zeros((actual_n_bins, actual_n_bins), dtype=np.float32)
        for i in range(actual_n_bins):
            if i % 100 == 0:
                print(f"    Progress: {i}/{actual_n_bins} ({100*i/actual_n_bins:.1f}%)")
            sim[i, i] = 1.0
            for j in range(i + 1, actual_n_bins):
                a, b = binned_sets[i], binned_sets[j]
                inter = len(a & b)
                union = len(a | b)
                v = inter / union if union > 0 else 0.0
                sim[i, j] = sim[j, i] = v

    else:  # js
        n_cats = arrays.shape[1]
        n_codes = int(arrays.max()) + 1
        actual_n_bins = (n_samples + bin_size - 1) // bin_size
        bin_freqs = np.zeros((actual_n_bins, n_cats * n_codes), dtype=np.float32)
        bin_indices = []

        for b, start in enumerate(range(0, n_samples, bin_size)):
            end = min(start + bin_size, n_samples)
            chunk = arrays[start:end]
            count = end - start
            for c in range(n_codes):
                bin_freqs[b, c * n_cats:(c + 1) * n_cats] = (chunk == c).sum(axis=0) / count
            bin_indices.append((start, end))

        row_sums = bin_freqs.sum(axis=1, keepdims=True)
        bin_freqs /= np.maximum(row_sums, 1e-12)

        print(f"  Computing {actual_n_bins:,}×{actual_n_bins:,} JS similarity matrix via cdist...")
        freqs = np.clip(bin_freqs, 1e-10, None)
        freqs /= freqs.sum(axis=1, keepdims=True)
        js_dist = cdist(freqs, freqs, metric="jensenshannon").astype(np.float32)
        sim = 1.0 - js_dist
        np.fill_diagonal(sim, 1.0)

    avg = float(np.mean(sim[np.triu_indices(len(sim), k=1)]))
    metric_label = "Jaccard" if metric == "jaccard" else "JS"
    print(f"  ✓ Average pairwise {metric_label} similarity: {avg:.3f}")
    return sim, bin_indices
