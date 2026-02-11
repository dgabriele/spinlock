"""Checkpoint save/load utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from pydantic import BaseModel, Field, ConfigDict

from .config import TokenizerConfig

logger = logging.getLogger(__name__)


class CleaningOperation(BaseModel):
    """Record of a single cleaning operation applied to features.

    Captures detailed information about what was done during feature cleaning,
    including which features were removed and the parameters used.
    """
    operation_type: str = Field(
        description="Type of cleaning operation: 'variance_filter', 'deduplication', 'nan_handling', 'outlier_capping'"
    )
    features_removed: List[int] = Field(
        description="Indices of features removed by this operation (in input space)"
    )
    features_kept: List[int] = Field(
        description="Indices of features kept by this operation (in input space)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for this operation (e.g., thresholds, methods)"
    )
    num_removed: int = Field(
        description="Number of features removed by this operation"
    )
    num_kept: int = Field(
        description="Number of features kept by this operation"
    )


class FeatureFamilyMetadata(BaseModel):
    """Metadata for a feature family (initial_manual, temporal, theta).

    Tracks the complete processing pipeline for a family of features,
    including original feature names, cleaning operations, and final state.
    """
    family_name: str = Field(
        description="Name of feature family: 'initial_manual', 'temporal', 'theta'"
    )
    original_feature_count: int = Field(
        description="Number of features before cleaning"
    )
    cleaned_feature_count: int = Field(
        description="Number of features after cleaning"
    )
    original_feature_names: List[str] = Field(
        description="Names of all original features before cleaning"
    )
    kept_feature_indices: List[int] = Field(
        description="Indices of features kept after cleaning (in original space)"
    )
    kept_feature_names: List[str] = Field(
        description="Names of features kept after cleaning"
    )
    removed_feature_indices: List[int] = Field(
        description="Indices of features removed by cleaning (in original space)"
    )
    removed_feature_names: List[str] = Field(
        description="Names of features removed by cleaning"
    )
    cleaning_operations: List[CleaningOperation] = Field(
        default_factory=list,
        description="Ordered list of cleaning operations applied"
    )


class CategoryMetadata(BaseModel):
    """Metadata for a feature category after grouping.

    Categories are created by grouping cleaned features into semantic groups
    (e.g., temporal_group_1_L0, temporal_group_1_L1, initial_manual_features).
    """
    category_name: str = Field(
        description="Name of this category (e.g., 'temporal_group_1_L0')"
    )
    feature_indices: List[int] = Field(
        description="Indices into cleaned feature space (after all cleaning)"
    )
    feature_names: List[str] = Field(
        description="Names of features in this category"
    )
    original_indices: List[int] = Field(
        description="Original indices before cleaning (for traceability)"
    )
    num_features: int = Field(
        description="Number of features in this category"
    )


class FeatureMetadata(BaseModel):
    """Complete feature processing metadata for a VQTokenizer.

    This is the root metadata structure that captures everything about
    how features were processed: cleaning, grouping, transformations.
    It provides a single source of truth for feature organization.
    """
    families: Dict[str, FeatureFamilyMetadata] = Field(
        description="Metadata for each feature family (temporal, initial_manual, theta)"
    )
    categories: Dict[str, CategoryMetadata] = Field(
        description="Metadata for each category after grouping"
    )
    total_original_features: int = Field(
        description="Total number of features before any cleaning"
    )
    total_cleaned_features: int = Field(
        description="Total number of features after cleaning, before grouping"
    )
    total_features_removed: int = Field(
        description="Total number of features removed by cleaning"
    )
    cleaning_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Feature cleaning configuration used"
    )
    grouping_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Feature grouping configuration used"
    )


class TokenizerCheckpoint(BaseModel):
    """Pydantic schema for VQTokenizer checkpoint data.

    Provides type-safe access to checkpoint contents with validation.
    """
    model_state_dict: Dict[str, Any] = Field(
        description="PyTorch model state dict"
    )
    config: TokenizerConfig = Field(
        description="Tokenizer configuration"
    )
    group_indices: Dict[str, list[int]] = Field(
        description="Feature group indices mapping"
    )
    normalization_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Feature normalization statistics"
    )
    temporal_input_dim: Optional[int] = Field(
        default=None,
        description="Temporal feature input dimension"
    )
    initial_input_dim: Optional[int] = Field(
        default=None,
        description="Initial feature input dimension"
    )
    feature_metadata: Optional[FeatureMetadata] = Field(
        default=None,
        description="Complete feature processing metadata (v2.1+)"
    )
    optimizer_state_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optimizer state dict (training only)"
    )
    epoch: Optional[int] = Field(
        default=None,
        description="Training epoch number"
    )
    val_loss: Optional[float] = Field(
        default=None,
        description="Validation loss"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (training history, etc.)"
    )
    version: str = Field(
        default="v2",
        description="Checkpoint format version"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow torch tensors in state_dict


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: TokenizerConfig,
    group_indices: Dict[str, list[int]],
    normalization_stats: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    metadata: Optional[Dict] = None,
    temporal_input_dim: Optional[int] = None,
    initial_input_dim: Optional[int] = None,
    feature_metadata: Optional[FeatureMetadata] = None,
):
    """Save tokenizer checkpoint in V2 format.

    Args:
        path: Output path for checkpoint
        model: VQTokenizer model
        config: Tokenizer configuration
        group_indices: Feature group indices mapping
        normalization_stats: Feature normalization statistics
        optimizer: Optimizer state (training only)
        epoch: Training epoch number
        val_loss: Validation loss
        metadata: Additional metadata (training history, etc.)
        temporal_input_dim: Temporal feature input dimension
        initial_input_dim: Initial feature input dimension
        feature_metadata: Complete feature processing metadata (NEW in v2.1)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.model_dump(),
        'group_indices': group_indices,
        'normalization_stats': normalization_stats,
        'temporal_input_dim': temporal_input_dim,
        'initial_input_dim': initial_input_dim,
        'version': 'v2',
    }

    # Add explicit dimension validation
    dimension_validation = {}
    if hasattr(model, 'initial_encoder'):
        from spinlock.tokens.encoders.initial import InitialHybridEncoder
        if isinstance(model.initial_encoder, InitialHybridEncoder):
            if hasattr(model.initial_encoder, 'manual_encoder'):
                dimension_validation['initial_manual_dim'] = model.initial_encoder.manual_encoder[0].in_features
    if hasattr(model, 'temporal_encoder') and hasattr(model.temporal_encoder, 'input_dim'):
        dimension_validation['temporal_input_dim'] = model.temporal_encoder.input_dim
    if hasattr(model, 'theta_encoder') and hasattr(model.theta_encoder, 'param_dim'):
        dimension_validation['theta_param_dim'] = model.theta_encoder.param_dim

    if dimension_validation:
        checkpoint['dimension_validation'] = dimension_validation
        logger.info(f"Saving dimension validation: {dimension_validation}")

    # Add feature metadata if provided (v2.1+)
    if feature_metadata is not None:
        checkpoint['feature_metadata'] = feature_metadata.model_dump()
        logger.info(f"Saving feature metadata: {feature_metadata.total_cleaned_features} features across {len(feature_metadata.families)} families")

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    if metadata is not None:
        checkpoint['metadata'] = metadata

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(path: Path) -> TokenizerCheckpoint:
    """Load tokenizer checkpoint with Pydantic validation.

    Args:
        path: Path to checkpoint file

    Returns:
        TokenizerCheckpoint with validated, type-safe data

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint version is not v2
        ValidationError: If checkpoint data is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    raw_checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    version = raw_checkpoint.get('version', 'v1')
    if version != 'v2':
        raise ValueError(
            f"Checkpoint version {version} not supported. Use V2 checkpoints only."
        )

    # Parse config as TokenizerConfig
    config = TokenizerConfig(**raw_checkpoint['config'])

    # Validate dimensions if available
    if 'dimension_validation' in raw_checkpoint:
        dims = raw_checkpoint['dimension_validation']
        logger.info(f"Checkpoint dimensions: {dims}")

    # Parse feature_metadata if present (v2.1+)
    feature_metadata = None
    if 'feature_metadata' in raw_checkpoint and raw_checkpoint['feature_metadata'] is not None:
        feature_metadata = FeatureMetadata(**raw_checkpoint['feature_metadata'])
        logger.info(f"Loaded feature metadata: {feature_metadata.total_cleaned_features} features across {len(feature_metadata.families)} families")
    else:
        logger.warning(f"Checkpoint does not contain feature_metadata (v2.0 format). Pretokenization will use fallback cleaning.")

    # Build TokenizerCheckpoint with validation
    return TokenizerCheckpoint(
        model_state_dict=raw_checkpoint['model_state_dict'],
        config=config,
        group_indices=raw_checkpoint['group_indices'],
        normalization_stats=raw_checkpoint.get('normalization_stats'),
        temporal_input_dim=raw_checkpoint.get('temporal_input_dim'),
        initial_input_dim=raw_checkpoint.get('initial_input_dim'),
        feature_metadata=feature_metadata,
        optimizer_state_dict=raw_checkpoint.get('optimizer_state_dict'),
        epoch=raw_checkpoint.get('epoch'),
        val_loss=raw_checkpoint.get('val_loss'),
        metadata=raw_checkpoint.get('metadata', {}),
        version=version,
    )


def inspect_checkpoint_features(checkpoint: TokenizerCheckpoint) -> str:
    """Generate human-readable summary of checkpoint feature metadata.

    Args:
        checkpoint: Loaded TokenizerCheckpoint

    Returns:
        Formatted string summary of feature organization
    """
    if checkpoint.feature_metadata is None:
        return "⚠ Checkpoint does not contain feature_metadata (v2.0 format)"

    meta = checkpoint.feature_metadata
    lines = []

    lines.append("=" * 80)
    lines.append("FEATURE METADATA SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\nTotal Features:")
    lines.append(f"  Original: {meta.total_original_features}")
    lines.append(f"  After Cleaning: {meta.total_cleaned_features}")
    lines.append(f"  Removed: {meta.total_features_removed} ({100*meta.total_features_removed/meta.total_original_features:.1f}%)")

    # Feature Families
    lines.append(f"\nFeature Families: {len(meta.families)}")
    for family_name, family in meta.families.items():
        lines.append(f"\n  {family_name.upper()}:")
        lines.append(f"    Original: {family.original_feature_count} features")
        lines.append(f"    Cleaned: {family.cleaned_feature_count} features")
        lines.append(f"    Removed: {len(family.removed_feature_indices)} features")

        # Cleaning operations
        if family.cleaning_operations:
            lines.append(f"    Cleaning Operations:")
            for i, op in enumerate(family.cleaning_operations, 1):
                lines.append(f"      {i}. {op.operation_type}: removed {op.num_removed} features")

        # Sample feature names
        if family.kept_feature_names:
            sample_names = family.kept_feature_names[:5]
            lines.append(f"    Sample Features (kept): {', '.join(sample_names)}")
            if len(family.kept_feature_names) > 5:
                lines.append(f"      ... and {len(family.kept_feature_names) - 5} more")

        if family.removed_feature_names:
            sample_removed = family.removed_feature_names[:3]
            lines.append(f"    Sample Features (removed): {', '.join(sample_removed)}")
            if len(family.removed_feature_names) > 3:
                lines.append(f"      ... and {len(family.removed_feature_names) - 3} more")

    # Categories
    lines.append(f"\nFeature Categories: {len(meta.categories)}")
    for cat_name, cat in sorted(meta.categories.items())[:10]:  # Show first 10
        lines.append(f"\n  {cat_name}:")
        lines.append(f"    Features: {cat.num_features}")
        lines.append(f"    Indices: {cat.feature_indices[:5]}{'...' if len(cat.feature_indices) > 5 else ''}")

    if len(meta.categories) > 10:
        lines.append(f"\n  ... and {len(meta.categories) - 10} more categories")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def export_feature_metadata_to_json(checkpoint: TokenizerCheckpoint, output_path: Path):
    """Export feature metadata to JSON for external analysis.

    Args:
        checkpoint: Loaded TokenizerCheckpoint
        output_path: Path to output JSON file
    """
    import json

    if checkpoint.feature_metadata is None:
        raise ValueError("Checkpoint does not contain feature_metadata")

    # Convert to dict and save
    metadata_dict = checkpoint.feature_metadata.model_dump()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)

    logger.info(f"Feature metadata exported to {output_path}")


def verify_pretrained_cnn(checkpoint_path: Path, embedding_dim: int) -> Dict:
    """Verify pretrained CNN checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained CNN checkpoint not found: {checkpoint_path}\n\n"
            f"To pretrain the CNN encoder, run:\n"
            f"  poetry run spinlock pretrain-initial-features-cnn \\\n"
            f"    --dataset data/your_dataset.h5 \\\n"
            f"    --embedding-dim {embedding_dim} \\\n"
            f"    --output {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'encoder_state_dict' not in checkpoint:
        raise ValueError("Invalid CNN checkpoint: missing 'encoder_state_dict'")

    checkpoint_dim = checkpoint.get('embedding_dim')
    if checkpoint_dim != embedding_dim:
        raise ValueError(
            f"CNN embedding dimension mismatch: "
            f"checkpoint={checkpoint_dim}, config={embedding_dim}"
        )

    metadata = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'val_loss': checkpoint.get('val_loss', 'unknown'),
        'embedding_dim': checkpoint_dim,
    }

    logger.info(f"Verified pretrained CNN: {checkpoint_path}")
    logger.info(f"  Epoch: {metadata['epoch']}, Val Loss: {metadata['val_loss']}")

    return metadata
