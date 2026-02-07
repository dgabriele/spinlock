"""Checkpoint save/load utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from pydantic import BaseModel, Field

from .config import TokenizerConfig

logger = logging.getLogger(__name__)


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

    class Config:
        arbitrary_types_allowed = True  # Allow torch tensors in state_dict


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
):
    """Save tokenizer checkpoint in V2 format."""
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

    # Build TokenizerCheckpoint with validation
    return TokenizerCheckpoint(
        model_state_dict=raw_checkpoint['model_state_dict'],
        config=config,
        group_indices=raw_checkpoint['group_indices'],
        normalization_stats=raw_checkpoint.get('normalization_stats'),
        temporal_input_dim=raw_checkpoint.get('temporal_input_dim'),
        initial_input_dim=raw_checkpoint.get('initial_input_dim'),
        optimizer_state_dict=raw_checkpoint.get('optimizer_state_dict'),
        epoch=raw_checkpoint.get('epoch'),
        val_loss=raw_checkpoint.get('val_loss'),
        metadata=raw_checkpoint.get('metadata', {}),
        version=version,
    )


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
