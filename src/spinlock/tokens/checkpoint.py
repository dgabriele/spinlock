"""Checkpoint save/load utilities."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch

from .config import TokenizerConfig

logger = logging.getLogger(__name__)


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
):
    """Save tokenizer checkpoint in V2 format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.model_dump(),
        'group_indices': group_indices,
        'normalization_stats': normalization_stats,
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


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load tokenizer checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location='cpu')

    version = checkpoint.get('version', 'v1')
    if version != 'v2':
        raise ValueError(
            f"Checkpoint version {version} not supported. Use V2 checkpoints only."
        )

    config = TokenizerConfig(**checkpoint['config'])

    return {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': config,
        'group_indices': checkpoint['group_indices'],
        'normalization_stats': checkpoint.get('normalization_stats'),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
        'epoch': checkpoint.get('epoch'),
        'val_loss': checkpoint.get('val_loss'),
        'metadata': checkpoint.get('metadata', {}),
    }


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

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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
