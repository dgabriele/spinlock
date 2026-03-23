"""Checkpoint save/load for NLTokenizer + LFM adapter state.

Saves the NLTokenizerModel, LFMAdapter, NLListener, optimizer state,
and config into a single checkpoint file. The LFM decoder weights are
included (even though frozen) so the checkpoint is self-contained.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .nl_config import NLTokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class NLTokenizerCheckpoint:
    """Contents of a saved NLTokenizer checkpoint."""

    config: NLTokenizerConfig
    model_state_dict: Dict[str, Any]
    adapter_state_dict: Dict[str, Any]
    listener_state_dict: Dict[str, Any]
    group_indices: Dict[str, list]
    normalization_stats: Optional[Dict]
    epoch: int
    val_loss: float
    metadata: Optional[Dict[str, Any]] = None
    # Auto-detected dimensions for model reconstruction
    temporal_input_dim: Optional[int] = None
    theta_param_dim: Optional[int] = None
    initial_input_dim: Optional[int] = None


def save_nl_checkpoint(
    path: Path,
    model: "torch.nn.Module",
    adapter: "torch.nn.Module",
    listener: "torch.nn.Module",
    config: NLTokenizerConfig,
    group_indices: Dict[str, list],
    optimizer: Optional["torch.optim.Optimizer"] = None,
    normalization_stats: Optional[Dict] = None,
    epoch: int = 0,
    val_loss: float = float("inf"),
    metadata: Optional[Dict[str, Any]] = None,
    temporal_input_dim: Optional[int] = None,
    theta_param_dim: Optional[int] = None,
    initial_input_dim: Optional[int] = None,
) -> None:
    """Save NLTokenizer checkpoint to disk.

    Args:
        path: Output file path
        model: NLTokenizerModel
        adapter: LFMAdapter
        listener: NLListener
        config: NLTokenizerConfig
        group_indices: Feature group mapping
        optimizer: Optimizer (for resume)
        normalization_stats: Feature normalization stats
        epoch: Current epoch
        val_loss: Current best validation loss
        metadata: Additional metadata (training history, scheduler state, etc.)
        temporal_input_dim: Auto-detected temporal dimension
        theta_param_dim: Auto-detected theta parameter dimension
        initial_input_dim: Auto-detected initial feature dimension
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": config.model_dump(),
        "model_state_dict": model.state_dict(),
        "adapter_state_dict": adapter.state_dict(),
        "listener_state_dict": listener.state_dict(),
        "group_indices": group_indices,
        "normalization_stats": normalization_stats,
        "epoch": epoch,
        "val_loss": val_loss,
        "metadata": metadata or {},
        "temporal_input_dim": temporal_input_dim,
        "theta_param_dim": theta_param_dim,
        "initial_input_dim": initial_input_dim,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"NL checkpoint saved: {path} (epoch={epoch}, val_loss={val_loss:.6f})")


def load_nl_checkpoint(path: Path) -> NLTokenizerCheckpoint:
    """Load NLTokenizer checkpoint from disk.

    Args:
        path: Checkpoint file path

    Returns:
        NLTokenizerCheckpoint with all saved state
    """
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)

    config = NLTokenizerConfig(**data["config"])

    return NLTokenizerCheckpoint(
        config=config,
        model_state_dict=data["model_state_dict"],
        adapter_state_dict=data["adapter_state_dict"],
        listener_state_dict=data["listener_state_dict"],
        group_indices=data["group_indices"],
        normalization_stats=data.get("normalization_stats"),
        epoch=data.get("epoch", 0),
        val_loss=data.get("val_loss", float("inf")),
        metadata=data.get("metadata"),
        temporal_input_dim=data.get("temporal_input_dim"),
        theta_param_dim=data.get("theta_param_dim"),
        initial_input_dim=data.get("initial_input_dim"),
    )
