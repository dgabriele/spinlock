"""RolloutProvider: strategy pattern for MNO vs GT simulator rollouts.

Two implementations:
- MNORolloutProvider: wraps V2MNO for learned surrogate rollouts
- SimulatorRolloutProvider: wraps GT replayers (LeniaReplayAdapter, CNOReplayer)

Both produce [B, T+1, C, H, W] trajectories (IC-prefixed).

Factory function ``build_rollout_provider`` auto-resolves the operator type
from the tokenizer checkpoint chain when ``mno_checkpoint`` is None.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol

import torch
import yaml
from torch import Tensor

logger = logging.getLogger(__name__)


class RolloutProvider(Protocol):
    """Uniform interface for trajectory rollout."""

    def rollout(
        self, conditioning: Dict[str, Any], steps: int
    ) -> Tensor:
        """Produce a trajectory from conditioning parameters.

        Args:
            conditioning: Dict with at least ``theta`` [B, D] and ``ic``
                [B, C, H, W].  May contain extra keys (e.g. ``token_indices``)
                which implementations may ignore.
            steps: Number of rollout timesteps.

        Returns:
            Trajectory tensor [B, T+1, C, H, W] (IC-prefixed).
        """
        ...


class MNORolloutProvider:
    """Wraps a trained V2MNO model."""

    def __init__(self, mno) -> None:
        self.mno = mno

    def rollout(
        self, conditioning: Dict[str, Any], steps: int
    ) -> Tensor:
        return self.mno.rollout(conditioning, steps=steps)


class SimulatorRolloutProvider:
    """Wraps a GT replayer (LeniaReplayAdapter, CNOReplayer, etc.).

    Translates the conditioning dict into the replayer's positional
    arguments (params_vector/params_batch, ic, timesteps).
    """

    def __init__(self, replayer: Any) -> None:
        self.replayer = replayer

    def rollout(
        self, conditioning: Dict[str, Any], steps: int
    ) -> Tensor:
        theta = conditioning["theta"]   # [B, D]
        ic = conditioning["ic"]         # [B, C, H, W]
        B = theta.shape[0]

        if B > 1 and hasattr(self.replayer, "rollout_batch"):
            return self.replayer.rollout_batch(
                theta, ic, timesteps=steps, return_all_steps=True,
            )
        # Single-sample or replayer without batch support.
        # Replayer.rollout() already returns [1, T+1, C, H, W] (batch dim
        # included because it internally unsqueezes ic).
        return self.replayer.rollout(
            theta.squeeze(0), ic.squeeze(0),
            timesteps=steps, return_all_steps=True,
        )


# ── Factory helpers ──────────────────────────────────────────────────────────


def create_replayer(
    config_path: str, device: str, cache_size: int = 0
) -> Any:
    """Create a GT replayer from a dataset generation config YAML.

    Reads ``simulation.operator_type`` and dispatches to the appropriate
    replayer class.

    Args:
        config_path: Path to dataset generation config YAML.
        device: Computation device.
        cache_size: Operator cache size (ignored by some replayers).

    Returns:
        Replayer instance with ``.rollout()`` method.
    """
    with open(config_path) as f:
        gen_cfg = yaml.safe_load(f)
    op_type = gen_cfg.get("simulation", {}).get("operator_type")

    match op_type:
        case "lenia":
            from spinlock.lenia.replay_adapter import LeniaReplayAdapter

            return LeniaReplayAdapter.from_config(
                config_path, device=device, cache_size=cache_size,
            )
        case "cnn":
            from spinlock.mno.cno_replay import CNOReplayer

            return CNOReplayer.from_config(
                config_path, device=device, cache_size=cache_size,
            )
        case "u_afno":
            raise NotImplementedError(
                "U-AFNO replayer not yet implemented"
            )
        case "qbm":
            raise NotImplementedError(
                "QBM replayer not yet implemented"
            )
        case None:
            raise ValueError(
                f"No simulation.operator_type found in {config_path}"
            )
        case _:
            raise ValueError(f"Unknown operator_type: {op_type!r}")


def resolve_dataset_config_path(tokenizer_checkpoint: str) -> str:
    """Extract the dataset generation config path from a VQTokenizer checkpoint.

    The tokenizer checkpoint stores ``config['generation_config_path']`` which
    points to the YAML that was used to generate the training dataset.  That
    YAML contains ``simulation.operator_type`` needed to dispatch the replayer.

    Args:
        tokenizer_checkpoint: Path to VQTokenizer ``.pt`` checkpoint.

    Returns:
        Path string to the dataset generation config YAML.

    Raises:
        ValueError: If the checkpoint doesn't contain a generation config path.
    """
    ckpt = torch.load(
        tokenizer_checkpoint, map_location="cpu", weights_only=False,
    )
    config = ckpt.get("config", {})
    path = config.get("generation_config_path")
    if path is None:
        raise ValueError(
            f"VQTokenizer checkpoint {tokenizer_checkpoint!r} does not contain "
            f"'config.generation_config_path'. Provide dataset_config_path "
            f"explicitly in RefinementConfig."
        )
    return path


def build_rollout_provider(
    mno_checkpoint: Optional[str],
    tokenizer_checkpoint: str,
    device: str,
    dataset_config_path: Optional[str] = None,
) -> RolloutProvider:
    """Build a RolloutProvider from config.

    When ``mno_checkpoint`` is set, loads V2MNO into MNORolloutProvider.
    When None, resolves the GT simulator from the tokenizer checkpoint
    chain (or explicit ``dataset_config_path`` override).

    Args:
        mno_checkpoint: Path to trained MNO, or None for GT simulator.
        tokenizer_checkpoint: Path to VQTokenizer checkpoint.
        device: Computation device.
        dataset_config_path: Explicit path to dataset generation YAML.
            Auto-resolved from tokenizer checkpoint when None.

    Returns:
        A RolloutProvider instance.
    """
    if mno_checkpoint is not None:
        from spinlock.mno.v2.model import V2MNO

        ckpt = torch.load(
            mno_checkpoint, map_location=device, weights_only=False,
        )
        config = ckpt["config"]
        dims = ckpt.get("dims", {})
        operator_type = ckpt.get("operator_type", None)

        mno = V2MNO.from_config(
            config, dims, device=device, operator_type=operator_type,
        )
        mno.load_state_dict(ckpt["model_state_dict"])
        mno.to(device)
        mno.eval()
        logger.info("RolloutProvider: V2MNO from %s", mno_checkpoint)
        return MNORolloutProvider(mno)

    # GT simulator path
    if dataset_config_path is None:
        dataset_config_path = resolve_dataset_config_path(
            tokenizer_checkpoint
        )
    logger.info(
        "RolloutProvider: GT simulator from %s", dataset_config_path,
    )
    replayer = create_replayer(dataset_config_path, device)
    return SimulatorRolloutProvider(replayer)
