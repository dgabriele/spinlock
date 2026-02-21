"""V2 MNO model wrapper with pluggable conditioning.

V2MNO is a thin wrapper around BaseMNOBackbone that delegates conditioning
preparation to a ConditioningAdapter. This keeps the backbone reusable across
different conditioning schemes while providing a clean rollout() API for the
trajectory-first trainer.
"""

import logging
from typing import Dict

import torch.nn as nn
from torch import Tensor

from spinlock.mno.backbone import MNOBackbone
from spinlock.mno.base_backbone import BaseMNOBackbone
from spinlock.mno.v2.conditioning import ConditioningAdapter, ThetaICAdapter
from spinlock.mno.v2.config import V2MNOConfig

logger = logging.getLogger(__name__)


class V2MNO(nn.Module):
    """Backbone wrapper with pluggable conditioning for v2 trajectory-first training."""

    def __init__(
        self,
        backbone: BaseMNOBackbone,
        adapter: ConditioningAdapter,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter

    def rollout(
        self,
        conditioning: Dict[str, Tensor],
        steps: int,
    ) -> Tensor:
        """Run autoregressive rollout with conditioned backbone.

        Args:
            conditioning: Raw conditioning dict (e.g., {"theta": ..., "ic": ...}).
            steps: Number of rollout steps.

        Returns:
            Trajectory tensor [B, T+1, C, H, W].
        """
        prepared = self.adapter.prepare(conditioning)
        return self.backbone.rollout(
            prepared["ic"],
            steps=steps,
            return_all_steps=True,
            params=prepared.get("params"),
        )

    @classmethod
    def from_config(
        cls,
        config: V2MNOConfig,
        dims: Dict[str, int],
        device: str = "cuda",
    ) -> "V2MNO":
        """Construct V2MNO from config and auto-detected dimensions.

        Args:
            config: Full V2 config.
            dims: Auto-detected dimensions from SpinlockDataset.infer_mno_dimensions().
            device: Target device.
        """
        mc = config.model
        backbone = MNOBackbone(
            in_channels=dims["in_channels"],
            out_channels=dims["out_channels"],
            base_channels=mc.base_channels,
            encoder_levels=mc.encoder_levels,
            modes=mc.modes,
            afno_blocks=mc.afno_blocks,
            dropout=mc.dropout,
            param_conditioning=mc.param_conditioning,
            param_dim=dims.get("param_dim", 14),
            param_embed_dim=mc.param_embed_dim,
            conditioning_mode=mc.conditioning_mode,
            film_config=mc.film.model_dump() if mc.film else None,
            update_mode=mc.update_mode,
            use_checkpointing=mc.use_checkpointing,
            checkpoint_every=getattr(mc, "checkpoint_every", 16),
        )
        backbone = backbone.to(device)
        model = cls(backbone=backbone, adapter=ThetaICAdapter())
        logger.info(
            "V2MNO: %s trainable params, in_ch=%d, param_dim=%s",
            f"{backbone.num_trainable_parameters:,}",
            dims["in_channels"],
            dims.get("param_dim"),
        )
        return model
