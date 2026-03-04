"""V2 MNO model wrapper with pluggable conditioning.

V2MNO is a thin wrapper around BaseMNOBackbone that delegates conditioning
preparation to a ConditioningAdapter. This keeps the backbone reusable across
different conditioning schemes while providing a clean rollout() API for the
trajectory-first trainer.

Backbone dispatch:
    - "u_afno" (default): U-AFNO neural operator (~145M params)
    - "neural_ca": Lightweight CA-matched backbone (~60K params)
    Auto-detected from operator_type when backbone_type is None.
"""

import logging
from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from spinlock.mno.backbone import MNOBackbone
from spinlock.mno.base_backbone import BaseMNOBackbone
from spinlock.mno.config import NCAConfig
from spinlock.mno.v2.conditioning import (
    ConditioningAdapter,
    ThetaICAdapter,
    TokenEmbeddingProjector,
    TokenThetaICAdapter,
)
from spinlock.mno.v2.config import V2MNOConfig

logger = logging.getLogger(__name__)


def _infer_backbone_type(operator_type: Optional[str]) -> str:
    """Map operator_type → default backbone architecture.

    Cellular automaton datasets (Lenia) default to the lightweight
    NeuralCABackbone. Everything else defaults to U-AFNO.
    """
    _BACKBONE_MAP = {
        "lenia": "neural_ca",
        "cnn": "u_afno",
        "u_afno": "u_afno",
        "qbm": "u_afno",
    }
    return _BACKBONE_MAP.get(operator_type, "u_afno")


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
        operator_type: Optional[str] = None,
    ) -> "V2MNO":
        """Construct V2MNO from config and auto-detected dimensions.

        Args:
            config: Full V2 config.
            dims: Auto-detected dimensions from SpinlockDataset.infer_mno_dimensions().
            device: Target device.
            operator_type: Dataset operator type (e.g. "lenia", "cnn").
                Used for backbone auto-detection when backbone_type is None.
        """
        mc = config.model
        backbone_type = mc.backbone_type

        # Auto-detect from operator_type if not explicitly set
        if backbone_type is None:
            backbone_type = _infer_backbone_type(operator_type)

        # Token conditioning: widen param_dim to accommodate projected token embeddings
        base_param_dim = dims.get("param_dim", 14)
        effective_param_dim = base_param_dim
        token_projector = None

        if config.token_conditioning:
            if config.tokenizer_checkpoint is None:
                raise ValueError(
                    "token_conditioning=True requires tokenizer_checkpoint"
                )
            token_projector = TokenEmbeddingProjector(
                vq_checkpoint_path=config.tokenizer_checkpoint,
                token_embed_dim=config.token_embed_dim,
            )
            effective_param_dim = base_param_dim + config.token_embed_dim
            logger.info(
                "Token conditioning: param_dim %d + token_embed_dim %d = %d",
                base_param_dim, config.token_embed_dim, effective_param_dim,
            )

        if backbone_type == "neural_ca":
            from spinlock.mno.nca_backbone import NeuralCABackbone, _auto_perception_specs

            nca_cfg = mc.nca or NCAConfig()
            kernel_sizes = nca_cfg.kernel_sizes
            dilations = nca_cfg.dilations

            # Auto-derive perception specs from spatial_dim when not explicit
            if kernel_sizes is None:
                spatial_dim = dims.get("spatial_dim", 128)
                kernel_sizes, dilations = _auto_perception_specs(spatial_dim)
                logger.info(
                    "NCA auto-config: spatial_dim=%d → kernel_sizes=%s, "
                    "dilations=%s, RFs=%s",
                    spatial_dim, kernel_sizes, dilations,
                    [d * (k - 1) + 1 for k, d in zip(kernel_sizes, dilations)],
                )

            backbone = NeuralCABackbone(
                in_channels=dims["in_channels"],
                out_channels=dims["out_channels"],
                hidden_channels=nca_cfg.hidden_channels,
                kernel_sizes=tuple(kernel_sizes),
                dilations=tuple(dilations) if dilations else None,
                growth_hidden=nca_cfg.growth_hidden,
                residual_scale=nca_cfg.residual_scale,
                clamp_output=nca_cfg.clamp_output,
                clamp_leak=nca_cfg.clamp_leak,
                padding_mode=nca_cfg.padding_mode,
                param_conditioning=mc.param_conditioning,
                param_dim=effective_param_dim,
                param_embed_dim=mc.param_embed_dim,
                conditioning_mode=mc.conditioning_mode,
                film_config=mc.film.model_dump() if mc.film else None,
                use_checkpointing=mc.use_checkpointing,
                checkpoint_every=getattr(mc, "checkpoint_every", 16),
            )
        else:
            backbone = MNOBackbone(
                in_channels=dims["in_channels"],
                out_channels=dims["out_channels"],
                base_channels=mc.base_channels,
                encoder_levels=mc.encoder_levels,
                modes=mc.modes,
                afno_blocks=mc.afno_blocks,
                dropout=mc.dropout,
                param_conditioning=mc.param_conditioning,
                param_dim=effective_param_dim,
                param_embed_dim=mc.param_embed_dim,
                conditioning_mode=mc.conditioning_mode,
                film_config=mc.film.model_dump() if mc.film else None,
                update_mode=mc.update_mode,
                use_checkpointing=mc.use_checkpointing,
                checkpoint_every=getattr(mc, "checkpoint_every", 16),
            )

        backbone = backbone.to(device)

        # Build adapter: token-conditioned or standard theta+IC
        if token_projector is not None:
            adapter: ConditioningAdapter = TokenThetaICAdapter(token_projector)
            adapter = adapter.to(device)  # type: ignore[union-attr]
        else:
            adapter = ThetaICAdapter()

        model = cls(backbone=backbone, adapter=adapter)
        logger.info(
            "V2MNO: %s (%s trainable params), in_ch=%d, param_dim=%d%s",
            type(backbone).__name__,
            f"{backbone.num_trainable_parameters:,}",
            dims["in_channels"],
            effective_param_dim,
            f" (token_cond: {base_param_dim}+{config.token_embed_dim})"
            if config.token_conditioning else "",
        )
        return model
