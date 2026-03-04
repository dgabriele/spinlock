"""CLI command for V2 MNO training (trajectory-first)."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import yaml

from spinlock.cli.base import CLICommand

logger = logging.getLogger(__name__)


def _create_replayer(config_path: str, device: str, cache_size: int) -> Any:
    """Create a replayer from the generation config YAML.

    Reads ``simulation.operator_type`` and dispatches to the appropriate
    replayer class. Supports duck-typed replayer interface (rollout method).

    Args:
        config_path: Path to dataset generation config YAML.
        device: Computation device.
        cache_size: Operator cache size (ignored by some replayers).

    Returns:
        Replayer instance with .rollout() method.
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
                "U-AFNO replayer not yet implemented for MNO training"
            )
        case "qbm":
            raise NotImplementedError(
                "QBM replayer not yet implemented for MNO training"
            )
        case None:
            raise ValueError(
                f"No simulation.operator_type found in {config_path}"
            )
        case _:
            raise ValueError(f"Unknown operator_type: {op_type!r}")


class TrainMNOV2Command(CLICommand):
    """Train V2 MNO with trajectory-first loss (MSE + contrastive)."""

    @property
    def name(self) -> str:
        return "train-mno-v2"

    @property
    def help(self) -> str:
        return "Train MNO v2 with trajectory-first loss"

    @property
    def description(self) -> str:
        return (
            "Train Meta Neural Operator v2 using trajectory MSE + InfoNCE "
            "contrastive as the training loss. Generates GT trajectories "
            "via CNOReplayer and uses TruncatedBPTT for long-horizon rollouts."
        )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to V2 MNO config YAML",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable debug logging",
        )

    def execute(self, args: Namespace) -> int:
        # Defer heavy imports to avoid CLI startup latency
        import torch

        from spinlock.data import SpinlockDataset
        from spinlock.mno.losses.components.contrastive import SoftTokenContrastiveLoss
        from spinlock.mno.truncated_bptt import TruncatedBPTT
        from spinlock.mno.v2.config import V2MNOConfig
        from spinlock.mno.v2.evaluation import TrajectoryEvaluator
        from spinlock.mno.v2.loss import TrajectoryLoss
        from spinlock.mno.v2.model import V2MNO
        from spinlock.mno.v2.trainer import V2Trainer

        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        # --- Load config ---
        config_path = Path(args.config)
        if not config_path.exists():
            return self.error(f"Config not found: {config_path}")

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        config = V2MNOConfig(**raw)

        # --- Seed ---
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        device = config.device
        tc = config.training
        print("V2 MNO Training (trajectory-first)")
        print(f"  Config: {config_path}")
        print(f"  Device: {device}")

        # --- Dataset + dimension inference ---
        has_feat_mse = (
            config.loss.lambda_feat_mse > 0
            and config.tokenizer_checkpoint is not None
        )
        has_contrastive = config.loss.lambda_contrastive > 0
        has_token_ce = config.loss.lambda_token_ce > 0
        has_centroid_mse = config.loss.lambda_centroid_mse > 0
        has_token_head = config.loss.lambda_token_head > 0
        dataset = SpinlockDataset(
            file_path=config.data.dataset_path,
            max_samples=tc.n_samples,
            load_gt_temporal_features=has_feat_mse,
        )
        dims_full = dataset.infer_mno_dimensions()
        dims = dims_full.get("model", dims_full)
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  Dimensions: {dims}")

        # --- Extract operator_type for backbone auto-detection ---
        with open(config.data.config) as f:
            gen_cfg = yaml.safe_load(f)
        operator_type = gen_cfg.get("simulation", {}).get("operator_type")

        # --- Model ---
        model = V2MNO.from_config(config, dims, device, operator_type=operator_type)
        print(
            f"  Backbone: {type(model.backbone).__name__} "
            f"({model.backbone.num_trainable_parameters:,} params)"
        )

        # --- Replayer (auto-detected from generation config) ---
        replayer = _create_replayer(
            config_path=config.data.config,
            device=device,
            cache_size=tc.replayer_cache_size,
        )
        print(f"  Replayer: {type(replayer).__name__}")

        # --- TruncatedBPTT ---
        bptt = TruncatedBPTT(
            model.backbone, tc.timesteps, tc.bptt_window or tc.timesteps,
            num_windows=tc.bptt_num_windows,
        )
        print(f"  BPTT: {bptt}")

        # --- VQ adapter (needed for contrastive, feature MSE, token CE, QA) ---
        vq_adapter = None
        needs_vq = (
            has_feat_mse or has_contrastive or has_token_ce
            or has_centroid_mse or has_token_head or config.quantization_aware
        )
        if needs_vq:
            if config.tokenizer_checkpoint is None:
                return self.error(
                    "tokenizer_checkpoint required when "
                    "lambda_contrastive > 0, lambda_feat_mse > 0, "
                    "or quantization_aware is true"
                )
            from spinlock.mno.vq_coherence import VQCoherenceAdapter

            vq_adapter = VQCoherenceAdapter.from_checkpoint(
                checkpoint_path=config.tokenizer_checkpoint,
                device=device,
            )
            print(f"  VQ adapter: {config.tokenizer_checkpoint}")

        # --- Token store (needed for contrastive indicators, QA, token CE, token cond) ---
        token_store = None
        needs_token_store = (
            config.quantization_aware or has_contrastive
            or has_token_ce or has_centroid_mse or has_token_head
            or config.token_conditioning
        )
        if needs_token_store:
            if config.token_dataset is None:
                return self.error(
                    "token_dataset required when lambda_contrastive > 0, "
                    "quantization_aware, lambda_token_ce > 0, "
                    "lambda_centroid_mse > 0, or token_conditioning"
                )
            from spinlock.tokens.pretokenized_store import PretokenizedTokenStore

            token_store = PretokenizedTokenStore(
                Path(config.token_dataset),
                truncation_length=config.token_truncation_length,
            )
            print(f"  Token dataset: {config.token_dataset}")
            if config.quantization_aware:
                print("  ** Quantization-aware mode enabled **")
                print("     MNO will train on VQ-decoded θ̂/IC_hat")

        # --- Soft Token Contrastive Loss ---
        cc = config.loss.contrastive
        feature_dim = vq_adapter.feature_dim if vq_adapter is not None else 960
        contrastive = SoftTokenContrastiveLoss(
            feature_dim=feature_dim,
            embed_dim=cc.embed_dim,
            hidden_dim=cc.hidden_dim,
            tau_pred=cc.tau_pred,
            tau_target=cc.tau_target,
            queue_size=cc.queue_size,
        ).to(device)
        print(f"  Contrastive: {contrastive}")

        # --- Optional learned token prediction head ---
        token_pred_head = None
        if has_token_head:
            from spinlock.mno.v2.token_head import TokenPredictionHead

            token_pred_head = TokenPredictionHead.from_vq_adapter(
                vq_adapter, in_channels=dims["in_channels"],
            ).to(device)
            n_head_params = sum(p.numel() for p in token_pred_head.parameters())
            print(f"  Token head: {n_head_params:,} params")

        if has_feat_mse:
            print("  Feature MSE enabled")
        if has_contrastive:
            print("  Soft token contrastive enabled (Jaccard targets)")
        if has_token_ce:
            print("  Token CE enabled (soft VQ cross-entropy)")
        if has_centroid_mse:
            print("  Centroid MSE enabled (VQ centroid supervision)")
        if has_token_head:
            print("  Token head CE enabled (learned bypass of frozen VQ encoder)")
        if config.token_conditioning:
            print(
                f"  Token conditioning enabled "
                f"(embed_dim={config.token_embed_dim})"
            )

        # --- Loss ---
        loss_fn = TrajectoryLoss(
            contrastive_loss=contrastive,
            lambda_traj=config.loss.lambda_traj,
            lambda_ic=config.loss.lambda_ic,
            lambda_contrastive=config.loss.lambda_contrastive,
            lambda_feat_mse=config.loss.lambda_feat_mse,
            lambda_token_ce=config.loss.lambda_token_ce,
            lambda_centroid_mse=config.loss.lambda_centroid_mse,
            lambda_token_head=config.loss.lambda_token_head,
            token_ce_temperature=config.loss.token_ce_temperature,
            gate_weight_token_ce=config.loss.gate_weight_token_ce,
            normalize_loss_scales=config.loss.normalize_loss_scales,
            loss_scale_ema_momentum=config.loss.loss_scale_ema_momentum,
            vq_adapter=vq_adapter if (has_feat_mse or has_token_ce or has_centroid_mse or has_contrastive) else None,
            token_pred_head=token_pred_head,
        )

        # --- Resume from checkpoint (warm-start model + loss weights) ---
        if config.resume_from is not None:
            ckpt_path = Path(config.resume_from)
            if not ckpt_path.exists():
                return self.error(f"Resume checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            # When token_conditioning is newly enabled, the old checkpoint has
            # param_embedding with shape [embed, 34] while the new model has
            # [embed, 34 + token_embed_dim]. Use strict=False and log skips.
            if config.token_conditioning:
                missing, unexpected = model.load_state_dict(
                    ckpt["model_state_dict"], strict=False,
                )
                if missing:
                    logger.info(
                        "Partial resume (token_conditioning): %d missing keys "
                        "(new token projector + widened param_embedding)",
                        len(missing),
                    )
                    for k in missing:
                        logger.debug("  missing: %s", k)
                if unexpected:
                    logger.warning(
                        "Partial resume: %d unexpected keys", len(unexpected),
                    )
            else:
                model.load_state_dict(ckpt["model_state_dict"])

            if "loss_fn_state_dict" in ckpt:
                loss_fn.load_state_dict(ckpt["loss_fn_state_dict"], strict=False)
            resumed_epoch = ckpt.get("epoch", "?")
            resumed_loss = ckpt.get("loss", float("nan"))
            print(
                f"  Resumed from: {ckpt_path} "
                f"(epoch {resumed_epoch}, loss {resumed_loss:.4f})"
            )
            print("  Optimizer/scheduler reset (warm-start, not full resume)")

        # --- Evaluator ---
        evaluator = TrajectoryEvaluator(
            replayer=replayer,
            bptt=bptt,
        )

        # --- Trainer ---
        trainer = V2Trainer(
            config=config,
            model=model,
            loss_fn=loss_fn,
            evaluator=evaluator,
            dataset=dataset,
            replayer=replayer,
            bptt=bptt,
            device=device,
            token_store=token_store,
            vq_adapter=vq_adapter if config.quantization_aware else None,
        )

        # --- Train ---
        metrics = trainer.train()
        final_loss = metrics.get("final_train_loss", float("inf"))
        print(f"\nTraining complete. Final loss: {final_loss:.4f}")
        return 0
