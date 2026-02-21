"""CLI command for V2 MNO training (trajectory-first)."""

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import yaml

from spinlock.cli.base import CLICommand

logger = logging.getLogger(__name__)


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
        from spinlock.mno.cno_replay import CNOReplayer
        from spinlock.mno.losses.components.contrastive import ContrastiveLoss
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
        has_token_ce = config.loss.lambda_token_ce > 0
        dataset = SpinlockDataset(
            file_path=config.data.dataset_path,
            max_samples=tc.n_samples,
            load_gt_temporal_features=has_feat_mse,
        )
        dims_full = dataset.infer_mno_dimensions()
        dims = dims_full.get("model", dims_full)
        print(f"  Dataset: {len(dataset)} samples")
        print(f"  Dimensions: {dims}")

        # --- Model ---
        model = V2MNO.from_config(config, dims, device)

        # --- CNOReplayer (required for GT trajectories) ---
        replayer = CNOReplayer.from_config(
            config_path=config.data.config,
            device=device,
            cache_size=tc.replayer_cache_size,
        )
        print("  Replayer: loaded")

        # --- TruncatedBPTT ---
        bptt = TruncatedBPTT(
            model.backbone, tc.timesteps, tc.bptt_window or tc.timesteps,
        )
        print(f"  BPTT: {bptt}")

        # --- ContrastiveLoss ---
        cc = config.loss.contrastive
        contrastive = ContrastiveLoss(
            param_dim=dims.get("param_dim", 14),
            embed_dim=cc.embed_dim,
            hidden_dim=cc.hidden_dim,
            temperature=cc.temperature,
            queue_size=cc.queue_size,
        ).to(device)
        print(f"  Contrastive: {contrastive}")

        # --- Optional VQ adapter (for feature MSE, token CE, and/or QA mode) ---
        vq_adapter = None
        needs_vq = has_feat_mse or has_token_ce or config.quantization_aware
        if needs_vq:
            if config.tokenizer_checkpoint is None:
                return self.error(
                    "tokenizer_checkpoint required when "
                    "lambda_feat_mse > 0 or quantization_aware is true"
                )
            from spinlock.mno.vq_coherence import VQCoherenceAdapter

            vq_adapter = VQCoherenceAdapter.from_checkpoint(
                checkpoint_path=config.tokenizer_checkpoint,
                device=device,
            )
            print(f"  VQ adapter: {config.tokenizer_checkpoint}")

        # --- Optional token store for quantization-aware mode ---
        token_store = None
        if config.quantization_aware:
            if config.token_dataset is None:
                return self.error(
                    "token_dataset required when quantization_aware is true"
                )
            from spinlock.tokens.pretokenized_store import PretokenizedTokenStore

            token_store = PretokenizedTokenStore(Path(config.token_dataset))
            print(f"  Token dataset: {config.token_dataset}")
            print("  ** Quantization-aware mode enabled **")
            print("     MNO will train on VQ-decoded θ̂/IC_hat")

        if has_feat_mse:
            print("  Feature MSE enabled")
        if has_token_ce:
            print("  Token CE enabled (soft VQ cross-entropy)")

        # --- Loss ---
        loss_fn = TrajectoryLoss(
            contrastive_loss=contrastive,
            lambda_traj=config.loss.lambda_traj,
            lambda_ic=config.loss.lambda_ic,
            lambda_contrastive=config.loss.lambda_contrastive,
            lambda_feat_mse=config.loss.lambda_feat_mse,
            lambda_token_ce=config.loss.lambda_token_ce,
            token_ce_temperature=config.loss.token_ce_temperature,
            normalize_loss_scales=config.loss.normalize_loss_scales,
            loss_scale_ema_momentum=config.loss.loss_scale_ema_momentum,
            vq_adapter=vq_adapter if (has_feat_mse or has_token_ce) else None,
        )

        # --- Evaluator ---
        evaluator = TrajectoryEvaluator(
            replayer=replayer,
            contrastive=contrastive,
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
