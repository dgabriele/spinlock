"""V2 MNO trainer: trajectory-first training loop.

Per-batch flow (standard mode):
  1. {ic, params} <- SpinlockDataset
  2. GT trajectory: replayer.rollout(params[b], ic[b], timesteps) per sample (no_grad)
  3. Pred trajectory: bptt.rollout(ic, params=params) → [B, W+1, C, H, W]
  4. Align: bptt.align_for_loss(pred, gt) → [B, W, C, H, W] each
  5. Loss: TrajectoryLoss.compute(pred, gt, params=params)
  6. Backward + gradient accumulation → optimizer.step()

Per-batch flow (quantization-aware mode):
  1. {ic, params, sample_idx} <- SpinlockDataset
  2. tokens <- PretokenizedTokenStore.get_batch(sample_idx)
  3. {θ̂, IC_hat} <- VQCoherenceAdapter.decode_tokens_to_params(tokens)
  4. GT trajectory: replayer.rollout(θ̂[b], IC_hat[b], timesteps) per sample (no_grad)
  5. Pred trajectory: bptt.rollout(IC_hat, params=θ̂) → [B, W+1, C, H, W]
  6. Same alignment, loss, backward as standard mode
"""

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from spinlock.data import SpinlockDataset
from spinlock.mno.truncated_bptt import TruncatedBPTT
from spinlock.mno.v2.config import V2MNOConfig
from spinlock.mno.v2.evaluation import TrajectoryEvaluator
from spinlock.mno.v2.loss import TrajectoryLoss
from spinlock.mno.v2.model import V2MNO

logger = logging.getLogger(__name__)


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + cosine annealing scheduler."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class V2Trainer:
    """Trajectory-first training loop for V2 MNO.

    Uses CNOReplayer for GT trajectory generation and TruncatedBPTT for
    memory-efficient long-horizon rollouts.
    """

    def __init__(
        self,
        config: V2MNOConfig,
        model: V2MNO,
        loss_fn: TrajectoryLoss,
        evaluator: TrajectoryEvaluator,
        dataset: SpinlockDataset,
        replayer: Any,
        bptt: TruncatedBPTT,
        device: str = "cuda",
        token_store: Optional[Any] = None,
        vq_adapter: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._model = model
        self._loss_fn = loss_fn
        self._evaluator = evaluator
        self._dataset = dataset
        self._replayer = replayer
        self._bptt = bptt
        self._device = device

        # Quantization-aware mode: decode VQ tokens → θ̂/IC_hat (cached)
        # For tokenizers that encode theta+initial families, predecoding maps
        # VQ tokens → quantized params/IC. For temporal-only tokenizers (e.g.
        # Lenia), predecoding is skipped and raw dataset params/IC are used.
        # The token store is always kept for GT token CE loss.
        self._token_store = token_store  # kept for GT token lookup during training
        self._qa_theta: Optional[torch.Tensor] = None  # [N, param_dim]
        self._qa_ic: Optional[torch.Tensor] = None  # [N, C, H, W]
        self._qa_mode = False
        if token_store is not None and vq_adapter is not None:
            try:
                self._qa_theta, self._qa_ic = self._predecode_tokens(
                    token_store, vq_adapter,
                )
                self._qa_mode = True
            except RuntimeError as e:
                if "inverse heads" in str(e).lower():
                    logger.warning(
                        "VQ adapter has no inverse heads — skipping QA "
                        "predecoding. Using raw dataset params/IC. "
                        "Token store will still be used for token CE loss."
                    )
                else:
                    raise

        tc = config.training

        # Train/val split
        n_total = len(dataset)
        n_val = max(1, int(n_total * config.data.val_split))
        n_train = n_total - n_val
        indices = list(range(n_total))
        train_set = Subset(dataset, indices[:n_train])
        val_set = Subset(dataset, indices[n_train:])

        self._train_loader = DataLoader(
            train_set,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self._val_loader = DataLoader(
            val_set,
            batch_size=tc.batch_size,
            shuffle=False,
            num_workers=0,
        )
        logger.info(
            "Data split: %d train, %d val (%.0f%% val)",
            n_train, n_val, config.data.val_split * 100,
        )

        # Optimizer with separate FiLM LR + contrastive projector params
        film_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "film" in name.lower():
                film_params.append(param)
            else:
                other_params.append(param)

        # Loss fn params: split token head (own LR) from contrastive projectors
        token_head_param_ids: set = set()
        token_head_params: list = []
        if hasattr(loss_fn, '_token_pred_head') and loss_fn._token_pred_head is not None:
            for p in loss_fn._token_pred_head.parameters():
                token_head_param_ids.add(id(p))
                token_head_params.append(p)
        contrastive_params = [
            p for p in loss_fn.parameters()
            if id(p) not in token_head_param_ids
        ]

        param_groups = [{"params": other_params, "lr": tc.learning_rate}]
        if film_params:
            param_groups.append({
                "params": film_params,
                "lr": tc.learning_rate * tc.film_lr_multiplier,
            })
        if contrastive_params:
            param_groups.append({
                "params": contrastive_params,
                "lr": tc.learning_rate,
            })
        if token_head_params:
            param_groups.append({
                "params": token_head_params,
                "lr": tc.learning_rate * tc.token_head_lr_multiplier,
            })

        logger.info(
            "Optimizer: %d base, %d FiLM (%.1fx LR), %d contrastive, "
            "%d token_head (%.1fx LR)",
            len(other_params), len(film_params), tc.film_lr_multiplier,
            len(contrastive_params),
            len(token_head_params), tc.token_head_lr_multiplier,
        )

        self._optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=tc.weight_decay,
        )

        # Scheduler
        steps_per_epoch = max(
            1, len(self._train_loader) // tc.gradient_accumulation_steps,
        )
        total_steps = steps_per_epoch * tc.epochs
        self._scheduler = _create_scheduler(
            self._optimizer, tc.warmup_optimizer_steps, total_steps,
        )

        # torch.compile
        if tc.use_torch_compile:
            self._model = torch.compile(self._model)  # type: ignore[assignment]
            logger.info("torch.compile enabled")

        # Checkpointing
        self._save_dir = Path(config.checkpointing.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._best_loss = float("inf")

    @torch.no_grad()
    def _predecode_tokens(
        self, token_store: Any, vq_adapter: Any,
    ) -> tuple:
        """Pre-decode all VQ tokens → θ̂/IC_hat and cache on CPU.

        Decodes in batches to avoid GPU OOM. Results are stored on CPU
        and indexed per-batch during training via sample_idx.

        Returns:
            (qa_theta [N, param_dim], qa_ic [N, C, H, W]) on CPU
        """
        n = token_store.num_samples
        batch_size = 256
        all_theta, all_ic = [], []

        logger.info(
            "Pre-decoding %d samples from token dataset → θ̂/IC_hat...", n,
        )

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            indices = torch.arange(start, end)
            tokens = token_store.get_batch(indices)
            tokens = {k: v.to(self._device) for k, v in tokens.items()}
            decoded = vq_adapter.decode_tokens_to_params(tokens)
            all_theta.append(decoded["theta"].cpu())
            all_ic.append(decoded["u0"].cpu())

        qa_theta = torch.cat(all_theta, dim=0)  # [N, param_dim]
        qa_ic = torch.cat(all_ic, dim=0)  # [N, C, H, W]

        logger.info(
            "Pre-decoded: θ̂ %s, IC_hat %s (cached on CPU)",
            list(qa_theta.shape), list(qa_ic.shape),
        )
        return qa_theta, qa_ic

    def train(self) -> Dict[str, Any]:
        """Run the full training loop. Returns final metrics."""
        tc = self._config.training
        logger.info(
            "Starting V2 trajectory training: %d epochs, %d samples, "
            "batch=%d, accum=%d, timesteps=%d, bptt_window=%s",
            tc.epochs, len(self._dataset),
            tc.batch_size, tc.gradient_accumulation_steps,
            tc.timesteps, tc.bptt_window,
        )

        all_metrics: Dict[str, Any] = {}
        train_metrics: Dict[str, float] = {}

        for epoch in range(tc.epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            elapsed = time.time() - t0

            lr = self._optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | loss=%.4f | traj_mse=%.4f | "
                "contrastive=%.4f | lr=%.2e | %.1fs",
                epoch + 1, tc.epochs,
                train_metrics["avg_loss"],
                train_metrics.get("traj_mse", 0),
                train_metrics.get("contrastive", 0),
                lr, elapsed,
            )

            # Evaluation
            if (epoch + 1) % tc.eval_every == 0 or epoch == tc.epochs - 1:
                eval_metrics = self._evaluator.evaluate(
                    self._model,
                    self._val_loader,
                    n_samples=tc.eval_samples,
                    device=self._device,
                )
                logger.info(
                    "Eval | traj_rmse=%.4f | relative_l2=%.4f | "
                    "ic_rmse=%.4f",
                    eval_metrics.get("traj_rmse", 0),
                    eval_metrics.get("relative_l2", 0),
                    eval_metrics.get("ic_rmse", 0),
                )
                all_metrics[f"eval_epoch_{epoch + 1}"] = eval_metrics

            # Periodic checkpointing
            if (epoch + 1) % self._config.checkpointing.save_every == 0:
                self._save_checkpoint(epoch, train_metrics["avg_loss"])

            # Best model
            if self._config.checkpointing.keep_best:
                if train_metrics["avg_loss"] < self._best_loss:
                    self._best_loss = train_metrics["avg_loss"]
                    self._save_checkpoint(
                        epoch, train_metrics["avg_loss"], best=True,
                    )

        all_metrics["final_train_loss"] = train_metrics.get(
            "avg_loss", float("inf"),
        )
        logger.info("Training complete. Best loss: %.4f", self._best_loss)
        return all_metrics

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Single training epoch. Returns averaged metrics."""
        self._model.train()
        tc = self._config.training
        self._optimizer.zero_grad()

        running_loss = 0.0
        running_components: Dict[str, float] = {}
        n_steps = 0
        n_optim_steps = 0
        n_total = len(self._train_loader)

        for step, batch in enumerate(self._train_loader):
            ic = batch["ic"].to(self._device)
            params = batch["params"].to(self._device)
            b = ic.shape[0]

            # Quantization-aware: use pre-decoded θ̂/IC_hat from cache
            if self._qa_mode:
                sample_indices = batch["sample_idx"]  # [B] original HDF5 indices
                params = self._qa_theta[sample_indices].to(self._device)
                ic = self._qa_ic[sample_indices].to(self._device)

            # GT trajectory: generate before forward pass (needed by both
            # single-window and multi-window paths for alignment)
            needs_gt = (
                self._loss_fn._lambda_traj > 0
                or self._loss_fn._lambda_ic > 0
                or self._loss_fn._lambda_feat_mse > 0
            )
            gt_trajectory = None
            if needs_gt:
                with torch.no_grad():
                    if hasattr(self._replayer, "rollout_batch"):
                        gt_trajectory = self._replayer.rollout_batch(
                            params_batch=params,
                            ics=ic,
                            timesteps=tc.timesteps,
                            return_all_steps=True,
                        )  # [B, T+1, C, H, W] on CPU
                        gt_trajectory = gt_trajectory.to(self._device)
                    else:
                        gt_trajs = []
                        for i in range(b):
                            gt_traj = self._replayer.rollout(
                                params_vector=params[i],
                                ic=ic[i],
                                timesteps=tc.timesteps,
                            )  # [1, T+1, C, H, W]
                            gt_trajs.append(gt_traj)
                        gt_trajectory = torch.cat(gt_trajs, dim=0)

            # GT raw features for optional feature MSE
            gt_raw_features = batch.get("gt_raw_temporal")
            if gt_raw_features is not None:
                gt_raw_features = gt_raw_features.to(self._device)

            # GT tokens for optional token CE loss (available with token store,
            # regardless of whether full QA predecoding is active)
            gt_tokens = None
            gt_indicators = None
            if self._token_store is not None:
                sample_indices = batch["sample_idx"]
                gt_tokens = self._token_store.get_batch(sample_indices)
                gt_tokens = {k: v.to(self._device) for k, v in gt_tokens.items()}
                # Binary indicator vectors for soft contrastive loss
                if hasattr(self._token_store, 'get_indicators'):
                    gt_indicators = self._token_store.get_indicators(sample_indices)

            # Forward + Loss + Backward
            # Branches on multi-window vs single-window BPTT
            if self._bptt.num_windows > 1:
                # Multi-window: per-window forward + backward.
                # Each window's backward() frees its computation graph,
                # keeping peak activation memory = O(W) regardless of N.
                segments = self._bptt.multi_window_rollout(
                    ic, params=params,
                )
                window_loss_total = 0.0
                window_components: Dict[str, float] = {}

                for pred_seg, win_start in segments:
                    if needs_gt:
                        pred_w, gt_w = self._bptt.align_window_for_loss(
                            pred_seg, gt_trajectory, win_start,
                        )
                    else:
                        pred_w = pred_seg[:, 1:]
                        gt_w = None

                    loss_out = self._loss_fn.compute(
                        pred_w, gt_w,
                        params=params,
                        gt_raw_features=gt_raw_features,
                        gt_tokens=gt_tokens,
                        gt_indicators=gt_indicators,
                    )
                    # Scale by both num_windows and grad accumulation so the
                    # effective gradient magnitude matches single-window
                    scale = (
                        tc.gradient_accumulation_steps * self._bptt.num_windows
                    )
                    (loss_out.total / scale).backward()
                    window_loss_total += loss_out.total.item()
                    for k, v in loss_out.metrics.items():
                        window_components[k] = (
                            window_components.get(k, 0.0) + v
                        )

                # Average across windows for running metrics
                n_win = self._bptt.num_windows
                running_loss += window_loss_total / n_win
                for k, v in window_components.items():
                    running_components[k] = (
                        running_components.get(k, 0.0) + v / n_win
                    )
            else:
                # Single-window (legacy path, unchanged)
                pred_trajectory = self._bptt.rollout(
                    ic, params=params,
                )  # [B, W+1, C, H, W]

                if needs_gt:
                    pred_aligned, gt_aligned = self._bptt.align_for_loss(
                        pred_trajectory, gt_trajectory,
                    )  # [B, W, C, H, W] each
                else:
                    pred_aligned = pred_trajectory[:, 1:]
                    gt_aligned = None

                loss_output = self._loss_fn.compute(
                    pred_aligned, gt_aligned,
                    params=params,
                    gt_raw_features=gt_raw_features,
                    gt_tokens=gt_tokens,
                    gt_indicators=gt_indicators,
                )
                scaled_loss = loss_output.total / tc.gradient_accumulation_steps
                scaled_loss.backward()

                running_loss += loss_output.total.item()
                for k, v in loss_output.metrics.items():
                    running_components[k] = running_components.get(k, 0.0) + v

            n_steps += 1

            # Per-step logging (running averages)
            # Only compute gnorm at accumulation boundaries — that's where
            # the gradient is fully accumulated and meaningful. On intermediate
            # micro-steps the gradient is partial (1/accum_steps scaled), so
            # reporting it is misleading and wastes time iterating all params.
            is_accum_step = (step + 1) % tc.gradient_accumulation_steps == 0
            avg = running_loss / n_steps
            if is_accum_step:
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self._model.parameters()
                    if p.grad is not None
                ) ** 0.5
            else:
                grad_norm = float("nan")
            lr_now = self._optimizer.param_groups[0]["lr"]
            avg_components = {
                k: v / n_steps
                for k, v in sorted(running_components.items())
                if k != "total"
            }
            parts = " | ".join(
                f"{k}={v:.4f}" for k, v in sorted(avg_components.items())
            )
            logger.info(
                "  [%d/%d] step %d/%d | avg=%.4f | gnorm=%.4f | "
                "lr=%.2e | %s",
                epoch + 1, tc.epochs, step + 1, n_total, avg,
                grad_norm, lr_now, parts,
            )

            # Optimizer step (after gradient accumulation)
            if (step + 1) % tc.gradient_accumulation_steps == 0:
                # Multi-window BPTT: 3× backward passes per micro-batch
                # fragments the CUDA allocator cache. Flush before
                # optimizer.step() allocates Adam state tensors.
                if self._bptt.num_windows > 1:
                    torch.cuda.empty_cache()
                if tc.clip_grad:
                    clip_grad_norm_(self._model.parameters(), tc.clip_grad)
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()
                n_optim_steps += 1

        # Handle leftover accumulated gradients
        if n_steps % tc.gradient_accumulation_steps != 0:
            if tc.clip_grad:
                clip_grad_norm_(self._model.parameters(), tc.clip_grad)
            self._optimizer.step()
            self._scheduler.step()
            self._optimizer.zero_grad()
            n_optim_steps += 1

        # Average metrics
        metrics = {"avg_loss": running_loss / max(n_steps, 1)}
        for k, v in running_components.items():
            metrics[k] = v / max(n_steps, 1)
        metrics["n_optim_steps"] = float(n_optim_steps)
        return metrics

    def _save_checkpoint(
        self,
        epoch: int,
        loss: float,
        best: bool = False,
    ) -> None:
        """Save model + optimizer + scheduler + config checkpoint."""
        name = "best.pt" if best else f"epoch_{epoch + 1:03d}.pt"
        path = self._save_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "loss_fn_state_dict": self._loss_fn.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scheduler_state_dict": self._scheduler.state_dict(),
                "loss": loss,
                "config": self._config.model_dump(),
            },
            path,
        )
        logger.info("Checkpoint saved: %s (loss=%.4f)", path, loss)
