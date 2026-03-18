"""Trajectory evaluation for V2 MNO.

Metrics:
  - traj_rmse: Root mean squared error between predicted and GT trajectories
  - relative_l2: Per-sample ||pred - gt||₂ / ||gt||₂, averaged (0.10 = 10% error)
  - gt_rms: RMS magnitude of GT trajectories (contextualizes RMSE)
  - ic_rmse: RMSE at the first supervised timestep (warmup endpoint quality)

Note: Contrastive quality (mean_jaccard, rank_corr) is a training-time concern
measured via SoftTokenContrastiveLoss. Eval focuses on trajectory accuracy.
"""

import logging
from typing import Dict

import torch
from torch.utils.data import DataLoader

from spinlock.mno.truncated_bptt import TruncatedBPTT

logger = logging.getLogger(__name__)


class TrajectoryEvaluator:
    """Trajectory-level evaluation for V2 MNO."""

    def __init__(
        self,
        replayer,
        bptt: TruncatedBPTT,
    ) -> None:
        self._replayer = replayer
        self._bptt = bptt

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        n_samples: int = 50,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Evaluate trajectory quality on a sample of the dataset.

        Args:
            model: V2MNO model (switched to eval mode internally).
            dataloader: Validation DataLoader.
            n_samples: Max samples to evaluate.
            device: Computation device.

        Returns:
            Dict of metric name -> value.
        """
        model.eval()
        n_evaluated = 0
        sum_traj_mse = 0.0
        sum_ic_mse = 0.0
        sum_relative_l2 = 0.0
        sum_gt_sq = 0.0

        for batch in dataloader:
            if n_evaluated >= n_samples:
                break

            ic = batch["ic"].to(device)
            params = batch["params"].to(device)
            b = ic.shape[0]
            remaining = n_samples - n_evaluated
            if b > remaining:
                ic, params = ic[:remaining], params[:remaining]
                b = remaining

            # Generate GT trajectories via replayer (per-sample)
            gt_trajs = []
            for i in range(b):
                gt_traj = self._replayer.rollout(
                    params_vector=params[i],
                    ic=ic[i],
                    timesteps=self._bptt.timesteps,
                )  # [1, T+1, C, H, W]
                gt_trajs.append(gt_traj)
            gt_trajectory = torch.cat(gt_trajs, dim=0)  # [B, T+1, C, H, W]

            # MNO rollout via BPTT
            pred_trajectory = self._bptt.rollout(
                ic, params=params,
            )  # [B, W+1, C, H, W]

            # Align for comparison
            pred_aligned, gt_aligned = self._bptt.align_for_loss(
                pred_trajectory, gt_trajectory,
            )  # [B, W, C, H, W] each

            # Trajectory RMSE (per-element, averaged over batch)
            mse = (pred_aligned - gt_aligned).pow(2).mean()
            sum_traj_mse += mse.item() * b

            # IC RMSE (first supervised timestep)
            ic_mse = (pred_aligned[:, 0] - gt_aligned[:, 0]).pow(2).mean()
            sum_ic_mse += ic_mse.item() * b

            # Per-sample relative L2: ||pred_i - gt_i||₂ / ||gt_i||₂
            per_sample_diff = (
                (pred_aligned - gt_aligned).pow(2).flatten(1).sum(1).sqrt()
            )  # [B]
            per_sample_gt = (
                gt_aligned.pow(2).flatten(1).sum(1).sqrt().clamp(min=1e-8)
            )  # [B]
            sum_relative_l2 += (per_sample_diff / per_sample_gt).sum().item()

            # GT signal magnitude (for contextualizing RMSE)
            sum_gt_sq += gt_aligned.pow(2).mean().item() * b

            n_evaluated += b

        metrics: Dict[str, float] = {"n_evaluated": float(n_evaluated)}
        if n_evaluated > 0:
            metrics["traj_rmse"] = (sum_traj_mse / n_evaluated) ** 0.5
            metrics["ic_rmse"] = (sum_ic_mse / n_evaluated) ** 0.5
            metrics["relative_l2"] = sum_relative_l2 / n_evaluated
            metrics["gt_rms"] = (sum_gt_sq / n_evaluated) ** 0.5

        model.train()
        return metrics
