"""NLTokenizer training orchestration.

Handles the complete training loop for NLTokenizerModel + LFMAdapter + NLListener,
including:
- VAE warmup (KL weight ramp, no listener)
- Full VAE + listener (listener roundtrip enabled)
- Gradient flow through Gumbel-Softmax to generator input projection and listener
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base_trainer import BaseTokenizerTrainer
from .nl_config import NLTokenizerConfig
from .nl_model import NLTokenizerModel
from .nl_losses import NLTokenizerLoss
from .nl_lfm_adapter import LFMAdapter, NLListener
from .nl_checkpoint import save_nl_checkpoint

logger = logging.getLogger(__name__)


class NLTokenizerTrainer(BaseTokenizerTrainer):
    """Training orchestrator for NLTokenizer.

    Training stages:
        1. VAE warmup (epochs 0 → kl_warmup_epochs): KL weight ramps 0 → full.
           Recon + inverse losses only. Generator and listener NOT connected.
        2. Full VAE + listener (epochs kl_warmup_epochs+): Enable listener
           roundtrip loss. Generator input projection and listener train jointly.
           Frozen decoder produces token_probs; gradients flow through
           Gumbel-Softmax to projection and through soft embeddings to listener.

    Args:
        model: NLTokenizerModel instance
        adapter: LFMAdapter (wraps LFM generator)
        listener: NLListener (text → z decoder)
        config: NLTokenizerConfig
        group_indices: Feature group mapping
        normalization_stats: Optional normalization stats
    """

    def __init__(
        self,
        model: NLTokenizerModel,
        adapter: LFMAdapter,
        listener: NLListener,
        config: NLTokenizerConfig,
        group_indices: Dict[str, list],
        normalization_stats: Optional[Dict] = None,
        replayer=None,
    ):
        # Shared: device, optimizer, scheduler, warmup, tracking
        super().__init__(
            model, config, group_indices,
            normalization_stats=normalization_stats,
        )

        self.replayer = replayer
        self.adapter = adapter.to(self.device)
        self.listener = listener.to(self.device)

        # Add adapter and listener trainable params to optimizer
        # (frozen decoder params already have requires_grad=False)
        extra_params = (
            list(adapter.parameters())
            + list(listener.parameters())
        )
        trainable_extra = [p for p in extra_params if p.requires_grad]
        if trainable_extra:
            self.optimizer.add_param_group({"params": trainable_extra})
            logger.info(
                "Added %d trainable params from adapter+listener to optimizer",
                sum(p.numel() for p in trainable_extra),
            )

        # Loss function
        self.loss_fn = NLTokenizerLoss(config.loss)

        # Training stage tracking
        self._kl_warmup_epochs = config.training.kl_warmup_epochs
        self._listener_start_epoch = config.training.listener_start_epoch

    # ──────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────

    def train(
        self,
        *,
        temporal_features: Optional[torch.Tensor] = None,
        initial_manual: Optional[torch.Tensor] = None,
        theta_features: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        temporal_lengths: Optional[torch.Tensor] = None,
        dataset: Optional[Dataset] = None,
        output_dir: Path = Path("checkpoints"),
        checkpoint_prefix: str = "nl_tokenizer",
    ) -> Dict[str, Any]:
        """Run complete NLTokenizer training loop.

        Two data paths (same as VQ):
        1. Tensor path (manual mode): pre-extracted features as tensors.
        2. Dataset path (learned mode): lazy Dataset with dict batches.
           Trajectories generated on-the-fly via replayer.

        Args:
            temporal_features: [N, T, D_t] temporal sequences (manual mode)
            initial_manual: [N, D_i] initial condition features (manual mode)
            theta_features: [N, param_dim] operator parameters (manual mode)
            temporal_mask: [N, T] validity mask (manual mode)
            temporal_lengths: [N] actual sequence lengths (manual mode)
            dataset: Lazy Dataset returning dict batches (learned mode)
            output_dir: Checkpoint save directory
            checkpoint_prefix: Checkpoint filename prefix

        Returns:
            Training history dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir
        self._checkpoint_prefix = checkpoint_prefix

        # Create dataloaders — dataset path (learned) or tensor path (manual)
        if dataset is not None:
            train_loader, val_loader = self._create_dataloaders_from_dataset(dataset)
        else:
            train_loader, val_loader = self._create_dataloaders(
                temporal_features=temporal_features,
                initial_manual=initial_manual,
                theta_features=theta_features,
                temporal_mask=temporal_mask,
                temporal_lengths=temporal_lengths,
            )

        cfg = self.config.training
        best_val_loss = float("inf")

        for epoch in range(cfg.num_epochs):
            # ── KL warmup schedule ──
            if self._kl_warmup_epochs > 0 and epoch < self._kl_warmup_epochs:
                kl_scale = epoch / self._kl_warmup_epochs
            else:
                kl_scale = 1.0
            self.loss_fn.set_kl_weight_scale(kl_scale)

            # ── Listener enable ──
            listener_enabled = epoch >= self._listener_start_epoch

            # ── Train epoch ──
            train_metrics = self._train_epoch(
                train_loader, epoch, listener_enabled=listener_enabled,
            )
            self.training_history["train_losses"].append(train_metrics["loss"])
            self.training_history["train_metrics"].append(train_metrics)

            # ── Validation ──
            if (epoch + 1) % cfg.val_every_n_epochs == 0 or epoch == cfg.num_epochs - 1:
                val_metrics = self._validate_epoch(
                    val_loader, listener_enabled=listener_enabled,
                )
                self.training_history["val_losses"].append(val_metrics["loss"])
                self.training_history["val_metrics"].append(val_metrics)

                val_loss = val_metrics["loss"]
                if val_loss < best_val_loss - cfg.early_stopping_min_delta:
                    best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(
                        output_dir / f"{checkpoint_prefix}_best.pt",
                        epoch, val_loss,
                    )
                else:
                    self.epochs_without_improvement += 1

                logger.info(
                    "Epoch %d/%d — train_loss=%.4f val_loss=%.4f recon=%.4f "
                    "kl=%.4f(×%.2f) θ_inv=%.4f listener=%.4f%s",
                    epoch + 1, cfg.num_epochs,
                    train_metrics["loss"], val_loss,
                    val_metrics["reconstruction"], val_metrics["kl"], kl_scale,
                    val_metrics["theta_inverse"],
                    val_metrics["listener_roundtrip"],
                    " [listener ON]" if listener_enabled else "",
                )

                if self.epochs_without_improvement >= cfg.early_stopping_patience:
                    logger.info("Early stopping after %d epochs without improvement", epoch + 1)
                    break

            # ── LR scheduler ──
            if self.scheduler is not None:
                warmup_epochs = getattr(cfg, "warmup_epochs", 0)
                if epoch >= warmup_epochs:
                    self.scheduler.step()

            # ── Periodic checkpoint ──
            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self._save_checkpoint(
                    output_dir / f"{checkpoint_prefix}_epoch{epoch+1}.pt",
                    epoch, train_metrics["loss"],
                )

        # Final checkpoint
        self._save_checkpoint(
            output_dir / f"{checkpoint_prefix}_final.pt",
            cfg.num_epochs - 1, train_metrics["loss"],
        )

        logger.info("NLTokenizer training complete")
        return self.training_history

    # ──────────────────────────────────────────────────────────────
    # Trajectory generation (learned mode)
    # ──────────────────────────────────────────────────────────────

    def _generate_trajectories(
        self,
        theta_feats: torch.Tensor,
        initial_raw: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Generate trajectories on-the-fly using the replayer.

        Args:
            theta_feats: [B, param_dim] on device
            initial_raw: [B, C, H, W] on device

        Returns:
            [B, T+1, C, H, W] on CPU, or None if replayer unavailable.
        """
        if self.replayer is None or theta_feats is None or initial_raw is None:
            return None

        timesteps = getattr(self.config, "generation_timesteps", None) or 64

        if hasattr(self.replayer, "rollout_batch"):
            return self.replayer.rollout_batch(
                params_batch=theta_feats.cpu(),
                ics=initial_raw,
                timesteps=timesteps,
                return_all_steps=True,
            )
        else:
            trajectories = []
            B = theta_feats.shape[0]
            for i in range(B):
                traj = self.replayer.rollout(
                    params_vector=theta_feats[i].cpu(),
                    ic=initial_raw[i],
                    timesteps=timesteps,
                    return_all_steps=True,
                )
                trajectories.append(traj.squeeze(0).cpu())
            return torch.stack(trajectories, dim=0)

    # ──────────────────────────────────────────────────────────────
    # Forward step (shared by train and val)
    # ──────────────────────────────────────────────────────────────

    def _forward_step(
        self,
        batch,
        listener_enabled: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run one forward step: unpack → generate trajectories → model → loss.

        Returns:
            (loss_dict, outputs) tuple
        """
        unpacked = self._unpack_batch(batch)

        # In learned mode, generate trajectories on-the-fly
        temporal_raw = None
        if self.replayer is not None:
            temporal_raw = self._generate_trajectories(
                unpacked.get("theta_features"),
                unpacked.get("initial_raw"),
            )
            if temporal_raw is not None:
                temporal_raw = temporal_raw.to(self.device)

        # Model forward
        outputs = self.model(
            temporal_features=unpacked.get("temporal_features"),
            initial_manual=unpacked.get("initial_manual"),
            theta_features=unpacked.get("theta_features"),
            temporal_mask=unpacked.get("temporal_mask"),
            temporal_lengths=unpacked.get("temporal_lengths"),
            temporal_raw=temporal_raw,
        )

        # Listener roundtrip
        z_hat = None
        if listener_enabled:
            gen_out = self.adapter.generate(outputs["z"])
            z_hat = self.listener(gen_out["token_probs"], gen_out["mask"])

        # Loss — uses behavioral equivalence for inverse losses
        # theta_encoded: ground truth theta's encoding (from family_embeddings)
        # theta_hat_encoded: re-encoded predicted theta (from model forward)
        family_embs = outputs.get("family_embeddings", {})
        loss_dict = self.loss_fn(
            h=outputs["h"],
            h_hat=outputs["h_hat"],
            mu=outputs["mu"],
            logvar=outputs["logvar"],
            theta_encoded=family_embs.get("theta"),
            theta_hat_encoded=outputs.get("theta_hat_encoded"),
            ic=unpacked.get("initial_manual"),
            ic_hat=outputs.get("ic_hat"),
            z=outputs["z"],
            z_hat=z_hat,
            listener_enabled=listener_enabled,
        )

        return loss_dict, outputs

    # ──────────────────────────────────────────────────────────────
    # Epoch-level methods
    # ──────────────────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        listener_enabled: bool = False,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.adapter.train()
        self.listener.train()

        accum_steps = self.config.training.gradient_accumulation_steps
        log_every = self.config.training.log_every_n_batches
        running: Dict[str, float] = {}
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            loss_dict, _ = self._forward_step(batch, listener_enabled)

            loss = loss_dict["loss"] / accum_steps
            loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(loader) - 1:
                clip_norm = self.config.training.gradient_clip_norm
                if clip_norm is not None:
                    all_params = (
                        list(self.model.parameters())
                        + list(self.adapter.parameters())
                        + list(self.listener.parameters())
                    )
                    nn.utils.clip_grad_norm_(all_params, clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self._warmup_scheduler is not None and self._warmup_steps_done < self._warmup_batches:
                    self._warmup_scheduler.step()
                    self._warmup_steps_done += 1

            # Accumulate metrics
            for key, val in loss_dict.items():
                v = val.item() if hasattr(val, "item") else float(val)
                running[key] = running.get(key, 0.0) + v
            n_batches += 1

            if (batch_idx + 1) % log_every == 0:
                logger.info(
                    "  [%d/%d] loss=%.4f recon=%.4f kl=%.4f θ=%.4f listener=%.4f",
                    batch_idx + 1, len(loader),
                    running["loss"] / n_batches,
                    running["reconstruction"] / n_batches,
                    running["kl"] / n_batches,
                    running["theta_inverse"] / n_batches,
                    running["listener_roundtrip"] / n_batches,
                )

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def _validate_epoch(
        self,
        loader: DataLoader,
        listener_enabled: bool = False,
    ) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        self.adapter.eval()
        self.listener.eval()

        running: Dict[str, float] = {}
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                loss_dict, _ = self._forward_step(batch, listener_enabled)
                for key, val in loss_dict.items():
                    v = val.item() if hasattr(val, "item") else float(val)
                    running[key] = running.get(key, 0.0) + v
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ──────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────

    def _save_checkpoint(
        self, path: Path, epoch: int, val_loss: float,
    ) -> None:
        """Save NLTokenizer checkpoint."""
        save_nl_checkpoint(
            path=path,
            model=self.model,
            adapter=self.adapter,
            listener=self.listener,
            config=self.config,
            group_indices=self.group_indices,
            optimizer=self.optimizer,
            normalization_stats=self.normalization_stats,
            epoch=epoch,
            val_loss=val_loss,
            metadata=self._build_resume_metadata(),
            temporal_input_dim=getattr(self.model, "_temporal_input_dim", None),
            theta_param_dim=getattr(self.model, "_theta_param_dim", None),
            initial_input_dim=getattr(self.model, "_initial_input_dim", None),
        )

    def _build_resume_metadata(self) -> Dict[str, Any]:
        """Build metadata for checkpoint resume."""
        meta: Dict[str, Any] = {
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        if self.scheduler is not None:
            meta["scheduler_state_dict"] = self.scheduler.state_dict()
        return meta
