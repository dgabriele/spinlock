"""Retrain ThetaInverseMLP with wider hidden_dim on frozen tokenizer.

Loads a trained VQTokenizer checkpoint, replaces the theta inverse head
with a wider MLP (hidden_dim=128), trains only the inverse head while
keeping the encoder/VQ/decoder frozen, and saves the updated checkpoint.

Strategy:
  1. Load full model + dataset (streaming raw ICs from HDF5 in batches)
  2. Apply feature cleaning (247→152 temporal features) from checkpoint metadata
  3. Precompute reconstructed_theta representations through frozen pipeline
  4. Train only the inverse head on (reconstructed_theta → original_theta) pairs

Usage:
    poetry run python scripts/retrain_theta_inverse.py \
        --checkpoint checkpoints/qbm_tokenizer_50k/vq_tokenizer_best.pt \
        --dataset datasets/qbm_50k.h5 \
        --hidden-dim 128 \
        --epochs 100 \
        --output checkpoints/qbm_tokenizer_50k/vq_tokenizer_wider_inverse.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from spinlock.tokens.inverse_models import ThetaInverseMLP
from spinlock.tokens.checkpoint import load_checkpoint, save_checkpoint
from spinlock.tokens.model import JointHierarchicalVQVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(args):
    device = torch.device(args.device)

    # 1. Load existing checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    ckpt = load_checkpoint(Path(args.checkpoint))
    config = ckpt.config

    # 2. Update config: widen theta inverse hidden_dim
    old_hidden_dim = config.inverse_heads.theta_hidden_dim
    config.inverse_heads.theta_hidden_dim = args.hidden_dim
    logger.info(f"Theta inverse hidden_dim: {old_hidden_dim} → {args.hidden_dim}")

    # 3. Create model with ORIGINAL config first (to load all weights)
    original_config = ckpt.config.model_copy(deep=True)
    original_config.inverse_heads.theta_hidden_dim = old_hidden_dim
    model_original = JointHierarchicalVQVAE(
        original_config,
        ckpt.group_indices,
        temporal_input_dim=ckpt.temporal_input_dim,
        initial_input_dim=ckpt.initial_input_dim,
    )
    model_original.load_state_dict(ckpt.model_state_dict)
    model_original.eval()
    model_original = model_original.to(device)
    logger.info("Loaded original model for precomputing reconstructed theta")

    # 4. Get feature cleaning indices from checkpoint metadata
    kept_indices = None
    if ckpt.feature_metadata and "temporal" in ckpt.feature_metadata.families:
        kept_indices = ckpt.feature_metadata.families["temporal"].kept_feature_indices
        logger.info(f"Temporal feature cleaning: 247 → {len(kept_indices)} features")

    # 5. Get normalization stats
    norm_stats = ckpt.normalization_stats
    t_mean = t_std = i_mean = i_std = None
    if norm_stats and "temporal" in norm_stats:
        t_mean = torch.tensor(norm_stats["temporal"]["mean"], dtype=torch.float32)
        t_std = torch.tensor(norm_stats["temporal"]["std"], dtype=torch.float32).clamp(min=1e-8)
    if norm_stats and "initial_manual" in norm_stats:
        i_mean = torch.tensor(norm_stats["initial_manual"]["mean"], dtype=torch.float32)
        i_std = torch.tensor(norm_stats["initial_manual"]["std"], dtype=torch.float32).clamp(min=1e-8)

    # 6. Precompute reconstructed_theta by streaming through HDF5 in batches
    # (raw ICs are ~49GB, can't load all at once)
    logger.info(f"Precomputing reconstructed theta from {args.dataset} (streaming)...")
    reconstructed_theta_list = []
    theta_list = []
    batch_size = args.batch_size

    with h5py.File(args.dataset, "r") as f:
        N = f["parameters/params"].shape[0]
        theta_all_np = np.array(f["parameters/params"])  # [N, 9] — small, load fully
        temporal_all_np = np.array(f["features/temporal/features"])  # [N, T, F]
        initial_manual_np = np.array(f["features/initial/aggregated/features"])  # [N, D_i]

        logger.info(f"Dataset: {N} samples. Streaming raw ICs in batches of {batch_size}...")

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)

            # Load raw ICs for this batch from HDF5 (streaming)
            raw_ics = torch.tensor(
                np.array(f["inputs/fields"][start:end]),  # [B, 3, 2, H, W]
                dtype=torch.float32,
            )
            # The initial_hybrid encoder expects [B, C, H, W] — use first realization
            # inputs/fields is [B, M=3, C=2, H=64, W=64], take first realization
            initial_raw = raw_ics[:, 0, :, :, :]  # [B, 2, H, W]

            # Theta
            theta_b = torch.tensor(theta_all_np[start:end], dtype=torch.float32)

            # Temporal: apply cleaning + normalization
            temporal_b = torch.tensor(temporal_all_np[start:end], dtype=torch.float32)
            if kept_indices is not None:
                temporal_b = temporal_b[:, :, kept_indices]
            if t_mean is not None:
                temporal_b = (temporal_b - t_mean) / t_std

            # Initial manual: apply normalization
            initial_b = torch.tensor(initial_manual_np[start:end], dtype=torch.float32)
            if i_mean is not None:
                initial_b = (initial_b - i_mean) / i_std

            with torch.no_grad():
                outputs = model_original(
                    temporal_features=temporal_b.to(device),
                    theta_features=theta_b.to(device),
                    initial_manual=initial_b.to(device),
                    initial_raw=initial_raw.to(device),
                )
                reconstructed_theta_list.append(
                    outputs["reconstructed_split"]["theta"].cpu()
                )
                theta_list.append(theta_b)

            if (start // batch_size) % 20 == 0:
                logger.info(f"  Processed {end}/{N} samples...")

    reconstructed_theta_all = torch.cat(reconstructed_theta_list, dim=0)
    theta_all = torch.cat(theta_list, dim=0)
    logger.info(f"Precomputed: reconstructed_theta={reconstructed_theta_all.shape}, theta={theta_all.shape}")

    # Free original model memory
    del model_original
    torch.cuda.empty_cache()

    # 7. Create new model with wider inverse head
    model_new = JointHierarchicalVQVAE(
        config,  # Has updated theta_hidden_dim
        ckpt.group_indices,
        temporal_input_dim=ckpt.temporal_input_dim,
        initial_input_dim=ckpt.initial_input_dim,
    )
    # Load all weights except theta_inverse
    filtered_state = {
        k: v for k, v in ckpt.model_state_dict.items() if "theta_inverse" not in k
    }
    model_new.load_state_dict(filtered_state, strict=False)
    model_new = model_new.to(device)

    # Only theta_inverse is trainable
    inverse_head = model_new.theta_inverse
    logger.info(f"New inverse head: {inverse_head}")
    num_params = sum(p.numel() for p in inverse_head.parameters())
    logger.info(f"Trainable parameters: {num_params:,}")

    # 8. Train/val split
    val_size = int(N * 0.1)
    indices = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = indices[:N - val_size], indices[N - val_size:]

    train_dataset = TensorDataset(
        reconstructed_theta_all[train_idx], theta_all[train_idx]
    )
    val_dataset = TensorDataset(
        reconstructed_theta_all[val_idx], theta_all[val_idx]
    )
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    # 9. Optimizer
    optimizer = torch.optim.Adam(inverse_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    # 10. Training loop (fast — only inverse head on precomputed inputs)
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        inverse_head.train()
        train_loss = 0.0
        n_batches = 0
        for recon_b, theta_b in train_loader:
            recon_b = recon_b.to(device)
            theta_b = theta_b.to(device)

            theta_hat = inverse_head(recon_b)
            loss = criterion(theta_hat, theta_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # Validate
        inverse_head.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for recon_b, theta_b in val_loader:
                recon_b = recon_b.to(device)
                theta_b = theta_b.to(device)
                theta_hat = inverse_head(recon_b)
                val_loss += criterion(theta_hat, theta_b).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        scheduler.step(avg_val)

        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state = {k: v.cpu().clone() for k, v in inverse_head.state_dict().items()}
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or marker:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:3d}: train_mse={avg_train:.6f}, "
                f"val_mse={avg_val:.6f}, lr={lr:.2e}{marker}"
            )

    logger.info(f"Best val MSE: {best_val_mse:.6f}")

    # 11. Load best weights and save updated checkpoint
    inverse_head.load_state_dict(best_state)
    model_new = model_new.cpu()

    output_path = Path(args.output)
    save_checkpoint(
        path=output_path,
        model=model_new,
        config=config,
        group_indices=ckpt.group_indices,
        normalization_stats=ckpt.normalization_stats,
        val_loss=ckpt.val_loss,
        epoch=ckpt.epoch,
        metadata={
            **(ckpt.metadata or {}),
            "theta_inverse_retrained": True,
            "theta_inverse_hidden_dim": args.hidden_dim,
            "theta_inverse_val_mse": best_val_mse,
            "theta_inverse_old_hidden_dim": old_hidden_dim,
        },
        temporal_input_dim=ckpt.temporal_input_dim,
        initial_input_dim=ckpt.initial_input_dim,
        feature_metadata=ckpt.feature_metadata,
    )
    logger.info(f"Saved updated checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain theta inverse head with wider hidden_dim")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained tokenizer checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to QBM dataset HDF5")
    parser.add_argument("--hidden-dim", type=int, default=128, help="New hidden_dim for ThetaInverseMLP")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Precompute batch size (streaming ICs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    args = parser.parse_args()
    main(args)
