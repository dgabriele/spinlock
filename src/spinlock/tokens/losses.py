"""Loss functions for VQ-VAE training.

Implements the 6-component loss used in spinlock V2:
1. Reconstruction loss - MSE between input and reconstructed features
2. VQ loss - Vector quantization commitment loss (from quantizers)
3. Orthogonality loss - Minimize correlation between category representations
4. Informativeness loss - Maximize variance within each category
5. Topographic loss - Optional spatial organization of codebook
6. Roundtrip loss - Ensure decoded values re-encode to same tokens
"""

import logging
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig, RoundtripLossConfig, AuxHeadConfig
from .fsq import FiniteScalarQuantizer

logger = logging.getLogger(__name__)


def compute_reconstruction_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute reconstruction loss between original and reconstructed features.

    Args:
        original: Original encoded features [B, D]
        reconstructed: Reconstructed features [B, D]
        normalize: If True, normalize by feature dimension

    Returns:
        Scalar reconstruction loss
    """
    mse = F.mse_loss(reconstructed, original, reduction='mean')

    if normalize:
        # Normalize by feature dimension to make loss scale-invariant
        mse = mse / original.shape[1]

    return mse


def compute_inverse_reconstruction_loss(
    decoded: Dict[str, torch.Tensor],
    original_theta: Optional[torch.Tensor] = None,
    original_initial: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute reconstruction loss on inverse head outputs vs actual physical inputs.

    Unlike the encoded-space reconstruction loss, this measures whether the
    discrete tokens contain enough information to recover the actual physical
    inputs (operator parameters θ and initial conditions).

    Args:
        decoded: Dict with inverse head outputs:
            - "theta": [B, param_dim] decoded parameters (Sigmoid → [0,1])
            - "initial": [B, C, H, W] decoded initial conditions
        original_theta: [B, param_dim] ground-truth parameters in [0,1]
        original_initial: [B, C, H, W] ground-truth initial conditions

    Returns:
        (total_loss, metrics_dict) where metrics_dict has per-family losses
    """
    losses = []
    metrics = {}

    if "theta" in decoded and original_theta is not None:
        theta_loss = F.mse_loss(decoded["theta"], original_theta)
        losses.append(theta_loss)
        metrics["recon/theta"] = theta_loss.item()

    if "initial" in decoded and original_initial is not None:
        initial_loss = F.mse_loss(decoded["initial"], original_initial)
        losses.append(initial_loss)
        metrics["recon/initial"] = initial_loss.item()

    if losses:
        total = torch.stack(losses).mean()
    else:
        device = next(iter(decoded.values())).device if decoded else torch.device("cpu")
        total = torch.tensor(0.0, device=device)

    metrics["recon/total"] = total.item()
    return total, metrics


def compute_aux_head_losses(
    decoded: Dict[str, torch.Tensor],
    theta_gt: Optional[torch.Tensor],
    ic_gt: Optional[torch.Tensor],
    aux_config: AuxHeadConfig,
    trajectory_targets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute auxiliary cross-family supervision losses.

    Aux heads predict theta/IC from temporal tokens. When inverse heads are
    also active, aux outputs use "_aux" suffix keys (e.g. "theta_aux").

    Args:
        decoded: Dict with aux head outputs:
            - "theta" or "theta_aux": [B, param_dim] predicted parameters
            - "initial" or "initial_aux": [B, C, H, W] predicted ICs
            - "trajectory_prototype": [B, K, C, H, W] predicted keyframes
            - "theta_probe": [B, param_dim] pre-VQ theta prediction
            - "initial_probe": [B, C, H, W] pre-VQ IC prediction
        theta_gt: [B, param_dim] ground-truth parameters in [0,1]
        ic_gt: [B, C, H, W] ground-truth initial conditions
        trajectory_targets: [B, K, C, H, W] ground-truth keyframe images
        aux_config: Auxiliary head configuration with per-family weights

    Returns:
        (weighted_total_loss, metrics_dict)
    """
    losses = []
    metrics = {}

    # Check both "theta" and "theta_aux" keys (aux uses suffix when inverse also active)
    theta_key = "theta_aux" if "theta_aux" in decoded else "theta"
    if theta_key in decoded and theta_gt is not None:
        theta_loss = F.mse_loss(decoded[theta_key], theta_gt)
        losses.append(aux_config.theta_weight * theta_loss)
        metrics["aux/theta"] = theta_loss.item()

    initial_key = "initial_aux" if "initial_aux" in decoded else "initial"
    if initial_key in decoded and ic_gt is not None:
        ic_loss = F.mse_loss(decoded[initial_key], ic_gt)
        losses.append(aux_config.initial_weight * ic_loss)
        metrics["aux/initial"] = ic_loss.item()

    # Pre-VQ theta probe: direct CNN → theta gradient (no VQ bottleneck)
    if "theta_probe" in decoded and theta_gt is not None:
        probe_loss = F.mse_loss(decoded["theta_probe"], theta_gt)
        losses.append(aux_config.theta_probe_weight * probe_loss)
        metrics["aux/theta_probe"] = probe_loss.item()

    # Pre-VQ IC probe: direct CNN → IC gradient (no VQ bottleneck)
    if "initial_probe" in decoded and ic_gt is not None:
        ic_probe_loss = F.mse_loss(decoded["initial_probe"], ic_gt)
        losses.append(aux_config.initial_probe_weight * ic_probe_loss)
        metrics["aux/initial_probe"] = ic_probe_loss.item()

    # Trajectory prototype: K keyframe images decoded from quantized latents
    # Multi-scale MSE: penalize structure at multiple resolutions for sharper prototypes.
    # Scales are derived dynamically from the actual spatial size (H) to stay
    # adaptive to any grid resolution rather than hardcoding 32/64/128.
    if "trajectory_prototype" in decoded and trajectory_targets is not None:
        pred = decoded["trajectory_prototype"]   # [B, K, C, H, W]
        tgt = trajectory_targets                 # [B, K, C, H, W]

        B_t, K_t, C_t, H_t, W_t = pred.shape
        mse_full = F.mse_loss(pred, tgt)
        metrics["aux/traj_full"] = mse_full.item()

        # Build downsample scales: repeatedly halve until size < 16
        scale_losses = [mse_full]
        pred_flat = pred.reshape(B_t * K_t, C_t, H_t, W_t)
        tgt_flat = tgt.reshape(B_t * K_t, C_t, H_t, W_t)
        ds_size = H_t // 2
        scale_idx = 0
        while ds_size >= 16:
            pred_ds = F.interpolate(pred_flat, size=ds_size, mode='bilinear', align_corners=False)
            tgt_ds = F.interpolate(tgt_flat, size=ds_size, mode='bilinear', align_corners=False)
            mse_ds = F.mse_loss(pred_ds, tgt_ds)
            scale_losses.append(mse_ds)
            metrics[f"aux/traj_{ds_size}"] = mse_ds.item()
            ds_size //= 2
            scale_idx += 1

        traj_loss = sum(scale_losses) / len(scale_losses)
        losses.append(aux_config.trajectory_weight * traj_loss)
        metrics["aux/trajectory"] = traj_loss.item()

    if losses:
        total = sum(losses)
    else:
        device = next(iter(decoded.values())).device if decoded else torch.device("cpu")
        total = torch.tensor(0.0, device=device)

    metrics["aux/total"] = total.item() if hasattr(total, 'item') else total
    return total, metrics


def compute_orthogonality_loss(
    category_embeddings: Dict[str, torch.Tensor],
    target_correlation: float = 0.0,
) -> torch.Tensor:
    """Compute orthogonality loss between category representations.

    Encourages different categories to learn decorrelated representations
    by penalizing high correlations between category embeddings.

    Args:
        category_embeddings: Dict mapping category_name → embeddings [B, D_cat]
        target_correlation: Target correlation (default: 0.0 for full orthogonality)

    Returns:
        Scalar orthogonality loss
    """
    if len(category_embeddings) <= 1:
        return torch.tensor(0.0, device=next(iter(category_embeddings.values())).device)

    categories = sorted(category_embeddings.keys())
    correlations = []

    for i, cat_i in enumerate(categories):
        for cat_j in categories[i + 1:]:
            emb_i = category_embeddings[cat_i]  # [B, D_i]
            emb_j = category_embeddings[cat_j]  # [B, D_j]

            # Flatten each category's embeddings across features
            # Then compute correlation across batch dimension
            emb_i_flat = emb_i.view(emb_i.size(0), -1)  # [B, D_i]
            emb_j_flat = emb_j.view(emb_j.size(0), -1)  # [B, D_j]

            # Normalize across batch dimension for each feature
            emb_i_norm = F.normalize(emb_i_flat, p=2, dim=0)  # Normalize across batch
            emb_j_norm = F.normalize(emb_j_flat, p=2, dim=0)  # Normalize across batch

            # Compute correlation between categories (average across features)
            # For different dimensions, compute the correlation matrix and average
            corr_matrix = torch.matmul(emb_i_norm.t(), emb_j_norm)  # [D_i, D_j]
            corr = corr_matrix.abs().mean()
            correlations.append(corr)

    if not correlations:
        return torch.tensor(0.0, device=emb_i.device)

    # Penalize deviation from target correlation
    correlations = torch.stack(correlations)
    loss = (correlations - target_correlation).pow(2).mean()

    return loss


def compute_informativeness_loss(
    category_embeddings: Dict[str, torch.Tensor],
    min_variance: float = 0.01,
    mode: str = "log_barrier",
) -> torch.Tensor:
    """Compute informativeness loss to encourage high variance within categories.

    Two modes:
      - ``"floor"``: Original ReLU formulation. Only activates when variance
        drops below ``min_variance`` (collapse prevention safety net).
      - ``"log_barrier"``: Continuously active log-barrier. Penalizes any
        group whose variance falls below 1.0 with ``clamp(-log(var), min=0)``.
        Provides smooth gradients throughout training, preventing the CNN
        from drifting toward low-variance representations before collapse
        actually occurs.

    Args:
        category_embeddings: Dict mapping category_name → embeddings [B, D_cat]
        min_variance: Minimum target variance threshold (used in floor mode)
        mode: ``"floor"`` or ``"log_barrier"``

    Returns:
        Scalar informativeness loss (lower variance = higher loss)
    """
    variances = []

    for cat_name, embeddings in category_embeddings.items():
        # Skip direct theta param groups — their variance is fixed (raw params)
        # and would create irreducible loss that wastes optimization pressure.
        if cat_name.startswith("theta_param_"):
            continue
        # Compute variance along batch dimension for each feature
        var = embeddings.var(dim=0, unbiased=False).mean()  # Average across features
        variances.append(var)

    if not variances:
        return torch.tensor(0.0, device=next(iter(category_embeddings.values())).device)

    variances = torch.stack(variances)

    if mode == "floor":
        # Original: hard floor, only activates on near-collapse
        loss = F.relu(min_variance - variances).mean()
    else:
        # Log-barrier: -log(var) is 0 at var=1, positive for var<1
        # Clamp at 0 so we never reward variance > 1 (only penalize low variance)
        loss = torch.clamp(-torch.log(variances + 1e-8), min=0.0).mean()

    return loss


def compute_group_balance_loss(
    cnn_features: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Penalize per-group variance imbalance in CNN output features.

    Computes the coefficient of variation (CV) of per-group variances.
    CV = std(group_vars) / mean(group_vars).  CV=0 means perfectly balanced
    groups; the CNN learns to distribute variance evenly across groups.

    Supports both 3D [B, T, D] (legacy per-frame mode) and 2D [B, D]
    (pyramid-first mode where D = num_levels * D_agg).

    Args:
        cnn_features: CNN output [B, T, D] or [B, D] before group slicing.
        num_groups: Number of groups (D must be divisible by num_groups).

    Returns:
        Scalar balance loss (CV of per-group variances).
    """
    if cnn_features.dim() == 3:
        # Legacy: [B, T, D] — variance across batch and time
        D = cnn_features.shape[-1]
        d_sub = D // num_groups
        group_vars = [
            cnn_features[:, :, g * d_sub:(g + 1) * d_sub].var()
            for g in range(num_groups)
        ]
    elif cnn_features.dim() == 2:
        # Pyramid-first: [B, D_multi_res] — variance across batch only
        D = cnn_features.shape[-1]
        d_sub = D // num_groups
        group_vars = [
            cnn_features[:, g * d_sub:(g + 1) * d_sub].var()
            for g in range(num_groups)
        ]
    else:
        return torch.tensor(0.0, device=cnn_features.device)

    group_vars = torch.stack(group_vars)
    cv = group_vars.std() / (group_vars.mean() + 1e-8)
    return cv


def compute_topographic_loss(
    original: torch.Tensor,
    latent_vectors: torch.Tensor,
    quantized_vectors: torch.Tensor,
    n_samples: int = 64,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Compute topographic similarity loss (PRE and POST quantization).

    Preserves topology at TWO stages (matching v1 approach):
    1. PRE-quantization: Input → Latent (encoder quality)
    2. POST-quantization: Latent → Code (VQ quality)

    Args:
        original: Original features [B, D_in]
        latent_vectors: Pre-quantization latent vectors [B, D_latent]
        quantized_vectors: Post-quantization vectors [B, D_latent]
        n_samples: Number of samples for pairwise distance computation

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics contains:
            - topo_pre: Pre-quantization correlation [0, 1]
            - topo_post: Post-quantization correlation [0, 1]
    """
    batch_size = original.shape[0]
    device = original.device

    if batch_size < n_samples:
        n_samples = batch_size

    # Sample random indices for efficiency
    indices = torch.randperm(batch_size, device=device)[:n_samples]
    sampled_original = original[indices]
    sampled_latent = latent_vectors[indices]
    sampled_quantized = quantized_vectors[indices]

    # Compute pairwise distances in input space
    input_dists = torch.cdist(sampled_original, sampled_original, p=2)  # [n, n]

    # Compute pairwise distances in PRE-quantization latent space
    latent_dists = torch.cdist(sampled_latent, sampled_latent, p=2)  # [n, n]

    # Compute pairwise distances in POST-quantization space
    quantized_dists = torch.cdist(sampled_quantized, sampled_quantized, p=2)  # [n, n]

    # Flatten for correlation computation
    input_flat = input_dists.view(-1)
    latent_flat = latent_dists.view(-1)
    quantized_flat = quantized_dists.view(-1)

    # PRE-quantization correlation (input → latent)
    input_mean = input_flat.mean()
    latent_mean = latent_flat.mean()
    input_centered = input_flat - input_mean
    latent_centered = latent_flat - latent_mean

    pre_correlation = (input_centered * latent_centered).sum() / (
        input_centered.norm() * latent_centered.norm() + 1e-8
    )

    # POST-quantization correlation (latent → quantized)
    quantized_mean = quantized_flat.mean()
    quantized_centered = quantized_flat - quantized_mean

    post_correlation = (latent_centered * quantized_centered).sum() / (
        latent_centered.norm() * quantized_centered.norm() + 1e-8
    )

    # Total loss: penalize low correlation (correlation in [0, 1], loss in [0, 2])
    # Higher correlation = better topology preservation
    # Weight POST-quantization more heavily (0.75) since quantization quality is more critical
    # than encoder topology preservation (0.25)
    pre_loss = 1.0 - pre_correlation
    post_loss = 1.0 - post_correlation
    total_loss = 0.25 * pre_loss + 0.75 * post_loss

    metrics = {
        'topo_pre': pre_correlation.item(),
        'topo_post': post_correlation.item(),
    }

    return total_loss, metrics


class RoundtripConsistencyLoss(nn.Module):
    """
    Roundtrip consistency loss: decoded values should re-encode to same tokens.

    Ensures that decode(tokens) → encode(decode(tokens)) produces the same tokens,
    creating self-consistent equivalence classes in the latent space.

    For each family:
      - **theta**: decoded params → ThetaEncoder → per-group projector → VQ → check tokens
      - **initial**: decoded ICs → InitialCNNEncoder → per-group projector → VQ → check tokens
      - **temporal** (learned mode): decoded CNN features [B, T_rt, D] → split per group →
        real PyramidTemporalEncoders → per-group projector → VQ → check tokens
    """

    def __init__(
        self,
        theta_weight: float = 1.0,
        initial_weight: float = 1.0,
        temporal_weight: float = 1.0,
    ):
        super().__init__()
        self.theta_weight = theta_weight
        self.initial_weight = initial_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        model: Any,
        tokens: Dict[str, torch.Tensor],
        decoded: Dict[str, torch.Tensor],
        initial_manual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute roundtrip consistency loss for all families.

        Args:
            model: JointHierarchicalVQVAE instance
            tokens: Original tokens per quantizer
            decoded: Decoded continuous values (theta → params, initial → ICs,
                     temporal → synthetic CNN features [B, T_rt, D_cnn])
            initial_manual: Manual features for initial encoder (if needed)

        Returns:
            (total_loss, metrics_dict)
        """
        losses = []
        metrics = {}
        device = next(iter(decoded.values())).device

        # ── Re-encode each family through its REAL encoder ──
        rt_per_group: Dict[str, torch.Tensor] = {}

        # Theta: decoded params → re-encode for roundtrip
        if 'theta' in decoded:
            if getattr(model, '_theta_direct', False):
                # Direct mode: each param is its own VQ group.
                # Re-encoding is identity — split decoded params per group.
                theta_decoded = decoded['theta']  # [B, param_dim]
                param_idx = 0
                for fc in sorted(model.group_indices):
                    if not fc.startswith("theta_"):
                        continue
                    n_features = len(model.group_indices[fc])
                    rt_per_group[fc] = theta_decoded[:, param_idx:param_idx + n_features]
                    param_idx += n_features
            else:
                theta_encoded_rt = model.theta_encoder(decoded['theta'])
                for fc in model.group_indices:
                    if fc.startswith("theta_"):
                        rt_per_group[fc] = theta_encoded_rt

        # Initial: prefer latent-space path (skip CNN) if available.
        # The latent decoder maps FSQ codes → pre-encoder latent space directly,
        # bypassing the lossy pixel-space decode → CNN re-encode cycle.
        if 'initial_latent' in decoded:
            initial_encoded_rt = decoded['initial_latent']
        elif 'initial' in decoded:
            initial_encoded_rt = self._encode_initial(
                model, decoded['initial'], initial_manual
            )
        else:
            initial_encoded_rt = None

        if initial_encoded_rt is not None:
            # SpatialICEncoder returns flat [B, G²×D] — split per position
            from .encoders.initial_spatial import SpatialICEncoder
            if isinstance(model.initial_encoder, SpatialICEncoder):
                enc = model.initial_encoder
                for i in range(enc.num_positions):
                    fc = f"initial_spatial_{i}"
                    if fc in model.group_indices:
                        start = i * enc.spatial_token_dim
                        end = start + enc.spatial_token_dim
                        rt_per_group[fc] = initial_encoded_rt[:, start:end]
            else:
                for fc in model.group_indices:
                    if fc.startswith("initial_"):
                        rt_per_group[fc] = initial_encoded_rt

        # Temporal: prefer latent bypass (skip shared decoder) if available.
        # Latent path: quantized temporal → MLP → per-group D_group embeddings.
        # Fallback: decoded CNN features → per_group_pyramid_encoders → project.
        if 'temporal_latent' in decoded:
            temporal_latent = decoded['temporal_latent']  # [B, num_groups × D_group]
            learned_cfg = model.config.encoder.temporal.learned
            group_dim = getattr(
                model, '_temporal_group_dim',
                learned_cfg.embedding_dim // learned_cfg.num_groups,
            )
            group_idx = 0
            for fc in sorted(model.group_indices):
                if not fc.startswith("temporal_"):
                    continue
                start = group_idx * group_dim
                end = start + group_dim
                rt_per_group[fc] = temporal_latent[:, start:end]  # [B, D_group]
                group_idx += 1
        elif 'temporal' in decoded and model.per_group_pyramid_encoders is not None:
            temporal_cnn_rt = decoded['temporal']  # [B, T_rt, D_cnn]
            learned_cfg = model.config.encoder.temporal.learned
            # In pyramid_first mode, _temporal_group_dim = D_group (embedding_dim).
            # In per_frame mode, group_dim = embedding_dim // num_groups.
            group_dim = getattr(
                model, '_temporal_group_dim',
                learned_cfg.embedding_dim // learned_cfg.num_groups,
            )
            # Projection layers for pyramid_first mode: sum(level_dims) → D_group
            rt_projections = getattr(model, '_temporal_rt_projections', None)
            group_idx = 0
            for fc in sorted(model.group_indices):
                if not fc.startswith("temporal_"):
                    continue
                start = group_idx * group_dim
                end = start + group_dim
                group_features = temporal_cnn_rt[:, :, start:end]  # [B, T_rt, group_dim]
                encoder = model.per_group_pyramid_encoders[fc]
                enc = encoder(group_features)  # [B, pyramid_out_dim]
                # In pyramid_first mode, project from sum(level_dims) → D_group
                # so that the projector (created with input_dim=D_group) accepts it.
                if rt_projections is not None and fc in rt_projections:
                    enc = rt_projections[fc](enc)
                rt_per_group[fc] = enc
                group_idx += 1

        # ── Compute roundtrip loss for each group ──
        for family_cat in model.group_indices:
            if family_cat not in rt_per_group:
                continue

            family = family_cat.split('_', 1)[0]
            if family == 'theta':
                weight = self.theta_weight
            elif family == 'initial':
                weight = self.initial_weight
            elif family == 'temporal':
                weight = self.temporal_weight
            else:
                continue

            # Skip families with zero weight — avoids diluting the mean
            # with zero-valued losses when only some families are active.
            if weight == 0.0:
                continue

            cat_losses = self._compute_category_roundtrip_direct(
                model, tokens, rt_per_group[family_cat], family_cat, weight
            )
            losses.extend(cat_losses['losses'])
            metrics.update(cat_losses['metrics'])

        # Mean (not sum) over quantizers: makes the loss scale independent of
        # #quantizers, so the weight is interpretable as "per-quantizer CE importance".
        # Critical when exempt from EMA normalization — sum would inflate the raw
        # value to ~115 (190 quantizers × 0.6 CE each), requiring a tiny weight.
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        metrics['roundtrip/total'] = total_loss.item()
        metrics['roundtrip/n_quantizers'] = len(losses)

        return total_loss, metrics

    def _encode_initial(
        self,
        model: Any,
        u0_decoded: torch.Tensor,
        cached_manual_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode initial conditions using cached features.

        Args:
            model: VQ tokenizer model
            u0_decoded: Decoded initial conditions [B, C, H, W]
            cached_manual_features: Pre-extracted manual features [B, D] from dataset

        Returns:
            Encoded features [B, embedding_dim]

        Raises:
            ValueError: If InitialHybridEncoder requires cached features but none provided
        """
        from spinlock.tokens.encoders.initial import InitialHybridEncoder

        if isinstance(model.initial_encoder, InitialHybridEncoder):
            # Use cached manual features (same as training!)
            if cached_manual_features is None:
                raise ValueError(
                    "InitialHybridEncoder requires cached_manual_features for roundtrip loss. "
                    "These should be passed from the training batch (same features used during encoding)."
                )
            # Pass cached features + raw ICs (exactly as training does)
            return model.initial_encoder(cached_manual_features, u0_decoded)
        else:
            # CNN-only mode: only needs raw ICs
            return model.initial_encoder(u0_decoded)

    def _compute_category_roundtrip_direct(
        self,
        model: Any,
        tokens: Dict[str, torch.Tensor],
        cat_features_rt: torch.Tensor,
        family_cat: str,
        weight: float,
    ) -> Dict[str, Any]:
        """Compute roundtrip loss for a single category across all hierarchy levels.

        Uses cross-entropy over codebook distances: for each sample, compute
        squared distances to ALL codebook entries and use CE with the original
        token as the target. This directly optimizes for landing in the correct
        Voronoi cell (same token identity) rather than matching the exact
        embedding vector.

        Args:
            model: JointHierarchicalVQVAE instance
            tokens: Original token indices per quantizer key
            cat_features_rt: Re-encoded features for this group [B, D_group]
            family_cat: Group key like ``"temporal_group_0"``
            weight: Family-specific loss weight

        Returns:
            Dict with ``losses`` (list of weighted loss tensors) and ``metrics``
        """
        losses = []
        metrics = {}

        # Project to hierarchical latents
        projector = model.projectors[family_cat]
        latents_rt = projector(cat_features_rt)

        # Compute loss for each hierarchy level
        for level_idx, latent_rt in enumerate(latents_rt):
            quantizer_key = f"{family_cat}_L{level_idx}"
            if quantizer_key not in model.quantizers:
                continue
            quantizer = model.quantizers[quantizer_key]

            # Get codebook and target tokens based on quantizer type
            if isinstance(quantizer, FiniteScalarQuantizer):
                if quantizer_key not in tokens:
                    continue
                target_tokens = tokens[quantizer_key]
                # FSQ implicit codebook: enumerate all codes, compute CE in post-tanh space
                all_indices = torch.arange(quantizer.codebook_size, device=latent_rt.device)
                codebook = quantizer.indices_to_values(all_indices)  # [K, D] in [-1, 1]
                # Apply tanh to match FSQ's internal mapping (latent_rt is pre-tanh)
                latent_rt_bounded = torch.tanh(latent_rt)  # [B, D] in (-1, 1)
                dists = torch.cdist(
                    latent_rt_bounded.unsqueeze(0), codebook.unsqueeze(0)
                ).squeeze(0).pow(2)
                logits = -dists  # [B, K]
            else:
                target_tokens = tokens[quantizer_key]  # [B]
                # VQ learned codebook: CE over embedding distances
                codebook = quantizer.embedding.weight  # [K, D]
                dists = torch.cdist(
                    latent_rt.unsqueeze(0), codebook.unsqueeze(0)
                ).squeeze(0).pow(2)
                logits = -dists  # [B, K]

            loss = F.cross_entropy(logits, target_tokens)
            losses.append(weight * loss)

            # Track token match accuracy as a metric
            predicted_tokens = logits.argmax(dim=-1)
            accuracy = (predicted_tokens == target_tokens).float().mean().item()
            metrics[f'roundtrip/{quantizer_key}'] = loss.item()
            metrics[f'roundtrip_acc/{quantizer_key}'] = accuracy

        return {'losses': losses, 'metrics': metrics}


class VQVAELoss:
    """Combined loss function for VQ-VAE training.

    Computes weighted sum of 6 loss components:
    1. Reconstruction loss (MSE)
    2. VQ loss (from quantizers)
    3. Orthogonality loss
    4. Informativeness loss
    5. Topographic loss (optional)
    6. Roundtrip loss (optional)

    When ``normalize_loss_scales`` is enabled, each loss L_i is replaced by
    L_i / EMA(L_i) before weighting. This makes all losses roughly unit-scale,
    so the config weights reflect actual gradient ratios regardless of raw
    loss magnitudes.

    Args:
        config: Loss configuration with weights
    """

    _EMA_KEYS = ("reconstruction", "vq", "orthogonality", "informativeness",
                 "topographic", "roundtrip", "aux", "group_balance",
                 "gate_sparsity")

    def __init__(self, config: LossConfig, aux_config: Optional[AuxHeadConfig] = None):
        self.config = config
        self.aux_config = aux_config  # For temporal-only mode

        # EMA loss-scale normalization
        self._normalize = config.normalize_loss_scales
        self._ema_momentum = config.loss_scale_ema_momentum
        self._ema: Dict[str, float] = {k: 1.0 for k in self._EMA_KEYS}
        self._ema_exempt: set[str] = set(config.loss_scale_ema_exempt)

        # Create roundtrip loss if enabled
        self.roundtrip_loss = None
        if config.roundtrip is not None and config.roundtrip.enabled:
            self.roundtrip_loss = RoundtripConsistencyLoss(
                theta_weight=config.roundtrip.theta_weight,
                initial_weight=config.roundtrip.initial_weight,
                temporal_weight=config.roundtrip.temporal_weight,
            )

    def _scale_loss(self, name: str, raw_loss: torch.Tensor) -> torch.Tensor:
        """Normalize a loss component by its running EMA magnitude.

        When normalize_loss_scales is enabled, each loss L_i is replaced by
        L_i / EMA(L_i), so that weights reflect actual gradient ratios
        regardless of raw loss scale differences.

        Exempt losses (configured via ``loss_scale_ema_exempt``) are returned
        unchanged. This is necessary for losses where |∇L| is not proportional
        to L, such as sum-of-CEs (roundtrip) or near-zero converged metrics.

        Args:
            name: Loss component name (must be in _EMA_KEYS).
            raw_loss: Raw loss tensor (scalar).

        Returns:
            raw_loss / EMA if normalizing and not exempt, else raw_loss.
        """
        if not self._normalize or name in self._ema_exempt:
            return raw_loss
        raw_val = raw_loss.item()
        if raw_val == 0.0:
            return raw_loss
        # Update EMA (no grad — bookkeeping, not part of loss graph)
        m = self._ema_momentum
        self._ema[name] = m * self._ema[name] + (1.0 - m) * raw_val
        return raw_loss / max(self._ema[name], 1e-8)

    def __call__(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        vq_loss: torch.Tensor,
        category_embeddings: Dict[str, torch.Tensor],
        quantized_vectors: Optional[Dict[str, torch.Tensor]] = None,
        token_indices: Optional[Dict[str, torch.Tensor]] = None,
        codebooks: Optional[Dict[str, torch.Tensor]] = None,
        latent_vectors: Optional[Dict[str, torch.Tensor]] = None,
        model: Optional[Any] = None,
        tokens: Optional[Dict[str, torch.Tensor]] = None,
        decoded: Optional[Dict[str, torch.Tensor]] = None,
        initial_manual: Optional[torch.Tensor] = None,
        original_theta: Optional[torch.Tensor] = None,
        original_initial: Optional[torch.Tensor] = None,
        cnn_features: Optional[torch.Tensor] = None,
        num_groups: Optional[int] = None,
        gate_values: Optional[torch.Tensor] = None,
        trajectory_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total VQ-VAE loss.

        Args:
            original: Original encoded features [B, D]
            reconstructed: Reconstructed features [B, D]
            vq_loss: VQ commitment loss from quantizers (scalar)
            category_embeddings: Dict mapping category → embeddings [B, D_cat]
            quantized_vectors: Optional dict of quantized vectors per category
            token_indices: Optional dict of token indices per category
            codebooks: Optional dict of codebook tensors per category
            latent_vectors: Optional dict of pre-quantization latent vectors per category
            model: Optional model instance (needed for roundtrip loss)
            tokens: Optional original tokens (needed for roundtrip loss)
            decoded: Optional decoded values (needed for roundtrip loss)
            initial_manual: Optional manual features for initial encoder
            original_theta: Optional original θ params [B, param_dim] for inverse recon
            original_initial: Optional original ICs [B, C, H, W] for inverse recon
            cnn_features: Optional CNN output [B, T, D] for group balance loss
            num_groups: Number of groups for group balance loss
            trajectory_targets: Optional [B, K, C, H, W] keyframe targets for trajectory head

        Returns:
            Dict with loss components and metrics
        """
        # 1. Reconstruction loss
        recon_metrics = {}
        aux_loss = torch.tensor(0.0, device=original.device)
        aux_metrics = {}

        # Compute reconstruction losses from all available decode heads.
        # Inverse heads and aux heads can both be active simultaneously.
        has_inverse = decoded is not None and (
            "theta" in decoded or "initial" in decoded or "temporal" in decoded
        ) and not all(k.endswith("_aux") or k.endswith("_probe") or k == "trajectory_prototype" for k in decoded)
        has_aux = self.aux_config is not None and decoded and any(
            k in decoded for k in ("theta_aux", "initial_aux", "theta", "initial",
                                    "theta_probe", "initial_probe", "trajectory_prototype")
        )

        if has_inverse and (original_theta is not None or original_initial is not None):
            recon_loss, recon_metrics = compute_inverse_reconstruction_loss(
                decoded=decoded,
                original_theta=original_theta,
                original_initial=original_initial,
            )
        else:
            # Encoded-space reconstruction (when no inverse heads, or no GT available)
            recon_loss = compute_reconstruction_loss(
                original, reconstructed, normalize=self.config.normalize_reconstruction
            )

        if has_aux:
            aux_loss, aux_metrics = compute_aux_head_losses(
                decoded=decoded,
                theta_gt=original_theta,
                ic_gt=original_initial,
                aux_config=self.aux_config,
                trajectory_targets=trajectory_targets,
            )

        # 2. VQ loss (already computed by quantizers)
        # This includes commitment cost from VectorQuantizer

        # 3. Orthogonality loss
        ortho_loss = compute_orthogonality_loss(category_embeddings)

        # 4. Informativeness loss
        info_loss = compute_informativeness_loss(
            category_embeddings, mode=self.config.informativeness_mode
        )

        # 5. Topographic loss (optional)
        topo_loss = torch.tensor(0.0, device=original.device)
        topo_pre_corr = 0.0
        topo_post_corr = 0.0

        if self.config.topographic_weight > 0:
            if quantized_vectors and latent_vectors:
                # Aggregate all categories' features for topology computation
                # Concatenate along feature dimension to get full representation
                all_latent = []
                all_quantized = []

                for cat in sorted(quantized_vectors.keys()):
                    if cat in latent_vectors and cat in quantized_vectors:
                        all_latent.append(latent_vectors[cat])
                        all_quantized.append(quantized_vectors[cat])

                if all_latent and all_quantized:
                    # Concatenate to form full latent and quantized representations
                    full_latent = torch.cat(all_latent, dim=1)  # [B, total_latent_dim]
                    full_quantized = torch.cat(all_quantized, dim=1)  # [B, total_latent_dim]

                    # Compute topographic loss with PRE and POST correlations
                    topo_loss, topo_metrics = compute_topographic_loss(
                        original=original,
                        latent_vectors=full_latent,
                        quantized_vectors=full_quantized,
                        n_samples=self.config.topographic_n_samples,
                    )

                    topo_pre_corr = topo_metrics['topo_pre']
                    topo_post_corr = topo_metrics['topo_post']

        # 6. Roundtrip loss (optional, NEW!)
        roundtrip_loss = torch.tensor(0.0, device=original.device)
        roundtrip_metrics = {}
        if self.roundtrip_loss is not None:
            if decoded is not None and tokens is not None and model is not None:
                roundtrip_loss, roundtrip_metrics = self.roundtrip_loss(
                    model=model,
                    tokens=tokens,
                    decoded=decoded,
                    initial_manual=initial_manual,
                )
            else:
                # Roundtrip loss requires decoded values, tokens, and model
                roundtrip_metrics['roundtrip/total'] = 0.0

        # 7. Group balance loss (learned mode only)
        balance_loss = torch.tensor(0.0, device=original.device)
        if (
            self.config.group_balance_weight > 0
            and cnn_features is not None
            and num_groups is not None
        ):
            balance_loss = compute_group_balance_loss(cnn_features, num_groups)

        # 8. Gate sparsity loss (gated pyramid_first mode only)
        gate_sparsity_loss = torch.tensor(0.0, device=original.device)
        if self.config.gate_sparsity_weight > 0 and gate_values is not None:
            # L1 penalty on gate activations: mean(sigmoid(g_i))
            gate_sparsity_loss = gate_values.mean()

        # Weighted combination (with optional EMA loss-scale normalization)
        total_loss = (
            self.config.reconstruction_weight * self._scale_loss('reconstruction', recon_loss)
            + self._scale_loss('vq', vq_loss)
            + self.config.orthogonality_weight * self._scale_loss('orthogonality', ortho_loss)
            + self.config.informativeness_weight * self._scale_loss('informativeness', info_loss)
            + self.config.topographic_weight * self._scale_loss('topographic', topo_loss)
            + self.config.group_balance_weight * self._scale_loss('group_balance', balance_loss)
            + self.config.gate_sparsity_weight * self._scale_loss('gate_sparsity', gate_sparsity_loss)
            + (self.config.roundtrip.weight * self._scale_loss('roundtrip', roundtrip_loss)
               if self.config.roundtrip else 0.0)
            + self._scale_loss('aux', aux_loss)  # 0.0 when not temporal-only
        )

        result = {
            "total": total_loss,
            "reconstruction": recon_loss,
            "vq": vq_loss,
            "orthogonality": ortho_loss,
            "informativeness": info_loss,
            "topographic": topo_loss,
            "topo_pre": topo_pre_corr,
            "topo_post": topo_post_corr,
            "group_balance": balance_loss,
            "gate_sparsity": gate_sparsity_loss,
        }

        # Add per-family inverse reconstruction metrics
        result.update(recon_metrics)

        # Add aux head metrics if computed (temporal-only mode)
        result.update(aux_metrics)

        # Add roundtrip metrics if computed
        result.update(roundtrip_metrics)

        return result
