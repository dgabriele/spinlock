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

from .config import LossConfig, RoundtripLossConfig

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
) -> torch.Tensor:
    """Compute informativeness loss to encourage high variance within categories.

    Penalizes low variance in category embeddings to prevent collapse
    to constant representations.

    Args:
        category_embeddings: Dict mapping category_name → embeddings [B, D_cat]
        min_variance: Minimum target variance threshold

    Returns:
        Scalar informativeness loss (lower variance = higher loss)
    """
    variances = []

    for cat_name, embeddings in category_embeddings.items():
        # Compute variance along batch dimension for each feature
        var = embeddings.var(dim=0, unbiased=False).mean()  # Average across features
        variances.append(var)

    if not variances:
        return torch.tensor(0.0, device=next(iter(category_embeddings.values())).device)

    variances = torch.stack(variances)

    # Penalize variance below threshold (ReLU ensures only low variance is penalized)
    loss = F.relu(min_variance - variances).mean()

    return loss


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
    pre_loss = 1.0 - pre_correlation
    post_loss = 1.0 - post_correlation
    total_loss = (pre_loss + post_loss) / 2.0

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
    """

    def __init__(
        self,
        theta_weight: float = 1.0,
        initial_weight: float = 1.0,
    ):
        super().__init__()
        self.theta_weight = theta_weight
        self.initial_weight = initial_weight

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
            decoded: Decoded continuous values
            initial_manual: Manual features for initial encoder (if needed)

        Returns:
            (total_loss, metrics_dict)
        """
        losses = []
        metrics = {}

        if 'theta' in decoded:
            theta_losses = self._compute_theta_roundtrip(model, tokens, decoded['theta'])
            losses.extend(theta_losses['losses'])
            metrics.update(theta_losses['metrics'])

        if 'initial' in decoded:
            initial_losses = self._compute_initial_roundtrip(model, tokens, decoded['initial'])
            losses.extend(initial_losses['losses'])
            metrics.update(initial_losses['metrics'])

        device = next(iter(decoded.values())).device
        total_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        metrics['roundtrip/total'] = total_loss.item()

        return total_loss, metrics

    def _compute_theta_roundtrip(
        self,
        model: Any,
        tokens: Dict[str, torch.Tensor],
        theta_decoded: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute roundtrip loss for theta family."""
        losses = []
        metrics = {}

        # Re-encode decoded theta
        theta_encoded_rt = model.theta_encoder(theta_decoded)

        # Compute loss for each theta category
        for family_cat, indices in model.group_indices.items():
            if not family_cat.startswith('theta_'):
                continue

            cat_losses = self._compute_category_roundtrip(
                model, tokens, theta_encoded_rt, family_cat, indices, self.theta_weight
            )
            losses.extend(cat_losses['losses'])
            metrics.update(cat_losses['metrics'])

        return {'losses': losses, 'metrics': metrics}

    def _compute_initial_roundtrip(
        self,
        model: Any,
        tokens: Dict[str, torch.Tensor],
        u0_decoded: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute roundtrip loss for initial family."""
        losses = []
        metrics = {}

        # Re-encode decoded initial conditions
        initial_encoded_rt = self._encode_initial(model, u0_decoded)

        # Compute loss for each initial category
        for family_cat, indices in model.group_indices.items():
            if not family_cat.startswith('initial_'):
                continue

            cat_losses = self._compute_category_roundtrip(
                model, tokens, initial_encoded_rt, family_cat, indices, self.initial_weight
            )
            losses.extend(cat_losses['losses'])
            metrics.update(cat_losses['metrics'])

        return {'losses': losses, 'metrics': metrics}

    def _encode_initial(self, model: Any, u0_decoded: torch.Tensor) -> torch.Tensor:
        """Encode initial conditions (handle hybrid vs CNN-only modes)."""
        # Check if this is a hybrid encoder (has both manual and CNN)
        from spinlock.tokens.encoders.initial import InitialHybridEncoder

        if isinstance(model.initial_encoder, InitialHybridEncoder):
            # Hybrid mode: need to extract manual features
            from spinlock.features.initial.manual_extractors import InitialManualExtractor
            extractor = InitialManualExtractor(device=u0_decoded.device)
            u0_expanded = u0_decoded.unsqueeze(1)  # [B,3,64,64] → [B,1,3,64,64]
            manual_features = extractor.extract_all(u0_expanded).squeeze(1)  # [B, 42]
            # InitialHybridEncoder.forward(manual_features, raw_ics)
            return model.initial_encoder(manual_features, u0_decoded)
        else:
            # CNN-only mode: InitialCNNEncoder.forward(raw_ics)
            return model.initial_encoder(u0_decoded)

    def _compute_category_roundtrip(
        self,
        model: Any,
        tokens: Dict[str, torch.Tensor],
        encoded_rt: torch.Tensor,
        family_cat: str,
        indices: List[int],
        weight: float,
    ) -> Dict[str, Any]:
        """Compute roundtrip loss for a single category across all levels."""
        losses = []
        metrics = {}

        # Extract category features and project to hierarchical latents
        cat_features_rt = encoded_rt[:, indices]
        projector = model.projectors[family_cat]
        latents_rt = projector(cat_features_rt)

        # Compute loss for each hierarchy level
        for level_idx, latent_rt in enumerate(latents_rt):
            quantizer_key = f"{family_cat}_L{level_idx}"
            quantizer = model.quantizers[quantizer_key]

            # Target: embeddings of original tokens
            target_tokens = tokens[quantizer_key]
            target_embeddings = quantizer.embedding(target_tokens)

            # MSE loss between roundtrip latents and target embeddings
            loss = F.mse_loss(latent_rt, target_embeddings)
            losses.append(weight * loss)
            metrics[f'roundtrip/{quantizer_key}'] = loss.item()

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

    Args:
        config: Loss configuration with weights
    """

    def __init__(self, config: LossConfig):
        self.config = config

        # Create roundtrip loss if enabled
        self.roundtrip_loss = None
        if config.roundtrip is not None and config.roundtrip.enabled:
            self.roundtrip_loss = RoundtripConsistencyLoss(
                theta_weight=config.roundtrip.theta_weight,
                initial_weight=config.roundtrip.initial_weight,
            )

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

        Returns:
            Dict with keys:
                - total: Total weighted loss
                - reconstruction: Reconstruction loss component
                - vq: VQ loss component
                - orthogonality: Orthogonality loss component
                - informativeness: Informativeness loss component
                - topographic: Topographic loss component (if enabled)
                - topo_pre: Pre-quantization correlation (if topographic enabled)
                - topo_post: Post-quantization correlation (if topographic enabled)
                - roundtrip/total: Roundtrip loss component (if enabled)
                - roundtrip/*: Per-quantizer roundtrip losses (if enabled)
        """
        # 1. Reconstruction loss
        recon_loss = compute_reconstruction_loss(
            original, reconstructed, normalize=self.config.normalize_reconstruction
        )

        # 2. VQ loss (already computed by quantizers)
        # This includes commitment cost from VectorQuantizer

        # 3. Orthogonality loss
        ortho_loss = compute_orthogonality_loss(category_embeddings)

        # 4. Informativeness loss
        info_loss = compute_informativeness_loss(category_embeddings)

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
                        n_samples=64,
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

        # Weighted combination
        total_loss = (
            self.config.reconstruction_weight * recon_loss
            + vq_loss  # VQ loss already weighted by commitment_cost in quantizer
            + self.config.orthogonality_weight * ortho_loss
            + self.config.informativeness_weight * info_loss
            + self.config.topographic_weight * topo_loss
            + (self.config.roundtrip.weight * roundtrip_loss if self.config.roundtrip else 0.0)
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
        }

        # Add roundtrip metrics if computed
        result.update(roundtrip_metrics)

        return result
