"""Loss computation for NLTokenizer training.

Key design principles:
1. **Behavioral equivalence**: Inverse losses compare in *encoding space*,
   not parameter space. ‖encoder(θ_hat) - encoder(θ_true)‖² handles the
   many-to-many mapping from (theta, IC) → behavior.
2. **Topographic pressure**: Pearson correlation between pairwise distances
   in h-space (behavioral) and z-space (latent). Ensures z-space topology
   reflects behavioral similarity, enabling smooth interpolation.
3. **Listener roundtrip**: NL fidelity via ‖z - listener(NL(z))‖².

Loss components:
    1. Feature reconstruction: ‖h - ĥ‖²
    2. KL divergence: KL(q(z|x) ‖ N(0,I)) with free-bits
    3. Behavioral theta inverse: ‖encoder(θ_hat) - encoder(θ_true)‖²
    4. IC inverse: ‖IC - IC_hat‖² (when applicable)
    5. Listener roundtrip: ‖z - listener(generator(z).token_probs)‖²
    6. Topographic: 1 - ρ(d_h, d_z) over sampled pairs
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional

from .nl_config import NLLossConfig


class NLTokenizerLoss:
    """Computes all NLTokenizer loss components.

    Args:
        config: NLLossConfig with per-component weights
    """

    def __init__(self, config: NLLossConfig):
        self.config = config
        self._kl_weight_scale = 0.0  # Ramps during warmup

    def set_kl_weight_scale(self, scale: float):
        """Set KL weight multiplier (0→1 during warmup)."""
        self._kl_weight_scale = max(0.0, min(1.0, scale))

    def __call__(
        self,
        *,
        h: torch.Tensor,
        h_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        theta_encoded: Optional[torch.Tensor] = None,
        theta_hat_encoded: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        z_hat: Optional[torch.Tensor] = None,
        listener_enabled: bool = False,
    ) -> Dict[str, Any]:
        """Compute all loss components.

        Args:
            h: [B, D] original concatenated family embeddings
            h_hat: [B, D] reconstructed family embeddings
            mu: [B, latent_dim] VAE mean
            logvar: [B, latent_dim] VAE log-variance
            theta_encoded: [B, theta_dim] ground truth theta *encoding*
                (not raw params — the encoder output, detached)
            theta_hat_encoded: [B, theta_dim] re-encoded predicted theta
                (encoder(theta_inverse(z)) — gradients flow through inverse)
            z: [B, z_full_dim] full latent vector
            z_hat: [B, latent_dim] listener-predicted latent (optional)
            listener_enabled: Whether to include listener roundtrip loss

        Returns:
            Dict with 'loss' (total) and per-component scalars.
        """
        cfg = self.config
        components: Dict[str, torch.Tensor] = {}
        device = h.device

        # ── 1. Feature reconstruction ──
        recon_loss = F.mse_loss(h_hat, h)
        components["reconstruction"] = recon_loss

        # ── 2. KL divergence with free-bits ──
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        if cfg.kl_free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=cfg.kl_free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()
        components["kl"] = kl_loss

        # ── 3. Behavioral theta inverse ──
        # Compare in ENCODING space, not parameter space.
        # Two different θ that produce the same encoding → zero loss.
        theta_inv_loss = torch.tensor(0.0, device=device)
        if theta_encoded is not None and theta_hat_encoded is not None:
            theta_inv_loss = F.mse_loss(theta_hat_encoded, theta_encoded.detach())
        components["theta_inverse"] = theta_inv_loss

        # ── 4. Listener roundtrip ──
        listener_loss = torch.tensor(0.0, device=device)
        if listener_enabled and z is not None and z_hat is not None:
            listener_loss = F.mse_loss(z_hat, z.detach())
        components["listener_roundtrip"] = listener_loss

        # ── 5. Topographic: preserve behavioral neighborhoods in z ──
        topo_loss = torch.tensor(0.0, device=device)
        if cfg.topographic_weight > 0 and z is not None:
            topo_loss = self._topographic_loss(h, z)
        components["topographic"] = topo_loss

        # ── Total ──
        total = (
            cfg.reconstruction_weight * recon_loss
            + cfg.kl_weight * self._kl_weight_scale * kl_loss
            + cfg.theta_inverse_weight * theta_inv_loss
            + cfg.topographic_weight * topo_loss
        )
        if listener_enabled:
            total = total + cfg.listener_roundtrip_weight * listener_loss

        components["loss"] = total
        return components

    def _topographic_loss(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Topographic similarity: Pearson correlation of pairwise distances.

        Encourages the z-space topology to mirror h-space (behavioral)
        topology. Uses the same Pearson correlation approach as VQTokenizer's
        topographic loss, but without the pre/post quantization split
        (continuous z has no quantization boundary).

        Args:
            h: [B, D_h] behavioral embeddings (concatenated family encodings)
            z: [B, D_z] VAE latent vectors

        Returns:
            Scalar loss = 1 - ρ(d_h, d_z), in [0, 2].
        """
        B = h.shape[0]
        n = min(B, self.config.topographic_n_samples)
        if n < 4:
            return torch.tensor(0.0, device=h.device)

        # Sample indices
        idx = torch.randperm(B, device=h.device)[:n]
        h_sample = h[idx]
        z_sample = z[idx]

        # Pairwise L2 distances
        d_h = torch.cdist(h_sample, h_sample, p=2)  # [n, n]
        d_z = torch.cdist(z_sample, z_sample, p=2)  # [n, n]

        # Flatten upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones(n, n, device=h.device, dtype=torch.bool), diagonal=1)
        d_h_flat = d_h[mask]
        d_z_flat = d_z[mask]

        # Pearson correlation
        corr = self._pearson_correlation(d_h_flat, d_z_flat)

        # Loss = 1 - correlation (want to maximize correlation)
        return 1.0 - corr

    @staticmethod
    def _pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pearson correlation between two 1D tensors."""
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        return (
            (a_centered * b_centered).sum()
            / (a_centered.norm() * b_centered.norm() + 1e-8)
        )
