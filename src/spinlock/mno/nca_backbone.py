"""Neural CA backbone — inductive bias matched to cellular automata.

Mirrors the Lenia computation graph (simulator.py _step()) with learned components:
    state [B, C, H, W]
      ├── 1. Perception: K depthwise convolutions (↔ Lenia FFT convolution)
      ├── 2. Channel mixing: 1×1 conv (↔ Lenia coupling matrix)
      ├── 3. Growth function: per-pixel MLP + tanh (↔ Lenia growth G)
      └── 4. Euler update: state + scale * growth, clamp(0, 1)

~60K params (without conditioning) vs 145M for U-AFNO.
Shared weights across time — same single_step() applied at every timestep.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Any, List, Tuple

from spinlock.mno.base_backbone import BaseMNOBackbone


def _auto_perception_specs(spatial_dim: int) -> Tuple[List[int], List[int]]:
    """Multi-scale perception kernels covering half-grid receptive field.

    On a toroidal grid of size S, max meaningful distance = S//2.
    Uses geometrically spaced dilations (×3) on a base kernel_size=7.

    For spatial_dim=128:
        → kernel_sizes=[3, 7, 7, 7], dilations=[1, 1, 3, 9]
        → RFs = [3, 7, 19, 55]. After 2 NCA steps, effective RF > 128.

    Args:
        spatial_dim: Spatial resolution of the grid (e.g. 128).

    Returns:
        (kernel_sizes, dilations) — parallel lists for Conv2d construction.
    """
    target_rf = max(spatial_dim // 2, 7)
    kernel_sizes = [3, 7]
    dilations = [1, 1]
    base_k = 7
    d = 3
    while True:
        rf = d * (base_k - 1) + 1
        if rf > target_rf:
            break
        kernel_sizes.append(base_k)
        dilations.append(d)
        d *= 3
    return kernel_sizes, dilations


class NeuralCABackbone(BaseMNOBackbone):
    """Neural CA backbone — inductive bias matched to cellular automata.

    Mirrors the Lenia computation graph with learned components:
    depthwise perception → channel mixing → growth MLP → Euler update.

    Args:
        in_channels: Number of state channels (e.g. 3 for Lenia).
        out_channels: Number of output channels (typically == in_channels).
        hidden_channels: Width of the mixing layer.
        kernel_sizes: Depthwise convolution kernel sizes for multi-scale perception.
        dilations: Per-kernel dilation factors (default: all 1s).
        growth_hidden: Hidden width of the growth MLP.
        residual_scale: Euler step size (default 0.1, like MNOBackbone).
        clamp_output: Whether to clamp output to [0, 1] (Lenia state domain).
        clamp_leak: Gradient leak for soft clamping.
        padding_mode: Conv2d padding mode ("zeros", "circular", etc.).
        param_conditioning: Whether to condition on operator parameters θ.
        param_dim: Dimension of parameter vector θ.
        param_embed_dim: Dimension of parameter embedding.
        conditioning_mode: "film", "concat", or "both" (same as MNOBackbone).
        film_config: Optional FiLM configuration dict (unused, accepted for API compat).
        use_checkpointing: Gradient checkpointing for long rollouts.
        checkpoint_every: Checkpoint interval (unused — NCA checkpoints every step).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dilations: Optional[Tuple[int, ...]] = None,
        growth_hidden: int = 64,
        residual_scale: float = 0.1,
        clamp_output: bool = True,
        clamp_leak: float = 0.01,
        padding_mode: str = "zeros",
        # Parameter conditioning (same interface as MNOBackbone)
        param_conditioning: bool = False,
        param_dim: int = 14,
        param_embed_dim: int = 128,
        conditioning_mode: str = "film",
        film_config: Optional[Dict[str, Any]] = None,
        use_checkpointing: bool = True,
        checkpoint_every: int = 16,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self.conditioning_mode = conditioning_mode
        self.param_conditioning = param_conditioning
        self.use_checkpointing = use_checkpointing
        self.checkpoint_every = checkpoint_every
        self.residual_scale = residual_scale
        self.clamp_output = clamp_output
        self.clamp_leak = clamp_leak

        if dilations is None:
            dilations = tuple(1 for _ in kernel_sizes)

        K = len(kernel_sizes)
        perception_out = in_channels * K

        # 1. Perception bank: K dilated depthwise convolutions
        #    Each conv has groups=in_channels → separate filter per channel,
        #    matching Lenia's per-channel kernel structure.
        #    Multi-scale kernels with dilations cover large receptive fields
        #    without the parameter cost of huge dense kernels.
        #    padding = dilation * (kernel_size // 2) preserves spatial dims.
        self.perception = nn.ModuleList([
            nn.Conv2d(
                in_channels, in_channels, k,
                padding=d * (k // 2),
                dilation=d,
                groups=in_channels,
                padding_mode=padding_mode,
            )
            for k, d in zip(kernel_sizes, dilations)
        ])

        # 2. Channel mixing: pointwise 1×1 conv (↔ Lenia coupling matrix)
        self.mixing = nn.Sequential(
            nn.Conv2d(perception_out, hidden_channels, 1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            nn.GELU(),
        )

        # 3. Growth function: per-pixel MLP → softsign
        #    Stored as individual layers (not Sequential) for FiLM injection
        #    between GroupNorm and GELU.
        #    Softsign instead of tanh: both bound to [-1, 1], but softsign has
        #    polynomial gradient decay (1/(1+|x|)²) vs tanh's exponential
        #    (1 - tanh²(x) → 0 for |x| > 2). Over 32 BPTT steps, tanh causes
        #    float32 gradient underflow (0.02³² ≈ 4e-55 → exactly 0.0).
        self.growth_conv1 = nn.Conv2d(hidden_channels, growth_hidden, 1)
        self.growth_norm = nn.GroupNorm(min(8, growth_hidden), growth_hidden)
        self.growth_act = nn.GELU()
        self.growth_conv2 = nn.Conv2d(growth_hidden, out_channels, 1)
        self.growth_out = nn.Softsign()

        # Parameter conditioning (mirrors MNOBackbone pattern exactly)
        if param_conditioning:
            self.param_embedding = nn.Sequential(
                nn.Linear(param_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, param_embed_dim),
                nn.LayerNorm(param_embed_dim),
            )
            self.param_embed_dim = param_embed_dim
            # FiLM generators: conditioning → (gamma, beta) for mixing + growth
            self.film_mixing = nn.Linear(param_embed_dim, hidden_channels * 2)
            self.film_growth = nn.Linear(param_embed_dim, growth_hidden * 2)
        else:
            self.param_embedding = None
            self.param_embed_dim = None

    def single_step(
        self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CA update: perceive → mix → grow → Euler step.

        Args:
            x: Current state [B, C, H, W].
               May have extra channels if conditioning_mode="concat"/"both".
            conditioning: Optional FiLM embedding [B, param_embed_dim].

        Returns:
            Next state [B, out_channels, H, W].
        """
        # Extract base state (handles concat conditioning channels)
        base = x[:, :self._in_channels]

        # 1. Perception — multi-scale depthwise convolutions
        feats = torch.cat([conv(base) for conv in self.perception], dim=1)

        # 2. Channel mixing
        h = self.mixing(feats)
        if conditioning is not None and hasattr(self, "film_mixing"):
            gb = self.film_mixing(conditioning)  # [B, 2*hidden]
            gamma, beta = gb.chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        # 3. Growth (FiLM injected between GroupNorm and GELU)
        h = self.growth_conv1(h)
        h = self.growth_norm(h)
        if conditioning is not None and hasattr(self, "film_growth"):
            gb = self.film_growth(conditioning)
            gamma, beta = gb.chunk(2, dim=1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.growth_act(h)
        g = self.growth_out(self.growth_conv2(h))

        # 4. Euler update
        out = base + self.residual_scale * g
        if self.clamp_output:
            # Soft clamp: identity in [0,1], gradient=leak outside.
            # Hard clamp kills gradients at saturation → vanishing gnorm
            # over 32 chained BPTT steps. Leak keeps gradient alive.
            out_clamped = out.clamp(0.0, 1.0)
            out = out_clamped + self.clamp_leak * (out - out_clamped)
        return out

    def rollout(
        self,
        u0: torch.Tensor,
        steps: int = 64,
        return_all_steps: bool = True,
        num_realizations: int = 1,
        params: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive rollout with conditioning + checkpointing.

        Same API as MNOBackbone.rollout() for TruncatedBPTT compatibility.

        Args:
            u0: Initial condition [B, C, H, W].
            steps: Number of timesteps to generate.
            return_all_steps: If True, return full trajectory including u0.
            num_realizations: Number of independent realizations.
            params: Optional parameter vector θ [B, param_dim].
            tokens: Unused, accepted for API compatibility.

        Returns:
            Trajectory [B, T+1, C, H, W] or final state [B, C, H, W].
        """
        B, C, H, W = u0.shape

        # Prepare parameter embeddings
        param_embed = None
        param_spatial = None
        if self.param_conditioning:
            if params is None:
                raise ValueError("params required when param_conditioning=True")
            param_embed = self.param_embedding(params)
            if self.conditioning_mode in ("concat", "both"):
                param_spatial = param_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

        if num_realizations > 1:
            return self._rollout_multi_realization(
                u0, steps, return_all_steps, num_realizations,
                param_embed, param_spatial,
            )

        use_ckpt = (
            self.use_checkpointing and self.training and torch.is_grad_enabled()
        )

        if return_all_steps:
            trajectory: List[torch.Tensor] = [u0]

        x = u0

        for _ in range(steps):
            # Augment input for concat conditioning
            x_augmented = x
            if self.conditioning_mode in ("concat", "both") and param_spatial is not None:
                x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

            # Step with optional checkpointing
            if use_ckpt and return_all_steps:
                if self.conditioning_mode in ("film", "both"):
                    # Capture param_embed in closure for checkpoint
                    _pe = param_embed
                    x = checkpoint(
                        lambda x_aug: self.single_step(x_aug, conditioning=_pe),
                        x_augmented,
                        use_reentrant=False,
                    )
                else:
                    x = checkpoint(
                        self.single_step,
                        x_augmented,
                        use_reentrant=False,
                    )
            else:
                if self.conditioning_mode in ("film", "both"):
                    x = self.single_step(x_augmented, conditioning=param_embed)
                else:
                    x = self.single_step(x_augmented)

            if return_all_steps:
                trajectory.append(x)

        if return_all_steps:
            return torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]
        return x

    def _rollout_multi_realization(
        self,
        u0: torch.Tensor,
        steps: int,
        return_all_steps: bool,
        num_realizations: int,
        param_embed: Optional[torch.Tensor] = None,
        param_spatial: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate multiple independent realizations from same IC."""
        realizations = []

        for _ in range(num_realizations):
            if return_all_steps:
                trajectory: List[torch.Tensor] = [u0]

            x = u0
            for _ in range(steps):
                x_augmented = x
                if self.conditioning_mode in ("concat", "both") and param_spatial is not None:
                    x_augmented = torch.cat([x_augmented, param_spatial], dim=1)

                if self.conditioning_mode in ("film", "both"):
                    x = self.single_step(x_augmented, conditioning=param_embed)
                else:
                    x = self.single_step(x_augmented)

                if return_all_steps:
                    trajectory.append(x)

            if return_all_steps:
                realizations.append(torch.stack(trajectory, dim=1))
            else:
                realizations.append(x)

        if return_all_steps:
            return torch.stack(realizations, dim=1)  # [B, M, T+1, C, H, W]
        else:
            return torch.stack(realizations, dim=1)  # [B, M, C, H, W]

    def get_intermediate_features(
        self,
        x: torch.Tensor,
        extract_from: str = "bottleneck",
    ) -> Dict[str, torch.Tensor]:
        """Extract perception + mixing features for alignment losses.

        Args:
            x: Input state [B, C, H, W].
            extract_from: Feature level ("bottleneck", "perception", "all").

        Returns:
            Dict of feature tensors. "bottleneck" is the mixing output.
        """
        base = x[:, :self._in_channels]
        feats = torch.cat([conv(base) for conv in self.perception], dim=1)
        h = self.mixing(feats)
        return {"perception": feats, "mixing": h, "bottleneck": h}

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels
