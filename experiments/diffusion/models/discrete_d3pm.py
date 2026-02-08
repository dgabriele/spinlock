"""Discrete D3PM diffusion process for hierarchical dict tokens.

Implements discrete diffusion with:
- Per-category-level transition matrices Q_t (variable vocab sizes)
- Forward process: Add categorical noise to tokens
- Reverse process: Iterative denoising with RePaint-style inpainting
- Flexible noise schedules (linear, cosine, sqrt)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Noise schedule types for diffusion process."""
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"


@dataclass
class DiffusionSchedule:
    """Configuration for diffusion noise schedule.

    Args:
        num_timesteps: Total diffusion steps T
        beta_start: Initial noise level
        beta_end: Final noise level
        schedule_type: Type of schedule (linear, cosine, sqrt)
    """
    num_timesteps: int = 50
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = "cosine"


class DiscreteD3PM(nn.Module):
    """Discrete D3PM diffusion for hierarchical dict tokens.

    Manages the diffusion process for VQTokenizer v2 dict format:
    - Tokens: Dict[str, Tensor[B]] where keys are "family_category_Ll"
    - Per-category-level transition matrices handle variable vocab sizes
    - RePaint-style inpainting keeps observed tokens fixed

    Architecture:
        1. Build noise schedule β_t for t=0..T-1
        2. Compute transition matrices Q_t per category-level
        3. Forward process: x_0 → x_t via Q_t sampling
        4. Reverse process: x_t → x_{t-1} via denoiser + RePaint

    Args:
        vocab_sizes: Dict mapping "family_category_Ll" → vocab_size
        schedule: Diffusion schedule configuration
        category_level_info: Dict mapping key → {family, category, level}

    Example:
        >>> vocab_sizes = {
        ...     "temporal_group_1_L0": 28,
        ...     "temporal_group_1_L1": 14,
        ...     "initial_group_1_L0": 20
        ... }
        >>> diffusion = DiscreteD3PM(vocab_sizes, DiffusionSchedule())
        >>> noisy_dict = diffusion.forward_process(tokens_dict, t=25)
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        schedule: DiffusionSchedule,
        category_level_info: Optional[Dict[str, Dict[str, any]]] = None,
    ):
        super().__init__()

        self.vocab_sizes = vocab_sizes
        self.schedule = schedule
        self.category_level_info = category_level_info or {}

        # Build noise schedule
        self.register_buffer("betas", self._build_schedule())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

        # Build per-category-level transition matrices
        self._build_transition_matrices()

        logger.info(
            f"DiscreteD3PM initialized: T={schedule.num_timesteps}, "
            f"schedule={schedule.schedule_type}, "
            f"categories={len(vocab_sizes)}"
        )

    def _build_schedule(self) -> torch.Tensor:
        """Build noise schedule β_t for t=0..T-1.

        Returns:
            Tensor[T] with β_t values
        """
        T = self.schedule.num_timesteps

        if self.schedule.schedule_type == ScheduleType.LINEAR:
            # Linear interpolation from beta_start to beta_end
            betas = torch.linspace(
                self.schedule.beta_start,
                self.schedule.beta_end,
                T
            )

        elif self.schedule.schedule_type == ScheduleType.COSINE:
            # Cosine schedule (Nichol & Dhariwal, 2021)
            s = 0.008  # Offset to prevent β_t=0 at t=0
            steps = torch.arange(T + 1, dtype=torch.float32)
            alphas_cumprod = torch.cos(
                ((steps / T) + s) / (1 + s) * torch.pi / 2
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)

        elif self.schedule.schedule_type == ScheduleType.SQRT:
            # Square root schedule
            steps = torch.arange(T, dtype=torch.float32)
            betas = torch.sqrt(
                torch.linspace(
                    self.schedule.beta_start ** 2,
                    self.schedule.beta_end ** 2,
                    T
                )
            )

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule.schedule_type}")

        return betas

    def _build_transition_matrices(self):
        """Build per-category-level transition matrices Q_t.

        Each Q_t[k] is a [T, vocab_size, vocab_size] matrix where:
        - Q_t[k][i, j] = P(x_t = j | x_{t-1} = i) at timestep t
        - Absorbing state model: Q_t = (1 - β_t) * I + β_t * U
        - U = uniform distribution over vocab

        Stored as nn.ParameterDict for per-category-level handling.
        """
        self.transition_matrices = nn.ParameterDict()

        for key, vocab_size in self.vocab_sizes.items():
            # Create identity and uniform matrices
            I = torch.eye(vocab_size)  # [V, V]
            U = torch.ones(vocab_size, vocab_size) / vocab_size  # [V, V]

            # Q_t = (1 - β_t) * I + β_t * U for each timestep
            # Shape: [T, V, V]
            Q_t = []
            for t in range(self.schedule.num_timesteps):
                beta_t = self.betas[t]
                Q = (1 - beta_t) * I + beta_t * U
                Q_t.append(Q)

            Q_t = torch.stack(Q_t, dim=0)  # [T, V, V]

            # Register as non-trainable parameter (frozen transition matrices)
            self.transition_matrices[key] = nn.Parameter(Q_t, requires_grad=False)

    def forward_process(
        self,
        tokens_dict: Dict[str, torch.Tensor],
        t: int,
        mask_dict: Optional[Dict[str, torch.BoolTensor]] = None,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Add noise to tokens via forward diffusion.

        Uses efficient cumulative product: q(x_t | x_0) without iterating.

        Args:
            tokens_dict: Dict mapping key → token indices [B]
            t: Timestep (0 to T-1)
            mask_dict: Optional dict of masks [B] (True = apply noise, False = keep)

        Returns:
            Tuple of (noisy_tokens_dict, log_probs_dict):
            - noisy_tokens_dict: Dict mapping key → noisy token indices [B]
            - log_probs_dict: Dict mapping key → log P(x_t | x_0) [B, V]
        """
        if t < 0 or t >= self.schedule.num_timesteps:
            raise ValueError(f"Invalid timestep t={t} (must be 0 to {self.schedule.num_timesteps - 1})")

        noisy_dict = {}
        log_probs_dict = {}

        for key, tokens in tokens_dict.items():
            B = tokens.shape[0]
            V = self.vocab_sizes[key]

            # Get transition matrix Q_t for this category-level
            Q_t = self.transition_matrices[key][t]  # [V, V]

            # Convert tokens to one-hot [B, V]
            x_0_onehot = F.one_hot(tokens.long(), num_classes=V).float()

            # Apply transition: q(x_t | x_0) = x_0 @ Q_t
            # x_0_onehot: [B, V], Q_t: [V, V] → probs: [B, V]
            probs = torch.matmul(x_0_onehot, Q_t)

            # Sample noisy tokens
            noisy_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # Apply mask if provided (keep original tokens where mask=False)
            if mask_dict is not None and key in mask_dict:
                mask = mask_dict[key]
                noisy_tokens = torch.where(mask, noisy_tokens, tokens)

            # Store results
            noisy_dict[key] = noisy_tokens
            log_probs_dict[key] = torch.log(probs + 1e-8)  # [B, V]

        return noisy_dict, log_probs_dict

    def reverse_step(
        self,
        x_t: Dict[str, torch.Tensor],
        predicted_logits: Dict[str, torch.Tensor],
        t: int,
        observed_dict: Optional[Dict[str, torch.BoolTensor]] = None,
        x_0_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single reverse diffusion step with RePaint inpainting.

        Denoise x_t → x_{t-1} using predicted logits, then re-inject
        observed tokens via forward process to maintain consistency.

        Args:
            x_t: Current noisy tokens dict [B]
            predicted_logits: Denoiser predictions dict [B, V]
            t: Current timestep (t → t-1)
            observed_dict: Optional dict of observed masks [B] (True = observed)
            x_0_dict: Optional clean tokens dict [B] (for RePaint)

        Returns:
            Dict mapping key → denoised tokens x_{t-1} [B]
        """
        if t == 0:
            # Final step: return predicted tokens directly
            return {
                key: torch.argmax(logits, dim=-1)
                for key, logits in predicted_logits.items()
            }

        x_prev = {}

        for key, logits in predicted_logits.items():
            B = logits.shape[0]
            V = self.vocab_sizes[key]

            # Sample from predicted distribution
            probs = F.softmax(logits, dim=-1)  # [B, V]
            x_t_minus_1 = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

            # RePaint: Re-inject observed tokens via forward process
            if observed_dict is not None and x_0_dict is not None and key in observed_dict:
                observed_mask = observed_dict[key]  # [B]
                x_0 = x_0_dict[key]  # [B]

                # Re-noise observed tokens to t-1
                noisy_observed, _ = self.forward_process(
                    {key: x_0},
                    t=t - 1,
                    mask_dict={key: torch.ones_like(observed_mask)}  # Always noise
                )
                x_0_at_t_minus_1 = noisy_observed[key]

                # Replace observed positions with re-noised tokens
                x_t_minus_1 = torch.where(observed_mask, x_0_at_t_minus_1, x_t_minus_1)

            x_prev[key] = x_t_minus_1

        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        observed_dict: Optional[Dict[str, torch.BoolTensor]] = None,
        x_0_dict: Optional[Dict[str, torch.Tensor]] = None,
        denoising_network: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """Full sampling loop T → 0 with optional inpainting.

        Args:
            batch_size: Number of samples to generate
            observed_dict: Optional dict of observed masks [B] (for inpainting)
            x_0_dict: Optional clean tokens dict [B] (for inpainting)
            denoising_network: Trained denoising network
            device: Device for computation

        Returns:
            Dict mapping key → generated tokens [B]
        """
        if denoising_network is None:
            raise ValueError("denoising_network is required for sampling")

        denoising_network.eval()

        # Initialize with uniform random tokens
        x_t = {}
        for key, vocab_size in self.vocab_sizes.items():
            x_t[key] = torch.randint(
                0, vocab_size, (batch_size,), device=device
            )

        # Iterative denoising T → 0
        for t in reversed(range(self.schedule.num_timesteps)):
            # Predict clean tokens
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_logits = denoising_network(
                x_t,
                t_tensor,
                observed_dict=observed_dict
            )

            # Reverse step with RePaint
            x_t = self.reverse_step(
                x_t,
                predicted_logits,
                t,
                observed_dict=observed_dict,
                x_0_dict=x_0_dict,
            )

        return x_t

    def get_timestep_weights(self) -> torch.Tensor:
        """Get per-timestep loss weights for training.

        Weight by signal-to-noise ratio: w_t = 1 / β_t

        Returns:
            Tensor[T] with weights for each timestep
        """
        # Simple inverse beta weighting
        weights = 1.0 / self.betas

        # Normalize to mean 1.0
        weights = weights / weights.mean()

        return weights
